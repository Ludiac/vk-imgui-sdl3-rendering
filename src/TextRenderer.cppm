module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:TextRenderer;

import vulkan_hpp;
import std;

import :VulkanDevice;
import :VulkanPipeline;
import :VMA;
import :texture;
import :text; // For FontAtlasData

// The vertex layout for the single static quad.
struct TextQuadVertex {
  glm::vec2 pos;
  glm::vec2 uv;
};

// This struct defines the unique data for each character instance.
// It will be sent to the shader via a storage buffer.
struct TextInstanceData {
  glm::vec2 screenPos; // Top-left position of the quad
  glm::vec2 scale;     // width and height of the quad
  glm::vec2 uvTopLeft;
  glm::vec2 uvBottomRight;
  glm::vec4 color; // NEW: Color is now per-instance
};

export struct TextPushConstants {
  glm::mat4 projection;
};

export struct Sheet {
  glm::vec2 position{0.0f};                          // Top-left corner in screen pixels
  glm::vec2 size{100.0f, 100.0f};                    // Width and height in pixels
  glm::vec4 backgroundColor{0.1f, 0.1f, 0.1f, 0.8f}; // RGBA for the sheet's background

  // Margins (padding) from the edges of the sheet
  float marginTop = 5.0f;
  float marginRight = 5.0f;
  float marginBottom = 5.0f;
  float marginLeft = 5.0f;
};

// Represents a block of text to be rendered within a Sheet.
export struct TextBlock {
  std::string text;
  glm::vec4 color{1.0f, 1.0f, 1.0f, 1.0f}; // Default to white text
  float fontSize = 48.0f;                  // This is illustrative; actual size is from FontAtlas
  // Future properties could include alignment (left, center, right)
};

export class TextRenderer {
private:
  VulkanDevice &device;
  const FontAtlasData &fontAtlasData;
  u32 frameCount; // Number of frames in flight

  std::shared_ptr<Texture> fontAtlasTexture;
  // One descriptor set for the font atlas texture, shared across all frames.
  vk::raii::DescriptorSet atlasTextureDescriptorSet{nullptr};

  // NEW: A descriptor set for the instance data buffer, one per frame.
  std::vector<vk::raii::DescriptorSet> instanceDataDescriptorSets;
  vk::raii::DescriptorSetLayout instanceDataLayout{nullptr}; // Layout for the instance SSBO

  // NEW: A single, static, device-local buffer for the quad mesh.
  VmaBuffer staticVertexBuffer;
  VmaBuffer staticIndexBuffer; // NEW: Added an index buffer for the quad

  // NEW: A vector of dynamic buffers for instance data, one for each frame-in-flight.
  std::vector<VmaBuffer> instanceBuffers;
  u32 maxQuads; // Max characters per batch

  std::vector<TextInstanceData> queuedInstances;

public:
  TextRenderer(VulkanDevice &dev, const FontAtlasData &atlasData, u32 inFlightFrameCount,
               const vk::raii::DescriptorSetLayout &textureLayout,
               const vk::raii::DescriptorPool &pool, const vk::raii::Queue &transferQueue)
      : device(dev), fontAtlasData(atlasData), frameCount(inFlightFrameCount),
        maxQuads(2048) // Max 2048 chars per draw call
  {
    EXPECTED_VOID(createAtlasTexture(transferQueue));
    EXPECTED_VOID(createInstanceBuffers());
    EXPECTED_VOID(createInstanceDataDescriptorSetLayout());
    EXPECTED_VOID(allocateDescriptorSets(textureLayout, pool));
    EXPECTED_VOID(createStaticQuadBuffers(transferQueue));
  }

  void queueTextBlock(const Sheet &sheet, const TextBlock &textBlock) {
    float startX = sheet.position.x + sheet.marginLeft;
    float startY = sheet.position.y + sheet.marginTop + textBlock.fontSize;
    queueText(textBlock.text, startX, startY, textBlock.color);
  }

  void queueText(const std::string &text, float x, float y, const glm::vec4 &color) {
    if (text.empty()) {
      return;
    }

    float cursorX = x;
    float baselineY = y;

    for (char c : text) {
      if (queuedInstances.size() >= maxQuads)
        break;

      const GlyphInfo &gi =
          fontAtlasData.glyphs.count(c) ? fontAtlasData.glyphs.at(c) : fontAtlasData.glyphs.at('?');

      if (gi.width > 0 && gi.height > 0) {
        float xpos = cursorX + gi.bearing_x;
        float ypos = baselineY + gi.bearing_y - gi.height;

        // Add the color directly to the instance data
        queuedInstances.emplace_back(TextInstanceData{.screenPos = {xpos, ypos},
                                                      .scale = {gi.width, gi.height},
                                                      .uvTopLeft = {gi.uv_x0, gi.uv_y1},
                                                      .uvBottomRight = {gi.uv_x1, gi.uv_y0},
                                                      .color = color});
      }
      cursorX += gi.advance;
    }
  }

  void draw(const vk::raii::CommandBuffer &cmd, const VulkanPipeline &pipeline,
            vk::Extent2D windowSize, u32 frameIndex) {
    if (queuedInstances.empty()) {
      return;
    }

    // 1. Copy ALL instance data for the frame to the GPU at once
    VmaBuffer &currentInstanceBuffer = instanceBuffers[frameIndex];
    size_t dataSize = queuedInstances.size() * sizeof(TextInstanceData);
    std::memcpy(currentInstanceBuffer.getMappedData(), queuedInstances.data(), dataSize);

    // 2. Bind pipeline and static buffers once
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.pipeline);
    cmd.bindVertexBuffers(0, {staticVertexBuffer.get()}, {0});
    cmd.bindIndexBuffer(staticIndexBuffer.get(), 0, vk::IndexType::eUint32);

    std::array<vk::DescriptorSet, 2> descriptorSets = {*atlasTextureDescriptorSet,
                                                       *instanceDataDescriptorSets[frameIndex]};
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.pipelineLayout, 0,
                           descriptorSets, {});

    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(windowSize.width),
                                 static_cast<float>(windowSize.height), 0.0f);

    // 3. Push the projection matrix once
    TextPushConstants constants{.projection = ortho};
    cmd.pushConstants<TextPushConstants>(*pipeline.pipelineLayout, vk::ShaderStageFlagBits::eVertex,
                                         0, constants);

    // 4. Issue a SINGLE draw call for all queued characters
    cmd.drawIndexed(6,                      // indexCount
                    queuedInstances.size(), // instanceCount
                    0,                      // firstIndex
                    0,                      // vertexOffset
                    0                       // firstInstance
    );

    queuedInstances.clear();
  }

  std::shared_ptr<Texture> getFontTexture() { return fontAtlasTexture; }

private:
  [[nodiscard]] std::expected<void, std::string> createInstanceDataDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding instanceBinding{.binding = 0,
                                                   .descriptorType =
                                                       vk::DescriptorType::eStorageBuffer,
                                                   .descriptorCount = 1,
                                                   .stageFlags = vk::ShaderStageFlagBits::eVertex};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = 1, .pBindings = &instanceBinding};

    auto layoutResult = device.logical().createDescriptorSetLayout(layoutInfo);
    if (!layoutResult) {
      return std::unexpected("Failed to create text instance data descriptor set layout.");
    }
    instanceDataLayout = std::move(layoutResult.value());
    return {};
  }

  [[nodiscard]] std::expected<void, std::string>
  allocateDescriptorSets(const vk::raii::DescriptorSetLayout &textureLayout,
                         const vk::raii::DescriptorPool &pool) {
    // --- Allocate descriptor set for the TEXTURE ---
    vk::DescriptorSetAllocateInfo textureAllocInfo{
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &*textureLayout};
    auto texSetResult = device.logical().allocateDescriptorSets(textureAllocInfo);
    if (!texSetResult) {
      return std::unexpected("Failed to allocate text texture descriptor set.");
    }
    atlasTextureDescriptorSet = std::move(texSetResult.value().front());

    // Update the texture descriptor set
    vk::DescriptorImageInfo imageInfo{.sampler = *fontAtlasTexture->sampler,
                                      .imageView = *fontAtlasTexture->view,
                                      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet write{.dstSet = *atlasTextureDescriptorSet,
                                 .dstBinding = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                 .pImageInfo = &imageInfo};
    device.logical().updateDescriptorSets({write}, nullptr);

    // --- Allocate descriptor sets for the INSTANCE DATA (one per frame) ---
    std::vector<vk::DescriptorSetLayout> layouts(frameCount, *instanceDataLayout);
    vk::DescriptorSetAllocateInfo instanceAllocInfo{
        .descriptorPool = pool, .descriptorSetCount = frameCount, .pSetLayouts = layouts.data()};

    auto instanceSetResult = device.logical().allocateDescriptorSets(instanceAllocInfo);
    if (!instanceSetResult) {
      return std::unexpected("Failed to allocate text instance descriptor sets.");
    }
    instanceDataDescriptorSets = std::move(instanceSetResult.value());

    // Update each instance descriptor set to point to its corresponding buffer
    for (u32 i = 0; i < frameCount; ++i) {
      vk::DescriptorBufferInfo bufferInfo{
          .buffer = instanceBuffers[i].get(), .offset = 0, .range = vk::WholeSize};
      vk::WriteDescriptorSet instanceWrite{.dstSet = *instanceDataDescriptorSets[i],
                                           .dstBinding = 0,
                                           .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &bufferInfo};
      device.logical().updateDescriptorSets({instanceWrite}, nullptr);
    }

    return {};
  }

  [[nodiscard]] std::expected<void, std::string>
  createStaticQuadBuffers(const vk::raii::Queue &transferQueue) {
    const std::vector<TextQuadVertex> vertices = {
        {{0.0f, 1.0f}, {0.0f, 1.0f}}, // Bottom-left
        {{1.0f, 1.0f}, {1.0f, 1.0f}}, // Bottom-right
        {{1.0f, 0.0f}, {1.0f, 0.0f}}, // Top-right
        {{0.0f, 0.0f}, {0.0f, 0.0f}}  // Top-left
    };
    const std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};

    auto createStagingBuffer = [&](vk::DeviceSize size,
                                   const void *data) -> std::expected<VmaBuffer, std::string> {
      vk::BufferCreateInfo bufInfo{.size = size, .usage = vk::BufferUsageFlagBits::eTransferSrc};
      vma::AllocationCreateInfo allocInfo{.flags = vma::AllocationCreateFlagBits::eMapped,
                                          .usage = vma::MemoryUsage::eCpuOnly};
      auto res = device.createBufferVMA(bufInfo, allocInfo);
      if (!res)
        return std::unexpected("Failed to create staging buffer");
      VmaBuffer buf = std::move(*res);
      std::memcpy(buf.getMappedData(), data, static_cast<size_t>(size));
      return std::expected<VmaBuffer, std::string>(std::move(buf));
    };

    auto createDeviceBuffer =
        [&](vk::DeviceSize size,
            vk::BufferUsageFlags usage) -> std::expected<VmaBuffer, std::string> {
      vk::BufferCreateInfo bufInfo{.size = size,
                                   .usage = usage | vk::BufferUsageFlagBits::eTransferDst};
      vma::AllocationCreateInfo allocInfo{.usage = vma::MemoryUsage::eGpuOnly};
      auto res = device.createBufferVMA(bufInfo, allocInfo);
      if (!res)
        return std::unexpected("Failed to create device-local buffer");
      return std::expected<VmaBuffer, std::string>(std::move(*res));
    };

    // Vertex buffer
    const vk::DeviceSize vertexSize = sizeof(TextQuadVertex) * vertices.size();
    auto stagingVb = createStagingBuffer(vertexSize, vertices.data());
    if (!stagingVb)
      return std::unexpected("Failed to create text staging VB");

    auto vertexBuf = createDeviceBuffer(vertexSize, vk::BufferUsageFlagBits::eVertexBuffer);
    if (!vertexBuf)
      return std::unexpected("Failed to create text static VB");

    EXPECTED_VOID(device.copyBuffer(stagingVb->get(), vertexBuf->get(), vertexSize));
    staticVertexBuffer = std::move(*vertexBuf);

    // Index buffer
    const vk::DeviceSize indexSize = sizeof(uint32_t) * indices.size();
    auto stagingIb = createStagingBuffer(indexSize, indices.data());
    if (!stagingIb)
      return std::unexpected("Failed to create text staging IB");

    auto indexBuf = createDeviceBuffer(indexSize, vk::BufferUsageFlagBits::eIndexBuffer);
    if (!indexBuf)
      return std::unexpected("Failed to create text static IB");

    EXPECTED_VOID(device.copyBuffer(stagingIb->get(), indexBuf->get(), indexSize));
    staticIndexBuffer = std::move(*indexBuf);

    return {};
  }

  [[nodiscard]] std::expected<void, std::string> createInstanceBuffers() {
    instanceBuffers.resize(frameCount);
    vk::DeviceSize bufferSize = maxQuads * sizeof(TextInstanceData);

    for (u32 i = 0; i < frameCount; ++i) {
      vk::BufferCreateInfo bufferInfo{
          .size = bufferSize,
          .usage = vk::BufferUsageFlagBits::eStorageBuffer // It's a storage buffer now
      };
      // These buffers need to be updated from the CPU every frame.
      vma::AllocationCreateInfo allocInfo{
          .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                   vma::AllocationCreateFlagBits::eMapped,
          .usage = vma::MemoryUsage::eAuto};

      auto bufferResult = device.createBufferVMA(bufferInfo, allocInfo);
      if (!bufferResult) {
        return std::unexpected("Failed to create text instance buffer " + std::to_string(i));
      }
      instanceBuffers[i] = std::move(*bufferResult);
    }
    return {};
  }

  [[nodiscard]] std::expected<void, std::string>
  createAtlasTexture(const vk::raii::Queue &transferQueue) {
    auto texResult = createTexture(
        device, fontAtlasData.atlasBitmap.data(), fontAtlasData.atlasBitmap.size(),
        vk::Extent3D{(u32)fontAtlasData.atlasWidth, (u32)fontAtlasData.atlasHeight, 1},
        vk::Format::eR8Unorm, transferQueue, false);
    if (!texResult) {
      return std::unexpected("Failed to create font atlas texture: " + texResult.error());
    }
    fontAtlasTexture = std::make_shared<Texture>(std::move(texResult.value()));
    return {};
  }
};
