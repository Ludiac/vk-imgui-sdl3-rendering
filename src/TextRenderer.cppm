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
import :ui;

export class TextRenderer {
private:
  VulkanDevice &device;
  u32 frameCount; // Number of frames in flight

  std::vector<std::unique_ptr<Font>> registeredFonts;

  std::vector<VmaBuffer> instanceBuffers;
  // Batching data, cleared each frame
  using InstanceVector = std::vector<TextInstanceData>;
  std::map<Font *, InstanceVector> frameBatch;

  VmaBuffer staticVertexBuffer;
  VmaBuffer staticIndexBuffer; // NEW: Added an index buffer for the quad
  std::vector<vk::raii::DescriptorSet> instanceDataDescriptorSets;
  vk::raii::DescriptorSetLayout instanceDataLayout{nullptr}; // Layout for the instance SSBO

  u32 maxQuadsPerFrame;

public:
  TextRenderer(VulkanDevice &dev, u32 inFlightFrameCount, const vk::raii::DescriptorPool &pool)
      : device(dev), frameCount(inFlightFrameCount),
        maxQuadsPerFrame(2048) // Max 2048 chars per draw call
  {
    EXPECTED_VOID(createInstanceBuffers(device, frameCount, maxQuadsPerFrame, instanceBuffers));
    EXPECTED_VOID(createInstanceDataDescriptorSetLayout());
    EXPECTED_VOID(allocateDescriptorSets(pool));
    EXPECTED_VOID(createStaticQuadBuffers(device, staticVertexBuffer, staticIndexBuffer));
  }

  [[nodiscard]] std::expected<Font *, std::string>
  registerFont(const std::string &fontPath, int pixelHeight,
               const vk::raii::DescriptorSetLayout &textureLayout,
               const vk::raii::DescriptorPool &pool, const vk::raii::Queue &transferQueue) {
    auto font = std::make_unique<Font>();

    // 1. Create Font Atlas from TTF
    auto atlasResult = createFontAtlas(fontPath, pixelHeight);
    if (!atlasResult)
      return std::unexpected("Failed to create font atlas: " + atlasResult.error());
    font->atlasData = std::move(*atlasResult);

    // 2. Create GPU Texture from Atlas
    auto texResult = createTexture(
        device, font->atlasData.atlasBitmap.data(), font->atlasData.atlasBitmap.size(),
        vk::Extent3D{(u32)font->atlasData.atlasWidth, (u32)font->atlasData.atlasHeight, 1},
        vk::Format::eR8Unorm, transferQueue, false);
    if (!texResult)
      return std::unexpected("Failed to create font texture: " + texResult.error());
    font->texture = std::make_shared<Texture>(std::move(*texResult));

    // 3. Create and update descriptor set for this font's texture
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &*textureLayout};
    auto setResult = device.logical().allocateDescriptorSets(allocInfo);
    if (!setResult)
      return std::unexpected("Failed to allocate font descriptor set.");
    font->textureDescriptorSet = std::move(setResult.value().front());

    vk::DescriptorImageInfo imageInfo{.sampler = *font->texture->sampler,
                                      .imageView = *font->texture->view,
                                      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet write{.dstSet = *font->textureDescriptorSet,
                                 .dstBinding = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                 .pImageInfo = &imageInfo};
    device.logical().updateDescriptorSets({write}, nullptr);

    // Store font and return raw pointer as a handle
    registeredFonts.push_back(std::move(font));
    return registeredFonts.back().get();
  }

  void queueText(Font *font, const std::string &text, float x, float y, const glm::vec4 &color) {
    if (!font || text.empty())
      return;

    const auto &glyphs = font->getAtlasData().glyphs;
    auto &instanceVec = frameBatch[font]; // Creates entry if not present

    float cursorX = x;
    float baselineY = y;

    for (char c : text) {
      if (instanceVec.size() + frameBatch.size() > maxQuadsPerFrame)
        break;

      const GlyphInfo &gi = glyphs.count(c) ? glyphs.at(c) : glyphs.at('?');

      if (gi.width > 0 && gi.height > 0) {
        float xpos = cursorX + gi.bearing_x;
        float ypos = baselineY + gi.bearing_y - gi.height;

        // Add the color directly to the instance data
        instanceVec.emplace_back(TextInstanceData{.screenPos = {xpos, ypos},
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
    if (frameBatch.empty())
      return;

    // 1. Bind pipeline and static buffers once.
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.pipeline);
    cmd.bindVertexBuffers(0, {staticVertexBuffer.get()}, {0});
    cmd.bindIndexBuffer(staticIndexBuffer.get(), 0, vk::IndexType::eUint32);

    // 2. Bind the common instance data descriptor set once.
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.pipelineLayout, 1,
                           {*instanceDataDescriptorSets[frameIndex]}, {});

    // 3. Push projection matrix once.
    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(windowSize.width),
                                 static_cast<float>(windowSize.height), 0.0f);
    TextPushConstants constants{.projection = ortho};
    cmd.pushConstants<TextPushConstants>(*pipeline.pipelineLayout, vk::ShaderStageFlagBits::eVertex,
                                         0, constants);

    u32 currentFirstInstance = 0;
    VmaBuffer &currentInstanceBuffer = instanceBuffers[frameIndex];

    // 4. Iterate through each font batch.
    for (auto const &[font, instances] : frameBatch) {
      if (instances.empty())
        continue;

      // 4a. Update the SSBO with data for the *current batch*.
      // We copy into the single large buffer at an offset.
      size_t dataSize = instances.size() * sizeof(TextInstanceData);
      size_t offset = currentFirstInstance * sizeof(TextInstanceData);
      if (offset + dataSize > currentInstanceBuffer.getAllocationInfo().size) {
        std::println("Text instance data exceeds buffer capacity. Some text will not be rendered.");
        break; // Stop rendering if we run out of space
      }
      std::memcpy(static_cast<char *>(currentInstanceBuffer.getMappedData()) + offset,
                  instances.data(), dataSize);

      // 4b. Bind the unique descriptor set for this font's texture.
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.pipelineLayout, 0,
                             {*font->textureDescriptorSet}, {});

      // 4c. Issue a draw call for this font's batch.
      cmd.drawIndexed(6, static_cast<u32>(instances.size()), 0, 0, currentFirstInstance);

      currentFirstInstance += static_cast<u32>(instances.size());
    }
    frameBatch.clear();
  }

  // Texture* getFontTexture() { return registeredFonts[0]; }

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
  allocateDescriptorSets(const vk::raii::DescriptorPool &pool) {
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
};
