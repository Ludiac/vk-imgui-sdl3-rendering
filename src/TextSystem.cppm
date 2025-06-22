module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>

export module vulkan_app:TextSystem;

import vulkan_hpp;
import std;

import :VulkanDevice;
import :VulkanPipeline;
import :VMA;
import :texture;
import :text;
import :ui;

// This class is responsible for laying out text and preparing it for rendering.
// It no longer issues draw calls itself.
export class TextSystem {
private:
  VulkanDevice &device;
  u32 frameCount;
  u32 maxQuadsPerFrame;
  size_t minStorageBufferOffsetAlignment;
  std::vector<std::unique_ptr<Font>> registeredFonts;
  std::vector<VmaBuffer> instanceBuffers;
  using InstanceVector = std::vector<TextInstanceData>;
  std::map<Font *, InstanceVector> frameBatch;
  VmaBuffer staticVertexBuffer;
  VmaBuffer staticIndexBuffer;
  std::vector<vk::raii::DescriptorSet> instanceDataDescriptorSets;
  vk::raii::DescriptorSetLayout instanceDataLayout{nullptr};

public:
  TextSystem(VulkanDevice &dev, u32 inFlightFrameCount, const vk::raii::DescriptorPool &pool)
      : device(dev), frameCount(inFlightFrameCount), maxQuadsPerFrame(2048) {
    // Implementations for these helpers are unchanged from your original TextRenderer.cpp
    EXPECTED_VOID(createInstanceBuffers(device, frameCount, maxQuadsPerFrame, instanceBuffers,
                                        sizeof(TextInstanceData)));
    EXPECTED_VOID(createInstanceDataDescriptorSetLayout());
    EXPECTED_VOID(allocateDescriptorSets(pool));
    EXPECTED_VOID(createStaticQuadBuffers(device, staticVertexBuffer, staticIndexBuffer));
  }

  void beginFrame() {
    // for (decltype(auto) i : frameBatch)
    //   i.second.clear();
    frameBatch.clear();
  }

  // The public API for registering fonts and queueing text remains unchanged.
  [[nodiscard]] std::expected<Font *, std::string>
  registerFont(const std::string &fontPath, int pixelHeight,
               const vk::raii::DescriptorSetLayout &textureLayout,
               const vk::raii::DescriptorPool &pool, const vk::raii::Queue &transferQueue) {
    // Unchanged from original TextRenderer.cpp
    auto font = std::make_unique<Font>();
    auto atlasResult = createFontAtlas(fontPath, pixelHeight);
    if (!atlasResult)
      return std::unexpected("Failed to create font atlas: " + atlasResult.error());
    font->atlasData = std::move(*atlasResult);
    auto texResult = createTexture(
        device, font->atlasData.atlasBitmap.data(), font->atlasData.atlasBitmap.size(),
        vk::Extent3D{(u32)font->atlasData.atlasWidth, (u32)font->atlasData.atlasHeight, 1},
        vk::Format::eR8Unorm, transferQueue, false);
    if (!texResult)
      return std::unexpected("Failed to create font texture: " + texResult.error());
    font->texture = std::make_shared<Texture>(std::move(*texResult));
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
    registeredFonts.push_back(std::move(font));
    return registeredFonts.back().get();
  }

  void queueText(Font *font, const std::string &text, float x, float y, const glm::vec4 &color) {
    // Unchanged from original TextRenderer.cpp
    if (!font || text.empty())
      return;
    const auto &glyphs = font->getAtlasData().glyphs;
    auto &instanceVec = frameBatch[font];
    float cursorX = x;
    float baselineY = y;
    for (char c : text) {
      if (instanceVec.size() + frameBatch.size() > maxQuadsPerFrame)
        break;
      const GlyphInfo &gi = glyphs.count(c) ? glyphs.at(c) : glyphs.at('?');
      if (gi.width > 0 && gi.height > 0) {
        float xpos = cursorX + gi.bearing_x;
        float ypos = baselineY - gi.bearing_y;
        instanceVec.emplace_back(TextInstanceData{.screenPos = {xpos, ypos},
                                                  .scale = {gi.width, gi.height},
                                                  .uvTopLeft = {gi.uv_x0, gi.uv_y0},
                                                  .uvBottomRight = {gi.uv_x1, gi.uv_y1},
                                                  .color = color});
      }
      cursorX += gi.advance;
    }
  }

  // REPLACES the old `draw` method.
  void prepareBatches(RenderQueue &queue, const VulkanPipeline &textPipeline, u32 frameIndex) {
    if (frameBatch.empty())
      return;

    u32 currentFirstInstance = 0;
    uint32_t currentByteOffset = 0; // <--- Track byte offset
    VmaBuffer &currentInstanceBuffer = instanceBuffers[frameIndex];

    for (auto const &[font, instances] : frameBatch) {
      if (instances.empty())
        continue;

      // 1. Copy instance data for this font batch to the GPU SSBO at the correct offset
      size_t dataSize = instances.size() * sizeof(TextInstanceData);
      if (currentByteOffset + dataSize > currentInstanceBuffer.getAllocationInfo().size) {
        // Consider logging this warning instead of printing to stdout
        break;
      }
      std::memcpy(static_cast<char *>(currentInstanceBuffer.getMappedData()) + currentByteOffset,
                  instances.data(), dataSize);

      // 2. Create a batch for THIS FONT and add it to the queue
      queue.emplace_back(RenderBatch{
          .sortKey = 200, // Text on top of UI
          .pipeline = &textPipeline.pipeline,
          .pipelineLayout = &textPipeline.pipelineLayout,
          .instanceDataSet = &instanceDataDescriptorSets[frameIndex], // Points to the big SSBO
          .textureSet = &font->textureDescriptorSet, // The unique texture for this font!
          .vertexBuffer = &staticVertexBuffer,
          .indexBuffer = &staticIndexBuffer,
          .indexCount = 6,
          .instanceCount = static_cast<u32>(instances.size()),
          .firstInstance = 0, // <--- IMPORTANT: This is now 0!
          .dynamicOffset = currentByteOffset,
      });

      currentFirstInstance += static_cast<u32>(instances.size());
      size_t alignedSize = pad_uniform_buffer_size(dataSize, this->minStorageBufferOffsetAlignment);
      currentByteOffset += static_cast<uint32_t>(alignedSize);

      // Bounds check
      if (currentByteOffset > currentInstanceBuffer.getAllocationInfo().size) {
        // Log a warning/error, you've run out of buffer space
        break;
      }
    }
    updateDescriptorSets(currentFirstInstance, frameIndex);
  }

private:
  // All private helper methods for buffer and descriptor set creation
  // remain unchanged from your original TextRenderer.cpp. They should be copied over.
  [[nodiscard]] std::expected<void, std::string> createInstanceDataDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding instanceBinding{.binding = 0,
                                                   .descriptorType =
                                                       vk::DescriptorType::eStorageBufferDynamic,
                                                   .descriptorCount = 1,
                                                   .stageFlags = vk::ShaderStageFlagBits::eVertex};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = 1, .pBindings = &instanceBinding};
    auto layoutResult = device.logical().createDescriptorSetLayout(layoutInfo);
    if (!layoutResult)
      return std::unexpected("Failed to create text instance data descriptor set layout.");
    instanceDataLayout = std::move(layoutResult.value());
    return {};
  }

  [[nodiscard]] std::expected<void, std::string>
  allocateDescriptorSets(const vk::raii::DescriptorPool &pool) {
    std::vector<vk::DescriptorSetLayout> layouts(frameCount, *instanceDataLayout);
    vk::DescriptorSetAllocateInfo instanceAllocInfo{
        .descriptorPool = pool, .descriptorSetCount = frameCount, .pSetLayouts = layouts.data()};
    auto instanceSetResult = device.logical().allocateDescriptorSets(instanceAllocInfo);
    if (!instanceSetResult)
      return std::unexpected("Failed to allocate text instance descriptor sets.");
    instanceDataDescriptorSets = std::move(instanceSetResult.value());
    return {};
  }

  void updateDescriptorSets(u32 instanceNumber, u32 frameIndex) {
    vk::DescriptorBufferInfo bufferInfo{
        .buffer = instanceBuffers[frameIndex].get(),
        .offset = 0,
        .range = instanceNumber * sizeof(TextInstanceData),
    };
    vk::WriteDescriptorSet instanceWrite{.dstSet = instanceDataDescriptorSets[frameIndex],
                                         .dstBinding = 0,
                                         .descriptorCount = 1,
                                         .descriptorType =
                                             vk::DescriptorType::eStorageBufferDynamic,
                                         .pBufferInfo = &bufferInfo};
    device.logical().updateDescriptorSets({instanceWrite}, nullptr);
  }
};
