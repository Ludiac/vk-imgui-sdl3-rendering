module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:TextSystem;

import vulkan_hpp;
import std;

import :VulkanDevice;
import :VulkanPipeline;
import :VMA;
import :texture;
import :text;
import :ui;

struct TextPushConstants2D {
  glm::mat4 projection;
  float sdf_weight;
  i32 antiAliasingToggle;
};
// This class is responsible for laying out text and preparing it for rendering.
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
    EXPECTED_VOID(createInstanceBuffers(device, frameCount, maxQuadsPerFrame, instanceBuffers,
                                        sizeof(TextInstanceData)));
    EXPECTED_VOID(createInstanceDataDescriptorSetLayout());
    EXPECTED_VOID(allocateDescriptorSets(pool));
    EXPECTED_VOID(createStaticQuadBuffers(device, staticVertexBuffer, staticIndexBuffer));
  }

  void beginFrame() { frameBatch.clear(); }

  const vk::SamplerCreateInfo msdfFontSamplerCreateInfo{
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear, // Use Linear for smoother scaling
      .addressModeU = vk::SamplerAddressMode::eClampToEdge,
      .addressModeV = vk::SamplerAddressMode::eClampToEdge,
      .addressModeW = vk::SamplerAddressMode::eClampToEdge,
      .mipLodBias = 0.0f,
      .anisotropyEnable = true, // Anisotropy can help with slanted views of text
      .maxAnisotropy = 4.0f,    // A modest value
      .compareEnable = false,
      .compareOp = vk::CompareOp::eAlways,
      .minLod = 0.0f,
      .maxLod = vk::LodClampNone, // Allow sampler to use all mip levels if they were generated
      .borderColor = vk::BorderColor::eFloatTransparentBlack,
      .unnormalizedCoordinates = false,
  };

  [[nodiscard]] std::expected<Font *, std::string>
  registerFont(const std::string &fontPath, int pixelHeight,
               const vk::raii::DescriptorSetLayout &textureLayout) {
    auto font = std::make_unique<Font>();
    auto atlasResult = createFontAtlasMSDF(fontPath, pixelHeight);
    if (!atlasResult)
      return std::unexpected("Failed to create font atlas: " + atlasResult.error());
    font->atlasData = std::move(*atlasResult);

    auto texResult = createTexture(
        device, font->atlasData.atlasBitmap.data(), font->atlasData.atlasBitmap.size(),
        vk::Extent3D{(u32)font->atlasData.atlasWidth, (u32)font->atlasData.atlasHeight, 1},
        vk::Format::eR8G8B8A8Unorm, device.queue_,
        false, // generateMipmaps = false for MSDF
        {}, {}, 1, vk::ImageViewType::e2D, &msdfFontSamplerCreateInfo);

    if (!texResult)
      return std::unexpected("Failed to create font texture: " + texResult.error());
    font->texture = std::make_shared<Texture>(std::move(*texResult));

    vk::DescriptorSetAllocateInfo allocInfo{.descriptorPool = device.descriptorPool_,
                                            .descriptorSetCount = 1,
                                            .pSetLayouts = &*textureLayout};
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

  // Example DPI, adjust to your target display if known, otherwise 96.0 is a safe default.
  const float SYSTEM_DPI = 96.0f * 8;

  void queueText(Font *font, const std::string &text, u32 pointSize, float x, float y,
                 const glm::vec4 &color) {
    if (!font || text.empty())
      return;

    // 1. Convert font point size to a target pixel height for the EM square.
    const float desiredPixelHeightForEm = static_cast<float>(pointSize) * (SYSTEM_DPI / 72.0f);

    const auto &metrics = font->atlasData;
    const float baselineY = y;

    auto &instanceVec = frameBatch[font];
    float cursorX = x;

    // 2. Calculate the scale factor to convert from abstract font units to screen pixels.
    const float fontUnitToPixelScale = desiredPixelHeightForEm / metrics.unitsPerEm;

    for (char c : text) {
      auto it = metrics.glyphs.find(static_cast<u32>(c));
      if (it == metrics.glyphs.end()) {
        it = metrics.glyphs.find(static_cast<u32>('?')); // Fallback
        if (it == metrics.glyphs.end())
          continue;
      }

      const auto &gi = it->second;

      // 3. Calculate the final on-screen size of the glyph quad in pixels.
      float scaledQuadWidth = gi.width * fontUnitToPixelScale;
      float scaledQuadHeight = gi.height * fontUnitToPixelScale;

      // 4. Calculate the on-screen position of the quad's top-left corner.
      float xpos = cursorX + (gi.bearing_x * fontUnitToPixelScale);
      float ypos = baselineY - (gi.bearing_y * fontUnitToPixelScale);

      // 5. **CRITICAL FIX**: Calculate the correct pixel range for the shader.
      // This converts the atlas pxRange into the on-screen pxRange.
      // It considers the atlas's internal scale and the final font-to-pixel scale.
      const float shaderPxRange =
          font->atlasData.pxRange * font->atlasData.atlasScale * fontUnitToPixelScale;

      instanceVec.emplace_back(TextInstanceData{
          .screenPos = {xpos, ypos},
          .size = {scaledQuadWidth, scaledQuadHeight},
          .uvTopLeft = {gi.uv_x0, gi.uv_y0},
          .uvBottomRight = {gi.uv_x1, gi.uv_y1},
          .color = color,
          .pxRange = shaderPxRange,
      });

      // 6. Advance the cursor for the next character.
      cursorX += gi.advance * fontUnitToPixelScale;
    }
  }

  void prepareBatches(RenderQueue &queue, const VulkanPipeline &textPipeline, u32 frameIndex,
                      vk::Extent2D windowSize, float sdf_weight, i32 antiAliasingToggle) {
    if (frameBatch.empty())
      return;

    glm::mat4 ortho = glm::ortho(0.0f, (float)windowSize.width, 0.0f, (float)windowSize.height);

    u32 currentFirstInstance = 0;
    uint32_t currentByteOffset = 0;
    VmaBuffer &currentInstanceBuffer = instanceBuffers[frameIndex];

    for (auto const &[font, instances] : frameBatch) {
      if (instances.empty())
        continue;

      size_t dataSize = instances.size() * sizeof(TextInstanceData);
      if (currentByteOffset + dataSize > currentInstanceBuffer.getAllocationInfo().size) {
        break;
      }
      std::memcpy(static_cast<char *>(currentInstanceBuffer.getMappedData()) + currentByteOffset,
                  instances.data(), dataSize);

      RenderBatch batch;
      batch.sortKey = 200;
      batch.pipeline = &textPipeline.pipeline;
      batch.pipelineLayout = &textPipeline.pipelineLayout;
      batch.instanceDataSet = &instanceDataDescriptorSets[frameIndex];
      batch.textureSet = &font->textureDescriptorSet;
      batch.vertexBuffer = &staticVertexBuffer;
      batch.indexBuffer = &staticIndexBuffer;
      batch.indexCount = 6;
      batch.instanceCount = static_cast<u32>(instances.size());
      batch.firstInstance = 0;
      batch.dynamicOffset = currentByteOffset;

      TextPushConstants2D pc;
      pc.projection = ortho;
      pc.sdf_weight = sdf_weight;
      pc.antiAliasingToggle = antiAliasingToggle;

      batch.pushConstantSize = sizeof(TextPushConstants2D);
      batch.pushConstantStages =
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
      std::memcpy(batch.pushConstantData.data(), &pc, sizeof(TextPushConstants2D));

      queue.push_back(batch);

      currentFirstInstance += static_cast<u32>(instances.size());
      size_t alignedSize = pad_uniform_buffer_size(dataSize, this->minStorageBufferOffsetAlignment);
      currentByteOffset += static_cast<u32>(alignedSize);

      if (currentByteOffset > currentInstanceBuffer.getAllocationInfo().size) {
        std::println("run out of buffer space");
        break;
      }
    }
    updateDescriptorSets(currentFirstInstance, frameIndex);
  }

private:
  [[nodiscard]] std::expected<void, std::string> createInstanceDataDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding instanceBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eStorageBufferDynamic,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex}; // Also needed in fragment
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
                                         .dstArrayElement = 0,
                                         .descriptorCount = 1,
                                         .descriptorType =
                                             vk::DescriptorType::eStorageBufferDynamic,
                                         .pBufferInfo = &bufferInfo};
    device.logical().updateDescriptorSets({instanceWrite}, nullptr);
  }
};
