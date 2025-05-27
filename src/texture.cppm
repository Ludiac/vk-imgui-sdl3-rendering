module;

#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:texture;

import vulkan_hpp;
import std;
import :VulkanDevice;

export struct Texture {
  vk::raii::Image image{nullptr};
  vk::raii::DeviceMemory memory{nullptr};
  vk::raii::ImageView view{nullptr};
  vk::raii::Sampler sampler{nullptr};

  vk::Format format{vk::Format::eUndefined};
  vk::Extent3D extent{0, 0, 0};
  uint32_t mipLevels{0};
  uint32_t arrayLayers{0};
};

export struct PBRTextures {
  std::shared_ptr<Texture> baseColor;
  std::shared_ptr<Texture> metallicRoughness;
  std::shared_ptr<Texture> normal;
  std::shared_ptr<Texture> occlusion;
  std::shared_ptr<Texture> emissive;
};

namespace TextureHelpers {

// Finds a suitable memory type index for a given type filter and properties.
[[nodiscard]] std::expected<uint32_t, std::string>
findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t typeFilter,
               vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return std::unexpected("TextureHelpers::findMemoryType: Failed to find suitable memory type.");
}

// Begins a command buffer for single-time submission.
[[nodiscard]] std::expected<vk::raii::CommandBuffer, std::string>
beginSingleTimeCommands(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool) {
  vk::CommandBufferAllocateInfo allocInfo{.commandPool = *commandPool,
                                          .level = vk::CommandBufferLevel::ePrimary,
                                          .commandBufferCount = 1};

  // Use vk::raii::CommandBuffers to allocate and manage command buffers.
  // This returns a ResultValue containing a vector of vk::raii::CommandBuffer.
  auto cmdBuffersResult = device.allocateCommandBuffers(allocInfo);
  if (!cmdBuffersResult) {
    return std::unexpected(
        "TextureHelpers::beginSingleTimeCommands: Failed to allocate command buffers: " +
        vk::to_string(cmdBuffersResult.error()));
  }
  // We requested one command buffer, so take the first (and only) one.
  // std::move is essential here as CommandBuffersUnique holds unique_ptr-like objects.
  vk::raii::CommandBuffer commandBuffer = std::move(cmdBuffersResult.value().front());

  vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

  commandBuffer.begin(beginInfo);
  return std::move(commandBuffer); // Return by value (moved)
}

// Ends, submits, and waits for a single-time command buffer to complete.
[[nodiscard]] std::expected<void, std::string>
endSingleTimeCommands(const vk::raii::Device &device,
                      vk::raii::CommandBuffer &&commandBuffer, // Consumes the command buffer
                      const vk::raii::Queue &queue) {
  commandBuffer.end();

  vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};

  vk::raii::Fence fence{nullptr};
  auto fenceResultValue = device.createFence(vk::FenceCreateInfo{});
  if (!fenceResultValue) {
    return std::unexpected("TextureHelpers::endSingleTimeCommands: Failed to create fence: " +
                           vk::to_string(fenceResultValue.error()));
  }
  fence = std::move(fenceResultValue.value());

  queue.submit(submitInfo, *fence);

  // Use true instead of VK_TRUE
  vk::Result waitResult = device.waitForFences({*fence}, true, UINT64_MAX);
  if (waitResult != vk::Result::eSuccess) {
    return std::unexpected("TextureHelpers::endSingleTimeCommands: Failed to wait for fence: " +
                           vk::to_string(waitResult));
  }
  return {};
}

// Transitions the layout of a Vulkan image.
void transitionImageLayout(const vk::raii::CommandBuffer &commandBuffer, vk::Image image,
                           vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                           const vk::ImageSubresourceRange &subresourceRange) {
  vk::ImageMemoryBarrier barrier{.srcAccessMask = {},
                                 .dstAccessMask = {},
                                 .oldLayout = oldLayout,
                                 .newLayout = newLayout,
                                 .srcQueueFamilyIndex = 0,
                                 .dstQueueFamilyIndex = 0,
                                 .image = image,
                                 .subresourceRange = subresourceRange};

  vk::PipelineStageFlags sourceStage;
  vk::PipelineStageFlags destinationStage;

  if (oldLayout == vk::ImageLayout::eUndefined &&
      newLayout == vk::ImageLayout::eTransferDstOptimal) {
    barrier.srcAccessMask = {};
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    destinationStage = vk::PipelineStageFlagBits::eTransfer;
  } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    sourceStage = vk::PipelineStageFlagBits::eTransfer;
    destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
  } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::eTransferSrcOptimal) {
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    sourceStage = vk::PipelineStageFlagBits::eTransfer;
    destinationStage = vk::PipelineStageFlagBits::eTransfer;
  } else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal &&
             newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    sourceStage = vk::PipelineStageFlagBits::eTransfer;
    destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
  } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eGeneral) {
    barrier.srcAccessMask = {};
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    destinationStage = vk::PipelineStageFlagBits::eComputeShader;
  } else if (oldLayout == vk::ImageLayout::eUndefined &&
             (newLayout == vk::ImageLayout::eColorAttachmentOptimal ||
              newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)) {
    barrier.srcAccessMask = {};
    barrier.dstAccessMask = (newLayout == vk::ImageLayout::eColorAttachmentOptimal)
                                ? vk::AccessFlagBits::eColorAttachmentWrite
                                : vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    destinationStage = (newLayout == vk::ImageLayout::eColorAttachmentOptimal)
                           ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                           : vk::PipelineStageFlagBits::eEarlyFragmentTests;
  } else {
    barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
    sourceStage = vk::PipelineStageFlagBits::eAllCommands;
    destinationStage = vk::PipelineStageFlagBits::eAllCommands;
  }

  // This check might be redundant if subresourceRange.aspectMask is always correctly set by the
  // caller.
  if (barrier.subresourceRange.aspectMask == vk::ImageAspectFlags{}) {
    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
        oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
        format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint ||
        format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD32Sfloat ||
        format == vk::Format::eD16Unorm) {
      barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
      if (format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint ||
          format == vk::Format::eD16UnormS8Uint) {
        barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
      }
    } else {
      barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    }
  }

  commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, barrier);
}

// Copies data from a Vulkan buffer to a Vulkan image.
void copyBufferToImage(const vk::raii::CommandBuffer &commandBuffer, vk::Buffer buffer,
                       vk::Image image, vk::Extent3D extent,
                       const vk::ImageSubresourceLayers &imageSubresource) {
  vk::BufferImageCopy region{.bufferOffset = 0,
                             .bufferRowLength = 0,
                             .bufferImageHeight = 0,
                             .imageSubresource = imageSubresource,
                             .imageOffset = {0, 0, 0},
                             .imageExtent = extent};
  commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

// Generates mipmaps for a given image using blitting.
void generateMipmaps(VulkanDevice &vulkanDevice, const vk::raii::CommandBuffer &commandBuffer,
                     vk::Image image, vk::Extent2D texExtent, uint32_t totalMipLevels,
                     vk::Format format, uint32_t arrayLayers = 1, uint32_t baseArrayLayer = 0) {
  if (totalMipLevels <= 1)
    return;

  vk::FormatProperties formatProperties = vulkanDevice.physical().getFormatProperties(format);
  if (!(formatProperties.optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
    // std::cerr << "Warning: Image format " << vk::to_string(format)
    //           << " does not support linear blitting for mipmap generation." << std::endl;
  }

  vk::ImageSubresourceRange subresourceRangeBase{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                 .baseMipLevel = 0,
                                                 .levelCount = 1,
                                                 .baseArrayLayer = baseArrayLayer,
                                                 .layerCount = arrayLayers};

  int32_t mipWidth = static_cast<int32_t>(texExtent.width);
  int32_t mipHeight = static_cast<int32_t>(texExtent.height);

  for (uint32_t i = 1; i < totalMipLevels; ++i) {
    subresourceRangeBase.baseMipLevel = i - 1;
    transitionImageLayout(commandBuffer, image, format, vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eTransferSrcOptimal, subresourceRangeBase);

    int32_t nextMipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
    int32_t nextMipHeight = mipHeight > 1 ? mipHeight / 2 : 1;

    vk::ImageBlit blit{
        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .mipLevel = i - 1,
                           .baseArrayLayer = baseArrayLayer,
                           .layerCount = arrayLayers},
        .srcOffsets = std::array<vk::Offset3D, 2>{{{0, 0, 0}, {mipWidth, mipHeight, 1}}},
        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .mipLevel = i,
                           .baseArrayLayer = baseArrayLayer,
                           .layerCount = arrayLayers},
        .dstOffsets = std::array<vk::Offset3D, 2>{{{0, 0, 0}, {nextMipWidth, nextMipHeight, 1}}}};

    commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image,
                            vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

    transitionImageLayout(commandBuffer, image, format, vk::ImageLayout::eTransferSrcOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal, subresourceRangeBase);

    mipWidth = nextMipWidth;
    mipHeight = nextMipHeight;
  }

  subresourceRangeBase.baseMipLevel = totalMipLevels - 1;
  transitionImageLayout(commandBuffer, image, format, vk::ImageLayout::eTransferDstOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal, subresourceRangeBase);
}
} // namespace TextureHelpers

// Creates a Vulkan texture from pixel data or as an uninitialized image.
export [[nodiscard]] std::expected<Texture, std::string>
createTexture(VulkanDevice &vulkanDevice, const void *pixels, vk::DeviceSize imageSize,
              vk::Extent3D texExtent, vk::Format texFormat, vk::raii::CommandPool &commandPool,
              const vk::raii::Queue &transferQueue, bool generateMipmaps = true,
              vk::ImageUsageFlags additionalImageUsage = {},
              vk::ImageCreateFlags imageCreateFlags = {}, uint32_t arrayLayers = 1,
              vk::ImageViewType viewType = vk::ImageViewType::e2D,
              const vk::SamplerCreateInfo *pCustomSamplerInfo = nullptr) {
  if (texExtent.width == 0 || texExtent.height == 0 || texExtent.depth == 0 || arrayLayers == 0) {
    return std::unexpected(
        "createTexture: Texture dimensions and arrayLayers must be greater than zero.");
  }
  if (pixels == nullptr && imageSize > 0) {
    return std::unexpected("createTexture: Pixel data is null but imageSize is non-zero.");
  }
  if (pixels != nullptr && imageSize == 0) {
    return std::unexpected("createTexture: Pixel data is provided but imageSize is zero.");
  }

  Texture textureOut;
  textureOut.format = texFormat;
  textureOut.extent = texExtent;
  textureOut.arrayLayers = arrayLayers;

  if (generateMipmaps && texExtent.width > 0 && texExtent.height > 0 && texExtent.depth == 1) {
    textureOut.mipLevels =
        static_cast<uint32_t>(std::floor(std::log2(std::max(texExtent.width, texExtent.height)))) +
        1;
  } else {
    textureOut.mipLevels = 1;
  }

  vk::raii::Buffer stagingBuffer{nullptr};
  vk::raii::DeviceMemory stagingBufferMemory{nullptr};

  if (pixels && imageSize > 0) {
    auto bufferResourcesExpected = vulkanDevice.createBuffer(
        imageSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    if (!bufferResourcesExpected) {
      return std::unexpected("createTexture: Failed to create staging buffer: " +
                             bufferResourcesExpected.error());
    }
    BufferResources stagingBufferResources = std::move(*bufferResourcesExpected);
    stagingBuffer = std::move(stagingBufferResources.buffer);
    stagingBufferMemory = std::move(stagingBufferResources.memory);

    stagingBuffer.bindMemory(*stagingBufferMemory, 0);

    // mapMemory on vk::raii::DeviceMemory returns void* and throws on error if exceptions are
    // enabled. With VULKAN_HPP_NO_EXCEPTIONS, it might behave differently (e.g. abort or specific
    // error handling). The vkMapMemory function itself returns VkResult. If your Vulkan-Hpp is
    // configured to not throw, you might need a wrapper around mapMemory that checks a result or
    // the pointer. For now, proceeding with the direct call, assuming a valid pointer or prior
    // failure. A more robust check for `nullptr` after mapMemory is good practice if it can return
    // it on failure without throwing.
    void *mappedData = vulkanDevice.logical().mapMemory2KHR({
        .memory = *stagingBufferMemory,
        .offset = 0,
        .size = imageSize,
    });
    if (!mappedData && imageSize > 0) { // Check if mapMemory failed (returned nullptr)
      return std::unexpected("createTexture: Failed to map staging buffer memory (got nullptr).");
    }
    std::memcpy(mappedData, pixels, static_cast<size_t>(imageSize));
    vulkanDevice.logical().unmapMemory2KHR({
        .memory = *stagingBufferMemory,
    });
  }

  vk::ImageCreateInfo imageInfo{.flags = imageCreateFlags,
                                .imageType =
                                    (texExtent.depth > 1 && viewType != vk::ImageViewType::e2DArray)
                                        ? vk::ImageType::e3D
                                        : vk::ImageType::e2D,
                                .format = textureOut.format,
                                .extent = textureOut.extent,
                                .mipLevels = textureOut.mipLevels,
                                .arrayLayers = textureOut.arrayLayers,
                                .samples = vk::SampleCountFlagBits::e1,
                                .tiling = vk::ImageTiling::eOptimal,
                                .usage = vk::ImageUsageFlagBits::eSampled | additionalImageUsage,
                                .sharingMode = vk::SharingMode::eExclusive,
                                .initialLayout = vk::ImageLayout::eUndefined};
  if (pixels && imageSize > 0) {
    imageInfo.usage |= vk::ImageUsageFlagBits::eTransferDst;
  }
  if (generateMipmaps && textureOut.mipLevels > 1) {
    imageInfo.usage |= vk::ImageUsageFlagBits::eTransferSrc;
  }

  auto imageResultValue = vulkanDevice.logical().createImage(imageInfo);
  if (!imageResultValue) {
    return std::unexpected("createTexture: Failed to create vk::Image: " +
                           vk::to_string(imageResultValue.error()));
  }
  textureOut.image = std::move(imageResultValue.value());

  vk::MemoryRequirements memRequirements = textureOut.image.getMemoryRequirements();
  auto memoryTypeIndexExpected =
      TextureHelpers::findMemoryType(vulkanDevice.physical(), memRequirements.memoryTypeBits,
                                     vk::MemoryPropertyFlagBits::eDeviceLocal);
  if (!memoryTypeIndexExpected) {
    return std::unexpected("createTexture: " + memoryTypeIndexExpected.error());
  }

  vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                   .memoryTypeIndex = *memoryTypeIndexExpected};
  auto memoryResultValue = vulkanDevice.logical().allocateMemory(allocInfo);
  if (!memoryResultValue) {
    return std::unexpected("createTexture: Failed to allocate image memory: " +
                           vk::to_string(memoryResultValue.error()));
  }
  textureOut.memory = std::move(memoryResultValue.value());

  textureOut.image.bindMemory(*textureOut.memory, 0);

  auto commandBufferExpected =
      TextureHelpers::beginSingleTimeCommands(vulkanDevice.logical(), commandPool);
  if (!commandBufferExpected) {
    return std::unexpected("createTexture: " + commandBufferExpected.error());
  }
  vk::raii::CommandBuffer commandBuffer = std::move(*commandBufferExpected);

  vk::ImageSubresourceRange baseSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                 .baseMipLevel = 0,
                                                 .levelCount = textureOut.mipLevels,
                                                 .baseArrayLayer = 0,
                                                 .layerCount = textureOut.arrayLayers};
  if (additionalImageUsage & vk::ImageUsageFlagBits::eDepthStencilAttachment ||
      texFormat == vk::Format::eD32Sfloat || texFormat == vk::Format::eD16Unorm ||
      texFormat == vk::Format::eD32SfloatS8Uint || texFormat == vk::Format::eD24UnormS8Uint ||
      texFormat == vk::Format::eD16UnormS8Uint) {
    baseSubresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    if (texFormat == vk::Format::eD32SfloatS8Uint || texFormat == vk::Format::eD24UnormS8Uint ||
        texFormat == vk::Format::eD16UnormS8Uint) {
      baseSubresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
    }
  }

  if (pixels && imageSize > 0) {
    TextureHelpers::transitionImageLayout(
        commandBuffer, *textureOut.image, textureOut.format, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, baseSubresourceRange);

    vk::ImageSubresourceLayers copySubresourceLayers{.aspectMask = baseSubresourceRange.aspectMask,
                                                     .mipLevel = 0,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = textureOut.arrayLayers};
    TextureHelpers::copyBufferToImage(commandBuffer, *stagingBuffer, *textureOut.image, texExtent,
                                      copySubresourceLayers);

    if (generateMipmaps && textureOut.mipLevels > 1) {
      TextureHelpers::generateMipmaps(vulkanDevice, commandBuffer, *textureOut.image,
                                      vk::Extent2D{texExtent.width, texExtent.height},
                                      textureOut.mipLevels, textureOut.format,
                                      textureOut.arrayLayers);
    } else {
      TextureHelpers::transitionImageLayout(
          commandBuffer, *textureOut.image, textureOut.format, vk::ImageLayout::eTransferDstOptimal,
          vk::ImageLayout::eShaderReadOnlyOptimal, baseSubresourceRange);
    }
  } else {
    vk::ImageLayout targetLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    if (additionalImageUsage & vk::ImageUsageFlagBits::eColorAttachment) {
      targetLayout = vk::ImageLayout::eColorAttachmentOptimal;
    } else if (additionalImageUsage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
      targetLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    } else if (additionalImageUsage & vk::ImageUsageFlagBits::eStorage) {
      targetLayout = vk::ImageLayout::eGeneral;
    }
    TextureHelpers::transitionImageLayout(commandBuffer, *textureOut.image, textureOut.format,
                                          vk::ImageLayout::eUndefined, targetLayout,
                                          baseSubresourceRange);
  }

  auto endCommandsExpected = TextureHelpers::endSingleTimeCommands(
      vulkanDevice.logical(), std::move(commandBuffer), transferQueue);
  if (!endCommandsExpected) {
    // Staging buffer and its memory are RAII and will be cleaned up.
    // Texture image and memory are also RAII and will be cleaned up if this function returns an
    // error.
    return std::unexpected("createTexture: " + endCommandsExpected.error());
  }

  vk::ImageViewCreateInfo viewInfo{.image = *textureOut.image,
                                   .viewType = viewType,
                                   .format = textureOut.format,
                                   .components = {.r = vk::ComponentSwizzle::eIdentity,
                                                  .g = vk::ComponentSwizzle::eIdentity,
                                                  .b = vk::ComponentSwizzle::eIdentity,
                                                  .a = vk::ComponentSwizzle::eIdentity},
                                   .subresourceRange = baseSubresourceRange};

  auto viewResultValue = vulkanDevice.logical().createImageView(viewInfo);
  if (!viewResultValue) {
    return std::unexpected("createTexture: Failed to create image view: " +
                           vk::to_string(viewResultValue.error()));
  }
  textureOut.view = std::move(viewResultValue.value());

  if (pCustomSamplerInfo) {
    auto samplerResultValue = vulkanDevice.logical().createSampler(*pCustomSamplerInfo);
    if (!samplerResultValue) {
      return std::unexpected("createTexture: Failed to create custom sampler: " +
                             vk::to_string(samplerResultValue.error()));
    }
    textureOut.sampler = std::move(samplerResultValue.value());
  } else {
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vulkanDevice.physical().getFeatures().samplerAnisotropy
                                ? true
                                : false, // Use true/false
        .maxAnisotropy = vulkanDevice.physical().getFeatures().samplerAnisotropy
                             ? vulkanDevice.physical().getProperties().limits.maxSamplerAnisotropy
                             : 1.0f,
        .compareEnable = false, // Use true/false
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(textureOut.mipLevels),
        .borderColor = vk::BorderColor::eFloatOpaqueBlack,
        .unnormalizedCoordinates = false // Use true/false
    };
    auto samplerResultValue = vulkanDevice.logical().createSampler(samplerInfo);
    if (!samplerResultValue) {
      return std::unexpected("createTexture: Failed to create default sampler: " +
                             vk::to_string(samplerResultValue.error()));
    }
    textureOut.sampler = std::move(samplerResultValue.value());
  }

  return textureOut;
}

// Creates a simple, default 1x1 texture with a specified color.
export [[nodiscard]] std::expected<Texture, std::string>
createDefaultTexture(VulkanDevice &vulkanDevice, vk::raii::CommandPool &commandPool,
                     const vk::raii::Queue &transferQueue,
                     vk::Format format = vk::Format::eR8G8B8A8Unorm,
                     std::array<uint8_t, 4> color = {255, 255, 255, 255},
                     const vk::SamplerCreateInfo *pCustomSamplerInfo = nullptr) {
  vk::Extent3D extent = {1, 1, 1};
  std::vector<uint8_t> pixels;
  uint32_t bytesPerPixel = 0;

  switch (format) {
  case vk::Format::eR8G8B8A8Unorm:
  case vk::Format::eR8G8B8A8Srgb:
    bytesPerPixel = 4;
    pixels.resize(bytesPerPixel);
    pixels[0] = color[0];
    pixels[1] = color[1];
    pixels[2] = color[2];
    pixels[3] = color[3];
    break;
  default:
    return std::unexpected(
        "createDefaultTexture: Unsupported format for default pixel data generation: " +
        vk::to_string(format));
  }

  vk::DeviceSize imageSize = pixels.size();

  return createTexture(vulkanDevice, pixels.data(), imageSize, extent, format, commandPool,
                       transferQueue, false, {}, {}, 1, vk::ImageViewType::e2D, pCustomSamplerInfo);
}

export [[nodiscard]] std::expected<Texture, std::string>
createTestPatternTexture(VulkanDevice &vulkanDevice, vk::raii::CommandPool &commandPool,
                         const vk::raii::Queue &transferQueue,
                         vk::Format format = vk::Format::eR8G8B8A8Unorm,
                         std::array<uint8_t, 4> color = {255, 255, 255, 255},
                         const vk::SamplerCreateInfo *pCustomSamplerInfo = nullptr) {
  vk::Extent3D extent = {32, 32, 1};
  std::vector<uint8_t> pixels;
  uint32_t bytesPerPixel = 0;

  switch (format) {
  case vk::Format::eR8G8B8A8Unorm:
  case vk::Format::eR8G8B8A8Srgb:
    bytesPerPixel = 4;
    pixels.resize(32 * 32 * bytesPerPixel);

    for (uint32_t y = 0; y < 32; ++y) {
      for (uint32_t x = 0; x < 32; ++x) {
        // Calculate position-based factors
        float xFactor = static_cast<float>(x) / 31.0f;
        float yFactor = static_cast<float>(y) / 31.0f;
        float xyFactor = static_cast<float>(x + y) / 62.0f;

        // Calculate color components using position factors
        uint8_t r = static_cast<uint8_t>(color[0] * xFactor + 0.5f);
        uint8_t g = static_cast<uint8_t>(color[1] * yFactor + 0.5f);
        uint8_t b = static_cast<uint8_t>(color[2] * xyFactor + 0.5f);
        uint8_t a = color[3];

        // Set pixel data
        size_t index = (y * 32 + x) * 4;
        pixels[index] = r;
        pixels[index + 1] = g;
        pixels[index + 2] = b;
        pixels[index + 3] = a;
      }
    }
    break;
  default:
    return std::unexpected(
        "createTestPatternTexture: Unsupported format for pixel data generation: " +
        vk::to_string(format));
  }

  vk::DeviceSize imageSize = pixels.size();

  return createTexture(vulkanDevice, pixels.data(), imageSize, extent, format, commandPool,
                       transferQueue, false, {}, {}, 1, vk::ImageViewType::e2D, pCustomSamplerInfo);
}
