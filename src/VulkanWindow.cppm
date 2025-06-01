module;

// #include "imgui.h"
#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:VulkanWindow;

import vulkan_hpp;
import :VMA;
import :VulkanDevice;
import std;

struct Frame {
  vk::raii::CommandPool CommandPool{nullptr};
  vk::raii::CommandBuffer CommandBuffer{nullptr};
  vk::raii::Fence Fence{nullptr};
  vk::Image Backbuffer{nullptr}; // no raii, destroyed by swapchain
  vk::raii::ImageView BackbufferView{nullptr};
  vk::raii::Framebuffer Framebuffer{nullptr};
};

struct FrameSemaphores {
  vk::raii::Semaphore ImageAcquiredSemaphore{nullptr};
  vk::raii::Semaphore RenderCompleteSemaphore{nullptr};
};

struct WindowConfig {
  vk::Extent2D swapchainExtent;
  vk::SurfaceFormatKHR SurfaceFormat;
  vk::PresentModeKHR PresentMode;
  bool UseDynamicRendering{};
  bool ClearEnable{};
  vk::ClearValue ClearValue{};
};

// In VulkanWindow.cppm, modify the Window struct:
struct Window {
  WindowConfig config;

  vk::raii::SurfaceKHR Surface{nullptr};
  vk::raii::SwapchainKHR Swapchain{nullptr};
  vk::raii::RenderPass RenderPass{nullptr};

  VmaImage depthVmaImage;
  vk::raii::ImageView depthImageView{nullptr};
  vk::Format depthFormat{}; // To store the chosen depth format

  std::vector<Frame> Frames;
  std::vector<FrameSemaphores> FrameSemaphores;

  uint32_t FrameIndex{};
  uint32_t SemaphoreIndex{};
};

// Add these helper functions (if not already present) in VulkanWindow.cppm
// or a module imported by VulkanWindow.cppm
vk::Format findSupportedFormat(const vk::raii::PhysicalDevice &physicalDevice,
                               const std::vector<vk::Format> &candidates, vk::ImageTiling tiling,
                               vk::FormatFeatureFlags features) {
  for (vk::Format format : candidates) {
    vk::FormatProperties props = physicalDevice.getFormatProperties(format);
    if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == vk::ImageTiling::eOptimal &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }
  std::exit(0);
}

vk::Format findDepthFormat(const vk::raii::PhysicalDevice &physicalDevice) {
  return findSupportedFormat(
      physicalDevice,
      {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
      vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

// Helper for finding memory type (if not already in your VulkanDevice or utils)
uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t typeFilter,
                        vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  std::exit(0);
  return (u32)-1;
}

vk::SurfaceFormatKHR selectSurfaceFormat(const vk::raii::PhysicalDevice &physical_device,
                                         const vk::raii::SurfaceKHR &surface,
                                         std::span<vk::Format> request_formats,
                                         vk::ColorSpaceKHR request_color_space) {
  // IM_ASSERT(!request_formats.empty());

  const auto avail_formats = physical_device.getSurfaceFormatsKHR(surface);

  if (avail_formats.size() == 1) {
    return avail_formats[0].format == vk::Format::eUndefined
               ? vk::SurfaceFormatKHR{request_formats.front(), request_color_space}
               : avail_formats[0];
  }

  for (decltype(auto) requested : request_formats) {
    for (decltype(auto) available : avail_formats) {
      if (available.format == requested && available.colorSpace == request_color_space) {
        return available;
      }
    }
  }

  return avail_formats.front();
}

vk::PresentModeKHR selectPresentMode(const vk::raii::PhysicalDevice &physical_device,
                                     const vk::raii::SurfaceKHR &surface,
                                     std::span<const vk::PresentModeKHR> request_modes) {
  // IM_ASSERT(!request_modes.empty());

  const auto avail_modes = physical_device.getSurfacePresentModesKHR(surface);

  for (decltype(auto) requested : request_modes) {
    for (decltype(auto) available : avail_modes) {
      if (available == requested) {
        return requested;
      }
    }
  }

  return vk::PresentModeKHR::eFifo;
}

[[nodiscard]] std::expected<void, std::string>
createWindowCommandBuffers(const VulkanDevice &device, Window &wd) {
  for (uint32_t i = 0; i < wd.Frames.size(); i++) {
    decltype(auto) fd = wd.Frames[i];

    if (auto commandPool = device.logical().createCommandPool({
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = device.queueFamily_,
        });
        commandPool) {
      fd.CommandPool = std::move(*commandPool);
    } else {
      return std::unexpected("Failed to create command pool: " +
                             vk::to_string(commandPool.error()));
    }

    if (auto commandBuffers = device.logical().allocateCommandBuffers({
            .commandPool = *fd.CommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        });
        commandBuffers) {
      fd.CommandBuffer = std::move(commandBuffers->front());
    } else {
      return std::unexpected("Failed to allocate command buffer: " +
                             vk::to_string(commandBuffers.error()));
    }

    if (auto fence = device.logical().createFence({
            .flags = vk::FenceCreateFlagBits::eSignaled,
        });
        fence) {
      fd.Fence = std::move(*fence);
    } else {
      return std::unexpected("Failed to create fence: " + vk::to_string(fence.error()));
    }
  }

  for (uint32_t i = 0; i < wd.FrameSemaphores.size(); i++) {
    FrameSemaphores *fsd = &wd.FrameSemaphores[i];

    if (auto imageSem = device.logical().createSemaphore({}); imageSem) {
      fsd->ImageAcquiredSemaphore = std::move(*imageSem);
    } else {
      return std::unexpected("Failed to create image semaphore: " +
                             vk::to_string(imageSem.error()));
    }

    if (auto renderSem = device.logical().createSemaphore({}); renderSem) {
      fsd->RenderCompleteSemaphore = std::move(*renderSem);
    } else {
      return std::unexpected("Failed to create render semaphore: " +
                             vk::to_string(renderSem.error()));
    }
  }

  return {};
}

int getMinImageCountFromPresentMode(vk::PresentModeKHR present_mode) {
  if (present_mode == vk::PresentModeKHR::eMailbox)
    return 3;
  if (present_mode == vk::PresentModeKHR::eFifo || present_mode == vk::PresentModeKHR::eFifoRelaxed)
    return 2;
  if (present_mode == vk::PresentModeKHR::eImmediate)
    return 1;
  // IM_ASSERT(0);
  return 1;
}

// Helper to transition image layout (you might have this in a texture utility or VulkanDevice)
void transitionImageLayout(const vk::raii::Device &device,
                           const vk::raii::CommandBuffer &commandBuffer, vk::Image image,
                           vk::Format format, vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout) {
  vk::ImageMemoryBarrier barrier{};
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
  barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
  barrier.image = image;
  barrier.subresourceRange.aspectMask =
      vk::ImageAspectFlagBits::eDepth; // Assuming depth only, adjust if stencil
  if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
    if (format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint) {
      barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
    }
  }
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  vk::PipelineStageFlags sourceStage;
  vk::PipelineStageFlags destinationStage;

  if (oldLayout == vk::ImageLayout::eUndefined &&
      newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
    barrier.srcAccessMask = vk::AccessFlagBits::eNone; // Updated from eNoneKHR
    barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead |
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
  } else {
    std::exit(0);
  }
  commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, barrier);
}

// In VulkanWindow.cppm or wherever Window struct and createDepthResources are
// Make sure Window struct stores VmaImage for depth:
/*
export module vulkan_app:VulkanWindow;
import :VMA; // For VmaImage
// ...
export struct Window {
    // ... other members ...
    VmaImage depthVmaImage; // Replaces depthImage and depthImageMemory
    vk::raii::ImageView depthImageView{nullptr}; // ImageView is still separate
    vk::Format depthFormat = vk::Format::eUndefined;
    // ...
};
*/

[[nodiscard]] std::expected<void, std::string> createDepthResources( // Renamed for clarity
    VulkanDevice &vulkan_device, Window &wd, vk::Extent2D extent) {
  wd.depthFormat = findDepthFormat(vulkan_device.physical());
  if (wd.depthFormat == vk::Format::eUndefined) {
    return std::unexpected("Failed to find suitable depth format.");
  }

  vk::ImageCreateInfo imageCi{
      .imageType = vk::ImageType::e2D,
      .format = wd.depthFormat,
      .extent = vk::Extent3D{extent.width, extent.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
      .sharingMode = vk::SharingMode::eExclusive,
      .initialLayout =
          vk::ImageLayout::eUndefined // VMA doesn't set initial layout for optimal images
  };

  vma::AllocationCreateInfo imageAllocInfo{
      .usage = vma::MemoryUsage::eAutoPreferDevice // Depth buffers are best in device local memory
      // No eMapped flag needed as CPU doesn't typically access depth buffer directly
  };

  auto vmaImageResult = vulkan_device.createImageVMA(imageCi, imageAllocInfo);
  if (!vmaImageResult) {
    return std::unexpected("VMA depth image creation failed: " + vmaImageResult.error());
  }
  wd.depthVmaImage = std::move(vmaImageResult.value()); // Store VmaImage

  // --- Create Depth Image View ---
  vk::ImageSubresourceRange depthSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                                  .baseMipLevel = 0,
                                                  .levelCount = 1,
                                                  .baseArrayLayer = 0,
                                                  .layerCount = 1};
  if (wd.depthFormat == vk::Format::eD32SfloatS8Uint ||
      wd.depthFormat == vk::Format::eD24UnormS8Uint) {
    depthSubresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
  }

  vk::ImageViewCreateInfo viewInfo{
      .image = wd.depthVmaImage.get(), // Get VkImage handle from VmaImage
      .viewType = vk::ImageViewType::e2D,
      .format = wd.depthVmaImage.getFormat(), // Get format from VmaImage
      .subresourceRange = depthSubresourceRange};

  // If wd.depthImageView is vk::raii::ImageView, create it as unique
  if (*wd.depthImageView)
    wd.depthImageView.clear(); // Clear existing if any
  auto viewResult = vulkan_device.logical().createImageView(viewInfo);
  if (!viewResult) {
    return std::unexpected("Failed to create depth image view (VMA): " +
                           vk::to_string(viewResult.error()));
  }
  wd.depthImageView = std::move(viewResult.value());

  // --- Transition Image Layout using VulkanDevice helpers ---
  auto cmdBufferExpected = vulkan_device.beginSingleTimeCommands();
  if (!cmdBufferExpected) {
    return std::unexpected("VMA Depth: Failed to begin single-time commands: " +
                           cmdBufferExpected.error());
  }
  vk::raii::CommandBuffer cmdBuffer = std::move(cmdBufferExpected.value());

  // Record barrier for layout transition
  vk::ImageMemoryBarrier barrier{.srcAccessMask = vk::AccessFlagBits::eNone,
                                 .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                                  vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                                 .oldLayout = vk::ImageLayout::eUndefined,
                                 .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                 .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                                 .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                                 .image =
                                     wd.depthVmaImage.get(), // Use VkImage handle from VmaImage
                                 .subresourceRange = depthSubresourceRange};
  cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            {}, nullptr, nullptr, {barrier});

  auto endCmdResult = vulkan_device.endSingleTimeCommands(std::move(cmdBuffer));
  if (!endCmdResult) {
    return std::unexpected("VMA Depth: Failed to end single-time commands: " +
                           endCmdResult.error());
  }

  std::println("Depth resources (VMA) created successfully.");
  return {};
}

[[nodiscard]] std::expected<void, std::string>
createFramebuffers(const vk::raii::Device &device,
                   Window &wd) { // Takes non-const Window&
  for (uint32_t i = 0; i < wd.Frames.size(); i++) {
    decltype(auto) fd = wd.Frames[i];

    // Attachments: 0 = Color, 1 = Depth
    std::array<vk::ImageView, 2> attachments = {
        *fd.BackbufferView, // Color attachment view
        *wd.depthImageView  // Depth attachment view (from the Window struct)
    };

    const vk::FramebufferCreateInfo createInfo = {
        .renderPass = *wd.RenderPass, // Use the render pass that includes depth
        .attachmentCount = static_cast<uint32_t>(attachments.size()), // Now 2 attachments
        .pAttachments = attachments.data(),
        .width = static_cast<uint32_t>(wd.config.swapchainExtent.width),
        .height = static_cast<uint32_t>(wd.config.swapchainExtent.height),
        .layers = 1};

    if (*fd.Framebuffer) {
      fd.Framebuffer.clear();
    }

    auto expected = device.createFramebuffer(createInfo);
    if (!expected) {
      return std::unexpected("Framebuffer creation failed for frame " + std::to_string(i) + ": " +
                             vk::to_string(expected.error()));
    }
    fd.Framebuffer = std::move(expected.value());
  }
  return {};
}

[[nodiscard]] std::expected<void, std::string> createImageViews(const vk::raii::Device &device,
                                                                Window &wd) {
  for (uint32_t i = 0; i < wd.Frames.size(); i++) {
    decltype(auto) fd = wd.Frames[i];

    const vk::ImageViewCreateInfo createInfo = {
        .image = fd.Backbuffer,
        .viewType = vk::ImageViewType::e2D,
        .format = wd.config.SurfaceFormat.format,
        .components = {.r = vk::ComponentSwizzle::eR,
                       .g = vk::ComponentSwizzle::eG,
                       .b = vk::ComponentSwizzle::eB,
                       .a = vk::ComponentSwizzle::eA},
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1}};

    auto expected = device.createImageView(createInfo);
    if (!expected) {
      return std::unexpected("ImageView creation failed: " + vk::to_string(expected.error()));
    }
    fd.BackbufferView = std::move(*expected);
  }
  return {};
}

[[nodiscard]] std::expected<void, std::string> createWindowSwapChain(VulkanDevice &device,
                                                                     Window &wd_old,
                                                                     vk::Extent2D extent,
                                                                     uint32_t min_image_count) {
  Window wd_new{};
  device.logical().waitIdle();

  std::swap(wd_old.Surface, wd_new.Surface);
  std::swap(wd_old.config, wd_new.config);

  if (min_image_count == 0) {
    min_image_count =
        getMinImageCountFromPresentMode(wd_new.config.PresentMode); // Use wd_new.config
  }

  auto cap = device.physical().getSurfaceCapabilitiesKHR(*wd_new.Surface); // Use *wd_new.Surface

  vk::SwapchainCreateInfoKHR createInfo = {
      .surface = *wd_new.Surface,
      .minImageCount = std::clamp(min_image_count, cap.minImageCount,
                                  cap.maxImageCount > 0 ? cap.maxImageCount : 0x7FFFFFFF),
      .imageFormat = wd_new.config.SurfaceFormat.format,
      .imageColorSpace = wd_new.config.SurfaceFormat.colorSpace,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = cap.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = wd_new.config.PresentMode,
      .clipped = true,
      .oldSwapchain = wd_old.Swapchain,
  };

  if (cap.currentExtent.width == 0xFFFFFFFF) {
    createInfo.imageExtent = extent;
  } else {
    createInfo.imageExtent = cap.currentExtent;
  }
  wd_new.config.swapchainExtent = createInfo.imageExtent; // Update config with actual extent

  auto expectedSwapchain = device.logical().createSwapchainKHR(createInfo);
  if (!expectedSwapchain) {
    return std::unexpected("Swapchain creation failed: " +
                           vk::to_string(expectedSwapchain.error()));
  }
  wd_new.Swapchain = std::move(expectedSwapchain.value());

  auto images = wd_new.Swapchain.getImages();
  wd_new.Frames.resize(images.size());
  if (wd_new.FrameSemaphores.size() != images.size() + 1) {
    wd_new.FrameSemaphores.resize(images.size() + 1);
  }

  for (uint32_t i = 0; i < wd_new.Frames.size(); ++i) {
    wd_new.Frames[i].Backbuffer = images[i];
  }

  auto depthResult = createDepthResources(device, wd_new, wd_new.config.swapchainExtent);
  if (!depthResult) {
    return std::unexpected("Failed to create depth resources: " + depthResult.error());
  }
  // At this point, wd_new.depthImage, depthImageView, depthImageMemory, depthFormat are populated.
  // The depth image layout also needs to be transitioned to eDepthStencilAttachmentOptimal.
  // This transition should be part of createDepthResources or done immediately after using a
  // command buffer. For now, assuming createDepthResources handles the transition or you have a
  // mechanism.
  // **CRITICAL**: Ensure depth image is transitioned to
  // vk::ImageLayout::eDepthStencilAttachmentOptimal before being used in a render pass. This
  // usually requires a command buffer submission.

  if (!wd_new.config
           .UseDynamicRendering) { // Assuming you create RenderPass if not using dynamic rendering
    // *** MODIFIED RENDER PASS CREATION ***
    vk::AttachmentDescription colorAttachment{.format = wd_new.config.SurfaceFormat.format,
                                              .samples = vk::SampleCountFlagBits::e1,
                                              .loadOp = wd_new.config.ClearEnable
                                                            ? vk::AttachmentLoadOp::eClear
                                                            : vk::AttachmentLoadOp::eDontCare,
                                              .storeOp = vk::AttachmentStoreOp::eStore,
                                              .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                              .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                              .initialLayout = vk::ImageLayout::eUndefined,
                                              .finalLayout = vk::ImageLayout::ePresentSrcKHR};

    vk::AttachmentDescription depthAttachment{
        // New depth attachment
        .format = wd_new.depthFormat, // Use the format chosen by findDepthFormat
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,      // Clear depth at the start of the pass
        .storeOp = vk::AttachmentStoreOp::eDontCare, // We don't need to store depth after render
                                                     // pass (unless for post-processing)
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout =
            vk::ImageLayout::eDepthStencilAttachmentOptimal, // Or eDepthStencilAttachmentOptimal if
                                                             // transitioned already
        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

    std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

    vk::AttachmentReference colorAttachmentRef{.attachment = 0, // Color attachment is at index 0
                                               .layout = vk::ImageLayout::eColorAttachmentOptimal};
    vk::AttachmentReference depthAttachmentRef{                 // New depth attachment reference
                                               .attachment = 1, // Depth attachment is at index 1
                                               .layout =
                                                   vk::ImageLayout::eDepthStencilAttachmentOptimal};

    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pDepthStencilAttachment = &depthAttachmentRef // Set the depth stencil attachment
    };

    // Dependency to ensure depth buffer is ready before depth tests, and color is ready for writing
    vk::SubpassDependency dependency{
        .srcSubpass = vk::SubpassExternal,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eNone, // Updated from eNoneKHR
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite};

    vk::RenderPassCreateInfo renderPassInfo{
        .attachmentCount = static_cast<uint32_t>(attachments.size()), // Now 2 attachments
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency};

    auto expectedRenderPass = device.logical().createRenderPass(renderPassInfo);
    if (!expectedRenderPass) {
      return std::unexpected("RenderPass creation failed: " +
                             vk::to_string(expectedRenderPass.error()));
    }
    wd_new.RenderPass = std::move(expectedRenderPass.value());
  }

  auto imageViewResult = createImageViews(device.logical(), wd_new);
  if (!imageViewResult) {
    return std::unexpected(imageViewResult.error());
  }

  if (!wd_new.config.UseDynamicRendering) {
    auto framebufferResult = createFramebuffers(device.logical(), wd_new);
    if (!framebufferResult) {
      return std::unexpected(framebufferResult.error());
    }
  }

  std::swap(wd_old, wd_new);
  return {};
}

void createOrResizeWindow(const vk::raii::Instance &instance, VulkanDevice &device, Window &wd,
                          vk::Extent2D extent, uint32_t min_image_count) {
  EXPECTED_VOID(createWindowSwapChain(device, wd, extent, min_image_count));
  EXPECTED_VOID(createWindowCommandBuffers(device, wd));
}
