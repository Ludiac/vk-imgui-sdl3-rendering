module;

#include "imgui.h"
#include <vulkan/vulkan_core.h>

export module vulkan_app:DDX;

import vulkan_hpp;
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

struct Window {
  WindowConfig config;

  vk::raii::SurfaceKHR Surface{nullptr};
  vk::raii::SwapchainKHR Swapchain{nullptr};
  vk::raii::RenderPass RenderPass{nullptr};
  vk::raii::Pipeline Pipeline{nullptr};

  std::vector<Frame> Frames;
  std::vector<FrameSemaphores> FrameSemaphores;

  uint32_t FrameIndex{};
  uint32_t SemaphoreIndex{};
};

vk::SurfaceFormatKHR selectSurfaceFormat(const vk::raii::PhysicalDevice &physical_device,
                                         const vk::raii::SurfaceKHR &surface,
                                         std::span<vk::Format> request_formats,
                                         vk::ColorSpaceKHR request_color_space) {
  IM_ASSERT(!request_formats.empty());

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
  IM_ASSERT(!request_modes.empty());

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

std::expected<void, std::string> createWindowCommandBuffers(const VulkanDevice &device,
                                                            Window &wd) {
  for (uint32_t i = 0; i < wd.Frames.size(); i++) {
    decltype(auto) fd = wd.Frames[i];

    if (auto commandPool = device.logical().createCommandPool({
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = device.queueFamily,
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
  IM_ASSERT(0);
  return 1;
}

std::expected<void, std::string> createFramebuffers(const vk::raii::Device &device, Window &wd) {
  for (uint32_t i = 0; i < wd.Frames.size(); i++) {
    decltype(auto) fd = wd.Frames[i];

    const vk::FramebufferCreateInfo createInfo = {
        .renderPass = wd.RenderPass,
        .attachmentCount = 1,
        .pAttachments = &*fd.BackbufferView,
        .width = static_cast<uint32_t>(wd.config.swapchainExtent.width),
        .height = static_cast<uint32_t>(wd.config.swapchainExtent.height),
        .layers = 1};

    auto expected = device.createFramebuffer(createInfo);
    if (!expected) {
      return std::unexpected("Framebuffer creation failed: " + vk::to_string(expected.error()));
    }
    fd.Framebuffer = std::move(*expected);
  }
  return {};
}

std::expected<void, std::string> createImageViews(const vk::raii::Device &device, Window &wd) {
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

std::expected<void, std::string> createWindowSwapChain(const VulkanDevice &device, Window &wd_old,
                                                       vk::Extent2D extent,
                                                       uint32_t min_image_count) {
  Window wd_new{};
  device.logical().waitIdle();
  std::swap(wd_old.Surface, wd_new.Surface);
  std::swap(wd_old.config, wd_new.config);

  if (min_image_count == 0) {
    min_image_count = getMinImageCountFromPresentMode(wd_old.config.PresentMode);
  }

  auto cap = device.physical().getSurfaceCapabilitiesKHR(wd_new.Surface);

  vk::SwapchainCreateInfoKHR createInfo = {
      .surface = wd_new.Surface,
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
      .clipped = VK_TRUE,
      .oldSwapchain = *wd_old.Swapchain,
  };

  if (cap.currentExtent.width == 0xFFFFFFFF) {
    createInfo.imageExtent = extent;
    wd_new.config.swapchainExtent = extent;
  } else {
    createInfo.imageExtent = cap.currentExtent;
    wd_new.config.swapchainExtent = cap.currentExtent;
  }

  auto expected = device.logical().createSwapchainKHR(createInfo);
  if (!expected) {
    return std::unexpected("Swapchain creation failed: " + vk::to_string(expected.error()));
  }
  wd_new.Swapchain = std::move(*expected);

  auto images = wd_new.Swapchain.getImages();

  wd_new.Frames.resize(images.size());
  wd_new.FrameSemaphores.resize(images.size() + 1);

  for (uint32_t i = 0; i < wd_new.Frames.size(); ++i) {
    wd_new.Frames[i].Backbuffer = std::move(images[i]);
  }

  if (!wd_new.config.UseDynamicRendering) {
    const vk::AttachmentDescription attachment{.format = wd_new.config.SurfaceFormat.format,
                                               .samples = vk::SampleCountFlagBits::e1,
                                               .loadOp = wd_new.config.ClearEnable
                                                             ? vk::AttachmentLoadOp::eClear
                                                             : vk::AttachmentLoadOp::eDontCare,
                                               .storeOp = vk::AttachmentStoreOp::eStore,
                                               .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                               .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                               .initialLayout = vk::ImageLayout::eUndefined,
                                               .finalLayout = vk::ImageLayout::ePresentSrcKHR};

    const vk::AttachmentReference colorAttachmentRef{
        .attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};

    const vk::SubpassDescription subpass{.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                         .colorAttachmentCount = 1,
                                         .pColorAttachments = &colorAttachmentRef};

    const vk::SubpassDependency dependency{
        .srcSubpass = vk::SubpassExternal,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = {},
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite};

    const vk::RenderPassCreateInfo renderPassInfo{.attachmentCount = 1,
                                                  .pAttachments = &attachment,
                                                  .subpassCount = 1,
                                                  .pSubpasses = &subpass,
                                                  .dependencyCount = 1,
                                                  .pDependencies = &dependency};

    auto expected = device.logical().createRenderPass(renderPassInfo);
    if (!expected) {
      return std::unexpected("RenderPass creation failed: " + vk::to_string(expected.error()));
    }
    wd_new.RenderPass = std::move(*expected);
  }

  {
    createImageViews(device.logical(), wd_new);
  }

  if (!wd_new.config.UseDynamicRendering) {
    createFramebuffers(device.logical(), wd_new);
  }

  std::swap(wd_old, wd_new); // placing new window in place of old window
  return {};
}

void createOrResizeWindow(const vk::raii::Instance &instance, const VulkanDevice &device,
                          Window &wd, vk::Extent2D extent, uint32_t min_image_count) {
  createWindowSwapChain(device, wd, extent, min_image_count);
  createWindowCommandBuffers(device, wd);
}
