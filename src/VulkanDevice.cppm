module;

#include "imgui_impl_vulkan.h"
#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:VulkanDevice;

import vulkan_hpp;
import std;
import vk_mem_alloc_hpp;
import :VMA;

export struct BufferResources {
  vk::raii::Buffer buffer{nullptr};
  vk::raii::DeviceMemory memory{nullptr};
};

export struct VulkanDevice {
private:
  vk::raii::PhysicalDevice physicalDevice_{nullptr};
  vk::raii::Device device_{nullptr};
  const vk::raii::Instance &instance_;
  vma::Allocator vmaAllocator_;
  vk::raii::CommandPool transientCommandPool_{nullptr};

public:
  u32 queueFamily_ = (u32)-1;
  vk::raii::Queue queue_{nullptr};
  vk::raii::DescriptorPool descriptorPool_{nullptr};

  VulkanDevice(const vk::raii::Instance &instance) : instance_(instance) {}

private:
  u32 selectQueueFamilyIndex(const vk::raii::PhysicalDevice &physical_device) {
    auto queue_family_properties = physical_device.getQueueFamilyProperties();

    for (u32 i = 0; i < queue_family_properties.size(); i++) {
      if (queue_family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        return i;
      }
    }

    return (u32)-1;
  }

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice_.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    return -1;
    // throw std::runtime_error("Failed to find suitable memory type!");
  }

public:
  const vk::raii::Device &logical() const { return device_; }
  const vk::raii::PhysicalDevice &physical() const { return physicalDevice_; }

  ~VulkanDevice() { vmaAllocator_.destroy(); }

  ImGui_ImplVulkan_InitInfo init_info() {
    return ImGui_ImplVulkan_InitInfo{
        .PhysicalDevice = *physicalDevice_,
        .Device = *device_,
        .QueueFamily = queueFamily_,
        .Queue = *queue_,
        .DescriptorPool = *descriptorPool_,
    };
  }

  [[nodiscard]] std::expected<void, std::string> pickPhysicalDevice() {
    if (auto expected = instance_.enumeratePhysicalDevices(); expected) {
      if (expected->empty()) {
        return std::unexpected("No Vulkan-compatible physical devices found!");
      }
      physicalDevice_ = std::move(expected->front());
      return {};
    } else {
      return std::unexpected("Failed to enumerate physical devices: " +
                             vk::to_string(expected.error()));
    }
  }

  [[nodiscard]] std::expected<void, std::string> createLogicalDevice() {
    queueFamily_ = selectQueueFamilyIndex(physicalDevice_);
    if (queueFamily_ == (u32)-1) {
      return std::unexpected("Failed to select queue family index");
    }

    std::vector<const char *> deviceExtensions;
    deviceExtensions.push_back(vk::KHRSwapchainExtensionName);
    deviceExtensions.push_back(vk::KHRMapMemory2ExtensionName);

    u32 properties_count;
    std::vector<vk::ExtensionProperties> properties =
        physicalDevice_.enumerateDeviceExtensionProperties();

    const float queue_priority[] = {1.0f};
    vk::DeviceQueueCreateInfo queue_info[1] = {};
    queue_info[0].queueFamilyIndex = queueFamily_;
    queue_info[0].queueCount = 1;
    queue_info[0].pQueuePriorities = queue_priority;

    vk::PhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.samplerAnisotropy = true;
    if (auto expected = physicalDevice_.createDevice({
            .queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]),
            .pQueueCreateInfos = queue_info,
            .enabledExtensionCount = static_cast<u32>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &enabledFeatures,
        });
        expected) {
      device_ = std::move(*expected);
    } else {
      return std::unexpected(std::format("error with device {}", vk::to_string(expected.error())));
    }
    if (auto expected = device_.getQueue(queueFamily_, 0); expected) {
      queue_ = std::move(*expected);
    } else {
      return std::unexpected(std::format("error with queue {}", vk::to_string(expected.error())));
    }

    vk::CommandPoolCreateInfo poolCreateInfo{.flags =
                                                 vk::CommandPoolCreateFlagBits::eTransient |
                                                 vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                             .queueFamilyIndex = queueFamily_};
    auto transientPoolResult = device_.createCommandPool(poolCreateInfo);
    if (!transientPoolResult)
      return std::unexpected("Failed to create transient command pool: " +
                             vk::to_string(transientPoolResult.error()));
    transientCommandPool_ = std::move(transientPoolResult.value());

    if (auto expected = createVmaAllocator(instance_, physicalDevice_, device_); expected) {
      vmaAllocator_ = std::move(*expected);
      return {};
    } else {
      return std::unexpected(std::format("Failed to create Vma allocator:"));
    }
  }

  [[nodiscard]] std::expected<void, std::string>
  createDescriptorPool(uint32_t imageCountBasedFactor) {
    if (!*device_) {
      return std::unexpected("VulkanDevice::createDescriptorPool: Logical device is null. Call "
                             "createLogicalDevice first.");
    }

    // Calculate descriptor counts needed by the application
    uint32_t app_uniform_buffers = imageCountBasedFactor; // For static UBOs (1 per mesh * frame)
    uint32_t app_dynamic_uniform_buffers =
        imageCountBasedFactor; // For dynamic UBOs (1 descriptor per mesh * frame)
    uint32_t app_combined_image_samplers =
        imageCountBasedFactor; // For scene textures (1 per mesh * frame, assuming 1 texture for
                               // now)

    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        {vk::DescriptorType::eUniformBuffer, app_uniform_buffers},
        {vk::DescriptorType::eUniformBufferDynamic, app_dynamic_uniform_buffers},
        {
            vk::DescriptorType::eCombinedImageSampler,
            app_combined_image_samplers +
                IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE // App samplers + ImGui samplers
        }
        // Add other types if ImGui needs them from this pool, e.g.,
        // vk::DescriptorType::eInputAttachment
    };

    // Calculate maxSets:
    // Total sets your application will allocate for its meshes/nodes.
    // Each mesh/node allocates 'numSwapchainImages' static sets and 'numSwapchainImages' dynamic
    // sets. So, if imageCountBasedFactor = numMeshes * numSwapchainImages, then: Total app sets =
    // (numMeshes * numSwapchainImages for static) + (numMeshes * numSwapchainImages for dynamic)
    //                  = imageCountBasedFactor + imageCountBasedFactor = imageCountBasedFactor * 2.
    uint32_t application_max_sets = imageCountBasedFactor * 2;

    // ImGui also allocates descriptor sets. A common estimate is a few, e.g., for font texture.
    uint32_t imgui_estimated_sets = 10; // A small buffer for ImGui's own sets if it uses this pool.

    vk::DescriptorPoolCreateInfo pool_info{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = application_max_sets +
                   imgui_estimated_sets, // Total number of vkAllocateDescriptorSets calls allowed
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data() // Total individual descriptors of each type
    };

    // If descriptorPool can be re-created, ensure the old one is properly destroyed.
    if (*descriptorPool_) {
      // Consider device.waitIdle() here if descriptor sets from the old pool might still be in use.
      // For simplicity, assuming this is called during setup or a controlled resize.
      descriptorPool_.clear(); // Resets the RAII handle, old pool is destroyed.
    }

    auto poolResult = device_.createDescriptorPool(pool_info); // Using the member vk::raii::Device
    if (!poolResult) {
      return std::unexpected(
          std::format("VulkanDevice::createDescriptorPool: Failed to create descriptor pool: {} - "
                      "MaxSets: {}, PoolSizes: UBOs({}), DynUBOs({}), Samplers({})",
                      vk::to_string(poolResult.error()), pool_info.maxSets,
                      (pool_sizes.size() > 0 ? pool_sizes[0].descriptorCount : 0),
                      (pool_sizes.size() > 1 ? pool_sizes[1].descriptorCount : 0),
                      (pool_sizes.size() > 2 ? pool_sizes[2].descriptorCount : 0)));
    }
    descriptorPool_ = std::move(poolResult.value());
    return {};
  }

  [[nodiscard]] std::expected<BufferResources, std::string>
  createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags memoryProperties) NOEXCEPT {
    BufferResources resources;

    if (auto buf = device_.createBuffer(
            {.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive})) {
      resources.buffer = std::move(*buf);
    } else {
      return std::unexpected("Buffer creation failed: " + vk::to_string(buf.error()));
    }

    auto memReqs = device_.getBufferMemoryRequirements2({.buffer = resources.buffer});
    if (auto mem = device_.allocateMemory(
            {.allocationSize = memReqs.memoryRequirements.size,
             .memoryTypeIndex =
                 findMemoryType(memReqs.memoryRequirements.memoryTypeBits, memoryProperties)})) {
      resources.memory = std::move(*mem);
    } else {
      return std::unexpected("Memory allocation failed: " + vk::to_string(mem.error()));
    }

    // device.bindBufferMemory2(vk::BindBufferMemoryInfo{
    //     .buffer = resources.buffer, .memory = resources.memory, .memoryOffset = 0});
    return resources;
  }

  [[nodiscard]] std::expected<VmaBuffer, std::string> createBufferVMA(
      const vk::BufferCreateInfo &bufferCreateInfo,         // Pass vk::BufferCreateInfo directly
      const vma::AllocationCreateInfo &allocationCreateInfo // Pass vma::AllocationCreateInfo
      ) NOEXCEPT {
    if (!vmaAllocator_) {
      return std::unexpected("VMA Allocator not initialized in VulkanDevice::createBufferVMA.");
    }
    if (bufferCreateInfo.size == 0) {
      return std::unexpected("VulkanDevice::createBufferVMA: Buffer size cannot be zero.");
    }

    // vma::Allocator::createBuffer returns std::pair<vk::Buffer, vma::Allocation>
    // or throws on error if exceptions are enabled in vulkan-hpp.
    // With VULKAN_HPP_NO_EXCEPTIONS, it should return vk::ResultValue<std::pair<...>>

    // Need to convert vk::BufferCreateInfo to VkBufferCreateInfo (C-style) for VMA C++ bindings if
    // they expect it, or check if vma::Allocator::createBuffer can take vk::BufferCreateInfo
    // directly. The vk-mem-alloc-hpp createBuffer takes const VkBufferCreateInfo*.
    vk::BufferCreateInfo c_bufferCreateInfo = bufferCreateInfo;
    vma::AllocationCreateInfo c_allocationCreateInfo =
        allocationCreateInfo; // vma::AllocationCreateInfo is already a struct

    vk::Buffer outBuffer;
    vma::Allocation outAllocation;
    vma::AllocationInfo outAllocInfo; // To get mapped data if applicable

    // Use the vma::Allocator member function
    vk::Result result = vmaAllocator_.createBuffer(&c_bufferCreateInfo, &c_allocationCreateInfo,
                                                   &outBuffer,     // Takes vk::Buffer&
                                                   &outAllocation, // Takes vma::Allocation&
                                                   &outAllocInfo);

    if (result != vk::Result::eSuccess) {
      return std::unexpected("VMA failed to create buffer: " + vk::to_string(result));
    }

    return VmaBuffer(vmaAllocator_, outBuffer, outAllocation, outAllocInfo, bufferCreateInfo.size);
  }

  [[nodiscard]] std::expected<VmaImage, std::string> createImageVMA(
      const vk::ImageCreateInfo &imageCreateInfo,           // Pass vk::ImageCreateInfo directly
      const vma::AllocationCreateInfo &allocationCreateInfo // Pass vma::AllocationCreateInfo
      ) NOEXCEPT {
    if (!vmaAllocator_) {
      return std::unexpected("VMA Allocator not initialized in VulkanDevice::createImageVMA.");
    }

    vk::ImageCreateInfo c_imageCreateInfo = imageCreateInfo;
    vma::AllocationCreateInfo c_allocationCreateInfo = allocationCreateInfo;

    vk::Image outImage;
    vma::Allocation outAllocation;
    vma::AllocationInfo outAllocInfo;

    vk::Result result = vmaAllocator_.createImage(&c_imageCreateInfo, &c_allocationCreateInfo,
                                                  &outImage, &outAllocation, &outAllocInfo);

    if (result != vk::Result::eSuccess) {
      return std::unexpected("VMA failed to create image: " + vk::to_string(result));
    }

    return VmaImage(vmaAllocator_, outImage, outAllocation, outAllocInfo, imageCreateInfo.format,
                    imageCreateInfo.extent);
  }

  [[nodiscard]] std::expected<vk::raii::CommandBuffer, std::string> beginSingleTimeCommands() {
    if (!*transientCommandPool_) {
      return std::unexpected("VulkanDevice: Transient command pool not initialized.");
    }

    vk::CommandBufferAllocateInfo allocInfo{.commandPool = *transientCommandPool_,
                                            .level = vk::CommandBufferLevel::ePrimary,
                                            .commandBufferCount = 1};

    // Use allocateCommandBuffersUnique for RAII vk::raii::CommandBuffer
    auto cmdBuffersResult = device_.allocateCommandBuffers(allocInfo);
    if (!cmdBuffersResult) {
      return std::unexpected("Failed to allocate single-time command buffer: " +
                             vk::to_string(cmdBuffersResult.error()));
    }
    // The vector contains unique_ptr-like RAII command buffers. Move the first one out.
    vk::raii::CommandBuffer commandBuffer = std::move(cmdBuffersResult.value().front());

    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo); // begin() on vk::raii::CommandBuffer returns vk::Result
    return std::move(commandBuffer);
  }

  // *** NEW: Helper to end and submit a single-time command buffer ***
  [[nodiscard]] std::expected<void, std::string> endSingleTimeCommands(
      vk::raii::CommandBuffer commandBuffer) {      // Takes by value to manage lifetime
    if (!*commandBuffer || !*queue_ || !*device_) { // Basic checks
      return std::unexpected("VulkanDevice: Invalid parameter for endSingleTimeCommands.");
    }

    commandBuffer.end();

    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffer // Get raw handle from RAII wrapper
    };

    // Create a fence for synchronization
    auto fenceResult = device_.createFence({}); // RAII fence
    if (!fenceResult) {
      return std::unexpected("Failed to create fence for single-time command: " +
                             vk::to_string(fenceResult.error()));
    }
    vk::raii::Fence fence = std::move(*fenceResult);

    queue_.submit(submitInfo, *fence);

    vk::Result waitResult =
        device_.waitForFences({*fence}, VK_TRUE, UINT64_MAX); // Use VK_TRUE or true
    if (waitResult != vk::Result::eSuccess) {
      return std::unexpected("Failed to wait for single-time command fence: " +
                             vk::to_string(waitResult));
    }

    // Command buffer (and fence) will be automatically destroyed by RAII when they go out of scope.
    // If the command buffer was allocated from transientCommandPool_ with eResetCommandBuffer,
    // it's good practice to reset the pool or individual buffers if they are to be reused,
    // but for true one-time use, just letting RAII free it is fine.
    // The transientCommandPool_ itself can be reset via transientCommandPool_.reset() if desired,
    // for example, at the end of a larger loading operation.
    return {};
  }
};
