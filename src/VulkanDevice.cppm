module;

#include "imgui_impl_vulkan.h"
#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:VulkanDevice;

import vulkan_hpp;
import std;

export struct BufferResources {
  vk::raii::Buffer buffer{nullptr};
  vk::raii::DeviceMemory memory{nullptr};
};

export struct VulkanDevice {
private:
  vk::raii::PhysicalDevice physicalDevice{nullptr};
  vk::raii::Device device{nullptr};
  const vk::raii::Instance &instance;

public:
  u32 queueFamily = (u32)-1;
  vk::raii::Queue queue{nullptr};
  vk::raii::DescriptorPool descriptorPool{nullptr};

  VulkanDevice(const vk::raii::Instance &instance) : instance(instance) {}

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
    auto memProperties = physicalDevice.getMemoryProperties();

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
  const vk::raii::Device &logical() const { return device; }
  const vk::raii::PhysicalDevice &physical() const { return physicalDevice; }

  ImGui_ImplVulkan_InitInfo init_info() {
    return ImGui_ImplVulkan_InitInfo{
        .PhysicalDevice = *physicalDevice,
        .Device = *device,
        .QueueFamily = queueFamily,
        .Queue = *queue,
        .DescriptorPool = *descriptorPool,
    };
  }

  [[nodiscard]] std::expected<void, std::string> pickPhysicalDevice() {
    if (auto expected = instance.enumeratePhysicalDevices(); expected) {
      if (expected->empty()) {
        return std::unexpected("No Vulkan-compatible physical devices found!");
      }
      physicalDevice = std::move(expected->front());
      return {};
    } else {
      return std::unexpected("Failed to enumerate physical devices: " +
                             vk::to_string(expected.error()));
    }
  }

  [[nodiscard]] std::expected<void, std::string> createLogicalDevice() {
    queueFamily = selectQueueFamilyIndex(physicalDevice);
    if (queueFamily == (u32)-1) {
      return std::unexpected("Failed to select queue family index");
    }

    std::vector<const char *> deviceExtensions;
    deviceExtensions.push_back(vk::KHRSwapchainExtensionName);
    deviceExtensions.push_back(vk::KHRMapMemory2ExtensionName);

    u32 properties_count;
    std::vector<vk::ExtensionProperties> properties =
        physicalDevice.enumerateDeviceExtensionProperties();

    const float queue_priority[] = {1.0f};
    vk::DeviceQueueCreateInfo queue_info[1] = {};
    queue_info[0].queueFamilyIndex = queueFamily;
    queue_info[0].queueCount = 1;
    queue_info[0].pQueuePriorities = queue_priority;

    vk::PhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.samplerAnisotropy = true;
    if (auto expected = physicalDevice.createDevice({
            .queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]),
            .pQueueCreateInfos = queue_info,
            .enabledExtensionCount = static_cast<u32>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &enabledFeatures,
        });
        expected) {
      device = std::move(*expected);
    } else {
      return std::unexpected(std::format("error with device {}", vk::to_string(expected.error())));
    }
    if (auto expected = device.getQueue(queueFamily, 0); expected) {
      queue = std::move(*expected);
      return {};
    } else {
      return std::unexpected(std::format("error with queue {}", vk::to_string(expected.error())));
    }
  }

  [[nodiscard]] std::expected<void, std::string>
  createDescriptorPool(uint32_t imageCountBasedFactor) {
    if (!*device) {
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
    if (*descriptorPool) {
      // Consider device.waitIdle() here if descriptor sets from the old pool might still be in use.
      // For simplicity, assuming this is called during setup or a controlled resize.
      descriptorPool.clear(); // Resets the RAII handle, old pool is destroyed.
    }

    auto poolResult = device.createDescriptorPool(pool_info); // Using the member vk::raii::Device
    if (!poolResult) {
      return std::unexpected(
          std::format("VulkanDevice::createDescriptorPool: Failed to create descriptor pool: {} - "
                      "MaxSets: {}, PoolSizes: UBOs({}), DynUBOs({}), Samplers({})",
                      vk::to_string(poolResult.error()), pool_info.maxSets,
                      (pool_sizes.size() > 0 ? pool_sizes[0].descriptorCount : 0),
                      (pool_sizes.size() > 1 ? pool_sizes[1].descriptorCount : 0),
                      (pool_sizes.size() > 2 ? pool_sizes[2].descriptorCount : 0)));
    }
    descriptorPool = std::move(poolResult.value());
    return {};
  }

  [[nodiscard]] std::expected<BufferResources, std::string>
  createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags memoryProperties) NOEXCEPT {
    BufferResources resources;

    if (auto buf = device.createBuffer(
            {.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive})) {
      resources.buffer = std::move(*buf);
    } else {
      return std::unexpected("Buffer creation failed: " + vk::to_string(buf.error()));
    }

    auto memReqs = device.getBufferMemoryRequirements2({.buffer = resources.buffer});
    if (auto mem = device.allocateMemory(
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
};
