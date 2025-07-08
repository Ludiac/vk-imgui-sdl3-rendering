module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:UISystem;

import vulkan_hpp;
import std;

import :VulkanDevice;
import :VulkanPipeline;
import :VMA;
import :ui; // For Sheet, UIInstanceData, etc.

export struct UIPushConstants2D {
  glm::mat4 projection;
};

// This class is responsible for managing and preparing UI geometry for rendering.
// It no longer issues draw calls itself.
export class UISystem {
private:
  VulkanDevice &device;
  u32 frameCount;
  u32 maxQuadsPerFrame;
  size_t minStorageBufferOffsetAlignment; // This isn't strictly needed if you always start at
                                          // offset 0 for the single UI batch.
  std::vector<VmaBuffer> instanceBuffers;
  std::vector<UIInstanceData> queuedInstances; // All UI instances for the current frame
  VmaBuffer staticVertexBuffer;
  VmaBuffer staticIndexBuffer;
  vk::raii::DescriptorSetLayout instanceDataLayout{nullptr};
  std::vector<vk::raii::DescriptorSet> instanceDataDescriptorSets;

public:
  UISystem(VulkanDevice &dev, u32 inFlightFrameCount, const vk::raii::DescriptorPool &pool)
      : device(dev), frameCount(inFlightFrameCount), maxQuadsPerFrame(2048) {
    this->minStorageBufferOffsetAlignment =
        device.limits.minStorageBufferOffsetAlignment; // Keep for consistency, but not used for
                                                       // alignment here.
    EXPECTED_VOID(createInstanceBuffers(device, frameCount, maxQuadsPerFrame, instanceBuffers,
                                        sizeof(UIInstanceData)));
    EXPECTED_VOID(createStaticQuadBuffers(device, staticVertexBuffer, staticIndexBuffer));
    EXPECTED_VOID(createInstanceDataDescriptorSetLayout());
    EXPECTED_VOID(allocateInstanceDataDescriptorSets(pool));
  }

  void beginFrame() { queuedInstances.clear(); }

  void queueQuad(Quad quad) {
    if (queuedInstances.size() >= maxQuadsPerFrame)
      return;
    queuedInstances.emplace_back(UIInstanceData{.quad = quad});
  }

  // REPLACES the old `draw` method.
  // This method prepares the data and adds a RenderBatch to the queue.
  void prepareBatches(RenderQueue &queue, const VulkanPipeline &uiPipeline, u32 frameIndex,
                      vk::Extent2D windowSize) {
    glm::mat4 ortho = glm::ortho(0.0f, (float)windowSize.width, 0.0f, (float)windowSize.height);

    if (frameIndex >= frameCount || queuedInstances.empty()) {
      return;
    }

    // 1. Copy instance data to the GPU buffer for this frame
    VmaBuffer &currentInstanceBuffer = instanceBuffers[frameIndex];
    size_t dataSize = queuedInstances.size() * sizeof(UIInstanceData);
    // Ensure the buffer is large enough for all queued instances
    if (dataSize > currentInstanceBuffer.getAllocationInfo().size) {
      // Log an error or resize the buffer (more complex)
      return;
    }
    std::memcpy(static_cast<char *>(currentInstanceBuffer.getMappedData()), queuedInstances.data(),
                dataSize);
    RenderBatch batch{
        .sortKey = 100, // UI background elements could have a low sort key
        .pipeline = &uiPipeline.pipeline,
        .pipelineLayout = &uiPipeline.pipelineLayout,
        .instanceDataSet = &instanceDataDescriptorSets[frameIndex],
        .textureSet = nullptr, // No texture for basic UI sheets
        .vertexBuffer = &staticVertexBuffer,
        .indexBuffer = &staticIndexBuffer,
        .indexCount = 6, // Quad has 6 indices
        .instanceCount = static_cast<u32>(queuedInstances.size()),
        .firstInstance = 0,
        .dynamicOffset = 0,
        // .scale = {0, 0},
    };

    UIPushConstants2D pc;
    pc.projection = ortho;
    batch.pushConstantSize = sizeof(UIPushConstants2D);
    batch.pushConstantStages = vk::ShaderStageFlagBits::eVertex;
    std::memcpy(batch.pushConstantData.data(), &pc, sizeof(UIPushConstants2D));

    queue.emplace_back(batch);

    // Update the descriptor set to reflect the *actual* number of instances we're drawing this
    // frame.
    updateDescriptorSets(static_cast<u32>(queuedInstances.size()), frameIndex);
  }

private:
  [[nodiscard]] std::expected<void, std::string> createInstanceDataDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding instanceBinding{.binding = 0,
                                                   .descriptorType =
                                                       vk::DescriptorType::eStorageBufferDynamic,
                                                   .descriptorCount = 1,
                                                   .stageFlags = vk::ShaderStageFlagBits::eVertex};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = 1, .pBindings = &instanceBinding};
    auto layoutResult = device.logical().createDescriptorSetLayout(layoutInfo);
    if (!layoutResult)
      return std::unexpected("Failed to create UI instance data descriptor set layout.");
    instanceDataLayout = std::move(layoutResult.value());
    return {};
  }

  [[nodiscard]] std::expected<void, std::string>
  allocateInstanceDataDescriptorSets(const vk::raii::DescriptorPool &pool) {
    std::vector<vk::DescriptorSetLayout> layouts(frameCount, *instanceDataLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = pool, .descriptorSetCount = frameCount, .pSetLayouts = layouts.data()};
    auto setsResult = device.logical().allocateDescriptorSets(allocInfo);
    if (!setsResult)
      return std::unexpected("Failed to allocate UI instance descriptor sets.");
    instanceDataDescriptorSets = std::move(setsResult.value());
    return {};
  }

  void updateDescriptorSets(u32 instanceNumber, u32 frameIndex) {
    vk::DescriptorBufferInfo bufferInfo{
        .buffer = instanceBuffers[frameIndex].get(),
        .offset = 0,
        .range = instanceNumber * sizeof(UIInstanceData), // <--- CHANGE to UIInstanceData
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
