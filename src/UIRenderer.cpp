module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:UIRenderer;

import vulkan_hpp;
import std;

import :VulkanDevice;
import :VulkanPipeline;
import :VMA;
import :ui;

// The vertex format for our static unit quad.
struct UIQuadVertex {
  glm::vec2 pos;
  glm::vec2 uv;
};

// Data for a single UI element instance.
// This gets sent to the GPU via an SSBO.
struct UIInstanceData {
  glm::vec2 screenPos;
  glm::vec2 scale;
  glm::vec4 color;
  float z_layer;
  // Future SDF parameters would go here.
};

export struct UIPushConstants {
  glm::mat4 projection;
};

export class UIRenderer {
private:
  VulkanDevice &device;
  u32 frameCount;

  // A single, large buffer for all instance data, one per frame-in-flight.
  std::vector<VmaBuffer> instanceBuffers;
  u32 maxQuadsPerFrame;

  // We store the instance data on the CPU first before uploading.
  std::vector<UIInstanceData> queuedInstances;

  // One static, unchanging quad mesh shared by all instances.
  VmaBuffer staticVertexBuffer;
  VmaBuffer staticIndexBuffer;

  // Descriptor set resources pointing to the instance buffers.
  vk::raii::DescriptorSetLayout instanceDataLayout{nullptr};
  std::vector<vk::raii::DescriptorSet> instanceDataDescriptorSets;

public:
  UIRenderer(VulkanDevice &dev, u32 inFlightFrameCount, const vk::raii::DescriptorPool &pool)
      : device(dev), frameCount(inFlightFrameCount), maxQuadsPerFrame(10000) {
    EXPECTED_VOID(createInstanceBuffers(device, frameCount, maxQuadsPerFrame, instanceBuffers));
    EXPECTED_VOID(createStaticQuadBuffers(device, staticVertexBuffer, staticIndexBuffer));
    EXPECTED_VOID(createInstanceDataDescriptorSetLayout());
    EXPECTED_VOID(allocateInstanceDataDescriptorSets(pool));
  }

  // Clears the CPU-side queue for the new frame.
  void beginFrame() { queuedInstances.clear(); }

  // Queues a Sheet to be rendered as a quad.
  void queueSheet(const Sheet &sheet, float z_layer) {
    if (queuedInstances.size() >= maxQuadsPerFrame)
      return;

    queuedInstances.emplace_back(UIInstanceData{.screenPos = sheet.position,
                                                .scale = sheet.size,
                                                .color = sheet.backgroundColor,
                                                .z_layer = z_layer});
  }

  // Uploads all queued data to the GPU and issues a single instanced draw call.
  void draw(const vk::raii::CommandBuffer &cmd, const VulkanPipeline &pipeline,
            vk::Extent2D windowSize, u32 frameIndex) {
    if (frameIndex >= frameCount || queuedInstances.empty()) {
      return;
    }

    // --- 1. Copy all instance data for the frame to the GPU ---
    VmaBuffer &currentInstanceBuffer = instanceBuffers[frameIndex];
    size_t dataSize = queuedInstances.size() * sizeof(UIInstanceData);
    std::memcpy(currentInstanceBuffer.getMappedData(), queuedInstances.data(), dataSize);

    // --- 2. Issue Draw Commands ---
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.pipeline);

    // Bind the descriptor set for the instance buffer
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.pipelineLayout, 0,
                           {*instanceDataDescriptorSets[frameIndex]}, {});

    // Bind the static quad mesh
    cmd.bindVertexBuffers(0, {staticVertexBuffer.get()}, {0});
    cmd.bindIndexBuffer(staticIndexBuffer.get(), 0, vk::IndexType::eUint32);

    // Push the projection matrix
    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(windowSize.width),
                                 static_cast<float>(windowSize.height), 0.0f, 1.0f, -1.0f);
    UIPushConstants constants{.projection = ortho};
    cmd.pushConstants<UIPushConstants>(*pipeline.pipelineLayout, vk::ShaderStageFlagBits::eVertex,
                                       0, constants);

    // Draw all instances in one go!
    cmd.drawIndexed(6, static_cast<u32>(queuedInstances.size()), 0, 0, 0);
  }

private:
  [[nodiscard]] std::expected<void, std::string> createInstanceDataDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding instanceBinding{.binding = 0,
                                                   .descriptorType =
                                                       vk::DescriptorType::eStorageBuffer,
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

    for (u32 i = 0; i < frameCount; ++i) {
      vk::DescriptorBufferInfo bufferInfo{
          .buffer = instanceBuffers[i].get(), .offset = 0, .range = vk::WholeSize};
      vk::WriteDescriptorSet write{.dstSet = *instanceDataDescriptorSets[i],
                                   .dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageBuffer,
                                   .pBufferInfo = &bufferInfo};
      device.logical().updateDescriptorSets({write}, nullptr);
    }
    return {};
  }
};
