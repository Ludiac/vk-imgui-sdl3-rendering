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
import :TextRenderer;

// A simple vertex for 2D UI elements. No UVs or normals needed for now.
struct UIVertex {
  glm::vec2 pos;
  glm::vec4 color;
};

export struct UIPushConstants {
  glm::mat4 projection;
};

export class UIRenderer {
private:
  VulkanDevice &device;
  u32 frameCount;

  // A buffer of vertices and indices for each frame-in-flight.
  std::vector<VmaBuffer> vertexBuffers;
  std::vector<VmaBuffer> indexBuffers;

  // We store the vertices and indices on the CPU first.
  std::vector<std::vector<UIVertex>> cpuVertices;
  std::vector<std::vector<u32>> cpuIndices;

  u32 maxVerticesPerFrame;

public:
  UIRenderer(VulkanDevice &dev, u32 inFlightFrameCount)
      : device(dev), frameCount(inFlightFrameCount), maxVerticesPerFrame(10000) {
    cpuVertices.resize(frameCount);
    cpuIndices.resize(frameCount);

    EXPECTED_VOID(createDynamicBuffers());
  }

  // Clears the CPU-side buffers for the current frame. Call this at the start of a frame.
  void beginFrame(u32 frameIndex) {
    if (frameIndex < frameCount) {
      cpuVertices[frameIndex].clear();
      cpuIndices[frameIndex].clear();
    }
  }

  // Queues a Sheet's background quad to be rendered.
  void queueSheet(const Sheet &sheet, u32 frameIndex) {
    if (frameIndex >= frameCount)
      return;

    auto &vertices = cpuVertices[frameIndex];
    auto &indices = cpuIndices[frameIndex];

    u32 firstVertexIndex = static_cast<u32>(vertices.size());

    // Add the 4 corners of the quad
    vertices.push_back({{sheet.position.x, sheet.position.y}, sheet.backgroundColor}); // Top-left
    vertices.push_back(
        {{sheet.position.x + sheet.size.x, sheet.position.y}, sheet.backgroundColor}); // Top-right
    vertices.push_back({{sheet.position.x + sheet.size.x, sheet.position.y + sheet.size.y},
                        sheet.backgroundColor}); // Bottom-right
    vertices.push_back({{sheet.position.x, sheet.position.y + sheet.size.y},
                        sheet.backgroundColor}); // Bottom-left

    // Add indices for two triangles (clockwise)
    indices.push_back(firstVertexIndex + 0);
    indices.push_back(firstVertexIndex + 1);
    indices.push_back(firstVertexIndex + 2);
    indices.push_back(firstVertexIndex + 2);
    indices.push_back(firstVertexIndex + 3);
    indices.push_back(firstVertexIndex + 0);
  }

  // Uploads the queued UI data to the GPU and issues a single draw call.
  void draw(const vk::raii::CommandBuffer &cmd, const VulkanPipeline &pipeline,
            vk::Extent2D windowSize, u32 frameIndex) {
    if (frameIndex >= frameCount || cpuVertices[frameIndex].empty()) {
      return;
    }

    // --- 1. Copy data to GPU buffers for the current frame ---
    auto &currentVertexBuffer = vertexBuffers[frameIndex];
    auto &currentIndexBuffer = indexBuffers[frameIndex];
    auto &vertices = cpuVertices[frameIndex];
    auto &indices = cpuIndices[frameIndex];

    size_t vertexDataSize = vertices.size() * sizeof(UIVertex);
    size_t indexDataSize = indices.size() * sizeof(u32);

    if (vertexDataSize > currentVertexBuffer.getAllocationInfo().size ||
        indexDataSize > currentIndexBuffer.getAllocationInfo().size) {
      std::println("UI data exceeds buffer size for this frame. Truncating.");
      // In a real app, you might re-allocate the buffer here.
      return;
    }

    std::memcpy(currentVertexBuffer.getMappedData(), vertices.data(), vertexDataSize);
    std::memcpy(currentIndexBuffer.getMappedData(), indices.data(), indexDataSize);

    // --- 2. Issue Draw Commands ---
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.pipeline);

    cmd.bindVertexBuffers(0, {currentVertexBuffer.get()}, {0});
    cmd.bindIndexBuffer(currentIndexBuffer.get(), 0, vk::IndexType::eUint32);

    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(windowSize.width),
                                 static_cast<float>(windowSize.height), 0.0f);

    UIPushConstants constants{.projection = ortho};
    cmd.pushConstants<UIPushConstants>(*pipeline.pipelineLayout, vk::ShaderStageFlagBits::eVertex,
                                       0, constants);

    cmd.drawIndexed(static_cast<u32>(indices.size()), 1, 0, 0, 0);
  }

private:
  [[nodiscard]] std::expected<void, std::string> createDynamicBuffers() {
    vertexBuffers.resize(frameCount);
    indexBuffers.resize(frameCount);

    vk::DeviceSize vertexBufferSize = maxVerticesPerFrame * sizeof(UIVertex);
    vk::DeviceSize indexBufferSize =
        maxVerticesPerFrame * 1.5 * sizeof(u32); // Approx 6 indices for 4 vertices

    for (u32 i = 0; i < frameCount; ++i) {
      // Create Vertex Buffer
      auto vbResult = device.createBufferVMA(
          {.size = vertexBufferSize, .usage = vk::BufferUsageFlagBits::eVertexBuffer},
          {.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                    vma::AllocationCreateFlagBits::eMapped,
           .usage = vma::MemoryUsage::eAuto});
      if (!vbResult)
        return std::unexpected("Failed to create UI vertex buffer " + std::to_string(i));
      vertexBuffers[i] = std::move(*vbResult);

      // Create Index Buffer
      auto ibResult = device.createBufferVMA(
          {.size = indexBufferSize, .usage = vk::BufferUsageFlagBits::eIndexBuffer},
          {.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                    vma::AllocationCreateFlagBits::eMapped,
           .usage = vma::MemoryUsage::eAuto});
      if (!ibResult)
        return std::unexpected("Failed to create UI index buffer " + std::to_string(i));
      indexBuffers[i] = std::move(*ibResult);
    }
    return {};
  }
};
