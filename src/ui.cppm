module;

#include "macros.hpp"
#include "primitive_types.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:ui;

import vulkan_hpp;
import std;

import :VulkanDevice;
import :VulkanPipeline;
import :VMA;
import :texture;
import :text; // For FontAtlasData

constexpr void convertToCCW(std::vector<uint32_t> &indices) {
  for (size_t i = 0; i + 2 < indices.size(); i += 3) {
    std::swap(indices[i + 1], indices[i + 2]);
  }
}

export struct Font {
  friend class TextSystem; // Allow renderer to access private members
private:
  FontAtlasData atlasData;
  std::shared_ptr<Texture> texture;
  vk::raii::DescriptorSet textureDescriptorSet{nullptr};

public:
  const FontAtlasData &getAtlasData() const { return atlasData; }
};

// The vertex layout for the single static quad.
export struct TextQuadVertex {
  glm::vec2 pos;
  glm::vec2 uv;
};

// This struct defines the unique data for each character instance.
// It will be sent to the shader via a storage buffer.
export struct TextInstanceData {
  glm::vec2 screenPos; // Top-left position of the quad
  glm::vec2 scale;     // width and height of the quad
  glm::vec2 uvTopLeft;
  glm::vec2 uvBottomRight;
  glm::vec4 color; // NEW: Color is now per-instance
};

// A shared push constant struct for a simple orthographic projection.
// All 2D pipelines can share a layout that expects this.
export struct OrthoPushConstants {
  glm::mat4 projection;
};

export struct Sheet {
  glm::vec2 position{0.0f};                          // Top-left corner in screen pixels
  glm::vec2 size{100.0f, 100.0f};                    // Width and height in pixels
  glm::vec4 backgroundColor{0.1f, 0.1f, 0.1f, 0.8f}; // RGBA for the sheet's background

  // Margins (padding) from the edges of the sheet
  float marginTop = 5.0f;
  float marginRight = 5.0f;
  float marginBottom = 5.0f;
  float marginLeft = 5.0f;
};

// Represents a block of text to be rendered within a Sheet.
export struct TextBlock {
  std::string text;
  glm::vec4 color{1.0f, 1.0f, 1.0f, 1.0f}; // Default to white text
  float fontSize = 48.0f;                  // This is illustrative; actual size is from FontAtlas
  // Future properties could include alignment (left, center, right)
};

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
  float _padding[3]; // Explicit padding to fill up to a 16-byte boundary or for future use
  // Future SDF parameters:
  float cornerRadius;
  float borderWidth;
  float _padding2[2]; // Padding for the next float4
};

export struct UIPushConstants {
  glm::mat4 projection;
};

export struct RenderBatch {
  // A key for sorting. Lower numbers are drawn first.
  // Can be used for layering (e.g., UI background = 100, UI foreground = 200)
  int sortKey{0};

  // Pipeline state
  vk::raii::Pipeline const *pipeline;
  vk::raii::PipelineLayout const *pipelineLayout;

  // Resources to bind (Descriptor Sets)
  vk::raii::DescriptorSet const *instanceDataSet{nullptr}; // Set 0: for instance SSBO
  vk::raii::DescriptorSet const *textureSet{nullptr};      // Set 1: for textures (optional)

  // Mesh data
  VmaBuffer const *vertexBuffer;
  VmaBuffer const *indexBuffer;
  uint32_t indexCount;

  // Instancing data
  uint32_t instanceCount;
  uint32_t firstInstance; // Base instance for vkCmdDrawIndexed
  uint32_t dynamicOffset{0};

  // Custom comparison operator to enable sorting the render queue.
  // This is the key to minimizing GPU state changes.
  bool operator<(const RenderBatch &other) const {
    if (sortKey != other.sortKey) {
      return sortKey < other.sortKey;
    }
    if (pipeline != other.pipeline) {
      return pipeline < other.pipeline;
    }
    if (textureSet != other.textureSet) {
      return textureSet < other.textureSet;
    }
    // Add other criteria if needed
    return false;
  }
};

// The RenderQueue is simply a vector of batches for a given frame.
export using RenderQueue = std::vector<RenderBatch>;

export size_t pad_uniform_buffer_size(size_t originalSize, size_t minAlignment) {
  if (minAlignment > 0) {
    return (originalSize + minAlignment - 1) & ~(minAlignment - 1);
  }
  return originalSize;
}

export [[nodiscard]] std::expected<void, std::string>
createStaticQuadBuffers(VulkanDevice &device, VmaBuffer &vertexBuffer, VmaBuffer &indexBuffer) {
  const std::vector<TextQuadVertex> vertices = {
      {{0.0f, 1.0f}, {0.0f, 1.0f}}, // Bottom-left
      {{1.0f, 1.0f}, {1.0f, 1.0f}}, // Bottom-right
      {{1.0f, 0.0f}, {1.0f, 0.0f}}, // Top-right
      {{0.0f, 0.0f}, {0.0f, 0.0f}}  // Top-left
  };
  std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
  convertToCCW(indices);

  auto createStagingBuffer = [&](vk::DeviceSize size,
                                 const void *data) -> std::expected<VmaBuffer, std::string> {
    vk::BufferCreateInfo bufInfo{.size = size, .usage = vk::BufferUsageFlagBits::eTransferSrc};
    vma::AllocationCreateInfo allocInfo{.flags = vma::AllocationCreateFlagBits::eMapped,
                                        .usage = vma::MemoryUsage::eCpuOnly};
    auto res = device.createBufferVMA(bufInfo, allocInfo);
    if (!res)
      return std::unexpected("Failed to create staging buffer");
    VmaBuffer buf = std::move(*res);
    std::memcpy(buf.getMappedData(), data, static_cast<size_t>(size));
    return std::expected<VmaBuffer, std::string>(std::move(buf));
  };

  auto createDeviceBuffer =
      [&](vk::DeviceSize size,
          vk::BufferUsageFlags usage) -> std::expected<VmaBuffer, std::string> {
    vk::BufferCreateInfo bufInfo{.size = size,
                                 .usage = usage | vk::BufferUsageFlagBits::eTransferDst};
    vma::AllocationCreateInfo allocInfo{.usage = vma::MemoryUsage::eGpuOnly};
    auto res = device.createBufferVMA(bufInfo, allocInfo);
    if (!res)
      return std::unexpected("Failed to create device-local buffer");
    return std::expected<VmaBuffer, std::string>(std::move(*res));
  };

  // Vertex buffer
  const vk::DeviceSize vertexSize = sizeof(TextQuadVertex) * vertices.size();
  auto stagingVb = createStagingBuffer(vertexSize, vertices.data());
  if (!stagingVb)
    return std::unexpected("Failed to create text staging VB");

  auto vertexBuf = createDeviceBuffer(vertexSize, vk::BufferUsageFlagBits::eVertexBuffer);
  if (!vertexBuf)
    return std::unexpected("Failed to create text static VB");

  EXPECTED_VOID(device.copyBuffer(stagingVb->get(), vertexBuf->get(), vertexSize));
  vertexBuffer = std::move(*vertexBuf);

  // Index buffer
  const vk::DeviceSize indexSize = sizeof(uint32_t) * indices.size();
  auto stagingIb = createStagingBuffer(indexSize, indices.data());
  if (!stagingIb)
    return std::unexpected("Failed to create text staging IB");

  auto indexBuf = createDeviceBuffer(indexSize, vk::BufferUsageFlagBits::eIndexBuffer);
  if (!indexBuf)
    return std::unexpected("Failed to create text static IB");

  EXPECTED_VOID(device.copyBuffer(stagingIb->get(), indexBuf->get(), indexSize));
  indexBuffer = std::move(*indexBuf);

  return {};
}

export [[nodiscard]] std::expected<void, std::string>
createInstanceBuffers(VulkanDevice &device, u32 frameCount, u32 size,
                      std::vector<VmaBuffer> &buffers, size_t instanceSize) {
  buffers.resize(frameCount);
  vk::DeviceSize bufferSize = size * instanceSize;

  for (u32 i = 0; i < frameCount; ++i) {
    vk::BufferCreateInfo bufferInfo{
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer // It's a storage buffer now
    };
    // These buffers need to be updated from the CPU every frame.
    vma::AllocationCreateInfo allocInfo{
        .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                 vma::AllocationCreateFlagBits::eMapped,
        .usage = vma::MemoryUsage::eAuto};

    auto bufferResult = device.createBufferVMA(bufferInfo, allocInfo);
    if (!bufferResult) {
      return std::unexpected("Failed to create text instance buffer " + std::to_string(i));
    }
    buffers[i] = std::move(*bufferResult);
  }
  return {};
}

// This is the core execution unit. It is stateless and simply processes
// the command list given to it.
export void processRenderQueue(const vk::raii::CommandBuffer &cmd, vk::Extent2D windowSize,
                               RenderQueue &queue) {
  if (queue.empty()) {
    return;
  }

  std::sort(queue.begin(), queue.end());

  vk::raii::Pipeline const *lastPipeline = nullptr;
  vk::raii::DescriptorSet const *lastTextureSet = nullptr;
  VmaBuffer const *lastVertexBuffer = nullptr;
  VmaBuffer const *lastIndexBuffer = nullptr;

  for (const auto &batch : queue) {
    // 1. Bind Pipeline (only if it has changed)
    if (batch.pipeline != lastPipeline) {
      cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *batch.pipeline);

      glm::mat4 ortho = glm::ortho(0.0f, float(windowSize.width), 0.0f, float(windowSize.height),
                                   0.0f, 1.0f // ← near=0, far=1
      );
      OrthoPushConstants pushConstant{ortho};
      cmd.pushConstants<OrthoPushConstants>(*batch.pipelineLayout, vk::ShaderStageFlagBits::eVertex,
                                            0, pushConstant);

      lastPipeline = batch.pipeline;
      // A new pipeline means we MUST rebind all descriptors and buffers.
      lastTextureSet = nullptr;
      lastVertexBuffer = nullptr;
      lastIndexBuffer = nullptr;
    }
    // Instance data is expected to be unique per-batch, so we always bind it.
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *batch.pipelineLayout, 0,
                           {*batch.instanceDataSet},
                           {batch.dynamicOffset}); // <--- PROVIDE OFFSET HERE
    // 2. Bind Descriptor Sets
    if (batch.textureSet && batch.textureSet != lastTextureSet) {
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *batch.pipelineLayout, 1,
                             {*batch.textureSet}, {});
      lastTextureSet = batch.textureSet;
    }

    // 3. Bind Buffers (only if they have changed)
    if (batch.vertexBuffer != lastVertexBuffer) {
      cmd.bindVertexBuffers(0, {batch.vertexBuffer->get()}, {0});
      lastVertexBuffer = batch.vertexBuffer;
    }
    if (batch.indexBuffer != lastIndexBuffer) {
      cmd.bindIndexBuffer(batch.indexBuffer->get(), 0, vk::IndexType::eUint32);
      lastIndexBuffer = batch.indexBuffer;
    }

    // 4. Draw!
    cmd.drawIndexed(batch.indexCount, batch.instanceCount, 0, 0, 0);
  }
}
