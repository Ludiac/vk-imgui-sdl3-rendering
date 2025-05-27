module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>

export module vulkan_app:mesh;
import vulkan_hpp;
import std;
import :VulkanDevice;
import :texture;
import :VulkanPipeline;
export class Mesh {
public:
  std::string name;
  Material material;
  PBRTextures textures;
  uint32_t indexCount{0};

private:
  VulkanDevice &device;
  u32 imageCount_member;
  std::vector<Vertex> vertices_data;
  std::vector<uint32_t> indices_data;
  vk::raii::Buffer vertexBuffer{nullptr};
  vk::raii::DeviceMemory vertexBufferMemory{nullptr};
  vk::raii::Buffer indexBuffer{nullptr};
  vk::raii::DeviceMemory indexBufferMemory{nullptr};
  vk::raii::Buffer mvpUniformBuffers{nullptr};
  vk::raii::DeviceMemory mvpUniformBuffersMemory{nullptr};
  vk::raii::Buffer materialUniformBuffer{nullptr};
  vk::raii::DeviceMemory materialUniformBufferMemory{nullptr};
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  [[nodiscard]] std::expected<void, std::string> createVertexBuffer() NOEXCEPT {
    if (vertices_data.empty()) {
      vertexBuffer = nullptr;
      vertexBufferMemory = nullptr;
      return {};
    }
    const auto bufferSize = sizeof(Vertex) * vertices_data.size();
    auto resources = device.createBuffer(bufferSize, vk::BufferUsageFlagBits::eVertexBuffer,
                                         vk::MemoryPropertyFlagBits::eHostVisible |
                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!resources) {
      std::string errorMsg =
          "Mesh " + name + ": Vertex buffer creation failed: " + resources.error();
      return std::unexpected(errorMsg);
    }
    device.logical().bindBufferMemory2(vk::BindBufferMemoryInfo{
        .buffer = resources->buffer, .memory = resources->memory, .memoryOffset = 0});
    void *mappedDataVoid = device.logical().mapMemory2KHR({
        .memory = resources->memory,
        .offset = 0,
        .size = bufferSize,
    });
    char *data = static_cast<char *>(mappedDataVoid);
    if (!data) {
      std::string errorMsg = "Mesh " + name + ": Failed to map vertex buffer memory.";
      return std::unexpected(errorMsg);
    }
    std::memcpy(data, vertices_data.data(), bufferSize);
    device.logical().unmapMemory2KHR({.memory = resources->memory});
    vertexBuffer = std::move(resources->buffer);
    vertexBufferMemory = std::move(resources->memory);
    return {};
  }
  [[nodiscard]] std::expected<void, std::string> createIndexBuffer() NOEXCEPT {
    if (indices_data.empty()) {
      indexBuffer = nullptr;
      indexBufferMemory = nullptr;
      this->indexCount = 0;
      return {};
    }
    this->indexCount = static_cast<uint32_t>(indices_data.size());
    const auto bufferSize = sizeof(uint32_t) * indices_data.size();
    auto resources = device.createBuffer(bufferSize, vk::BufferUsageFlagBits::eIndexBuffer,
                                         vk::MemoryPropertyFlagBits::eHostVisible |
                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!resources) {
      std::string errorMsg =
          "Mesh " + name + ": Index buffer creation failed: " + resources.error();
      return std::unexpected(errorMsg);
    }
    device.logical().bindBufferMemory2(vk::BindBufferMemoryInfo{
        .buffer = resources->buffer, .memory = resources->memory, .memoryOffset = 0});
    void *mappedDataVoid = device.logical().mapMemory2KHR({
        .memory = resources->memory,
        .offset = 0,
        .size = bufferSize,
    });
    char *data = static_cast<char *>(mappedDataVoid);
    if (!data) {
      std::string errorMsg = "Mesh " + name + ": Failed to map index buffer memory.";
      return std::unexpected(errorMsg);
    }
    std::memcpy(data, indices_data.data(), bufferSize);
    device.logical().unmapMemory2KHR({.memory = resources->memory});
    indexBuffer = std::move(resources->buffer);
    indexBufferMemory = std::move(resources->memory);
    return {};
  }
  [[nodiscard]] std::expected<void, std::string> createMvpUniformBuffers() NOEXCEPT {
    if (imageCount_member == 0) {
      std::string errorMsg = "Mesh " + name + ": Image count is zero for MVP UBO creation.";
      return std::unexpected(errorMsg);
    }
    auto bufferSize = sizeof(UniformBufferObject) * imageCount_member;
    if (bufferSize == 0) {
      mvpUniformBuffers = nullptr;
      mvpUniformBuffersMemory = nullptr;
      return {};
    }
    auto resources = device.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                         vk::MemoryPropertyFlagBits::eHostVisible |
                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!resources) {
      std::string errorMsg = "Mesh " + name + ": MVP UBO creation failed: " + resources.error();
      return std::unexpected(errorMsg);
    }
    device.logical().bindBufferMemory2(vk::BindBufferMemoryInfo{
        .buffer = resources->buffer, .memory = resources->memory, .memoryOffset = 0});
    mvpUniformBuffers = std::move(resources->buffer);
    mvpUniformBuffersMemory = std::move(resources->memory);
    return {};
  }
  [[nodiscard]] std::expected<void, std::string> createSingleMaterialUniformBuffer() NOEXCEPT {
    if (imageCount_member == 0) {
      std::string errorMsg = "Mesh " + name + ": Image count is zero for Material UBO creation.";
      return std::unexpected(errorMsg);
    }
    auto alignment = device.physical().getProperties().limits.minUniformBufferOffsetAlignment;
    auto alignedMaterialSize = (sizeof(Material) + alignment - 1) & ~(alignment - 1);
    auto bufferSize = alignedMaterialSize * imageCount_member;
    if (bufferSize == 0) {
      materialUniformBuffer = nullptr;
      materialUniformBufferMemory = nullptr;
      return {};
    }
    auto resources = device.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                         vk::MemoryPropertyFlagBits::eHostVisible |
                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!resources) {
      std::string errorMsg =
          "Mesh " + name + ": Material UBO creation failed: " + resources.error();
      return std::unexpected(errorMsg);
    }
    device.logical().bindBufferMemory2(vk::BindBufferMemoryInfo{
        .buffer = resources->buffer, .memory = resources->memory, .memoryOffset = 0});
    materialUniformBuffer = std::move(resources->buffer);
    materialUniformBufferMemory = std::move(resources->memory);
    for (uint32_t i = 0; i < imageCount_member; ++i) {
      updateMaterialUniformBufferData(i);
    }
    return {};
  }

public:
  Mesh(VulkanDevice &dev, std::string meshName, std::vector<Vertex> &&verts,
       std::vector<uint32_t> &&meshIndices, Material initMaterial, PBRTextures initPbrTextures,
       u32 numImages)
      : device(dev), name(std::move(meshName)), material(std::move(initMaterial)),
        textures(std::move(initPbrTextures)), imageCount_member(numImages),
        vertices_data(std::move(verts)), indices_data(std::move(meshIndices)) {
    EXPECTED_VOID(createVertexBuffer());
    EXPECTED_VOID(createIndexBuffer());
    EXPECTED_VOID(createMvpUniformBuffers());
    EXPECTED_VOID(createSingleMaterialUniformBuffer());
  }

  void updateMaterialUniformBufferData(uint32_t currentImage) {
    if (!*materialUniformBufferMemory || currentImage >= imageCount_member) {
      return;
    }
    auto alignment = device.physical().getProperties().limits.minUniformBufferOffsetAlignment;
    auto alignedMaterialSize = (sizeof(Material) + alignment - 1) & ~(alignment - 1);
    vk::DeviceSize offset = currentImage * alignedMaterialSize;
    void *mappedDataVoid = device.logical().mapMemory2KHR({
        .memory = *materialUniformBufferMemory,
        .offset = offset,
        .size = sizeof(Material),
    });
    char *data = static_cast<char *>(mappedDataVoid);
    if (!data) {
      std::println("Error: Mesh '{}': Failed to map material UBO for image index {}.", name,
                   currentImage);
      return;
    }
    std::memcpy(data, &this->material, sizeof(Material));
    device.logical().unmapMemory2KHR({.memory = *materialUniformBufferMemory});
  }

  void updateMvpUniformBuffer(uint32_t currentImage, const glm::mat4 &model, const glm::mat4 &view,
                              const glm::mat4 &projection) {
    if (!*mvpUniformBuffersMemory || currentImage >= imageCount_member) {
      return;
    }
    UniformBufferObject ubo{};
    ubo.model = model;
    ubo.view = view;
    ubo.projection = projection;
    vk::DeviceSize offset = sizeof(UniformBufferObject) * currentImage;
    void *mappedDataVoid = device.logical().mapMemory2KHR({
        .memory = *mvpUniformBuffersMemory,
        .offset = offset,
        .size = sizeof(ubo),
    });
    char *data = static_cast<char *>(mappedDataVoid);
    if (!data) {
      return;
    }
    std::memcpy(data, &ubo, sizeof(ubo));
    device.logical().unmapMemory2KHR({.memory = *mvpUniformBuffersMemory});
  }

  [[nodiscard]] std::expected<void, std::string>
  allocateDescriptorSets(const vk::raii::DescriptorPool &pool,
                         const vk::raii::DescriptorSetLayout &combinedLayout) {
    if (imageCount_member == 0) {
      std::string errorMsg =
          "Mesh " + name + ": Image count is zero for descriptor set allocation.";
      return std::unexpected(errorMsg);
    }
    if (!*combinedLayout) {
      std::string errorMsg =
          "Mesh " + name + ": Combined layout is null for descriptor set allocation.";
      return std::unexpected(errorMsg);
    }
    std::vector<vk::DescriptorSetLayout> layouts(imageCount_member, *combinedLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *pool,
        .descriptorSetCount = imageCount_member,
        .pSetLayouts = layouts.data(),
    };
    auto setsResult = device.logical().allocateDescriptorSets(allocInfo);
    if (!setsResult) {
      std::string errorMsg = "Mesh " + name + ": Failed to allocate combined descriptor sets: " +
                             vk::to_string(setsResult.error());
      return std::unexpected(errorMsg);
    }
    descriptorSets = std::move(setsResult.value());
    return {};
  }

  void updateDescriptorSetContents(u32 currentImage) {
    if (currentImage >= imageCount_member || descriptorSets.empty() ||
        currentImage >= descriptorSets.size() || !*descriptorSets[currentImage]) {
      return;
    }
    std::vector<vk::WriteDescriptorSet> writes;
    vk::DescriptorBufferInfo mvpUboInfoStorage;
    vk::DescriptorBufferInfo materialUboInfoStorage;
    vk::DescriptorImageInfo baseColorTextureInfoStorage;
    if (*mvpUniformBuffers) {
      mvpUboInfoStorage =
          vk::DescriptorBufferInfo{.buffer = *mvpUniformBuffers,
                                   .offset = currentImage * sizeof(UniformBufferObject),
                                   .range = sizeof(UniformBufferObject)};
      writes.emplace_back(
          vk::WriteDescriptorSet{.dstSet = *descriptorSets[currentImage],
                                 .dstBinding = 0,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = vk::DescriptorType::eUniformBuffer,
                                 .pBufferInfo = &mvpUboInfoStorage});
    } else {
    }
    if (*materialUniformBuffer) {
      auto alignment = device.physical().getProperties().limits.minUniformBufferOffsetAlignment;
      auto alignedMaterialSize = (sizeof(Material) + alignment - 1) & ~(alignment - 1);
      materialUboInfoStorage =
          vk::DescriptorBufferInfo{.buffer = *materialUniformBuffer,
                                   .offset = currentImage * alignedMaterialSize,
                                   .range = sizeof(Material)};
      writes.emplace_back(
          vk::WriteDescriptorSet{.dstSet = *descriptorSets[currentImage],
                                 .dstBinding = 1,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = vk::DescriptorType::eUniformBuffer,
                                 .pBufferInfo = &materialUboInfoStorage});
    } else {
    }
    if (textures.baseColor && *textures.baseColor->view && *textures.baseColor->sampler) {
      baseColorTextureInfoStorage =
          vk::DescriptorImageInfo{.sampler = *textures.baseColor->sampler,
                                  .imageView = *textures.baseColor->view,
                                  .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
      writes.emplace_back(
          vk::WriteDescriptorSet{.dstSet = *descriptorSets[currentImage],
                                 .dstBinding = 2,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                 .pImageInfo = &baseColorTextureInfoStorage});
    } else {
    }
    if (!writes.empty()) {
      device.logical().updateDescriptorSets(writes, nullptr);
    } else {
    }
  }

  void bind(vk::raii::CommandBuffer &cmd, VulkanPipeline *pipeline, u32 currentImage) const {
    if (!*vertexBuffer) {
      return;
    }
    if (!*indexBuffer && indexCount > 0) {
      return;
    }
    if (descriptorSets.empty() || currentImage >= descriptorSets.size() ||
        !*descriptorSets[currentImage]) {
      return;
    }
    if (!pipeline || !*pipeline->pipelineLayout) {
      return;
    }
    cmd.bindVertexBuffers(0, {*vertexBuffer}, {0});
    if (indexCount > 0 && *indexBuffer) {
      cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
    }
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->pipelineLayout, 0,
                           {*descriptorSets[currentImage]}, {});
  }
  void draw(vk::raii::CommandBuffer &cmd, VulkanPipeline * /*pipeline*/,
            u32 /*currentImage*/) const {
    if (indexCount > 0) {
      cmd.drawIndexed(indexCount, 1, 0, 0, 0);
    } else if (!vertices_data.empty()) {
    } else {
    }
  }
  [[nodiscard]] std::expected<void, std::string> setImageCount(u32 newCount) NOEXCEPT {
    if (newCount == imageCount_member) {
      return {};
    }
    imageCount_member = newCount;
    EXPECTED_VOID(createMvpUniformBuffers());
    EXPECTED_VOID(createSingleMaterialUniformBuffer());
    descriptorSets.clear();
    return {};
  }
  Material &getMaterial() { return material; }
  const std::string &getName() const { return name; }
};
