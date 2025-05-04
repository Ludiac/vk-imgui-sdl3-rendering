module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:mesh;

import vulkan_hpp;
import std;
import :DDX;
import :VulkanDevice;
import :extra;

export struct Submesh {
  uint32_t indexOffset;
  uint32_t indexCount;
  Material material;
};

export class Mesh {
  VulkanDevice &device;

  std::vector<Submesh> submeshes{
      {.indexOffset = 0,
       .indexCount = 36,
       .material =
           {
               .baseColorFactor = {1, 0, 0, 1},
           }},
  };

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  vk::raii::Buffer vertexBuffer{nullptr};
  vk::raii::DeviceMemory vertexBufferMemory{nullptr};

  vk::raii::Buffer indexBuffer{nullptr};
  vk::raii::DeviceMemory indexBufferMemory{nullptr};

  vk::raii::Buffer uniformBuffers{nullptr};
  vk::raii::DeviceMemory uniformBuffersMemory{nullptr};

  vk::raii::Buffer materialUniformBuffers{nullptr};
  vk::raii::DeviceMemory materialUniformBuffersMemory{nullptr};

  std::vector<vk::raii::DescriptorSet> descriptorSets;

public:
  Transform transform;

  Mesh(VulkanDevice &dev, std::vector<Vertex> &&verts, std::vector<u32> indices, Material mat = {})
      : device(dev), vertices(std::move(verts)), indices(std::move(indices)) {
    EXPECTED_VOID(createVertexBuffer());
    EXPECTED_VOID(createIndexBuffer());
  }

  std::expected<void, std::string> createVertexBuffer() NOEXCEPT {
    const auto bufferSize = sizeof(Vertex) * vertices.size();

    auto resources = device.createBuffer(bufferSize, vk::BufferUsageFlagBits::eVertexBuffer,
                                         vk::MemoryPropertyFlagBits::eHostVisible |
                                             vk::MemoryPropertyFlagBits::eHostCoherent);

    if (!resources)
      return std::unexpected(resources.error());

    device.logical().bindBufferMemory2(vk::BindBufferMemoryInfo{
        .buffer = resources->buffer, .memory = resources->memory, .memoryOffset = 0});

    auto data = device.logical().mapMemory2KHR({
        .memory = resources->memory,
        .offset = 0,
        .size = bufferSize,
    });

    std::memcpy(data, vertices.data(), bufferSize);
    device.logical().unmapMemory2KHR(vk::MemoryUnmapInfoKHR{.memory = resources->memory});

    vertexBuffer = std::move(resources->buffer);
    vertexBufferMemory = std::move(resources->memory);

    // std::println("vertexBuffer created successfully!");
    return {};
  }

  std::expected<void, std::string> createIndexBuffer() NOEXCEPT {
    const auto bufferSize = sizeof(uint32_t) * indices.size();

    auto resources = device.createBuffer(bufferSize,
                                         vk::BufferUsageFlagBits::eIndexBuffer, // <-- Key change
                                         vk::MemoryPropertyFlagBits::eHostVisible |
                                             vk::MemoryPropertyFlagBits::eHostCoherent);

    if (!resources)
      return std::unexpected(resources.error());

    device.logical().bindBufferMemory2(vk::BindBufferMemoryInfo{
        .buffer = resources->buffer, .memory = resources->memory, .memoryOffset = 0});

    auto data = device.logical().mapMemory2KHR({
        .memory = resources->memory,
        .offset = 0,
        .size = bufferSize,
    });
    std::memcpy(data, indices.data(), bufferSize);
    device.logical().unmapMemory2KHR({.memory = resources->memory});

    indexBuffer = std::move(resources->buffer);
    indexBufferMemory = std::move(resources->memory);

    // std::println("indexBuffer created successfully!");
    return {};
  }

  std::expected<void, std::string> createUniformBuffers(uint32_t imageCount) NOEXCEPT {
    auto resources = device.createBuffer(
        sizeof(UniformBufferObject) * imageCount, vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!resources)
      return std::unexpected(resources.error());

    vk::BindBufferMemoryInfo bindInfo{
        .buffer = resources->buffer, .memory = resources->memory, .memoryOffset = 0};

    uniformBuffers = std::move(resources->buffer);
    uniformBuffersMemory = std::move(resources->memory);
    device.logical().bindBufferMemory2(bindInfo);

    // std::println("uniformBuffers created successfully!");
    return {};
  }

  std::expected<void, std::string> createMaterialUniformBuffers(uint32_t imageCount) NOEXCEPT {
    auto alignment = device.physical().getProperties().limits.minUniformBufferOffsetAlignment;
    auto alignedSize = (sizeof(Material) + alignment - 1) & ~(alignment - 1);
    auto bufferSize = alignedSize * imageCount * submeshes.size();
    auto resources = device.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                         vk::MemoryPropertyFlagBits::eHostVisible |
                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!resources)
      return std::unexpected(resources.error());

    vk::BindBufferMemoryInfo bindInfo{
        .buffer = resources->buffer,
        .memory = resources->memory,
        .memoryOffset = 0,
    };

    materialUniformBuffers = std::move(resources->buffer);
    materialUniformBuffersMemory = std::move(resources->memory);
    device.logical().bindBufferMemory2(bindInfo);

    // std::println("material uniformBuffers created successfully!");
    return {};
  }

  void updateUniformBuffer(uint32_t currentImage, const glm::mat4 &model, const glm::mat4 &view,
                           const glm::mat4 &projection) {
    UniformBufferObject ubo{};
    ubo.model = model;
    ubo.view = view;
    ubo.projection = projection;

    auto data = device.logical().mapMemory2KHR({
        .memory = uniformBuffersMemory,
        .offset = sizeof(ubo) * currentImage,
        .size = sizeof(ubo),
    });
    std::memcpy(data, &ubo, sizeof(ubo));
    device.logical().unmapMemory2KHR({.memory = uniformBuffersMemory});
  }

  void updateMaterialBuffer(uint32_t currentImage, uint32_t submeshIndex) {
    void *data = device.logical().mapMemory2KHR({
        .memory = materialUniformBuffersMemory,
        .offset = (currentImage * submeshes.size() + submeshIndex) * sizeof(Material),
        .size = sizeof(Material),
    });
    std::memcpy(data, &submeshes[submeshIndex].material, sizeof(Material));
    device.logical().unmapMemory2KHR({.memory = materialUniformBuffersMemory});
  }

  void updateMaterialBuffers(uint32_t currentImage) {
    for (u32 i = 0; i < submeshes.size(); ++i) {
      updateMaterialBuffer(currentImage, i);
    }
  }

  void bind(vk::raii::CommandBuffer &cmdBuffer, const vk::raii::PipelineLayout &pipelineLayout,
            u32 currentImage) const {
    cmdBuffer.bindVertexBuffers(0, {*vertexBuffer}, {0});
    cmdBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                 *descriptorSets[currentImage], {});
  }

  void draw(vk::raii::CommandBuffer &cmdBuffer) const {
    cmdBuffer.drawIndexed(indices.size(), 1, 0, 0, 0);
  }
  // void translate(const glm::vec3 &translation) { transform.translation += translation; }
  // void rotate(const glm::vec3 &rotation) { transform.rotation += rotation; }
  // void scaleBy(const glm::vec3 &scaling) { transform.scale *= scaling; }

  std::expected<void, std::string>
  allocateDescriptorSets(const vk::raii::DescriptorPool &pool,
                         const vk::raii::DescriptorSetLayout &layout, u32 imageCount) {
    std::vector<vk::DescriptorSetLayout> layouts(imageCount, *layout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = pool, .descriptorSetCount = imageCount, .pSetLayouts = layouts.data()};
    if (auto expected = device.logical().allocateDescriptorSets(allocInfo); expected) {
      descriptorSets = std::move(*expected);
    } else {
      return std::unexpected("Failed to allocate descriptor sets: " +
                             vk::to_string(expected.error()));
    }
    return {};
  }

  void updateDescriptorSet(uint32_t currentImage) {
    vk::DescriptorBufferInfo bufferInfo{.buffer = *uniformBuffers,
                                        .offset = currentImage * sizeof(UniformBufferObject),
                                        .range = sizeof(UniformBufferObject)};

    vk::DescriptorBufferInfo bufferInfo2{.buffer = *materialUniformBuffers,
                                         .offset =
                                             currentImage * submeshes.size() * sizeof(Material),
                                         .range = sizeof(Material)};

    std::vector<vk::WriteDescriptorSet> descriptorWrites{
        {.dstSet = descriptorSets[currentImage],
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .pBufferInfo = &bufferInfo},
        {.dstSet = descriptorSets[currentImage],
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .pBufferInfo = &bufferInfo2}};

    device.logical().updateDescriptorSets(descriptorWrites, nullptr);
  }
};

class SceneNode;

struct SceneNodeCreateInfo {
  Mesh *mesh = nullptr;
  Transform *transform = nullptr; // Pointer to external transform (mesh's or custom)
  SceneNode *parent = nullptr;
};

class SceneNode {
public:
  Mesh *mesh = nullptr;
  Transform *transform; // Points to external transform (never null)
  std::vector<SceneNode *> children;

  explicit SceneNode(const SceneNodeCreateInfo &info) : mesh(info.mesh), transform(info.transform) {
    assert(transform != nullptr && "SceneNode must have a valid transform");
  }

  void addChild(SceneNode *child) { children.push_back(child); }

  void update(glm::mat4 parentTransform, uint32_t currentImage, const glm::mat4 &view,
              const glm::mat4 &projection, float deltaTime) {
    // Calculate local transformation matrix
    glm::mat4 localTransform = transform->matrix(deltaTime);

    // Combine with parent's world transform
    glm::mat4 worldTransform = parentTransform * localTransform;

    if (mesh) {
      mesh->updateUniformBuffer(currentImage, worldTransform, view, projection);
      mesh->updateMaterialBuffers(currentImage);
      mesh->updateDescriptorSet(currentImage);
    }

    for (SceneNode *child : children) {
      child->update(worldTransform, currentImage, view, projection, deltaTime);
    }
  }

  void draw(vk::raii::CommandBuffer &cmd, const vk::raii::PipelineLayout &layout,
            u32 currentImage) {
    if (mesh) {
      mesh->bind(cmd, layout, currentImage);
      mesh->draw(cmd);
    }
    for (SceneNode *child : children)
      child->draw(cmd, layout, currentImage);
  }

  void createUniformBuffers(u32 imageCount) {
    if (mesh) {
      EXPECTED_VOID(mesh->createUniformBuffers(imageCount));
      EXPECTED_VOID(mesh->createMaterialUniformBuffers(imageCount));
    }

    for (SceneNode *child : children)
      child->createUniformBuffers(imageCount);
  }

  void allocateDescriptorSets(const vk::raii::DescriptorPool &pool,
                              const vk::raii::DescriptorSetLayout &layout, u32 imageCount) {
    if (mesh)
      EXPECTED_VOID(mesh->allocateDescriptorSets(pool, layout, imageCount));

    for (SceneNode *child : children)
      child->allocateDescriptorSets(pool, layout, imageCount);
  }

  void updateDescriptorSets(uint32_t currentImage) {
    if (mesh)
      mesh->updateDescriptorSet(currentImage);

    for (SceneNode *child : children)
      child->updateDescriptorSets(currentImage);
  }
};

export class Scene {
public:
  SceneNode *createNode(const SceneNodeCreateInfo &info) {
    auto node = std::make_unique<SceneNode>(info);
    SceneNode *ptr = node.get();

    if (info.parent)
      info.parent->addChild(ptr);
    else
      roots.push_back(ptr);

    nodes.push_back(std::move(node));
    return ptr;
  }

  void createUniformBuffers(uint32_t imageCount) {
    for (auto &node : nodes)
      node->createUniformBuffers(imageCount);
  }

  void allocateDescriptorSets(const vk::raii::DescriptorPool &pool,
                              const vk::raii::DescriptorSetLayout &layout, u32 imageCount) {
    for (decltype(auto) root : roots)
      root->allocateDescriptorSets(pool, layout, imageCount);
  }

  void updateDescriptorSets(uint32_t currentImage) {
    for (decltype(auto) root : roots)
      root->updateDescriptorSets(currentImage);
  }

  void update(uint32_t currentImage, const glm::mat4 &view, const glm::mat4 &projection,
              float deltaTime) {
    for (decltype(auto) root : roots)
      root->update(glm::mat4(1.0f), currentImage, view, projection, deltaTime);
  }

  void draw(vk::raii::CommandBuffer &cmdBuffer, const vk::raii::PipelineLayout &layout,
            u32 currentImage) {
    for (decltype(auto) root : roots)
      root->draw(cmdBuffer, layout, currentImage);
  }

  void addRoot(SceneNode *node) { roots.push_back(node); }

  std::vector<std::unique_ptr<SceneNode>> nodes;
  std::vector<SceneNode *> roots;
};
