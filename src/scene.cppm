module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:scene;

import vulkan_hpp;
import std;
import :DDX;
import :VulkanDevice;
import :extra;
import :mesh;

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
