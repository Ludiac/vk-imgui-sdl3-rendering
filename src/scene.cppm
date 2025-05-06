module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:scene;

import vulkan_hpp;
import std;
import :VulkanWindow;
import :VulkanDevice;
import :extra;
import :mesh;

class SceneNode;

struct SceneNodeCreateInfo {
  Mesh *mesh = nullptr;
  Transform transform; // Pointer to external transform (mesh's or custom)
  VulkanPipeline *pipeline = nullptr;
  SceneNode *parent = nullptr;
};

class SceneNode {
public:
  Mesh *mesh = nullptr;
  Transform transform;
  VulkanPipeline *pipeline;
  std::vector<SceneNode *> children;

  explicit SceneNode(const SceneNodeCreateInfo &info)
      : mesh(info.mesh), transform(info.transform), pipeline(info.pipeline) {
    assert(transform != nullptr && "SceneNode must have a valid transform");
  }

  void addChild(SceneNode *child) { children.push_back(child); }

  void update(glm::mat4 parentTransform, uint32_t currentImage, const glm::mat4 &view,
              const glm::mat4 &projection, float deltaTime) {
    glm::mat4 localTransform = transform.matrix(deltaTime);

    glm::mat4 worldTransform = parentTransform * localTransform;

    if (mesh) {
      mesh->updateUniformBuffer(currentImage, worldTransform, view, projection);
      mesh->updateDescriptorSet(currentImage);
    }

    for (SceneNode *child : children) {
      child->update(worldTransform, currentImage, view, projection, deltaTime);
    }
  }

  void draw(vk::raii::CommandBuffer &cmd, u32 currentImage) {
    if (mesh) {
      mesh->bind(cmd, pipeline, currentImage);
      mesh->draw(cmd);
    }
  }

  void recreateUniformBuffers(u32 imageCount) {
    if (mesh) {
      EXPECTED_VOID(mesh->recreateUniformBuffers(imageCount));
      EXPECTED_VOID(mesh->recreateMaterialUniformBuffers(imageCount));
    }

    for (SceneNode *child : children)
      child->recreateUniformBuffers(imageCount);
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

  void recreateUniformBuffers(uint32_t imageCount) {
    for (auto &node : nodes)
      node->recreateUniformBuffers(imageCount);
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

  void draw(vk::raii::CommandBuffer &cmdBuffer, u32 currentImage) {
    std::sort(nodes.begin(), nodes.end(),
              [](const auto &a, const auto &b) { return a->pipeline < b->pipeline; });
    VulkanPipeline *currentPipeline = nullptr;

    for (decltype(auto) node : nodes) {
      if (!node->mesh)
        continue; // Skip non-renderables

      if (node->pipeline != currentPipeline) {
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *node->pipeline->pipeline);
        currentPipeline = node->pipeline;
      }

      node->draw(cmdBuffer, currentImage);
    }
  }

  void addRoot(SceneNode *node) { roots.push_back(node); }

  std::vector<std::unique_ptr<SceneNode>> nodes;
  std::vector<SceneNode *> roots;
};
