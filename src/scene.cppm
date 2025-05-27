module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:scene;

import vulkan_hpp;
import std;
import :VulkanDevice;
import :mesh; // For Mesh, Transform. Mesh will be refactored to use a single descriptor set.

class VulkanPipeline;
class SceneNode;

export struct SceneNodeCreateInfo {
  Mesh *mesh = nullptr;
  Transform transform; // Assuming Transform has a default constructor or is initialized
  VulkanPipeline *pipeline = nullptr;
  SceneNode *parent = nullptr;
  std::string name = "SceneNode";
};

class SceneNode {
public:
  std::string name;
  Mesh *mesh = nullptr;
  Transform transform; // Assuming Transform has a default constructor
  VulkanPipeline *pipeline = nullptr;
  SceneNode *parent_ = nullptr;
  std::vector<SceneNode *> children;

  explicit SceneNode(const SceneNodeCreateInfo &info)
      : name(info.name), mesh(info.mesh), transform(info.transform), pipeline(info.pipeline),
        parent_(info.parent) {
    // The original assert about transform being non-null is not applicable
    // if transform is a value type. You might assert that pipeline is not null
    // if it's always required for a node that will be rendered.
  }

  void addChild(SceneNode *child) {
    if (child) {
      // Prevent adding self as child or circular dependencies if necessary
      children.emplace_back(child);
      child->parent_ = this;
    }
  }

  void removeChild(SceneNode *childToRemove) {
    if (!childToRemove)
      return;
    auto it =
        std::remove_if(children.begin(), children.end(), [childToRemove](SceneNode *current_child) {
          return current_child == childToRemove;
        });
    if (it != children.end()) {
      children.erase(it, children.end());
      childToRemove->parent_ = nullptr;
    }
  }

  SceneNode *getParent() const { return parent_; }

  // updateNodeUniforms: Updates MVP UBO based on hierarchical transform.
  void updateNodeUniforms(uint32_t currentImage, const glm::mat4 &worldTransform,
                          const glm::mat4 &view, const glm::mat4 &projection) {
    if (mesh) {
      // Assumes Mesh class has updateMvpUniformBuffer
      mesh->updateMvpUniformBuffer(currentImage, worldTransform, view, projection);
    }
  }

  // updateNodeDescriptorSetContents: Calls Mesh to update its (single) descriptor set's contents.
  void updateNodeDescriptorSetContents(uint32_t currentImage) {
    if (mesh) {
      // Assumes Mesh class has updateDescriptorSetContents
      mesh->updateDescriptorSetContents(currentImage);
    }
  }

  // updateHierarchy: Recursively updates transforms and MVP UBOs.
  void updateHierarchy(glm::mat4 parentWorldTransform, uint32_t currentImage, const glm::mat4 &view,
                       const glm::mat4 &projection, float deltaTime) {
    // Assuming Transform has a 'matrix(deltaTime)' method that returns glm::mat4
    transform.update(deltaTime);
    glm::mat4 localMat = transform.getMatrix();
    glm::mat4 worldMat = parentWorldTransform * localMat;

    updateNodeUniforms(currentImage, worldMat, view, projection);

    for (SceneNode *child : children) {
      if (child)
        child->updateHierarchy(worldMat, currentImage, view, projection, deltaTime);
    }
  }

  // drawNode: Called by Scene::draw for a single node from the sorted list.
  // Handles pipeline binding and delegates to Mesh for its binding and drawing.
  void drawNode(vk::raii::CommandBuffer &cmd, VulkanPipeline *&currentPipeline, u32 currentImage) {
    // Bind pipeline if different from current
    if (pipeline && pipeline != currentPipeline) {
      if (*pipeline->pipeline) { // Ensure the pipeline handle itself is valid
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline->pipeline);
      } else {
        // std::println("Warning: Node '{}' has a null Vulkan pipeline handle.", name);
      }
      currentPipeline = pipeline;
    }

    if (mesh && pipeline && *pipeline->pipeline) {
      // Mesh::bind will now bind its single descriptor set (containing MVP, Material, Textures)
      // Assumes Mesh class has 'bind' and 'draw' methods.
      mesh->bind(cmd, pipeline, currentImage);
      mesh->draw(cmd, pipeline, currentImage);
    }
  }

  // allocateNodeDescriptorSets: Allocates the single combined descriptor set for the mesh.
  void allocateNodeDescriptorSets(const vk::raii::DescriptorPool &pool,
                                  const vk::raii::DescriptorSetLayout &combinedLayout) {
    if (mesh) {
      // Mesh::allocateDescriptorSets will be refactored to take one layout
      auto result = mesh->allocateDescriptorSets(pool, combinedLayout);
      if (!result)
        std::println("Error allocating descriptor sets for node '{}', mesh '{}': {}", name,
                     mesh->getName(), result.error());
    }
  }
};

export class Scene {
  u32 imageCount_member_{};

public:
  std::vector<std::unique_ptr<SceneNode>> nodes;
  std::vector<SceneNode *> roots;

  explicit Scene(u32 imageCount) : imageCount_member_(imageCount) {}

  void setImageCount(u32 newImageCount) {
    if (imageCount_member_ == newImageCount)
      return;
    imageCount_member_ = newImageCount;
    for (const auto &node : nodes) {
      if (node && node->mesh) {
        // Assumes Mesh class has setImageCount
        auto result = node->mesh->setImageCount(newImageCount);
        if (!result) {
          std::println("Error setting image count for mesh '{}': {}", node->mesh->getName(),
                       result.error());
          std::exit(0);
        }
      }
    }
  }

  SceneNode *createNode(const SceneNodeCreateInfo &info) {
    auto newNode = std::make_unique<SceneNode>(info);
    SceneNode *ptr = newNode.get();
    nodes.emplace_back(std::move(newNode)); // Add to the main ownership list

    if (info.parent) {
      info.parent->addChild(ptr); // addChild also sets child->parent_
    } else {
      addRoot(ptr);
    }
    return ptr;
  }

  void addRoot(SceneNode *node) NOEXCEPT {
    if (node && std::find(roots.begin(), roots.end(), node) == roots.end()) {
      if (node->getParent() == nullptr) { // Only add if it's truly a root
        roots.emplace_back(node);
      } else {
        // This case should ideally not happen if createNode and addChild manage parentage
        // correctly. std::println("Warning: Node '{}' has a parent, cannot add as root via
        // addRoot.", node->name);
      }
    }
  }

  // allocateAllDescriptorSets: Uses the single combined layout.
  void allocateAllDescriptorSets(const vk::raii::DescriptorPool &pool,
                                 const vk::raii::DescriptorSetLayout &combinedLayout) {
    for (const auto &node : nodes) {
      if (node)
        node->allocateNodeDescriptorSets(pool, combinedLayout);
    }
  }

  // updateAllDescriptorSetContents: Calls each node to update its mesh's descriptor set contents.
  void updateAllDescriptorSetContents(uint32_t currentImage) {
    for (const auto &node : nodes) {
      if (node)
        node->updateNodeDescriptorSetContents(currentImage);
    }
  }

  // updateHierarchy: Updates transforms and MVP UBOs for all nodes.
  void updateHierarchy(uint32_t currentImage, const glm::mat4 &view, const glm::mat4 &projection,
                       float deltaTime) NOEXCEPT {
    for (SceneNode *root : roots) {
      if (root)
        root->updateHierarchy(glm::mat4(1.0f), currentImage, view, projection, deltaTime);
    }
  }

  // draw: Sorts all nodes by pipeline, then draws them.
  void draw(vk::raii::CommandBuffer &cmdBuffer, u32 currentImage) NOEXCEPT {
    // Sort all nodes by pipeline.
    std::sort(
        nodes.begin(), nodes.end(),
        [](const std::unique_ptr<SceneNode> &a, const std::unique_ptr<SceneNode> &b) {
          if (!a || !a->pipeline)
            return b &&
                   b->pipeline; // Nodes without pipelines sort earlier (or later, adjust as needed)
          if (!b || !b->pipeline)
            return false;
          // Compare pipeline pointers for sorting.
          // If VulkanPipeline has an ID or comparable value, that could also be used.
          return a->pipeline < b->pipeline;
        });

    VulkanPipeline *currentPipeline = nullptr;

    for (const auto &node_ptr : nodes) {
      if (node_ptr) {
        node_ptr->drawNode(cmdBuffer, currentPipeline, currentImage);
      }
    }
  }
};
