module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app:scene;

import vulkan_hpp;
import std;
import :VulkanDevice;
import :mesh;

class VulkanPipeline;
class SceneNode;
export struct SceneNodeCreateInfo {
  Mesh *mesh = nullptr;
  Transform transform;
  VulkanPipeline *pipeline = nullptr;
  SceneNode *parent = nullptr;
  std::string name = "SceneNode";
};

class SceneNode {
public:
  std::string name;
  Mesh *mesh = nullptr;
  Transform transform;
  VulkanPipeline *pipeline = nullptr;
  SceneNode *parent_ = nullptr;
  std::vector<SceneNode *> children;

  explicit SceneNode(const SceneNodeCreateInfo &info,
                     const vk::raii::DescriptorPool &pool = nullptr,
                     const vk::raii::DescriptorSetLayout &combinedLayout = nullptr)
      : name(info.name), mesh(info.mesh), transform(info.transform), pipeline(info.pipeline),
        parent_(info.parent) {
    if (mesh) {
      EXPECTED_VOID(mesh->allocateDescriptorSets(pool, combinedLayout));
    }
  }

  void addChild(SceneNode *child) {
    if (child) {
      if (child->parent_ && child->parent_ != this) {
        child->parent_->removeChild(child);
      }

      bool alreadyChild = false;
      for (SceneNode *existingChild : children) {
        if (existingChild == child) {
          alreadyChild = true;
          break;
        }
      }
      if (!alreadyChild) {
        children.emplace_back(child);
        child->parent_ = this;
      }
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
      if (childToRemove->parent_ == this) {
        childToRemove->parent_ = nullptr;
      }
    }
  }

  SceneNode *getParent() const { return parent_; }

  void updateNodeUniforms(u32 currentImage, const glm::mat4 &worldTransform, const glm::mat4 &view,
                          const glm::mat4 &projection) {
    if (mesh) {
      mesh->updateMvpUniformBuffer(currentImage, worldTransform, view, projection);
    }
  }

  void updateNodeDescriptorSetContents(u32 currentImage) {
    if (mesh) {
      mesh->updateDescriptorSetContents(currentImage);
    }
  }

  void updateHierarchy(glm::mat4 parentWorldTransform, u32 currentImage, const glm::mat4 &view,
                       const glm::mat4 &projection, float deltaTime) {
    transform.update(deltaTime);
    glm::mat4 localMat = transform.getMatrix();
    glm::mat4 worldMat = parentWorldTransform * localMat;
    updateNodeUniforms(currentImage, worldMat, view, projection);
    for (SceneNode *child : children) {
      if (child)
        child->updateHierarchy(worldMat, currentImage, view, projection, deltaTime);
    }
  }

  void drawNode(vk::raii::CommandBuffer &cmd, VulkanPipeline *&currentPipeline, u32 currentImage) {
    if (pipeline && pipeline != currentPipeline) {
      if (*pipeline->pipeline) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline->pipeline);
      } else {
      }
      currentPipeline = pipeline;
    }

    if (mesh && pipeline && *pipeline->pipeline) {
      mesh->bind(cmd, pipeline, currentImage);
      mesh->draw(cmd, pipeline, currentImage);
    }
  }

  void allocateNodeDescriptorSets(const vk::raii::DescriptorPool &pool,
                                  const vk::raii::DescriptorSetLayout &combinedLayout) {
    if (mesh) {
      EXPECTED_VOID(mesh->allocateDescriptorSets(pool, combinedLayout));
    }
  }
};

export class Scene {
  u32 imageCount_member_{};

public:
  std::vector<std::unique_ptr<SceneNode>> nodes;
  std::unique_ptr<SceneNode> fatherNode_;
  explicit Scene(u32 imageCount) : imageCount_member_(imageCount) {
    SceneNodeCreateInfo fatherInfo;
    fatherInfo.name = "SceneRoot";
    fatherNode_ = std::make_unique<SceneNode>(fatherInfo);
  }

  void setImageCount(u32 newImageCount, const vk::raii::DescriptorPool &pool,
                     const vk::raii::DescriptorSetLayout &layout) {
    if (imageCount_member_ == newImageCount)
      return;
    imageCount_member_ = newImageCount;
    for (const auto &node : nodes) {
      if (node && node->mesh) {
        EXPECTED_VOID(node->mesh->setImageCount(newImageCount, pool, layout));
      }
    }
  }

  SceneNode *createNode(const SceneNodeCreateInfo &info,
                        const vk::raii::DescriptorPool &pool = nullptr,
                        const vk::raii::DescriptorSetLayout &combinedLayout = nullptr) {
    auto newNode = std::make_unique<SceneNode>(info, pool, combinedLayout);
    SceneNode *ptr = newNode.get();
    nodes.emplace_back(std::move(newNode));
    if (info.parent) {
      info.parent->addChild(ptr);
    } else {
      fatherNode_->addChild(ptr);
    }
    return ptr;
  }

  void updateAllDescriptorSetContents(u32 currentImage) {
    for (const auto &node : nodes) {
      if (node)
        node->updateNodeDescriptorSetContents(currentImage);
    }
  }

  void updateHierarchy(u32 currentImage, const glm::mat4 &view, const glm::mat4 &projection,
                       float deltaTime) NOEXCEPT {
    fatherNode_->updateHierarchy(glm::mat4(1.0f), currentImage, view, projection, deltaTime);
  }

  void draw(vk::raii::CommandBuffer &cmdBuffer, u32 currentImage) NOEXCEPT {
    std::sort(nodes.begin(), nodes.end(),
              [](const std::unique_ptr<SceneNode> &a, const std::unique_ptr<SceneNode> &b) {
                if (!a || !a->pipeline)
                  return b && b->pipeline;
                if (!b || !b->pipeline)
                  return false;
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
