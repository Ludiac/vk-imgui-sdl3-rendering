module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>

export module vulkan_app:SceneBuilder;

import std;
import :VulkanDevice;
import :mesh;
import :scene;
import :texture;
import :TextureStore;
import :ModelLoader; // For LoadedGltfScene, GltfImageData, etc.
import :VulkanPipeline;

export struct BuiltSceneMeshes { /* ... same ... */
  std::vector<std::unique_ptr<Mesh>> engineMeshes;
};

export [[nodiscard]] std::expected<BuiltSceneMeshes, std::string>
populateSceneFromGltf(Scene &targetScene, const LoadedGltfScene &gltfData, VulkanDevice &device,
                      TextureStore &textureStore, VulkanPipeline *defaultPipeline, u32 imageCount) {
  if (!defaultPipeline)
    return std::unexpected("SceneBuilder: Default pipeline is null.");

  BuiltSceneMeshes builtMeshesResult;
  std::map<int, std::vector<Mesh *>> gltfMeshIndexToEngineMeshesMap;

  // 1. Create Engine Meshes
  for (int gltfMeshIdx = 0; gltfMeshIdx < gltfData.meshes.size(); ++gltfMeshIdx) {
    const auto &gltfMesh = gltfData.meshes[gltfMeshIdx];
    std::vector<Mesh *> engineMeshesForThisGltfMesh;

    for (int primIdx = 0; primIdx < gltfMesh.primitives.size(); ++primIdx) {
      const auto &gltfPrimitive = gltfMesh.primitives[primIdx];
      if (gltfPrimitive.vertices.empty() || gltfPrimitive.indices.empty())
        continue;

      Material engineMaterial = gltfPrimitive.material;
      PBRTextures enginePbrTextures;

      // *** LOAD BASE COLOR TEXTURE FROM GLTF DATA ***
      if (gltfPrimitive.baseColorTextureGltfIndex >= 0) {
        // gltfPrimitive.baseColorTextureGltfIndex now stores the GLTF *image* index.
        auto imageIt = gltfData.images.find(gltfPrimitive.baseColorTextureGltfIndex);
        if (imageIt != gltfData.images.end()) {
          const GltfImageData &loadedImgData = imageIt->second;
          // Use image name or index as part of cache key
          std::string textureCacheKey = "gltf_img_" +
                                        std::to_string(gltfPrimitive.baseColorTextureGltfIndex) +
                                        "_" + loadedImgData.name;
          enginePbrTextures.baseColor =
              textureStore.getTextureFromData(textureCacheKey, loadedImgData);
        } else {
          std::println("Warning: GLTF material specified base color texture (image index {}), but "
                       "image data not found in loadedScene.images.",
                       gltfPrimitive.baseColorTextureGltfIndex);
          enginePbrTextures.baseColor = textureStore.getDefaultTexture(); // Fallback
        }
      } else {
        // No base color texture specified in GLTF material, use default white.
        enginePbrTextures.baseColor = textureStore.getDefaultTexture();
      }

      // Assign fallback/default for other PBR textures for now
      enginePbrTextures.metallicRoughness = textureStore.getDefaultTexture(); // Placeholder
      enginePbrTextures.normal = textureStore.getDefaultTexture(); // Placeholder (e.g. flat normal)
      enginePbrTextures.occlusion = textureStore.getDefaultTexture(); // Placeholder
      enginePbrTextures.emissive = textureStore.getDefaultTexture();  // Placeholder

      std::string engineMeshName = gltfMesh.name + "_Prim" + std::to_string(primIdx);
      auto newEngineMesh = std::make_unique<Mesh>(device, engineMeshName,
                                                  std::vector<Vertex>(gltfPrimitive.vertices),
                                                  std::vector<uint32_t>(gltfPrimitive.indices),
                                                  engineMaterial, enginePbrTextures, imageCount);

      engineMeshesForThisGltfMesh.push_back(newEngineMesh.get());
      builtMeshesResult.engineMeshes.emplace_back(std::move(newEngineMesh));
    }
    gltfMeshIndexToEngineMeshesMap[gltfMeshIdx] = engineMeshesForThisGltfMesh;
  }

  // 2. Create SceneNodes and Build Hierarchy (remains largely the same as previous version)
  std::vector<SceneNode *> createdEngineNodes(gltfData.nodes.size(), nullptr);
  for (size_t gltfNodeIdx = 0; gltfNodeIdx < gltfData.nodes.size();
       ++gltfNodeIdx) { /* ... same logic ... */
    const auto &gltfNode = gltfData.nodes[gltfNodeIdx];
    SceneNodeCreateInfo nodeCreateInfo;
    nodeCreateInfo.name =
        gltfNode.name.empty() ? ("GLTFNode_" + std::to_string(gltfNodeIdx)) : gltfNode.name;
    nodeCreateInfo.transform = decomposeFromMatrix(gltfNode.transform);
    nodeCreateInfo.pipeline = defaultPipeline;
    SceneNode *engineNodeForGltfNode = nullptr;
    if (gltfNode.meshIndex >= 0) {
      const auto &it = gltfMeshIndexToEngineMeshesMap.find(gltfNode.meshIndex);
      if (it != gltfMeshIndexToEngineMeshesMap.end()) {
        const auto &primitiveEngineMeshes = it->second;
        if (primitiveEngineMeshes.empty())
          engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
        else if (primitiveEngineMeshes.size() == 1) {
          nodeCreateInfo.mesh = primitiveEngineMeshes[0];
          engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
        } else {
          engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
          for (Mesh *primitiveMesh : primitiveEngineMeshes) {
            SceneNodeCreateInfo childNodeCi;
            childNodeCi.name = primitiveMesh->getName();
            childNodeCi.transform = Transform{};
            childNodeCi.mesh = primitiveMesh;
            childNodeCi.pipeline = defaultPipeline;
            childNodeCi.parent = engineNodeForGltfNode;
            targetScene.createNode(childNodeCi);
          }
        }
      } else
        engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
    } else
      engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
    createdEngineNodes[gltfNodeIdx] = engineNodeForGltfNode;
  }

  for (size_t gltfNodeIdx = 0; gltfNodeIdx < gltfData.nodes.size(); ++gltfNodeIdx) {
    const auto &gltfNode = gltfData.nodes[gltfNodeIdx];
    SceneNode *engineParentNode = createdEngineNodes[gltfNodeIdx];
    if (!engineParentNode)
      continue;

    for (int childGltfNodeIndex : gltfNode.childrenIndices) {
      if (childGltfNodeIndex >= 0 && childGltfNodeIndex < createdEngineNodes.size()) {
        SceneNode *engineChildNode = createdEngineNodes[childGltfNodeIndex];
        if (engineChildNode) {
          // If Scene::createNode with parent info doesn't set parent_ and add to children, do it
          // manually. Assuming Scene::createNode with .parent set already handles this. If not:
          // engineParentNode->addChild(engineChildNode);
          // Also, if engineChildNode was previously a root, it needs to be removed from
          // targetScene.roots. This part depends heavily on how Scene::createNode and
          // SceneNode::addChild interact with Scene::roots.
          if (engineChildNode->getParent() == nullptr) { // If child was made a root initially
            engineParentNode->addChild(engineChildNode); // This should set child->parent_
            // Remove child from roots if it's there
            targetScene.roots.erase(
                std::remove(targetScene.roots.begin(), targetScene.roots.end(), engineChildNode),
                targetScene.roots.end());
          }
        }
      }
    }
  }

  // Final pass to ensure targetScene.roots is correctly populated based on GLTF scene roots.
  // The previous logic in Scene::createNode might have added nodes to roots prematurely.
  targetScene.roots.clear();
  for (int rootGltfNodeIndex : gltfData.rootNodeIndices) {
    if (rootGltfNodeIndex >= 0 && rootGltfNodeIndex < createdEngineNodes.size()) {
      if (createdEngineNodes[rootGltfNodeIndex]) {
        targetScene.addRoot(createdEngineNodes[rootGltfNodeIndex]);
        // Ensure these root nodes indeed have no parent set
        if (createdEngineNodes[rootGltfNodeIndex]->getParent() != nullptr) {
          std::println(
              "Warning: GLTF root node {} was assigned a parent during scene construction.",
              createdEngineNodes[rootGltfNodeIndex]->name);
          createdEngineNodes[rootGltfNodeIndex]->parent_ = nullptr; // Force it to be a root
        }
      }
    }
  }
  // As a fallback, if no roots were defined by GLTF but nodes exist:
  if (targetScene.roots.empty() && !gltfData.nodes.empty()) {
    for (SceneNode *node : createdEngineNodes) {
      if (node && node->getParent() == nullptr) {
        targetScene.addRoot(node);
      }
    }
  }

  return builtMeshesResult;
}
