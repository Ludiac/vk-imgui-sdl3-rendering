module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>

export module vulkan_app:SceneBuilder;

import std;
import :VulkanDevice;
import :mesh;    // For Mesh, Material, Vertex (EXPECTS MODIFIED MESH CLASS FROM
                 // MESH_MODULE_REFACTORED)
import :scene;   // For Scene, SceneNode, SceneNodeCreateInfo, Transform (EXPECTS MODIFIED SCENE
                 // MODULE)
import :texture; // For PBRTextures, Texture
import :TextureStore;   // To get default/placeholder textures
import :ModelLoader;    // For LoadedGltfScene, GltfMeshData, GltfPrimitiveData, GltfNodeData
import :VulkanPipeline; // For VulkanPipeline* in SceneNodeCreateInfo

export struct BuiltSceneMeshes {
  std::vector<std::unique_ptr<Mesh>> engineMeshes;
  // Each Mesh in this vector now represents a GLTF primitive.
};

// Main function to convert LoadedGltfScene to your engine's Scene representation.
export std::expected<BuiltSceneMeshes, std::string> populateSceneFromGltf(
    Scene &targetScene, const LoadedGltfScene &gltfData, VulkanDevice &device,
    TextureStore &textureStore,
    VulkanPipeline *defaultPipeline, // A default pipeline for newly created SceneNodes
    u32 imageCount)                  // For Mesh resource creation (UBOs, descriptor sets per frame)
{
  if (!defaultPipeline) {
    return std::unexpected("SceneBuilder: Default pipeline is null.");
  }

  BuiltSceneMeshes builtMeshesResult;

  // This map will store a vector of Mesh pointers for each GLTF mesh index.
  // Each Mesh* in the vector corresponds to an engine Mesh created from a GLTF primitive.
  std::map<int, std::vector<Mesh *>> gltfMeshIndexToEngineMeshesMap;

  // 1. Create Engine Meshes: One engine Mesh per GLTF Primitive
  for (int gltfMeshIdx = 0; gltfMeshIdx < gltfData.meshes.size(); ++gltfMeshIdx) {
    const auto &gltfMesh = gltfData.meshes[gltfMeshIdx];
    std::vector<Mesh *> engineMeshesForThisGltfMesh; // Collects Mesh* for current gltfMesh

    for (int primIdx = 0; primIdx < gltfMesh.primitives.size(); ++primIdx) {
      const auto &gltfPrimitive = gltfMesh.primitives[primIdx];

      if (gltfPrimitive.vertices.empty() || gltfPrimitive.indices.empty()) {
        std::println("Warning: GLTF Mesh '{}', Primitive {} has no vertices or indices. Skipping.",
                     gltfMesh.name, primIdx);
        continue;
      }

      Material engineMaterial = gltfPrimitive.material;
      PBRTextures enginePbrTextures;

      // Assign default textures (e.g., white for baseColor)
      if (auto whiteTex =
              textureStore.getColorTexture("white")) { // Assuming getWhiteTexture exists
        enginePbrTextures.baseColor = whiteTex;
      } else {
        enginePbrTextures.baseColor = textureStore.getFallbackDefaultTexture();
      }
      enginePbrTextures.metallicRoughness = textureStore.getFallbackDefaultTexture();
      enginePbrTextures.normal = textureStore.getFallbackDefaultTexture();
      enginePbrTextures.occlusion = textureStore.getFallbackDefaultTexture();
      enginePbrTextures.emissive = textureStore.getFallbackDefaultTexture();

      std::string engineMeshName = gltfMesh.name + "_Prim" + std::to_string(primIdx);

      // EXPECTS MODIFIED ENGINE MESH CONSTRUCTOR:
      // Mesh(VulkanDevice&, std::string name, std::vector<Vertex>&&, std::vector<uint32_t>&&,
      //      Material, PBRTextures, u32 imageCount, uint32_t indexCount)
      auto newEngineMesh = std::make_unique<Mesh>(
          device, engineMeshName,
          std::vector<Vertex>(gltfPrimitive.vertices),  // Create copies for the new Mesh
          std::vector<uint32_t>(gltfPrimitive.indices), // Create copies
          engineMaterial, enginePbrTextures, imageCount);

      engineMeshesForThisGltfMesh.push_back(newEngineMesh.get()); // Store raw pointer for lookup
      builtMeshesResult.engineMeshes.emplace_back(
          std::move(newEngineMesh)); // Store unique_ptr for ownership
    }
    gltfMeshIndexToEngineMeshesMap[gltfMeshIdx] = engineMeshesForThisGltfMesh;
  }

  // 2. Create SceneNodes and Build Hierarchy
  std::vector<SceneNode *> createdEngineNodes(gltfData.nodes.size(),
                                              nullptr); // Map GLTF node index to engine SceneNode*

  // First pass: Create all SceneNode objects based on GLTF nodes.
  // If a GLTF node instantiates a mesh with multiple primitives, this node becomes a parent
  // transform, and child nodes are created for each primitive.
  for (size_t gltfNodeIdx = 0; gltfNodeIdx < gltfData.nodes.size(); ++gltfNodeIdx) {
    const auto &gltfNode = gltfData.nodes[gltfNodeIdx];

    SceneNodeCreateInfo nodeCreateInfo;
    nodeCreateInfo.name =
        gltfNode.name.empty() ? ("GLTFNode_" + std::to_string(gltfNodeIdx)) : gltfNode.name;
    nodeCreateInfo.transform =
        decomposeFromMatrix(gltfNode.transform); // Assuming Transform has setMatrix
    nodeCreateInfo.pipeline = defaultPipeline;
    // Parent will be set in the second pass if not handled by targetScene.createNode with
    // info.parent

    SceneNode *engineNodeForGltfNode =
        nullptr; // This will be the primary engine node for the GLTF node

    if (gltfNode.meshIndex >= 0) {
      // This GLTF node references a GLTF mesh.
      // Retrieve the engine Meshes (one per primitive) created for this GLTF mesh.
      const auto &primitiveEngineMeshes = gltfMeshIndexToEngineMeshesMap[gltfNode.meshIndex];

      if (primitiveEngineMeshes.empty()) {
        // GLTF node references a mesh, but it had no valid primitives. Create a transform-only
        // node.
        engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
      } else if (primitiveEngineMeshes.size() == 1) {
        // GLTF mesh had only one primitive. This engine node directly uses that Mesh.
        nodeCreateInfo.mesh = primitiveEngineMeshes[0];
        engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
      } else {
        // GLTF mesh had multiple primitives.
        // The current GLTF node becomes a parent transform node (without a mesh itself).
        engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo); // Mesh is nullptr

        // Create child SceneNodes for each primitive, parented to engineNodeForGltfNode.
        for (Mesh *primitiveMesh : primitiveEngineMeshes) {
          SceneNodeCreateInfo childNodeCi;
          childNodeCi.name = primitiveMesh->getName(); // Name child node after the primitive mesh
          childNodeCi.transform =
              decomposeFromMatrix(glm::mat4(1.0f)); // Child nodes have identity local transform
          childNodeCi.mesh = primitiveMesh;
          childNodeCi.pipeline = defaultPipeline;
          childNodeCi.parent = engineNodeForGltfNode; // Set parent relationship

          targetScene.createNode(childNodeCi); // This should add child to parent
        }
      }
    } else {
      // This GLTF node does not have a mesh (it's just a transform node).
      engineNodeForGltfNode = targetScene.createNode(nodeCreateInfo);
    }
    createdEngineNodes[gltfNodeIdx] = engineNodeForGltfNode;
  }

  // Second pass: Link children to their parents for the main GLTF node hierarchy.
  // This ensures nodes that are parents but weren't processed as mesh instances above still get
  // their children.
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
