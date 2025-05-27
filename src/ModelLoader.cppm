module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include "tiny_gltf.h" // Include the tinygltf header
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // For glm::make_vec3, glm::make_mat4 etc.

export module vulkan_app:ModelLoader;

import std;
import :extra;   // For Vertex, Material (ensure these are defined here or accessible)
import :mesh;    // For Submesh, PBRTextures (though PBRTextures might not be fully populated yet)
import :texture; // For Texture (if we were to load actual textures from GLTF)

// --- Helper Data Structures for Loaded GLTF Data ---
// These can be intermediate representations before converting to your engine's format.

// Structure to hold data for a single GLTF primitive (which maps to a Submesh)
export struct GltfPrimitiveData {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  Material material; // Basic material properties extracted from GLTF
  // PBRTextures pbrTextures; // For later, when loading textures from GLTF

  // Optional: GLTF material index, useful for debugging or advanced material handling
  int gltfMaterialIndex = -1;
};

// Structure to hold data for a single GLTF mesh (which can contain multiple primitives)
export struct GltfMeshData {
  std::string name;
  std::vector<GltfPrimitiveData> primitives;
};

// Structure to hold data for a GLTF node
export struct GltfNodeData {
  std::string name;
  glm::mat4 transform{1.0f}; // Local transform of the node
  int meshIndex = -1;        // Index into a list of loaded GltfMeshData, if this node has a mesh
  std::vector<int> childrenIndices; // Indices of child nodes in a flat list of GltfNodeData

  // For easier scene reconstruction
  int parentIndex = -1;
};

export struct LoadedGltfScene {
  std::vector<GltfMeshData> meshes;
  std::vector<GltfNodeData> nodes;
  std::vector<int> rootNodeIndices; // Indices of root nodes in the 'nodes' vector
  std::string error;                // If loading failed
};

// Helper to extract data from a GLTF accessor
template <typename T_Component,
          int N_Components> // e.g., T_Component=float, N_Components=3 for vec3
[[nodiscard]] std::expected<std::vector<glm::vec<N_Components, T_Component, glm::defaultp>>,
                            std::string>
getAccessorData(const tinygltf::Model &model, int accessorIndex) {
  if (accessorIndex < 0 || accessorIndex >= model.accessors.size()) {
    return std::unexpected("Accessor index out of bounds.");
  }
  const tinygltf::Accessor &accessor = model.accessors[accessorIndex];
  const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

  std::vector<glm::vec<N_Components, T_Component, glm::defaultp>> data(accessor.count);

  const unsigned char *bufferData =
      buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  size_t componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
  size_t numComponentsPerElement = tinygltf::GetNumComponentsInType(accessor.type);

  if (numComponentsPerElement != N_Components) {
    return std::unexpected("Accessor component count mismatch. Expected " +
                           std::to_string(N_Components) + ", got " +
                           std::to_string(numComponentsPerElement));
  }
  if (componentSize != sizeof(T_Component)) {
    // This check might be too strict if GLTF uses, e.g., FLOAT and T_Component is double, or SHORT
    // and T_Component is int. A more robust version would handle type conversions.
    // return std::unexpected("Accessor component type size mismatch.");
  }

  size_t stride = accessor.ByteStride(bufferView); // Stride between elements
  if (stride == 0) {                               // Data is tightly packed
    stride = componentSize * numComponentsPerElement;
  }

  for (size_t i = 0; i < accessor.count; ++i) {
    const unsigned char *elementData = bufferData + (i * stride);
    // This memcpy assumes the GLTF component type matches T_Component directly.
    // A production loader would handle different component types (float, double, int, short, etc.)
    // and perform necessary conversions.
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
        sizeof(T_Component) == sizeof(float)) {
      std::memcpy(glm::value_ptr(data[i]), elementData, sizeof(T_Component) * N_Components);
    } else {
      // Handle other types or return error if type mismatch is critical.
      // For simplicity, we'll assume float for positions, normals, uvs for now.
      // If T_Component is float and GLTF is double, this will be an issue.
      // This is a simplified example.
      // std::println("Warning: Unhandled component type in getAccessorData or T_Component
      // mismatch."); For now, attempt memcpy if sizes match, otherwise, it might lead to issues.
      if (sizeof(glm::vec<N_Components, T_Component, glm::defaultp>) ==
          componentSize * numComponentsPerElement) {
        std::memcpy(glm::value_ptr(data[i]), elementData, sizeof(T_Component) * N_Components);
      } else {
        return std::unexpected("Unhandled component type or size mismatch in getAccessorData.");
      }
    }
  }
  return data;
}

// Specialization for indices (unsigned int, or unsigned short, or unsigned char)
[[nodiscard]] std::expected<std::vector<uint32_t>, std::string>
getIndexAccessorData(const tinygltf::Model &model, int accessorIndex) {
  if (accessorIndex < 0 || accessorIndex >= model.accessors.size()) {
    return std::unexpected("Index accessor index out of bounds.");
  }
  const tinygltf::Accessor &accessor = model.accessors[accessorIndex];
  if (accessor.type != TINYGLTF_TYPE_SCALAR) {
    return std::unexpected("Index accessor is not scalar.");
  }

  const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
  const unsigned char *bufferData =
      buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

  std::vector<uint32_t> indices(accessor.count);
  size_t stride = accessor.ByteStride(bufferView);
  if (stride == 0) {
    stride = tinygltf::GetComponentSizeInBytes(accessor.componentType);
  }

  for (size_t i = 0; i < accessor.count; ++i) {
    const unsigned char *elementData = bufferData + (i * stride);
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
      uint32_t val;
      std::memcpy(&val, elementData, sizeof(uint32_t));
      indices[i] = val;
    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
      uint16_t val;
      std::memcpy(&val, elementData, sizeof(uint16_t));
      indices[i] = static_cast<uint32_t>(val);
    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
      uint8_t val;
      std::memcpy(&val, elementData, sizeof(uint8_t));
      indices[i] = static_cast<uint32_t>(val);
    } else {
      return std::unexpected("Unsupported index component type.");
    }
  }
  return indices;
}

// --- Main Loading Function ---
export [[nodiscard]] std::expected<LoadedGltfScene, std::string>
loadGltfFile(const std::string &filePath) {
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;
  LoadedGltfScene loadedScene;

  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filePath);
  // For binary GLB: bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);

  if (!warn.empty()) {
    std::println("TinyGLTF Warning: {}", warn);
  }
  if (!err.empty()) {
    loadedScene.error = "TinyGLTF Error: " + err;
    return std::unexpected(loadedScene.error);
  }
  if (!ret) {
    loadedScene.error = "Failed to parse GLTF file: " + filePath;
    return std::unexpected(loadedScene.error);
  }

  // 1. Process Meshes and their Primitives
  for (const auto &gltfMesh : model.meshes) {
    GltfMeshData meshData;
    meshData.name = gltfMesh.name.empty() ? ("Mesh_" + std::to_string(loadedScene.meshes.size()))
                                          : gltfMesh.name;

    for (const auto &gltfPrimitive : gltfMesh.primitives) {
      GltfPrimitiveData primitiveData;
      primitiveData.gltfMaterialIndex = gltfPrimitive.material;

      // --- Extract Indices ---
      if (gltfPrimitive.indices >= 0) {
        auto indicesResult = getIndexAccessorData(model, gltfPrimitive.indices);
        if (indicesResult) {
          primitiveData.indices = std::move(*indicesResult);
        } else {
          return std::unexpected("Failed to load indices for primitive: " + indicesResult.error());
        }
      } else {
        // Non-indexed geometry: need to generate indices or handle differently.
        // For now, we require indexed geometry.
        return std::unexpected("Primitive is not indexed. This loader requires indexed geometry.");
      }

      // --- Extract Vertex Attributes ---
      std::vector<glm::vec3> positions;
      std::vector<glm::vec3> normals;
      std::vector<glm::vec2> uvs;
      // Tangents are optional for now
      // std::vector<glm::vec4> tangents;

      if (auto posIt = gltfPrimitive.attributes.find("POSITION");
          posIt != gltfPrimitive.attributes.end()) {
        auto posResult = getAccessorData<float, 3>(model, posIt->second);
        if (posResult)
          positions = std::move(*posResult);
        else
          return std::unexpected("Failed to load POSITION attribute: " + posResult.error());
      } else {
        return std::unexpected("Primitive missing POSITION attribute.");
      }

      if (auto normIt = gltfPrimitive.attributes.find("NORMAL");
          normIt != gltfPrimitive.attributes.end()) {
        auto normResult = getAccessorData<float, 3>(model, normIt->second);
        if (normResult)
          normals = std::move(*normResult);
        // else return std::unexpected("Failed to load NORMAL attribute: " + normResult.error());
        // Normals can be optional; if missing, you might generate them or use a default.
        else
          std::println(
              "Warning: Primitive missing NORMAL attribute for mesh '{}'. Using default (0,1,0).",
              meshData.name);
      }

      if (auto uvIt = gltfPrimitive.attributes.find("TEXCOORD_0");
          uvIt != gltfPrimitive.attributes.end()) {
        auto uvResult = getAccessorData<float, 2>(model, uvIt->second);
        if (uvResult)
          uvs = std::move(*uvResult);
        // else return std::unexpected("Failed to load TEXCOORD_0 attribute: " + uvResult.error());
        else
          std::println(
              "Warning: Primitive missing TEXCOORD_0 attribute for mesh '{}'. Using default (0,0).",
              meshData.name);
      }

      // (Add tangent loading if needed, attribute "TANGENT")

      // --- Assemble Vertices ---
      // Ensure all required attributes have the same count.
      // The primary attribute count is from positions.
      size_t vertexCount = positions.size();
      primitiveData.vertices.resize(vertexCount);

      for (size_t i = 0; i < vertexCount; ++i) {
        primitiveData.vertices[i].pos = positions[i];
        primitiveData.vertices[i].normal = (!normals.empty() && i < normals.size())
                                               ? normals[i]
                                               : glm::vec3(0.0f, 1.0f, 0.0f); // Default normal
        primitiveData.vertices[i].uv =
            (!uvs.empty() && i < uvs.size()) ? uvs[i] : glm::vec2(0.0f, 0.0f); // Default UV
        primitiveData.vertices[i].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // Default tangent
      }

      // --- Extract Basic Material Properties (Base Color Factor) ---
      if (gltfPrimitive.material >= 0 && gltfPrimitive.material < model.materials.size()) {
        const auto &gltfMaterial = model.materials[gltfPrimitive.material];
        const auto &pbrMetallicRoughness = gltfMaterial.pbrMetallicRoughness;

        primitiveData.material.baseColorFactor =
            glm::vec4(static_cast<float>(pbrMetallicRoughness.baseColorFactor[0]),
                      static_cast<float>(pbrMetallicRoughness.baseColorFactor[1]),
                      static_cast<float>(pbrMetallicRoughness.baseColorFactor[2]),
                      static_cast<float>(pbrMetallicRoughness.baseColorFactor[3]));
        primitiveData.material.metallicFactor =
            static_cast<float>(pbrMetallicRoughness.metallicFactor);
        primitiveData.material.roughnessFactor =
            static_cast<float>(pbrMetallicRoughness.roughnessFactor);

        // We are not loading textures from the GLTF file yet.
        // You would inspect pbrMetallicRoughness.baseColorTexture.index here.
      } else {
        // Default material if none specified
        primitiveData.material = Material{};
      }
      meshData.primitives.emplace_back(std::move(primitiveData));
    }
    loadedScene.meshes.emplace_back(std::move(meshData));
  }

  // 2. Process Nodes (Scene Hierarchy and Transforms)
  loadedScene.nodes.resize(model.nodes.size());
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    const auto &gltfNode = model.nodes[i];
    GltfNodeData &nodeData = loadedScene.nodes[i];

    nodeData.name = gltfNode.name.empty() ? ("Node_" + std::to_string(i)) : gltfNode.name;
    nodeData.meshIndex = gltfNode.mesh; // Will be -1 if no mesh

    // GLTF node transform can be a matrix or T/R/S components
    if (!gltfNode.matrix.empty()) {
      // GLTF matrix is std::vector<double>, needs conversion to glm::mat4 (float)
      // GLTF matrices are column-major. glm::make_mat4 expects pointer to column-major data.
      std::array<float, 16> mat_data;
      for (int k = 0; k < 16; ++k)
        mat_data[k] = static_cast<float>(gltfNode.matrix[k]);
      nodeData.transform = glm::make_mat4(mat_data.data());
    } else {
      glm::mat4 translation = glm::mat4(1.0f);
      if (!gltfNode.translation.empty()) {
        translation =
            glm::translate(glm::mat4(1.0f), glm::vec3(static_cast<float>(gltfNode.translation[0]),
                                                      static_cast<float>(gltfNode.translation[1]),
                                                      static_cast<float>(gltfNode.translation[2])));
      }
      glm::mat4 rotation = glm::mat4(1.0f);
      if (!gltfNode.rotation.empty()) { // GLTF rotation is a quaternion (x, y, z, w)
        glm::quat q(static_cast<float>(gltfNode.rotation[3]),  // w
                    static_cast<float>(gltfNode.rotation[0]),  // x
                    static_cast<float>(gltfNode.rotation[1]),  // y
                    static_cast<float>(gltfNode.rotation[2])); // z
        rotation = glm::mat4_cast(q);
      }
      glm::mat4 scale = glm::mat4(1.0f);
      if (!gltfNode.scale.empty()) {
        scale = glm::scale(glm::mat4(1.0f), glm::vec3(static_cast<float>(gltfNode.scale[0]),
                                                      static_cast<float>(gltfNode.scale[1]),
                                                      static_cast<float>(gltfNode.scale[2])));
      }
      nodeData.transform = translation * rotation * scale;
    }

    for (int childIndex : gltfNode.children) {
      nodeData.childrenIndices.push_back(childIndex);
      if (childIndex >= 0 && childIndex < loadedScene.nodes.size()) {
        loadedScene.nodes[childIndex].parentIndex = static_cast<int>(i); // Set parent for child
      }
    }
  }

  // 3. Identify Root Nodes (nodes not a child of any other node)
  // This can also be taken from model.scenes[model.defaultScene].nodes if a default scene is
  // specified.
  if (model.defaultScene >= 0 && model.defaultScene < model.scenes.size()) {
    for (int rootNodeIdx : model.scenes[model.defaultScene].nodes) {
      loadedScene.rootNodeIndices.push_back(rootNodeIdx);
    }
  } else if (!model.nodes.empty() && model.scenes.empty()) {
    // If no scenes defined, but nodes exist, try to find roots manually (nodes with no parent)
    // This is a fallback, well-formed GLTFs usually have scenes.
    for (size_t i = 0; i < loadedScene.nodes.size(); ++i) {
      if (loadedScene.nodes[i].parentIndex == -1) { // Nodes we haven't marked as children
        loadedScene.rootNodeIndices.push_back(static_cast<int>(i));
      }
    }
    if (loadedScene.rootNodeIndices.empty() && !loadedScene.nodes.empty()) {
      loadedScene.rootNodeIndices.push_back(0); // Fallback to first node if no other roots found
    }
  }

  return loadedScene;
}
