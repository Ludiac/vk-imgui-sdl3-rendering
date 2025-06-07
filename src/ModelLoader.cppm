module;

#include "macros.hpp"
#include "primitive_types.hpp"
// #include "tiny_gltf.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

export module vulkan_app:ModelLoader;

import std;
import :utils;
import :mesh;
import :texture;
import :tinygltf;

// Structure to hold data for a single GLTF primitive
export struct GltfPrimitiveData {
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
  Material material;

  // Store the GLTF texture index for the base color map. -1 if not present.
  int baseColorTextureGltfIndex = -1;
  // Store the GLTF texture index for other maps as you add them
  // int metallicRoughnessTextureGltfIndex = -1;
  // int normalTextureGltfIndex = -1;
  // ... etc.

  int gltfMaterialIndex = -1;
};

// Structure to hold data for a single GLTF mesh
export struct GltfMeshData {
  std::string name;
  std::vector<GltfPrimitiveData> primitives;
};

// Structure to hold data for a GLTF node
export struct GltfNodeData {
  std::string name;
  glm::mat4 transform{1.0f};
  int meshIndex = -1;
  std::vector<int> childrenIndices;
  int parentIndex = -1;
};

// New struct to hold image data loaded by ModelLoader
export struct GltfImageData {
  std::vector<unsigned char> pixels;
  int width = 0;
  int height = 0;
  int component = 0;  // 3 for RGB, 4 for RGBA
  bool isSrgb = true; // Assume sRGB for color textures unless specified otherwise
  std::string name;   // From gltfImage.name or uri
};

export struct LoadedGltfScene {
  std::vector<GltfMeshData> meshes;
  std::vector<GltfNodeData> nodes;
  std::vector<int> rootNodeIndices;

  // Store loaded image data here, indexed by the GLTF image index
  std::map<int, GltfImageData> images;
  // Store GLTF material definitions if needed for more complex mapping later
  // std::vector<gltfm::Material> gltfMaterials;

  std::string error;
};

namespace GltfLoaderHelpers {
// ... (getAccessorData, getIndexAccessorData helpers remain the same as before) ...
template <typename T_Component, int N_Components>
[[nodiscard]] std::expected<std::vector<glm::vec<N_Components, T_Component, glm::defaultp>>,
                            std::string>
getAccessorData(const gltfm::Model &model, int accessorIndex) {
  if (accessorIndex < 0 || static_cast<size_t>(accessorIndex) >= model.accessors.size())
    return std::unexpected("Accessor index out of bounds.");
  const gltfm::Accessor &accessor = model.accessors[accessorIndex];
  const gltfm::BufferView &bufferView = model.bufferViews[accessor.bufferView];
  const gltfm::Buffer &buffer = model.buffers[bufferView.buffer];
  std::vector<glm::vec<N_Components, T_Component, glm::defaultp>> data(accessor.count);
  const unsigned char *bufferData =
      buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  size_t componentSize = gltfm::GetComponentSizeInBytes(accessor.componentType);
  size_t numComponentsPerElement = gltfm::GetNumComponentsInType(accessor.type);

  if (numComponentsPerElement != N_Components)
    return std::unexpected("Accessor component count mismatch.");

  size_t stride = accessor.ByteStride(bufferView);
  if (stride == 0)
    stride = componentSize * numComponentsPerElement;

  for (size_t i = 0; i < accessor.count; ++i) {
    const unsigned char *elementData = bufferData + (i * stride);
    if (accessor.componentType == gltfm::COMPONENT_TYPE_FLOAT &&
        sizeof(T_Component) == sizeof(float)) {
      std::memcpy(glm::value_ptr(data[i]), elementData, sizeof(T_Component) * N_Components);
    } else {
      // Simplified: Add more robust type handling/conversion here
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

[[nodiscard]] std::expected<std::vector<u32>, std::string>
getIndexAccessorData(const gltfm::Model &model, int accessorIndex) {
  if (accessorIndex < 0 || static_cast<size_t>(accessorIndex) >= model.accessors.size())
    return std::unexpected("Index accessor index out of bounds.");
  const gltfm::Accessor &accessor = model.accessors[accessorIndex];
  if (accessor.type != gltfm::TYPE_SCALAR)
    return std::unexpected("Index accessor is not scalar.");
  const gltfm::BufferView &bufferView = model.bufferViews[accessor.bufferView];
  const gltfm::Buffer &buffer = model.buffers[bufferView.buffer];
  const unsigned char *bufferData =
      buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  std::vector<u32> indices(accessor.count);
  size_t stride = accessor.ByteStride(bufferView);
  if (stride == 0)
    stride = gltfm::GetComponentSizeInBytes(accessor.componentType);

  for (size_t i = 0; i < accessor.count; ++i) {
    const unsigned char *elementData = bufferData + (i * stride);
    if (accessor.componentType == gltfm::COMPONENT_TYPE_UNSIGNED_INT) {
      u32 val;
      std::memcpy(&val, elementData, sizeof(u32));
      indices[i] = val;
    } else if (accessor.componentType == gltfm::COMPONENT_TYPE_UNSIGNED_SHORT) {
      uint16_t val;
      std::memcpy(&val, elementData, sizeof(uint16_t));
      indices[i] = static_cast<u32>(val);
    } else if (accessor.componentType == gltfm::COMPONENT_TYPE_UNSIGNED_BYTE) {
      uint8_t val;
      std::memcpy(&val, elementData, sizeof(uint8_t));
      indices[i] = static_cast<u32>(val);
    } else {
      return std::unexpected("Unsupported index component type.");
    }
  }
  return indices;
}
} // namespace GltfLoaderHelpers

export [[nodiscard]] std::expected<LoadedGltfScene, std::string>
loadGltfFile(const std::string &filePath, const std::string &baseDir = "") {
  gltfm::Model model;
  gltfm::TinyGLTF loader;
  std::string err, warn;
  LoadedGltfScene loadedScene;

  bool ret;
  if (filePath.size() > 4 && filePath.substr(filePath.size() - 4) == ".glb") {
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filePath);
  } else {
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filePath);
  }

  if (!warn.empty())
    std::println("TinyGLTF Warning for '{}': {}", filePath, warn);
  if (!err.empty())
    return std::unexpected("TinyGLTF Error for '" + filePath + "': " + err);
  if (!ret)
    return std::unexpected("Failed to parse GLTF file: " + filePath);

  // --- 0. Load Images ---
  // TinyGLTF can load images from URIs if compiled with STB_IMAGE, or access embedded data.
  for (size_t i = 0; i < model.images.size(); ++i) {
    const auto &gltfImage = model.images[i];
    GltfImageData imageData;
    imageData.name = gltfImage.name.empty()
                         ? (gltfImage.uri.empty() ? "Image_" + std::to_string(i) : gltfImage.uri)
                         : gltfImage.name;

    if (!gltfImage.uri.empty() && gltfImage.bufferView == -1) { // External image
      std::string imagePath = baseDir.empty() ? gltfImage.uri : baseDir + "/" + gltfImage.uri;
      // TinyGLTF's LoadImageData function can be used if STB is enabled.
      // Or you can use your own image loading library here (e.g. stb_image directly)
      // For simplicity, assuming tinygltf's internal STB usage or direct pixel access.
      // If gltfm::Image::image is populated, it means tinygltf loaded it (e.g. from URI or
      // embedded)
      if (!gltfImage.image.empty()) {
        imageData.pixels = gltfImage.image;
        imageData.width = gltfImage.width;
        imageData.height = gltfImage.height;
        imageData.component = gltfImage.component;
      } else {
        std::println(
            "Warning: GLTF image '{}' has URI but no pixel data loaded by tinygltf. Path: {}",
            gltfImage.name, imagePath);
        // Here you might attempt to load imagePath using stb_image directly if tinygltf didn't.
        // For now, we'll skip if tinygltf didn't load it.
        continue;
      }
    } else if (gltfImage.bufferView >= 0) { // Embedded image
      const auto &bufferView = model.bufferViews[gltfImage.bufferView];
      const auto &buffer = model.buffers[bufferView.buffer];
      const unsigned char *data_ptr = buffer.data.data() + bufferView.byteOffset;
      size_t data_len = bufferView.byteLength;

      // TinyGLTF can also load embedded images into gltfImage.image if mimeType is known.
      if (!gltfImage.image.empty()) {
        imageData.pixels = gltfImage.image;
        imageData.width = gltfImage.width;
        imageData.height = gltfImage.height;
        imageData.component = gltfImage.component;
      } else {
        // If gltfImage.image is empty, but bufferView is valid, it might be raw data
        // that tinygltf didn't decode (e.g. if mimeType was missing or unsupported by internal
        // STB). You might need to use stb_image_load_from_memory here.
        std::println("Warning: GLTF image '{}' is embedded (bufferView {}) but not decoded by "
                     "tinygltf. MimeType: {}",
                     gltfImage.name, gltfImage.bufferView, gltfImage.mimeType);
        // For now, skip if not decoded by tinygltf.
        continue;
      }
    } else {
      std::println("Warning: GLTF image '{}' has no URI and no bufferView.", gltfImage.name);
      continue;
    }

    // Determine if sRGB (simplistic check, GLTF extensions might specify color space)
    if (gltfImage.extras.IsObject() && gltfImage.extras.Has("colorspace")) {
      if (gltfImage.extras.Get("colorspace").IsString() &&
          gltfImage.extras.Get("colorspace").Get<std::string>() == "srgb") {
        imageData.isSrgb = true;
      } else {
        imageData.isSrgb = false; // Or linear
      }
    } else {
      // Heuristic: if it's a baseColorTexture, assume sRGB. Otherwise, could be linear.
      // This is a simplification. Proper color space handling is complex.
      // For now, assume color textures are sRGB.
      imageData.isSrgb = true;
    }

    loadedScene.images[static_cast<int>(i)] = std::move(imageData);
  }
  // loadedScene.gltfMaterials = model.materials; // Store raw GLTF materials if needed

  // 1. Process Meshes and their Primitives
  for (const auto &gltfMesh : model.meshes) {
    GltfMeshData meshData;
    meshData.name = gltfMesh.name.empty() ? ("Mesh_" + std::to_string(loadedScene.meshes.size()))
                                          : gltfMesh.name;

    for (const auto &gltfPrimitive : gltfMesh.primitives) {
      GltfPrimitiveData primitiveData;
      primitiveData.gltfMaterialIndex = gltfPrimitive.material;

      if (gltfPrimitive.indices >= 0) { /* ... load indices ... */
        auto indicesResult = GltfLoaderHelpers::getIndexAccessorData(model, gltfPrimitive.indices);
        if (indicesResult)
          primitiveData.indices = std::move(*indicesResult);
        else
          return std::unexpected("Indices load fail: " + indicesResult.error());
      } else
        return std::unexpected("Non-indexed primitive.");

      std::vector<glm::vec3> positions, normals;
      std::vector<glm::vec2> uvs;
      if (auto it = gltfPrimitive.attributes.find("POSITION");
          it != gltfPrimitive.attributes.end()) {
        auto res = GltfLoaderHelpers::getAccessorData<float, 3>(model, it->second);
        if (res)
          positions = std::move(*res);
        else
          return std::unexpected("POSITION: " + res.error());
      } else
        return std::unexpected("No POSITION attribute.");

      if (auto it = gltfPrimitive.attributes.find("NORMAL"); it != gltfPrimitive.attributes.end()) {
        auto res = GltfLoaderHelpers::getAccessorData<float, 3>(model, it->second);
        if (res)
          normals = std::move(*res);
        else
          std::println("Warning: No NORMAL for {}.", meshData.name);
      }
      if (auto it = gltfPrimitive.attributes.find("TEXCOORD_0");
          it != gltfPrimitive.attributes.end()) {
        auto res = GltfLoaderHelpers::getAccessorData<float, 2>(model, it->second);
        if (res)
          uvs = std::move(*res);
        else
          std::println("Warning: No TEXCOORD_0 for {}.", meshData.name);
      }

      size_t vertexCount = positions.size();
      primitiveData.vertices.resize(vertexCount);
      for (size_t i = 0; i < vertexCount; ++i) {
        primitiveData.vertices[i].pos = positions[i];
        primitiveData.vertices[i].normal = (i < normals.size()) ? normals[i] : glm::vec3(0, 1, 0);
        primitiveData.vertices[i].uv = (i < uvs.size()) ? uvs[i] : glm::vec2(0, 0);
        primitiveData.vertices[i].tangent = glm::vec4(1, 0, 0, 1); // Placeholder
      }

      if (gltfPrimitive.material >= 0 &&
          static_cast<size_t>(gltfPrimitive.material) < model.materials.size()) {
        const auto &gltfMaterial = model.materials[gltfPrimitive.material];
        const auto &pbr = gltfMaterial.pbrMetallicRoughness;
        primitiveData.material.baseColorFactor = glm::make_vec4(pbr.baseColorFactor.data());
        primitiveData.material.metallicFactor = static_cast<float>(pbr.metallicFactor);
        primitiveData.material.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

        // *** STORE BASE COLOR TEXTURE INDEX ***
        if (pbr.baseColorTexture.index >= 0) {
          primitiveData.baseColorTextureGltfIndex = pbr.baseColorTexture.index;
          // GLTF texture index -> GLTF image index
          if (static_cast<size_t>(pbr.baseColorTexture.index) < model.textures.size()) {
            primitiveData.baseColorTextureGltfIndex =
                model.textures[pbr.baseColorTexture.index].source;
            // Now baseColorTextureGltfIndex is actually the GLTF *image* index.
          } else {
            primitiveData.baseColorTextureGltfIndex = -1; // Invalid texture index
          }
        }
        // TODO: Extract other texture indices (metallicRoughnessTexture.index, normalTexture.index,
        // etc.)
      } else {
        primitiveData.material = Material{};
      }
      meshData.primitives.emplace_back(std::move(primitiveData));
    }
    loadedScene.meshes.emplace_back(std::move(meshData));
  }

  // 2. Process Nodes
  loadedScene.nodes.resize(model.nodes.size());
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    const auto &gltfNode = model.nodes[i];
    GltfNodeData &nodeData = loadedScene.nodes[i];
    nodeData.name = gltfNode.name.empty() ? ("Node_" + std::to_string(i)) : gltfNode.name;
    nodeData.meshIndex = gltfNode.mesh;

    if (!gltfNode.matrix.empty()) {
      std::array<float, 16> mat_data;
      for (int k = 0; k < 16; ++k)
        mat_data[k] = static_cast<float>(gltfNode.matrix[k]);
      nodeData.transform = glm::make_mat4(mat_data.data());
    } else {
      glm::mat4 T = glm::mat4(1.f), R = glm::mat4(1.f), S = glm::mat4(1.f);

      // Convert translation components to float
      if (!gltfNode.translation.empty()) {
        glm::vec3 translation(static_cast<float>(gltfNode.translation[0]),
                              static_cast<float>(gltfNode.translation[1]),
                              static_cast<float>(gltfNode.translation[2]));
        T = glm::translate(T, translation);
      }

      // Convert rotation components to float and reorder (xyzw -> wxyz)
      if (!gltfNode.rotation.empty()) {
        glm::quat rotation(static_cast<float>(gltfNode.rotation[3]), // w
                           static_cast<float>(gltfNode.rotation[0]), // x
                           static_cast<float>(gltfNode.rotation[1]), // y
                           static_cast<float>(gltfNode.rotation[2])  // z
        );
        R = glm::mat4_cast(rotation);
      }

      // Convert scale components to float
      if (!gltfNode.scale.empty()) {
        glm::vec3 scale(static_cast<float>(gltfNode.scale[0]),
                        static_cast<float>(gltfNode.scale[1]),
                        static_cast<float>(gltfNode.scale[2]));
        S = glm::scale(S, scale);
      }

      nodeData.transform = T * R * S;
    }

    for (int childIdx : gltfNode.children) {
      nodeData.childrenIndices.push_back(childIdx);
      if (childIdx >= 0 && static_cast<size_t>(childIdx) < loadedScene.nodes.size())
        loadedScene.nodes[childIdx].parentIndex = static_cast<int>(i);
    }
  }

  // 3. Identify Root Nodes
  if (model.defaultScene >= 0 && static_cast<size_t>(model.defaultScene) < model.scenes.size()) {
    for (int rootNodeIdx : model.scenes[model.defaultScene].nodes)
      loadedScene.rootNodeIndices.push_back(rootNodeIdx);
  } else if (!model.nodes.empty() && model.scenes.empty()) { /* ... fallback same as before ... */
    for (size_t i = 0; i < loadedScene.nodes.size(); ++i)
      if (loadedScene.nodes[i].parentIndex == -1)
        loadedScene.rootNodeIndices.push_back(static_cast<int>(i));
    if (loadedScene.rootNodeIndices.empty() && !loadedScene.nodes.empty())
      loadedScene.rootNodeIndices.push_back(0);
  }
  return loadedScene;
}
