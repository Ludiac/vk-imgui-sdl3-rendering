module;

// #include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:quads;

import vulkan_hpp;
import std;

export struct Vertex2 {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
};

std::vector<Vertex> createQuadVertices(vk::Extent2D extent) {}

std::vector<u32> createQuadIndices() { return {0, 1, 2, 2, 3, 0}; }
