module;

#include "primitive_types.hpp"
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

export module vulkan_app:extra;

import std;
import vulkan_hpp;
import :Types;

export std::vector<Vertex> create_cuboid_vertices(float width, float height, float length) {
  const float w = width / 2.0f;
  const float h = height / 2.0f;
  const float l = length / 2.0f;

  std::vector<Vertex> vertices;
  vertices.reserve(24); // 6 faces * 4 vertices per face

  // Front face (z = -l)
  vertices.push_back({{-w, -h, -l}, {0, 0, -1}, {0, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, -h, -l}, {0, 0, -1}, {1, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, h, -l}, {0, 0, -1}, {1, 1}, {1, 0, 0, 1}});
  vertices.push_back({{-w, h, -l}, {0, 0, -1}, {0, 1}, {1, 0, 0, 1}});

  // Back face (z = l)
  vertices.push_back({{-w, -h, l}, {0, 0, 1}, {0, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, -h, l}, {0, 0, 1}, {1, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, h, l}, {0, 0, 1}, {1, 1}, {1, 0, 0, 1}});
  vertices.push_back({{-w, h, l}, {0, 0, 1}, {0, 1}, {1, 0, 0, 1}});

  // Left face (x = -w)
  vertices.push_back({{-w, -h, -l}, {-1, 0, 0}, {0, 0}, {1, 0, 0, 1}});
  vertices.push_back({{-w, h, -l}, {-1, 0, 0}, {0, 1}, {1, 0, 0, 1}});
  vertices.push_back({{-w, h, l}, {-1, 0, 0}, {1, 1}, {1, 0, 0, 1}});
  vertices.push_back({{-w, -h, l}, {-1, 0, 0}, {1, 0}, {1, 0, 0, 1}});

  // Right face (x = w)
  vertices.push_back({{w, -h, -l}, {1, 0, 0}, {0, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, -h, l}, {1, 0, 0}, {1, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, h, l}, {1, 0, 0}, {1, 1}, {1, 0, 0, 1}});
  vertices.push_back({{w, h, -l}, {1, 0, 0}, {0, 1}, {1, 0, 0, 1}});

  // Top face (y = h)
  vertices.push_back({{-w, h, -l}, {0, 1, 0}, {0, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, h, -l}, {0, 1, 0}, {1, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, h, l}, {0, 1, 0}, {1, 1}, {1, 0, 0, 1}});
  vertices.push_back({{-w, h, l}, {0, 1, 0}, {0, 1}, {1, 0, 0, 1}});

  // Bottom face (y = -h)
  vertices.push_back({{-w, -h, -l}, {0, -1, 0}, {0, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, -h, -l}, {0, -1, 0}, {1, 0}, {1, 0, 0, 1}});
  vertices.push_back({{w, -h, l}, {0, -1, 0}, {1, 1}, {1, 0, 0, 1}});
  vertices.push_back({{-w, -h, l}, {0, -1, 0}, {0, 1}, {1, 0, 0, 1}});

  return vertices;
}

export std::vector<uint32_t> create_cuboid_indices() {
  std::vector<uint32_t> indices;
  indices.reserve(36); // 6 faces * 6 indices per face

  for (uint32_t face = 0; face < 6; ++face) {
    const uint32_t base = face * 4;
    // First triangle
    indices.push_back(base);
    indices.push_back(base + 1);
    indices.push_back(base + 2);
    // Second triangle
    indices.push_back(base + 2);
    indices.push_back(base + 3);
    indices.push_back(base);
  }

  return indices;
}

export void set_projection(glm::mat4 &projection, vk::Extent2D swapchainExtent) {
  projection = glm::perspective(glm::radians(45.0f),
                                swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 10.0f);
}

export void set_view(glm::mat4 &view) {
  view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                     glm::vec3(0.0f, 0.0f, 1.0f));
}
