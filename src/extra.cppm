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

export struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
  glm::vec4 tangent; // w-component stores handedness (1 or -1)
};

export struct Material {
  glm::vec4 baseColorFactor{1.0f}; // RGBA (sRGB color)
  // No textures/metallic/roughness yet
};

export class Camera {
public:
  // === Camera Configuration ===
  glm::vec3 Position{0.0f, 0.0f, 3.0f}; // World position
  float Yaw = -90.0f;                   // Horizontal rotation (degrees)
  float Pitch = 0.0f;                   // Vertical rotation (degrees)
  float Zoom = 45.0f;                   // Field of view (degrees)
  float Near = 0.1f;                    // Near clipping plane
  float Far = 100.0f;                   // Far clipping plane

  // === Control Parameters ===
  float MovementSpeed = 100.f;   // Base movement speed
  float MouseSensitivity = 0.1f; // Mouse look sensitivity
  bool MouseCaptured = false;    // Mouse control state

  // === Derived Vectors (updated internally) ===
  glm::vec3 Front{0.0f, 0.0f, -1.0f};  // Forward direction
  glm::vec3 Right{1.0f, 0.0f, 0.0f};   // Right direction
  glm::vec3 Up{0.0f, 1.0f, 0.0f};      // Up vector
  glm::vec3 WorldUp{0.0f, 1.0f, 0.0f}; // World up reference

  // === Matrix Generation Functions ===
  glm::mat4 GetViewMatrix() const { return glm::lookAt(Position, Position + Front, Up); }

  glm::mat4 GetProjectionMatrix(float aspectRatio) const {
    return glm::perspective(glm::radians(Zoom), // Vertical FOV
                            aspectRatio,        // Window aspect ratio (width/height)
                            Near,               // Near clipping distance
                            Far                 // Far clipping distance
    );
  }

private:
  void updateVectors() {
    // Calculate new Front vector
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front);

    // Recalculate right and up vectors
    Right = glm::normalize(glm::cross(Front, WorldUp));
    Up = glm::normalize(glm::cross(Right, Front));
  }
};

export struct Transform {
  glm::vec3 translation = glm::vec3(0.0f);
  glm::vec3 scale = glm::vec3(1.0f);
  glm::vec3 rotation = glm::vec3(0.0f);
  glm::vec3 rotation_speed = glm::vec3(0.0f);

  [[nodiscard]] glm::mat4 matrix(float delta_time = 0.0f) {
    rotation += rotation_speed * delta_time;

    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::scale(transform, scale);
    transform = glm::rotate(transform, glm::radians(rotation.x), {1, 0, 0});
    transform = glm::rotate(transform, glm::radians(rotation.y), {0, 1, 0});
    transform = glm::rotate(transform, glm::radians(rotation.z), {0, 0, 1});
    transform = glm::translate(transform, translation);

    return transform;
  }
};

export struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 projection;
};

export void set_projection(glm::mat4 &projection, vk::Extent2D swapchainExtent) {
  projection = glm::perspective(glm::radians(45.0f),
                                swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 10.0f);
}

export void set_view(glm::mat4 &view) {
  view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                     glm::vec3(0.0f, 0.0f, 1.0f));
}

// export void spin_model(glm::mat4 &model, float rotationAngle) {
//   glm::quat rotZ = glm::angleAxis(rotationAngle * glm::radians(90.0f), glm::vec3(0.0f,
//   0.0f, 1.0f)); glm::quat rotX = glm::angleAxis(rotationAngle * glm::radians(45.0f),
//   glm::vec3(1.0f, 0.0f, 0.0f)); glm::quat rotY = glm::angleAxis(rotationAngle *
//   glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
//
//   glm::quat combinedRotation = rotZ * rotX * rotY;
//   model = glm::mat4_cast(combinedRotation);
// }
