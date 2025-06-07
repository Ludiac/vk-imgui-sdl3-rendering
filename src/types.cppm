module;

// #include "macros.hpp"
// #include "primitive_types.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp> // For glm::decompose

export module vulkan_app:Types;

import vulkan_hpp;
import std;

export struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
  glm::vec4 tangent; // w-component stores handedness (1 or -1)
};

export struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 projection;
  glm::mat3 normalMatrix;
};

export struct Transform {
  glm::vec3 translation{0.0f};
  glm::vec3 scale{1.0f};
  glm::quat rotation{glm::identity<glm::quat>()};

  glm::vec3 rotation_speed_euler_dps{0.0f};

  void update(float delta_time) {
    if (glm::length(rotation_speed_euler_dps) > 0.0f) {

      glm::vec3 angular_change_rad = glm::radians(rotation_speed_euler_dps) * delta_time;

      // Create a delta quaternion from Euler angles
      // Order of application for Euler to quaternion can matter (e.g., ZYX, XYZ)
      // Common is ZYX:
      glm::quat delta_rotation =
          glm::quat(glm::vec3(angular_change_rad.x, angular_change_rad.y, angular_change_rad.z));

      rotation = delta_rotation * rotation;
      rotation = glm::normalize(rotation); // Normalize quaternion to prevent drift
    }
  }

  [[nodiscard]] glm::mat4 getMatrix() const {
    glm::mat4 trans_matrix = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 rot_matrix = glm::mat4_cast(rotation);
    glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), scale);

    return trans_matrix * rot_matrix * scale_matrix;
  }

  void setRotationEuler(const glm::vec3 &euler_angles_degrees) {
    rotation = glm::quat(glm::radians(euler_angles_degrees));
    rotation = glm::normalize(rotation);
  }

  [[nodiscard]] glm::vec3 getRotationEulerDegrees() const {
    return glm::degrees(glm::eulerAngles(rotation));
  }
};

export [[nodiscard]] Transform decomposeFromMatrix(const glm::mat4 &matrix) {
  Transform t;
  glm::vec3 skew;
  glm::vec4 perspective;

  if (glm::decompose(matrix, t.scale, t.rotation, t.translation, skew, perspective)) {

  } else {

    std::println("Warning: Matrix decomposition failed in Transform::decomposeFromMatrix. "
                 "Returning default transform.");
    return Transform{};
  }
  return t;
}

export struct Material {
  glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
  float metallicFactor{1.0f};
  float roughnessFactor{1.0f};
  float occlusionStrength{1.0f};
  glm::vec3 emissiveFactor{0.0f};
  float normalScale{1.0f};
  float heightScale{0.0f};
};

export class Camera {
public:
  // === Camera Configuration ===
  glm::vec3 Position{-10.0f, -10.0f, 60.0f}; // World position
  float Yaw = -75.0f;                        // Horizontal rotation (degrees)
  float Pitch = 10.0f;                       // Vertical rotation (degrees)
  float Roll = 0.0f;                         // Added: Roll rotation (degrees)
  float Zoom = 45.0f;                        // Field of view (degrees)
  float Near = 0.1f;                         // Near clipping plane
  float Far = 10000.0f;                      // Far clipping plane

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
    return glm::perspective(glm::radians(Zoom), aspectRatio, Near, Far);
  }

  void updateVectors() {
    // Calculate front vector from yaw and pitch
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front);

    // Create rotation quaternion for roll
    glm::quat rollQuat = glm::angleAxis(glm::radians(Roll), Front);

    // Apply roll rotation to world up
    glm::vec3 rolledUp = rollQuat * WorldUp;

    // Re-calculate right and up vectors
    Right = glm::normalize(glm::cross(Front, rolledUp));
    Up = glm::normalize(glm::cross(Right, Front));
  }
};
