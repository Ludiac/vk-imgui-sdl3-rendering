module;

#include "imgui.h"
#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>

export module vulkan_app:imgui;

import vulkan_hpp;
import std;
import :VulkanWindow;
import :VulkanDevice;
import :VulkanInstance;
import :scene;

void renderMeshControlsMenu(f32 framerate, const Scene &scene) {
  ImGui::Begin("mesh controls");
  for (size_t i = 0; i < scene.nodes.size(); ++i) {

    if (ImGui::CollapsingHeader(("Mesh " + std::to_string(i)).c_str())) {
      auto &transform = scene.nodes[i]->transform;

      std::string meshId = "##Mesh" + std::to_string(i);

      ImGui::SliderFloat3(("Rotation Speed (deg/s)" + meshId).c_str(),
                          &transform.rotation_speed_euler_dps.x, -360.0f, 360.0f, "%.1f");

      ImGui::SliderFloat3(("Rotation (deg)" + meshId).c_str(), &transform.rotation.x, -180.0f,
                          180.0f, "%.1f");

      ImGui::SliderFloat3(("Position" + meshId).c_str(), &transform.translation.x, -5.0f, 5.0f);

      ImGui::SliderFloat3(("Scale" + meshId).c_str(), &transform.scale.x, 1.f, 20.0f);

      if (ImGui::Button(("Reset Rotation" + meshId).c_str())) {
        transform.rotation = glm::vec3(0.f);
      }
      if (ImGui::Button(("Reset Rotation speed" + meshId).c_str())) {
        transform.rotation_speed_euler_dps = glm::vec3(0.0f);
      }
      ImGui::SameLine();
      if (ImGui::Button(("Reset Position" + meshId).c_str())) {
        transform.translation = glm::vec3(0.0f);
      }
      ImGui::SameLine();
      if (ImGui::Button(("Reset scale" + meshId).c_str())) {
        transform.scale = glm::vec3(1.0f);
      }
    }
  }
  ImGui::End();
}
void RenderCameraControlMenu(Camera &camera) {
  ImGui::Begin("Camera Controls");
  ImGui::Checkbox("Mouse Control", &camera.MouseCaptured);
  ImGui::SliderFloat3("Position", &camera.Position.x, -10.0f, 10.0f);
  ImGui::SliderFloat("Yaw", &camera.Yaw, -180.0f, 180.0f);
  ImGui::SliderFloat("Pitch", &camera.Pitch, -89.0f, 89.0f);
  ImGui::SliderFloat("FOV", &camera.Zoom, 1.0f, 120.0f);
  ImGui::SliderFloat("Near Plane", &camera.Near, 0.01f, 1000.0f);
  ImGui::SliderFloat("Far Plane", &camera.Far, 0.01f, 1000.0f);
  ImGui::SliderFloat("Move Speed", &camera.MovementSpeed, 0.1f, 10.0f);
  ImGui::SliderFloat("Mouse Sens", &camera.MouseSensitivity, 0.01f, 1.0f);
  ImGui::End();
}

void RenderVulkanStateWindow(VulkanDevice &device, Window &wd, int frameCap, float frameTime) {
  ImGui::Begin("Vulkan State Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  if (ImGui::CollapsingHeader("Physical Device", ImGuiTreeNodeFlags_DefaultOpen)) {
    vk::PhysicalDeviceProperties props = device.physical().getProperties();
    ImGui::Text("GPU: %s", props.deviceName.data());
    // ImGui::Text("API Version: %d.%d.%d", VK_VERSION_MAJOR(props.apiVersion),
    //             VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
  }

  if (ImGui::CollapsingHeader("Swapchain", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Present Mode: %s", vk::to_string(wd.config.PresentMode).c_str());
    ImGui::Text("Swapchain Images: %zu", wd.Frames.size());
    ImGui::Text("Extent: %dx%d", wd.config.swapchainExtent.width, wd.config.swapchainExtent.height);
    ImGui::Text("Format: %s", vk::to_string(wd.config.SurfaceFormat.format).c_str());
  }

  if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderInt("Frame Cap", &frameCap, 4, 500);
    ImGui::Text("Frame Time: %.3f ms", frameTime * 1000.0f);
    ImGui::Text("FPS: %.1f", 1.0f / frameTime);

    ImGui::Text("current image %d", wd.FrameIndex);
    static std::vector<float> frameTimes;
    frameTimes.push_back(frameTime * 1000.0f);
    if (frameTimes.size() > 90)
      frameTimes.erase(frameTimes.begin());
    ImGui::PlotLines("Frame Times", frameTimes.data(), frameTimes.size(), 0, nullptr, 0.0f, 33.3f,
                     ImVec2(300, 50));
  }

  if (ImGui::CollapsingHeader("Queue")) {
    ImGui::Text("Graphics Queue Family: %u", device.queueFamily_);
    auto props = device.physical().getQueueFamilyProperties()[device.queueFamily_];
    ImGui::Text("Queue Count: %u", props.queueCount);
    ImGui::Text("Timestamp Valid Bits: %u", props.timestampValidBits);
  }

  if (ImGui::CollapsingHeader("Memory")) {
    auto memProps = device.physical().getMemoryProperties();
    ImGui::Text("Memory Heaps: %u", memProps.memoryHeapCount);
    ImGui::Text("Memory Types: %u", memProps.memoryTypeCount);
  }

  ImGui::End();
}

bool EditMaterialProperties(const std::string &materialOwnerName, Material &material) {
  bool changed = false;
  ImGui::PushID(&material); // Unique ID scope for this material editor instance

  // You can use materialOwnerName to make the TreeNode label more specific if needed,
  // e.g., std::string title = materialOwnerName + " Material";
  // if (ImGui::TreeNode(title.c_str())) {
  // For now, using a generic "Material Properties" which is clear when nested under a node/mesh.
  if (ImGui::TreeNode("Material Properties")) {
    if (ImGui::ColorEdit4("Base Color Factor", &material.baseColorFactor.x)) {
      changed = true;
    }
    if (ImGui::SliderFloat("Metallic Factor", &material.metallicFactor, 0.0f, 1.0f)) {
      changed = true;
    }
    if (ImGui::SliderFloat("Roughness Factor", &material.roughnessFactor, 0.0f, 1.0f)) {
      changed = true;
    }

    // Add ImGui controls for other members of your Material struct here:
    // Example:
    // if (ImGui::ColorEdit3("Emissive Factor", &material.emissiveFactor.x)) changed = true;
    // if (ImGui::SliderFloat("Occlusion Strength", &material.occlusionStrength, 0.0f, 1.0f))
    // changed = true; if (ImGui::SliderFloat("Normal Scale", &material.normalScale, 0.0f, 2.0f))
    // changed = true; if (ImGui::SliderFloat("Height Scale", &material.heightScale, 0.0f, 0.1f))
    // changed = true;

    ImGui::TreePop();
  }
  ImGui::PopID();
  return changed;
}

/**
 * @brief Recursively renders ImGui UI for a SceneNode and its children.
 *
 * Displays the node's name. If the node has an associated mesh, it provides
 * controls to edit the mesh's material.
 *
 * @param node Pointer to the SceneNode to render.
 * @param currentFrameIndex The current swapchain image index, used for updating buffers.
 */
void RenderSceneNodeRecursive(SceneNode *node, u32 currentFrameIndex) {
  if (!node) {
    return;
  }

  ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow |
                                 ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                 ImGuiTreeNodeFlags_DefaultOpen;
  if (node->children.empty()) { // If the node has no children, display it as a leaf
    nodeFlags |= ImGuiTreeNodeFlags_Leaf;
  }

  // Use the node's pointer as a unique ID for ImGui::TreeNodeEx to handle nodes with identical
  // names. Display the node's name.
  bool nodeIsOpen = ImGui::TreeNodeEx((void *)(intptr_t)node, nodeFlags, "%s", node->name.c_str());

  if (nodeIsOpen) {
    // If the current node has an associated mesh, display material controls for it.
    if (node->mesh) { //
      // Pass the mesh's name and a reference to its material to the editor function.
      // The Mesh class has getName() and getMaterial() methods.
      if (EditMaterialProperties(node->mesh->getName(), node->mesh->getMaterial())) {
        // If the material was changed by ImGui:
        // 1. Update the material's Uniform Buffer Object (UBO) data for the current frame.
        node->mesh->updateMaterialUniformBufferData(currentFrameIndex);
        // 2. Update the mesh's descriptor set for the current frame. This is crucial
        //    as the descriptor set references the material UBO.
        node->mesh->updateDescriptorSetContents(currentFrameIndex);
      }
    } else {
      // Optionally, you can indicate that this node does not have a mesh.
      // ImGui::SameLine(); ImGui::TextDisabled("(No mesh)");
    }

    // Recursively call this function for all children of the current node.
    for (SceneNode *child : node->children) {
      RenderSceneNodeRecursive(child, currentFrameIndex);
    }
    ImGui::TreePop();
  }
}

export void RenderSceneHierarchyMaterialEditor(Scene &scene, u32 currentFrameIndex) {
  if (ImGui::Begin("Scene Inspector")) { // Main window for the scene editor
    // An initial TreeNode for the entire scene graph can be helpful.
    if (ImGui::TreeNodeEx("Scene Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (scene.roots.empty()) { // Check if there are any root nodes in the scene.
        ImGui::Text("Scene is empty or has no root nodes.");
      } else {
        // Iterate over the root nodes of the scene and render the hierarchy from there.
        for (SceneNode *rootNode : scene.roots) {
          RenderSceneNodeRecursive(rootNode, currentFrameIndex);
        }
      }
      ImGui::TreePop();
    }
    // You could add other scene-wide settings or information here, outside the "Scene Graph"
    // TreeNode.
  }
  ImGui::End(); // End of "Scene Inspector" window
}
