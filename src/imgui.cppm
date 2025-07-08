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
import :TextSystem;

/**
 * @brief Renders an ImGui window for toggling various shader features.
 */
export void RenderShaderTogglesMenu(ShaderTogglesUBO &toggles) {
    if (ImGui::Begin("Shader Toggles")) {
        ImGui::Checkbox("Normal Mapping", (bool *)&toggles.useNormalMapping);
        ImGui::Checkbox("Occlusion", (bool *)&toggles.useOcclusion);
        ImGui::Checkbox("Emission", (bool *)&toggles.useEmission);
        ImGui::Checkbox("Lights", (bool *)&toggles.useLights);
        ImGui::Checkbox("Ambient", (bool *)&toggles.useAmbient);
    }
    ImGui::End();
}

/**
 * @brief Renders an ImGui window for controlling text rendering parameters.
 */
export void RenderTextMenu(i32 &fontSizeMultiplier, TextToggles &textToggles) {
    const static i32 min_stages = 1;
    const static i32 max_stages = 16;
    if (ImGui::Begin("Text Parameters")) {
        ImGui::SliderInt("fontSizeMultiplier", &fontSizeMultiplier, -10, 50);
        ImGui::SliderFloat("sdf_weight", &textToggles.sdf_weight, 0.0f, 1.0f);
        ImGui::SliderFloat("pxRange direct additive", &textToggles.pxRangeDirectAdditive, -10.0f, 10.0f);

        ImGui::Text("Anti Aliasing Mode");
        ImGui::RadioButton("None", &textToggles.antiAliasingMode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Smoothstep", &textToggles.antiAliasingMode, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Linear Clamp", &textToggles.antiAliasingMode, 2);
        ImGui::SameLine();
        ImGui::RadioButton("Staged", &textToggles.antiAliasingMode, 3);

        if (textToggles.antiAliasingMode == 2) {
            ImGui::SliderFloat("Inner AA depth", &textToggles.start_fade_px, -2.0f, 2.0f);
            ImGui::SliderFloat("Outer AA depth", &textToggles.end_fade_px, -2.0f, 2.0f);
        }

        if (textToggles.antiAliasingMode == 3) {
            ImGui::Separator();
            ImGui::Text("Staged AA Parameters");
            ImGui::SliderScalar("Num Stages", ImGuiDataType_U32, &textToggles.num_stages, &min_stages, &max_stages, "%u");
            ImGui::Text("Rounding Direction");
            ImGui::RadioButton("Down", &textToggles.rounding_direction, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Up", &textToggles.rounding_direction, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Nearest", &textToggles.rounding_direction, 2);
        }
    }
    ImGui::End();
}

/**
 * @brief Renders an ImGui window for controlling camera properties.
 */
void RenderCameraControlMenu(Camera &camera) {
    ImGui::Begin("Camera Controls");
    ImGui::Checkbox("Mouse Control", &camera.MouseCaptured);
    ImGui::SliderFloat3("Position", &camera.Position.x, -100.0f, 100.0f);
    ImGui::SliderFloat("Yaw", &camera.Yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &camera.Pitch, -89.0f, 89.0f);
    ImGui::SliderFloat("Roll", &camera.Roll, -180.0f, 180.0f);
    ImGui::SliderFloat("FOV", &camera.Zoom, 1.0f, 120.0f);
    ImGui::SliderFloat("Near Plane", &camera.Near, 0.01f, 1000.0f);
    ImGui::SliderFloat("Far Plane", &camera.Far, 0.01f, 1000.0f);
    ImGui::SliderFloat("Move Speed", &camera.MovementSpeed, 0.1f, 1000.0f);
    ImGui::SliderFloat("Mouse Sens", &camera.MouseSensitivity, 0.01f, 1.0f);
    ImGui::End();
}

/**
 * @brief Renders an ImGui window with debug information about the Vulkan state.
 */
void RenderVulkanStateWindow(VulkanDevice &device, Window &wd, int frameCap, float frameTime) {
    ImGui::Begin("Vulkan State Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    if (ImGui::CollapsingHeader("Physical Device", ImGuiTreeNodeFlags_DefaultOpen)) {
        vk::PhysicalDeviceProperties props = device.physical().getProperties();
        ImGui::Text("GPU: %s", props.deviceName.data());
    }

    if (ImGui::CollapsingHeader("Swapchain", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Present Mode: %s", vk::to_string(wd.config.PresentMode).c_str());
        ImGui::Text("Swapchain Images: %zu", wd.Frames.size());
        ImGui::Text("Extent: %dx%d", wd.config.swapchainExtent.width, wd.config.swapchainExtent.height);
        ImGui::Text("Format: %s", vk::to_string(wd.config.SurfaceFormat.format).c_str());
    }

    if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Frame Time: %.3f ms", frameTime * 1000.0f);
        ImGui::Text("FPS: %.1f", 1.0f / frameTime);
        ImGui::Text("current image %d", wd.FrameIndex);
        static std::vector<float> frameTimes;
        frameTimes.push_back(frameTime * 1000.0f);
        if (frameTimes.size() > 90)
            frameTimes.erase(frameTimes.begin());
        ImGui::PlotLines("Frame Times", frameTimes.data(), frameTimes.size(), 0, nullptr, 0.0f, 33.3f, ImVec2(300, 50));
    }
    ImGui::End();
}

/**
 * @brief Renders ImGui controls for a single material's properties.
 */
bool EditMaterialProperties(const std::string &materialOwnerName, Material &material) {
    bool changed = false;
    if (ImGui::TreeNode("Material Properties")) {
        changed |= ImGui::ColorEdit4("Base Color Factor", &material.baseColorFactor.x);
        changed |= ImGui::SliderFloat("Metallic Factor", &material.metallicFactor, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Roughness Factor", &material.roughnessFactor, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Occlusion Strength", &material.occlusionStrength, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Emissive Factor", &material.emissiveFactor.x);
        ImGui::TreePop();
    }
    return changed;
}

/**
 * @brief Recursively renders an ImGui tree for a SceneNode and its children.
 */
void RenderSceneNodeRecursive(SceneNode *node, u32 currentFrameIndex) {
    if (!node) {
        return;
    }

    ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_DefaultOpen;
    if (node->children.empty()) {
        nodeFlags |= ImGuiTreeNodeFlags_Leaf;
    }

    bool nodeIsOpen = ImGui::TreeNodeEx((void *)(intptr_t)node, nodeFlags, "%s", node->name.c_str());

    if (nodeIsOpen) {
        if (node->mesh) {
            if (EditMaterialProperties(node->mesh->getName(), node->mesh->getMaterial())) {
                node->mesh->updateMaterialUniformBufferData(currentFrameIndex);
                node->mesh->updateDescriptorSetContents(currentFrameIndex);
            }
        }

        for (SceneNode *child : node->children) {
            RenderSceneNodeRecursive(child, currentFrameIndex);
        }
        ImGui::TreePop();
    }
}

/**
 * @brief Renders the main scene hierarchy and material editor window.
 */
export void RenderSceneHierarchyMaterialEditor(Scene &scene, u32 currentFrameIndex) {
    if (ImGui::Begin("Scene Inspector")) {
        if (ImGui::TreeNodeEx("Scene Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderSceneNodeRecursive(scene.fatherNode_.get(), currentFrameIndex);
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

/**
 * @brief Renders an ImGui window for controlling scene lights.
 */
void RenderLightControlMenu(SceneLightsUBO &lightUBO) {
    ImGui::Begin("Light Controls");
    for (int i = 0; i < lightUBO.lightCount; ++i) {
        std::string label = "Light " + std::to_string(i);
        ImGui::PushID(i);
        ImGui::Text("%s", label.c_str());
        ImGui::DragFloat3("Position", &lightUBO.lights[i].position.x, 0.5f);
        ImGui::ColorEdit3("Color", &lightUBO.lights[i].color.x);
        ImGui::DragFloat("Intensity", &lightUBO.lights[i].color.w, 1.0f, 0.0f, 1000.0f);
        ImGui::PopID();
    }
    ImGui::End();
}
