module;

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "macros.hpp"
#include "primitive_types.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module vulkan_app;

import vulkan_hpp;
import std;

import :VulkanWindow;
import :VulkanDevice;
import :VulkanInstance;
import :VulkanPipeline;
import :extra;        // For Vertex, Material, UniformBufferObject, Transform, Camera
import :mesh;         // For Mesh (refactored version)
import :scene;        // For Scene, SceneNode, SceneNodeCreateInfo (refactored version)
import :texture;      // For Texture, PBRTextures
import :TextureStore; // For managing textures
import :ModelLoader;  // Not used in this hardcoded example, but for future
import :SceneBuilder; // Not used in this hardcoded example, but for future
import :imgui;        // For ImGui helper functions
import :Logger;       // For ImGui helper functions

namespace {                             // Anonymous namespace for internal linkage
constexpr uint32_t MIN_IMAGE_COUNT = 2; // Renamed from minImageCount to avoid conflict

// Helper to create vertices for a single quad (face of a cube)
// Normal points outwards. UVs cover the quad. Tangent is basic.
std::vector<Vertex> createQuadVertices(float size, const glm::vec3 &normal, const glm::vec3 &up,
                                       const glm::vec3 &right) {
  float s = size / 2.0f;
  glm::vec3 p0 = -s * right - s * up; // Bottom-left
  glm::vec3 p1 = s * right - s * up;  // Bottom-right
  glm::vec3 p2 = s * right + s * up;  // Top-right
  glm::vec3 p3 = -s * right + s * up; // Top-left

  glm::vec4 tangent = glm::vec4(right, 1.0f); // Basic tangent
  if (glm::length(right) == 0.0f)
    tangent = glm::vec4(1, 0, 0, 1); // Fallback

  return {
      {p0, normal, {0.0f, 1.0f}, tangent}, // UVs: (0,1) BL
      {p1, normal, {1.0f, 1.0f}, tangent}, //      (1,1) BR
      {p2, normal, {1.0f, 0.0f}, tangent}, //      (1,0) TR
      {p3, normal, {0.0f, 0.0f}, tangent}  //      (0,0) TL
  };
}
// Indices for a quad (two triangles, CCW)
std::vector<uint32_t> createQuadIndices() { return {0, 1, 2, 2, 3, 0}; }

std::vector<Vertex> createAxisLineVertices(const glm::vec3 &start, const glm::vec3 &end,
                                           const glm::vec3 &normal_placeholder) {
  // Normals and UVs might not be strictly needed for a simple colored line,
  // but the Vertex struct requires them. Tangents also.
  return {{start, normal_placeholder, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
          {end, normal_placeholder, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}};
}
std::vector<uint32_t> createAxisLineIndices() {
  return {0, 1}; // A single line segment
}
} // anonymous namespace

void check_vk_result(VkResult err) {
  if (err == VK_SUCCESS)
    return;
  std::string errMsg = "[vulkan] Error: VkResult = " + std::to_string(err);
  std::print("{}", errMsg); // Keep console print for immediate visibility
  if (err < 0)
    std::exit(0);
}

void check_vk_result_hpp(vk::Result err) {
  if (err == vk::Result::eSuccess)
    return;
  std::string errMsg = "[vulkan] Error: VkResult = " + vk::to_string(err);
  std::print("{}", errMsg); // Keep console print
  std::exit(0);
}

export class App {
  VulkanInstance instance;
  VulkanDevice device{instance}; // Default constructor needs instance

  Window wd; // VulkanWindow related data
  bool swapChainRebuild = false;
  // int frameCap = 120;
  // float targetFrameDuration = 1.0f / static_cast<float>(frameCap);

  // Rendering Resources
  std::vector<VulkanPipeline> graphicsPipelines; // You might have multiple pipelines
  vk::raii::PipelineCache pipelineCache{nullptr};
  vk::raii::DescriptorSetLayout combinedMeshLayout{nullptr}; // Single layout for meshes

  // Scene and Assets
  Scene scene{0};
  TextureStore textureStore{device, device.queue_};
  std::vector<std::unique_ptr<Mesh>> appOwnedMeshes; // App owns all mesh objects

  Camera camera;

public:
  // Create the single combined descriptor set layout for meshes
  std::expected<void, std::string> createCombinedMeshDescriptorSetLayout() NOEXCEPT {

    std::vector<vk::DescriptorSetLayoutBinding> meshDataBindings = {
        {// Binding 0: MVP Uniform Buffer Object (Vertex Shader)
         .binding = 0,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eVertex},
        {// Binding 1: Material Uniform Buffer Object (Fragment Shader)
         .binding = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {// Binding 2: Base Color Texture (Fragment Shader)
         .binding = 2,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment}
        // Add more bindings here for other PBR textures (normal, metallicRoughness, etc.)
        // e.g., binding = 3 for normal map, etc.
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount =
                                                     static_cast<uint32_t>(meshDataBindings.size()),
                                                 .pBindings = meshDataBindings.data()};

    auto layoutResult = device.logical().createDescriptorSetLayout(layoutInfo);
    if (!layoutResult) {
      std::string errorMsg = "Failed to create combined mesh descriptor set layout: " +
                             vk::to_string(layoutResult.error());

      return std::unexpected(errorMsg);
    }
    combinedMeshLayout = std::move(layoutResult.value());

    return {};
  }

  void createPipelines() { // Simplified pipeline creation
    if (graphicsPipelines.empty()) {

      graphicsPipelines.resize(2); // For now, one main pipeline
    }
    VulkanPipeline &mainPipeline = graphicsPipelines[0];

    std::vector<vk::DescriptorSetLayout> layouts = {*combinedMeshLayout};

    EXPECTED_VOID(mainPipeline.createPipelineLayout(device.logical(), layouts));

    auto vertShaderModule = createShaderModuleFromFile(device.logical(), "shaders/vert.spv");
    auto fragShaderModule = createShaderModuleFromFile(device.logical(), "shaders/frag.spv");

    if (!vertShaderModule || !fragShaderModule) {
      std::println("Error loading shaders: {} & {}", vertShaderModule.error_or(""),
                   fragShaderModule.error_or("")); // Keep console print
      return;
    }

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
        {.stage = vk::ShaderStageFlagBits::eVertex,
         .module = *vertShaderModule.value(),
         .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment,
         .module = *fragShaderModule.value(),
         .pName = "main"}};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = false};

    EXPECTED_VOID(mainPipeline.createGraphicsPipeline(device.logical(), pipelineCache, shaderStages,
                                                      inputAssembly, wd.RenderPass));

    VulkanPipeline &linePipeline = graphicsPipelines[1];
    EXPECTED_VOID(
        linePipeline.createPipelineLayout(device.logical(), layouts)); // Reusing same layout

    vk::PipelineInputAssemblyStateCreateInfo lineInputAssembly{
        .topology = vk::PrimitiveTopology::eLineList, .primitiveRestartEnable = false};
    EXPECTED_VOID(linePipeline.createGraphicsPipeline(device.logical(), pipelineCache, shaderStages,
                                                      lineInputAssembly, wd.RenderPass));
  }

  void createTexturedCubeScene(u32 currentImageCount) {

    // if (textureStore) {
    //
    //   std::println("Error: TextureStore not initialized in createTexturedCubeScene.");
    //   return;
    // }
    if (graphicsPipelines.empty() || !*graphicsPipelines[0].pipeline) {

      std::println("Error: Default graphics pipeline not ready for createTexturedCubeScene.");
      return;
    }
    VulkanPipeline *defaultPipeline = &graphicsPipelines[0];

    SceneNode *cubeRootNode = scene.createNode(
        {.transform = Transform{}, .pipeline = defaultPipeline, .name = "CubeRoot"});
    if (!cubeRootNode) {
      return;
    }

    float cubeSize = 10.0f;

    std::array<std::shared_ptr<Texture>, 6> faceTextures = {
        textureStore.getColorTexture("red", {255, 0, 0, 255}),
        textureStore.getColorTexture("green", {0, 255, 0, 255}),
        textureStore.getColorTexture("blue", {0, 0, 255, 255}),
        textureStore.getColorTexture("yellow", {255, 255, 0, 255}),
        textureStore.getColorTexture("cyan", {0, 255, 255, 255}),
        textureStore.getColorTexture("magenta", {255, 0, 255, 255})};

    struct FaceDef {
      std::string name;
      glm::vec3 normal;
      glm::vec3 up;
      glm::vec3 right;
      glm::vec3 translation;
      std::shared_ptr<Texture> texture;
    };

    std::vector<FaceDef> faceDefs = {
        {"FrontFace", {0, 0, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -cubeSize / 2.0f}, faceTextures[0]},
        {"BackFace", {0, 0, 1}, {0, 1, 0}, {-1, 0, 0}, {0, 0, cubeSize / 2.0f}, faceTextures[1]},
        {"LeftFace", {-1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-cubeSize / 2.0f, 0, 0}, faceTextures[2]},
        {"RightFace", {1, 0, 0}, {0, 1, 0}, {0, 0, -1}, {cubeSize / 2.0f, 0, 0}, faceTextures[3]},
        {"TopFace", {0, 1, 0}, {0, 0, -1}, {1, 0, 0}, {0, cubeSize / 2.0f, 0}, faceTextures[4]},
        {"BottomFace", {0, -1, 0}, {0, 0, 1}, {1, 0, 0}, {0, -cubeSize / 2.f, 0}, faceTextures[5]}};

    for (const auto &def : faceDefs) {
      std::vector<Vertex> faceVertices =
          createQuadVertices(cubeSize, def.normal, def.up, def.right);
      std::vector<uint32_t> faceIndices = createQuadIndices();
      Material faceMaterial;
      faceMaterial.baseColorFactor = glm::vec4(1.0f);
      PBRTextures facePbrTextures;
      facePbrTextures.baseColor = def.texture ? def.texture : textureStore.getDefaultTexture();
      auto faceMesh =
          std::make_unique<Mesh>(device, def.name, std::move(faceVertices), std::move(faceIndices),
                                 faceMaterial, facePbrTextures, currentImageCount);
      Mesh *faceMeshPtr = faceMesh.get();
      appOwnedMeshes.emplace_back(std::move(faceMesh));
      Transform faceTransform;
      faceTransform.translation = def.translation;
      SceneNode *faceNode = scene.createNode({.mesh = faceMeshPtr,
                                              .transform = faceTransform,
                                              .pipeline = defaultPipeline,
                                              .parent = cubeRootNode,
                                              .name = def.name + "_Node"});
      if (!faceNode) {
      } else {
      }
    }
    if (cubeRootNode) {
      cubeRootNode->transform.rotation_speed_euler_dps = {10.f, 15.f, 5.f};
    }
  }

  void createDebugAxesScene(u32 currentImageCount) {
    if (graphicsPipelines.size() < 2 || !*graphicsPipelines[1].pipeline) {
      std::println(
          "Error: TextureStore or Line Pipeline not initialized for createDebugAxesScene.");
      return;
    }
    VulkanPipeline *linePipeline = &graphicsPipelines[1]; // Use the dedicated line pipeline
    float axisLength = 10000.0f;                          // Make axes reasonably large

    // X-Axis (Red)
    Material xAxisMaterial;
    PBRTextures xAxisTextures;
    xAxisTextures.baseColor = textureStore.getColorTexture("white", {255, 0, 0, 255});
    auto xAxisMesh = std::make_unique<Mesh>(
        device, "X_Axis",
        createAxisLineVertices({-axisLength, 0, 0}, {axisLength, 0, 0}, {0, 1, 0}),
        createAxisLineIndices(), xAxisMaterial, xAxisTextures, currentImageCount);
    appOwnedMeshes.emplace_back(std::move(xAxisMesh));
    scene.createNode(
        {.mesh = appOwnedMeshes.back().get(), .pipeline = linePipeline, .name = "X_Axis_Node"});

    // Y-Axis (Green)
    Material yAxisMaterial;
    PBRTextures yAxisTextures;
    yAxisTextures.baseColor = textureStore.getColorTexture("green", {0, 255, 0, 255});
    auto yAxisMesh = std::make_unique<Mesh>(
        device, "Y_Axis",
        createAxisLineVertices({0, -axisLength, 0}, {0, axisLength, 0}, {1, 0, 0}),
        createAxisLineIndices(), yAxisMaterial, yAxisTextures, currentImageCount);
    appOwnedMeshes.emplace_back(std::move(yAxisMesh));
    scene.createNode(
        {.mesh = appOwnedMeshes.back().get(), .pipeline = linePipeline, .name = "Y_Axis_Node"});

    // Z-Axis (Blue)
    Material zAxisMaterial;
    PBRTextures zAxisTextures;
    zAxisTextures.baseColor = textureStore.getColorTexture("blue", {0, 0, 255, 255});
    auto zAxisMesh = std::make_unique<Mesh>(
        device, "Z_Axis",
        createAxisLineVertices({0, 0, -axisLength}, {0, 0, axisLength}, {1, 0, 0}),
        createAxisLineIndices(), zAxisMaterial, zAxisTextures, currentImageCount);
    appOwnedMeshes.emplace_back(std::move(zAxisMesh));
    scene.createNode(
        {.mesh = appOwnedMeshes.back().get(), .pipeline = linePipeline, .name = "Z_Axis_Node"});
  }

  void loadAndInstanceGltfModel(const std::string &filePath, const std::string &baseDir,
                                u32 currentImageCount) {
    if (graphicsPipelines.empty() || !*graphicsPipelines[0].pipeline) {
      std::println("Error: Prerequisites not met for loading GLTF model '{}'.", filePath);
      return;
    }
    std::println("Attempting to load GLTF model: {}", filePath);

    auto loadedGltfDataResult = loadGltfFile(filePath, baseDir); // From ModelLoader.cppm
    if (!loadedGltfDataResult) {
      std::println("Failed to load GLTF file '{}': {}", filePath, loadedGltfDataResult.error());
      return;
    }

    const LoadedGltfScene &gltfData = *loadedGltfDataResult;
    if (gltfData.meshes.empty() && gltfData.nodes.empty()) {
      std::println("GLTF file '{}' loaded but contains no meshes or nodes.", filePath);
      return;
    }

    if (gltfData.images.empty()) {
      std::println("GLTF file '{}' loaded but contains no textures.", filePath);
      return;
    }

    // Populate the main scene with data from GLTF
    // This uses SceneBuilder.cppm
    auto builtMeshesResult =
        populateSceneFromGltf(scene, gltfData, device, textureStore,
                              &graphicsPipelines[0], // Use the main mesh pipeline for GLTF models
                              currentImageCount);

    if (!builtMeshesResult) {
      std::println("Failed to build engine scene from GLTF data for '{}': {}", filePath,
                   builtMeshesResult.error());
      return;
    }

    // Take ownership of the meshes created by SceneBuilder
    for (auto &mesh_ptr : builtMeshesResult->engineMeshes) {
      appOwnedMeshes.emplace_back(std::move(mesh_ptr));
    }

    std::println("Successfully processed and instanced GLTF model: {}", filePath);
    // After this, the scene graph (this->scene) will contain nodes from the GLTF.
    // Descriptor sets for these new meshes will need to be allocated and updated.
    // This should ideally happen as part of a broader "scene finalized" step or before first
    // render.
  }

  void SetupVulkan() {

    EXPECTED_VOID(instance.create());
    if (!NDEBUG) {
      EXPECTED_VOID(instance.setupDebugMessenger());
    }
    EXPECTED_VOID(device.pickPhysicalDevice());
    EXPECTED_VOID(device.createLogicalDevice());
    EXPECTED_VOID(createCombinedMeshDescriptorSetLayout());
    auto cacheResult = device.logical().createPipelineCache({});
    if (cacheResult) {
      pipelineCache = std::move(cacheResult.value());
    }
    EXPECTED_VOID(textureStore.createInternalCommandPool());
  }

  void SetupVulkanWindow(SDL_Window *sdl_window, vk::Extent2D extent) {

    VkSurfaceKHR surface_raw_handle;

    if (SDL_Vulkan_CreateSurface(sdl_window, instance.get_C_handle(), nullptr,
                                 &surface_raw_handle) == 0) {
      std::string errorMsg =
          "Failed to create Vulkan surface via SDL: " + std::string(SDL_GetError());

      std::println("{}", errorMsg);
      std::exit(EXIT_FAILURE);
    }
    wd.Surface = vk::raii::SurfaceKHR(instance, surface_raw_handle);

    std::vector<vk::Format> requestSurfaceImageFormat = {
        vk::Format::eB8G8R8A8Srgb, vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Unorm,
        vk::Format::eR8G8B8A8Unorm};

    wd.config.SurfaceFormat =
        selectSurfaceFormat(device.physical(), wd.Surface, requestSurfaceImageFormat,
                            vk::ColorSpaceKHR::eSrgbNonlinear);

#ifdef APP_USE_UNLIMITED_FRAME_RATE
    std::vector<vk::PresentModeKHR> present_modes = {vk::PresentModeKHR::eMailbox,
                                                     vk::PresentModeKHR::eFifo};
#else
    std::vector<vk::PresentModeKHR> present_modes = {vk::PresentModeKHR::eFifo};
#endif

    wd.config.PresentMode = selectPresentMode(device.physical(), wd.Surface, present_modes);
    wd.config.ClearEnable = true;
    wd.config.ClearValue.color = vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 1.0f});

    createOrResizeWindow(instance, device, wd, extent, MIN_IMAGE_COUNT);
  }

  void FrameRender(ImDrawData *draw_data, float deltaTime) {

    if (!*wd.Swapchain) {

      return;
    }

    auto &image_acquired_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    auto &render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;

    constexpr uint64_t timeout = std::numeric_limits<uint64_t>::max();

    auto [acquireRes, imageIndexx] = device.logical().acquireNextImage2KHR({
        .swapchain = wd.Swapchain,
        .timeout = timeout,
        .semaphore = image_acquired_semaphore,
        .deviceMask = 1,
    });
    uint32_t imageIndex = imageIndexx;

    if (acquireRes == vk::Result::eErrorOutOfDateKHR || acquireRes == vk::Result::eSuboptimalKHR) {
      swapChainRebuild = true;
      if (acquireRes == vk::Result::eErrorOutOfDateKHR) {

        return;
      }
    } else if (acquireRes != vk::Result::eSuccess) {

      std::println("Error acquiring swapchain image: {}", vk::to_string(acquireRes));
      return;
    }
    wd.FrameIndex = imageIndex;

    Frame &currentFrame = wd.Frames[wd.FrameIndex];

    check_vk_result_hpp(device.logical().waitForFences(*currentFrame.Fence, VK_TRUE, UINT64_MAX));
    device.logical().resetFences({*currentFrame.Fence});

    currentFrame.CommandPool.reset();
    currentFrame.CommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    scene.updateHierarchy(wd.FrameIndex, camera.GetViewMatrix(),
                          camera.GetProjectionMatrix((float)wd.config.swapchainExtent.width /
                                                     (float)wd.config.swapchainExtent.height),
                          deltaTime); // Use actual deltaTime

    scene.updateAllDescriptorSetContents(wd.FrameIndex);

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].color = wd.config.ClearValue.color;
    clearValues[1].depthStencil = {1.0f, 0};

    currentFrame.CommandBuffer.beginRenderPass(
        vk::RenderPassBeginInfo{
            .renderPass = *wd.RenderPass,
            .framebuffer = *currentFrame.Framebuffer,
            .renderArea = {.offset = {0, 0}, .extent = wd.config.swapchainExtent},
            .clearValueCount = static_cast<uint32_t>(clearValues.size()),
            .pClearValues = clearValues.data()},
        vk::SubpassContents::eInline);

    currentFrame.CommandBuffer.setViewport(
        0, vk::Viewport{0.0f, 0.0f, (float)wd.config.swapchainExtent.width,
                        (float)wd.config.swapchainExtent.height, 0.0f, 1.0f});
    currentFrame.CommandBuffer.setScissor(0, vk::Rect2D{{0, 0}, wd.config.swapchainExtent});

    scene.draw(currentFrame.CommandBuffer, wd.FrameIndex);

    ImGui_ImplVulkan_RenderDrawData(draw_data, *currentFrame.CommandBuffer);

    currentFrame.CommandBuffer.endRenderPass();
    currentFrame.CommandBuffer.end();

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submitInfo{.waitSemaphoreCount = 1,
                              .pWaitSemaphores = &*image_acquired_semaphore,
                              .pWaitDstStageMask = &waitStage,
                              .commandBufferCount = 1,
                              .pCommandBuffers = &*currentFrame.CommandBuffer,
                              .signalSemaphoreCount = 1,
                              .pSignalSemaphores = &*render_complete_semaphore};
    device.queue_.submit({submitInfo}, *currentFrame.Fence);
  }

  void FramePresent() {

    if (swapChainRebuild || !*wd.Swapchain) {
      return;
    }

    auto &render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;
    vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1,
                                   .pWaitSemaphores = &*render_complete_semaphore,
                                   .swapchainCount = 1,
                                   .pSwapchains = &*wd.Swapchain,
                                   .pImageIndices = &wd.FrameIndex};

    vk::Result presentResult = device.queue_.presentKHR(presentInfo);
    if (presentResult == vk::Result::eErrorOutOfDateKHR ||
        presentResult == vk::Result::eSuboptimalKHR) {
      swapChainRebuild = true;
    } else if (presentResult != vk::Result::eSuccess) {

      std::println("Error presenting swapchain image: {}", vk::to_string(presentResult));
    } else {
    }
    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.FrameSemaphores.size();
  }

  vk::Extent2D get_window_size(SDL_Window *window) {
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    return {static_cast<uint32_t>(w > 0 ? w : 1), static_cast<uint32_t>(h > 0 ? h : 1)};
  }

  void ProcessKeyboard(Camera &cam, SDL_Scancode key, float dt) {
    float velocity = cam.MovementSpeed * dt;
    if (key == SDL_SCANCODE_W)
      cam.Position += cam.Front * velocity;
    if (key == SDL_SCANCODE_S)
      cam.Position -= cam.Front * velocity;
    if (key == SDL_SCANCODE_A)
      cam.Position -= cam.Right * velocity;
    if (key == SDL_SCANCODE_D)
      cam.Position += cam.Right * velocity;
    if (key == SDL_SCANCODE_SPACE)
      cam.Position += cam.WorldUp * velocity;
    if (key == SDL_SCANCODE_LCTRL)
      cam.Position -= cam.WorldUp * velocity;
  }

  void updateCamera(float dt) {
    const auto keystate = SDL_GetKeyboardState(nullptr);
    if (keystate[SDL_SCANCODE_W])
      ProcessKeyboard(camera, SDL_SCANCODE_W, dt);
    if (keystate[SDL_SCANCODE_S])
      ProcessKeyboard(camera, SDL_SCANCODE_S, dt);
    if (keystate[SDL_SCANCODE_A])
      ProcessKeyboard(camera, SDL_SCANCODE_A, dt);
    if (keystate[SDL_SCANCODE_D])
      ProcessKeyboard(camera, SDL_SCANCODE_D, dt);
    if (keystate[SDL_SCANCODE_SPACE])
      ProcessKeyboard(camera, SDL_SCANCODE_SPACE, dt);
    if (keystate[SDL_SCANCODE_LCTRL])
      ProcessKeyboard(camera, SDL_SCANCODE_LCTRL, dt);
  }

  void mainLoop(SDL_Window *sdl_window) {
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    using Clock = std::chrono::high_resolution_clock;
    auto previousTime = Clock::now();
    float deltaTime = 0.0f;

    bool done = false;

    while (!done) {
      auto currentTime = Clock::now();
      deltaTime = std::chrono::duration<float>(currentTime - previousTime).count();
      previousTime = currentTime;

      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);
        if (event.type == SDL_EVENT_QUIT) {

          done = true;
        }
        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
            event.window.windowID == SDL_GetWindowID(sdl_window)) {

          done = true;
        }

        if (event.type == SDL_EVENT_WINDOW_MINIMIZED) {
        }
        if (event.type == SDL_EVENT_WINDOW_RESTORED) {

          swapChainRebuild = true;
        }
        if (event.type == SDL_EVENT_WINDOW_RESIZED ||
            event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
          swapChainRebuild = true;
        }
      }

      updateCamera(deltaTime);
      camera.updateVectors();

      if (SDL_GetWindowFlags(sdl_window) & SDL_WINDOW_MINIMIZED) {

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      vk::Extent2D currentExtent = get_window_size(sdl_window);
      if (currentExtent.width == 0 || currentExtent.height == 0) {

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      if (swapChainRebuild || wd.config.swapchainExtent.width != currentExtent.width ||
          wd.config.swapchainExtent.height != currentExtent.height) {

        device.logical().waitIdle();

        createOrResizeWindow(instance, device, wd, currentExtent, MIN_IMAGE_COUNT);

        scene.setImageCount(static_cast<u32>(wd.Frames.size()));

        scene.allocateAllDescriptorSets(device.descriptorPool_, combinedMeshLayout);

        for (u32 i = 0; i < wd.Frames.size(); ++i)
          scene.updateAllDescriptorSetContents(i);

        swapChainRebuild = false;
      }

      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplSDL3_NewFrame();
      ImGui::NewFrame();

      {
        RenderCameraControlMenu(camera);
        RenderSceneHierarchyMaterialEditor(scene, wd.FrameIndex);
      }

      ImGui::Render();
      FrameRender(ImGui::GetDrawData(), deltaTime);
      FramePresent();
      // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

public:
  int run(SDL_Window *sdl_window) {

    SetupVulkan();

    SetupVulkanWindow(sdl_window, get_window_size(sdl_window));

    u32 numMeshesEstimate = 30;

    EXPECTED_VOID(
        device.createDescriptorPool(static_cast<u32>(wd.Frames.size()) * numMeshesEstimate));

    createPipelines();

    scene = Scene(static_cast<u32>(wd.Frames.size()));
    // loadAndInstanceGltfModel("../assets/models/BoxVertexColors.gltf", "",
    //                          static_cast<u32>(wd.Frames.size()));
    loadAndInstanceGltfModel("../assets/models/sphinx-3d-model/scene.gltf",
                             "../assets/models/sphinx-3d-model/",
                             static_cast<u32>(wd.Frames.size()));
    // createTexturedCubeScene(static_cast<u32>(wd.Frames.size()));
    createDebugAxesScene(static_cast<u32>(wd.Frames.size()));

    scene.allocateAllDescriptorSets(device.descriptorPool_, combinedMeshLayout);

    for (u32 i = 0; i < wd.Frames.size(); ++i)
      scene.updateAllDescriptorSetContents(i);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForVulkan(sdl_window);

    ImGui_ImplVulkan_InitInfo init_info = device.init_info();
    init_info.Instance = instance.get_C_handle();
    init_info.PhysicalDevice = *device.physical();
    init_info.Device = *device.logical();
    init_info.Queue = *device.queue_;
    init_info.DescriptorPool = *device.descriptorPool_;
    init_info.RenderPass = *wd.RenderPass;
    init_info.MinImageCount = MIN_IMAGE_COUNT;
    init_info.ImageCount = static_cast<uint32_t>(wd.Frames.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.PipelineCache = *pipelineCache;
    init_info.Subpass = 0;
    init_info.CheckVkResultFn = check_vk_result; // Global one

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    mainLoop(sdl_window);

    device.logical().waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    // reliably on crash
    return 0;
  }
};

export struct SDL_Wrapper {
  SDL_Window *window{nullptr};

  int init() {
    if constexpr (VK_USE_PLATFORM_WAYLAND_KHR) {
      SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
      std::cerr << "[SDL_Wrapper] Error: SDL_Init(): " << SDL_GetError() << std::endl;

      return -1;
    }
    SDL_WindowFlags window_flags =
        (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                          SDL_WINDOW_HIGH_PIXEL_DENSITY); // Start hidden, show after Vulkan setup
    window = SDL_CreateWindow("Dear ImGui SDL3+Vulkan example", 1280, 720, window_flags);
    if (window == nullptr) {
      std::cerr << "[SDL_Wrapper] Error: SDL_CreateWindow(): " << SDL_GetError() << std::endl;

      return -1;
    }
    return 0;
  }

  void terminate() {
    if (window) {
      SDL_DestroyWindow(window);
      window = nullptr;
    }
    SDL_PumpEvents(); // Process any pending events
    SDL_Quit();
  }
};
