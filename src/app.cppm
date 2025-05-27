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

#define LOG_SCOPE Logger::LogScope log_scope_object(__func__, __FILE__, __LINE__)
#define LOG_MSG(message) Logger::Log(message, "DEBUG")
#define LOG_INFO(message) Logger::Log(message, "INFO")
#define LOG_WARN(message) Logger::Log(message, "WARN")
#define LOG_ERROR(message) Logger::Log(message, "ERROR")

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
  TextureStore textureStore{device, device.queue};
  std::vector<std::unique_ptr<Mesh>> appOwnedMeshes; // App owns all mesh objects

  Camera camera;

public:
  // Create the single combined descriptor set layout for meshes
  std::expected<void, std::string> createCombinedMeshDescriptorSetLayout() NOEXCEPT {
    LOG_SCOPE;
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

    LOG_MSG("Attempting to create combined mesh descriptor set layout.");
    auto layoutResult = device.logical().createDescriptorSetLayout(layoutInfo);
    if (!layoutResult) {
      std::string errorMsg = "Failed to create combined mesh descriptor set layout: " +
                             vk::to_string(layoutResult.error());
      LOG_ERROR(errorMsg);
      return std::unexpected(errorMsg);
    }
    combinedMeshLayout = std::move(layoutResult.value());
    LOG_INFO("Successfully created combined mesh descriptor set layout.");
    return {};
  }

  void createPipelines() { // Simplified pipeline creation
    LOG_SCOPE;
    if (graphicsPipelines.empty()) {
      LOG_MSG("Resizing graphicsPipelines to 1.");
      graphicsPipelines.resize(2); // For now, one main pipeline
    }
    VulkanPipeline &mainPipeline = graphicsPipelines[0];

    std::vector<vk::DescriptorSetLayout> layouts = {*combinedMeshLayout};
    LOG_MSG("Attempting to create pipeline layout.");
    EXPECTED_VOID(mainPipeline.createPipelineLayout(device.logical(), layouts));

    LOG_MSG("Attempting to load shader modules.");
    auto vertShaderModule = createShaderModuleFromFile(device.logical(), "shaders/vert.spv");
    auto fragShaderModule = createShaderModuleFromFile(device.logical(), "shaders/frag.spv");

    if (!vertShaderModule) {
      LOG_ERROR("Failed to load vertex shader module: " +
                vertShaderModule.error_or("Unknown error"));
    } else {
      LOG_INFO("Vertex shader module loaded successfully.");
    }
    if (!fragShaderModule) {
      LOG_ERROR("Failed to load fragment shader module: " +
                fragShaderModule.error_or("Unknown error"));
    } else {
      LOG_INFO("Fragment shader module loaded successfully.");
    }

    if (!vertShaderModule || !fragShaderModule) {
      std::println("Error loading shaders: {} & {}", vertShaderModule.error_or(""),
                   fragShaderModule.error_or("")); // Keep console print
      return;
    }
    LOG_INFO("Shaders loaded successfully.");

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
        {.stage = vk::ShaderStageFlagBits::eVertex,
         .module = *vertShaderModule.value(),
         .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment,
         .module = *fragShaderModule.value(),
         .pName = "main"}};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = false};

    LOG_MSG("Attempting to create graphics pipeline.");
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
    LOG_SCOPE;
    LOG_MSG("currentImageCount: " + std::to_string(currentImageCount));
    // if (textureStore) {
    //   LOG_ERROR("TextureStore not initialized in createTexturedCubeScene.");
    //   std::println("Error: TextureStore not initialized in createTexturedCubeScene.");
    //   return;
    // }
    if (graphicsPipelines.empty() || !*graphicsPipelines[0].pipeline) {
      LOG_ERROR("Default graphics pipeline not ready for createTexturedCubeScene.");
      std::println("Error: Default graphics pipeline not ready for createTexturedCubeScene.");
      return;
    }
    VulkanPipeline *defaultPipeline = &graphicsPipelines[0];
    LOG_INFO("Default pipeline obtained.");

    LOG_MSG("Creating cube root node.");
    SceneNode *cubeRootNode = scene.createNode(
        {.transform = Transform{}, .pipeline = defaultPipeline, .name = "CubeRoot"});
    if (!cubeRootNode) {
      LOG_ERROR("Failed to create cube root node.");
      return;
    }
    LOG_INFO("Cube root node created: " + cubeRootNode->name);

    float cubeSize = 10.0f;
    LOG_MSG("Cube size: " + std::to_string(cubeSize));

    std::array<std::shared_ptr<Texture>, 6> faceTextures = {
        textureStore.getColorTexture2({255, 0, 0, 255}),
        textureStore.getColorTexture("green", {0, 255, 0, 255}),
        textureStore.getColorTexture("blue", {0, 0, 255, 255}),
        textureStore.getColorTexture("yellow", {255, 255, 0, 255}),
        textureStore.getColorTexture("cyan", {0, 255, 255, 255}),
        textureStore.getColorTexture("magenta", {255, 0, 255, 255})};
    LOG_INFO("Face textures obtained from texture store.");

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
        {"BottomFace",
         {0, -1, 0},
         {0, 0, 1},
         {1, 0, 0},
         {0, -cubeSize / 2.0f, 0},
         faceTextures[5]}};
    LOG_INFO("Face definitions created.");

    for (const auto &def : faceDefs) {
      LOG_MSG("Processing face: " + def.name);
      std::vector<Vertex> faceVertices =
          createQuadVertices(cubeSize, def.normal, def.up, def.right);
      std::vector<uint32_t> faceIndices = createQuadIndices();
      LOG_MSG("Vertices and indices created for face: " + def.name);

      Material faceMaterial;
      faceMaterial.baseColorFactor = glm::vec4(1.0f);

      PBRTextures facePbrTextures;
      facePbrTextures.baseColor =
          def.texture ? def.texture : textureStore.getFallbackDefaultTexture();
      LOG_MSG("Material and PBR textures set for face: " + def.name);

      auto faceMesh =
          std::make_unique<Mesh>(device, def.name, std::move(faceVertices), std::move(faceIndices),
                                 faceMaterial, facePbrTextures, currentImageCount);
      LOG_INFO("Mesh created for face: " + def.name);
      Mesh *faceMeshPtr = faceMesh.get();
      appOwnedMeshes.emplace_back(std::move(faceMesh));
      LOG_MSG("Mesh added to appOwnedMeshes for face: " + def.name);

      Transform faceTransform;
      faceTransform.translation = def.translation;

      SceneNode *faceNode = scene.createNode({.mesh = faceMeshPtr,
                                              .transform = faceTransform,
                                              .pipeline = defaultPipeline,
                                              .parent = cubeRootNode,
                                              .name = def.name + "_Node"});
      if (!faceNode) {
        LOG_ERROR("Failed to create scene node for face: " + def.name);
      } else {
        LOG_INFO("Scene node created for face: " + faceNode->name);
      }
    }
    if (cubeRootNode) {
      cubeRootNode->transform.rotation_speed_euler_dps = {10.f, 15.f, 5.f};
      LOG_MSG("Set rotation speed for cube root node.");
    }
    LOG_INFO("Finished creating textured cube scene.");
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

  // --- added: placeholder for gltf loading and scene population ---
  void loadAndInstanceGltfModel(const std::string &filePath, u32 currentImageCount) {
    if (graphicsPipelines.empty() || !*graphicsPipelines[0].pipeline) {
      std::println("Error: Prerequisites not met for loading GLTF model '{}'.", filePath);
      return;
    }
    std::println("Attempting to load GLTF model: {}", filePath);

    auto loadedGltfDataResult = loadGltfFile(filePath); // From ModelLoader.cppm
    if (!loadedGltfDataResult) {
      std::println("Failed to load GLTF file '{}': {}", filePath, loadedGltfDataResult.error());
      return;
    }

    const LoadedGltfScene &gltfData = *loadedGltfDataResult;
    if (gltfData.meshes.empty() && gltfData.nodes.empty()) {
      std::println("GLTF file '{}' loaded but contains no meshes or nodes.", filePath);
      return;
    }

    // Populate the main scene with data from GLTF
    // This uses SceneBuilder.cppm
    auto builtMeshesResult = populateSceneFromGltf(
        this->scene, gltfData, this->device, textureStore,
        &this->graphicsPipelines[0], // Use the main mesh pipeline for GLTF models
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
    LOG_SCOPE;
    EXPECTED_VOID(instance.create());
    if (!NDEBUG) {
      EXPECTED_VOID(instance.setupDebugMessenger());
    }
    EXPECTED_VOID(device.pickPhysicalDevice());
    EXPECTED_VOID(device.createLogicalDevice());
    EXPECTED_VOID(createCombinedMeshDescriptorSetLayout());
    // auto cacheResult = device.logical().createPipelineCache({});
    // if (cacheResult) {
    //   pipelineCache = std::move(cacheResult.value());
    // }
    EXPECTED_VOID(textureStore.createInternalCommandPool());
  }

  void SetupVulkanWindow(SDL_Window *sdl_window, vk::Extent2D extent) {
    LOG_SCOPE;
    LOG_MSG("Window extent: " + std::to_string(extent.width) + "x" + std::to_string(extent.height));
    VkSurfaceKHR surface_raw_handle;
    LOG_MSG("Attempting to create Vulkan surface via SDL.");
    if (SDL_Vulkan_CreateSurface(sdl_window, instance.get_C_handle(), nullptr,
                                 &surface_raw_handle) == 0) {
      std::string errorMsg =
          "Failed to create Vulkan surface via SDL: " + std::string(SDL_GetError());
      LOG_ERROR(errorMsg);
      std::println("{}", errorMsg);
      std::exit(EXIT_FAILURE);
    }
    wd.Surface = vk::raii::SurfaceKHR(instance, surface_raw_handle);
    LOG_INFO("Vulkan surface created successfully.");

    std::vector<vk::Format> requestSurfaceImageFormat = {
        vk::Format::eB8G8R8A8Srgb, vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Unorm,
        vk::Format::eR8G8B8A8Unorm};
    LOG_MSG("Selecting surface format.");
    wd.config.SurfaceFormat =
        selectSurfaceFormat(device.physical(), wd.Surface, requestSurfaceImageFormat,
                            vk::ColorSpaceKHR::eSrgbNonlinear);
    LOG_INFO("Surface format selected: " + vk::to_string(wd.config.SurfaceFormat.format));

#ifdef APP_USE_UNLIMITED_FRAME_RATE
    std::vector<vk::PresentModeKHR> present_modes = {vk::PresentModeKHR::eMailbox,
                                                     vk::PresentModeKHR::eFifo};
    LOG_MSG("Requesting Mailbox or Fifo present mode.");
#else
    std::vector<vk::PresentModeKHR> present_modes = {vk::PresentModeKHR::eFifo};
    LOG_MSG("Requesting Fifo present mode.");
#endif
    wd.config.PresentMode = selectPresentMode(device.physical(), wd.Surface, present_modes);
    LOG_INFO("Present mode selected: " + vk::to_string(wd.config.PresentMode));

    wd.config.ClearEnable = true;
    wd.config.ClearValue.color = vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 1.0f});

    LOG_MSG("Attempting to create or resize window (swapchain, renderpass, etc.).");
    createOrResizeWindow(instance, device, wd, extent, MIN_IMAGE_COUNT);
    LOG_INFO("Window created/resized successfully.");
    LOG_INFO("SetupVulkanWindow completed.");
  }

  void FrameRender(ImDrawData *draw_data, float deltaTime) {
    LOG_SCOPE; // This is too frequent, log specific parts
    if (!*wd.Swapchain) {
      LOG_WARN("FrameRender: Swapchain not ready, skipping render.");
      return;
    }

    auto &image_acquired_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    auto &render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;

    constexpr uint64_t timeout = std::numeric_limits<uint64_t>::max();
    LOG_MSG("Acquiring next image. SemaphoreIndex: " + std::to_string(wd.SemaphoreIndex));
    auto [acquireRes, imageIndexx] = device.logical().acquireNextImage2KHR({
        .swapchain = wd.Swapchain,
        .timeout = timeout,
        .semaphore = image_acquired_semaphore,
        .deviceMask = 1,
    });
    uint32_t imageIndex = imageIndexx;

    if (acquireRes == vk::Result::eErrorOutOfDateKHR || acquireRes == vk::Result::eSuboptimalKHR) {
      LOG_WARN("AcquireNextImage: Swapchain out of date or suboptimal. Setting swapChainRebuild = "
               "true. Result: " +
               vk::to_string(acquireRes));
      swapChainRebuild = true;
      if (acquireRes == vk::Result::eErrorOutOfDateKHR) {
        LOG_MSG("AcquireNextImage: OutOfDateKHR, returning early from FrameRender.");
        return;
      }
    } else if (acquireRes != vk::Result::eSuccess) {
      LOG_ERROR("Error acquiring swapchain image: " + vk::to_string(acquireRes));
      std::println("Error acquiring swapchain image: {}", vk::to_string(acquireRes));
      return;
    }
    wd.FrameIndex = imageIndex;
    LOG_MSG("Image acquired successfully. FrameIndex: " + std::to_string(wd.FrameIndex));

    Frame &currentFrame = wd.Frames[wd.FrameIndex];

    LOG_MSG("Waiting for fence for frame " + std::to_string(wd.FrameIndex));
    check_vk_result_hpp(device.logical().waitForFences(*currentFrame.Fence, VK_TRUE, UINT64_MAX));
    device.logical().resetFences({*currentFrame.Fence});
    LOG_MSG("Fence reset for frame " + std::to_string(wd.FrameIndex));

    LOG_MSG("Resetting command pool and beginning command buffer for frame " +
            std::to_string(wd.FrameIndex));
    currentFrame.CommandPool.reset();
    currentFrame.CommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    LOG_MSG("Updating scene hierarchy for frame " + std::to_string(wd.FrameIndex));
    scene.updateHierarchy(wd.FrameIndex, camera.GetViewMatrix(),
                          camera.GetProjectionMatrix((float)wd.config.swapchainExtent.width /
                                                     (float)wd.config.swapchainExtent.height),
                          deltaTime); // Use actual deltaTime

    LOG_MSG("Updating all descriptor set contents for frame " + std::to_string(wd.FrameIndex));
    scene.updateAllDescriptorSetContents(wd.FrameIndex);

    LOG_MSG("Beginning render pass for frame " + std::to_string(wd.FrameIndex));
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

    LOG_MSG("Drawing scene for frame " + std::to_string(wd.FrameIndex));
    scene.draw(currentFrame.CommandBuffer, wd.FrameIndex);

    LOG_MSG("Rendering ImGui for frame " + std::to_string(wd.FrameIndex));
    ImGui_ImplVulkan_RenderDrawData(draw_data, *currentFrame.CommandBuffer);

    currentFrame.CommandBuffer.endRenderPass();
    currentFrame.CommandBuffer.end();
    LOG_MSG("Render pass and command buffer ended for frame " + std::to_string(wd.FrameIndex));

    LOG_MSG("Submitting command buffer for frame " + std::to_string(wd.FrameIndex));
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submitInfo{.waitSemaphoreCount = 1,
                              .pWaitSemaphores = &*image_acquired_semaphore,
                              .pWaitDstStageMask = &waitStage,
                              .commandBufferCount = 1,
                              .pCommandBuffers = &*currentFrame.CommandBuffer,
                              .signalSemaphoreCount = 1,
                              .pSignalSemaphores = &*render_complete_semaphore};
    device.queue.submit({submitInfo}, *currentFrame.Fence);
    LOG_MSG("Command buffer submitted for frame " + std::to_string(wd.FrameIndex));
  }

  void FramePresent() {
    LOG_SCOPE; // Too frequent
    if (swapChainRebuild || !*wd.Swapchain) {
      LOG_WARN("FramePresent: Swapchain rebuild pending or swapchain not ready. Skipping present. "
               "swapChainRebuild=" +
               std::to_string(swapChainRebuild));
      return;
    }

    auto &render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;
    vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1,
                                   .pWaitSemaphores = &*render_complete_semaphore,
                                   .swapchainCount = 1,
                                   .pSwapchains = &*wd.Swapchain,
                                   .pImageIndices = &wd.FrameIndex};

    LOG_MSG("Presenting image. SemaphoreIndex: " + std::to_string(wd.SemaphoreIndex) +
            ", FrameIndex: " + std::to_string(wd.FrameIndex));
    vk::Result presentResult = device.queue.presentKHR(presentInfo);
    if (presentResult == vk::Result::eErrorOutOfDateKHR ||
        presentResult == vk::Result::eSuboptimalKHR) {
      LOG_WARN("PresentKHR: Swapchain out of date or suboptimal. Setting swapChainRebuild = true. "
               "Result: " +
               vk::to_string(presentResult));
      swapChainRebuild = true;
    } else if (presentResult != vk::Result::eSuccess) {
      LOG_ERROR("Error presenting swapchain image: " + vk::to_string(presentResult));
      std::println("Error presenting swapchain image: {}", vk::to_string(presentResult));
    } else {
      LOG_MSG("Image presented successfully.");
    }
    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.FrameSemaphores.size();
  }

  vk::Extent2D get_window_size(SDL_Window *window) {
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    return {static_cast<uint32_t>(w > 0 ? w : 1), static_cast<uint32_t>(h > 0 ? h : 1)};
  }

  void ProcessKeyboard(Camera &cam, SDL_Scancode key, float dt) {
    LOG_SCOPE; // Too frequent
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
    LOG_SCOPE; // Too frequent
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
    LOG_SCOPE;
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    LOG_INFO("ImGui NavEnableKeyboard set.");

    using Clock = std::chrono::high_resolution_clock;
    auto previousTime = Clock::now();
    float deltaTime = 0.0f;

    bool done = false;
    // bool firstMouse = true; // Not used currently
    // float lastMouseX = static_cast<float>(wd.config.swapchainExtent.width / 2.0f); // Not used
    // float lastMouseY = static_cast<float>(wd.config.swapchainExtent.height / 2.0f); // Not used
    LOG_INFO("Main loop starting.");

    while (!done) {
      auto currentTime = Clock::now();
      deltaTime = std::chrono::duration<float>(currentTime - previousTime).count();
      previousTime = currentTime;

      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);
        if (event.type == SDL_EVENT_QUIT) {
          LOG_INFO("SDL_EVENT_QUIT received.");
          done = true;
        }
        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
            event.window.windowID == SDL_GetWindowID(sdl_window)) {
          LOG_INFO("SDL_EVENT_WINDOW_CLOSE_REQUESTED received.");
          done = true;
        }

        if (event.type == SDL_EVENT_WINDOW_MINIMIZED) {
          LOG_INFO("SDL_EVENT_WINDOW_MINIMIZED received.");
        }
        if (event.type == SDL_EVENT_WINDOW_RESTORED) {
          LOG_INFO("SDL_EVENT_WINDOW_RESTORED received, setting swapChainRebuild = true.");
          swapChainRebuild = true;
        }
        if (event.type == SDL_EVENT_WINDOW_RESIZED ||
            event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
          LOG_INFO("SDL_EVENT_WINDOW_RESIZED or SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED received, "
                   "setting swapChainRebuild = true.");
          swapChainRebuild = true;
        }
      }

      updateCamera(deltaTime);
      camera.updateVectors();

      if (SDL_GetWindowFlags(sdl_window) & SDL_WINDOW_MINIMIZED) {
        LOG_MSG("Window minimized, sleeping."); // Can be spammy
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      vk::Extent2D currentExtent = get_window_size(sdl_window);
      if (currentExtent.width == 0 || currentExtent.height == 0) {
        LOG_WARN("Window extent is zero, sleeping."); // Can be spammy
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      if (swapChainRebuild || wd.config.swapchainExtent.width != currentExtent.width ||
          wd.config.swapchainExtent.height != currentExtent.height) {
        LOG_INFO("Rebuilding swapchain. Current extent: " + std::to_string(currentExtent.width) +
                 "x" + std::to_string(currentExtent.height) +
                 ", Old extent: " + std::to_string(wd.config.swapchainExtent.width) + "x" +
                 std::to_string(wd.config.swapchainExtent.height) +
                 ", swapChainRebuild flag: " + std::to_string(swapChainRebuild));
        LOG_MSG("Waiting for logical device idle before rebuilding swapchain.");
        device.logical().waitIdle();
        LOG_MSG("Logical device idle. Proceeding with createOrResizeWindow.");
        createOrResizeWindow(instance, device, wd, currentExtent, MIN_IMAGE_COUNT);
        LOG_INFO("Swapchain/window resources recreated.");

        LOG_MSG("Updating scene image count to: " + std::to_string(wd.Frames.size()));
        scene.setImageCount(static_cast<u32>(wd.Frames.size()));
        LOG_MSG("Allocating all descriptor sets for the scene.");
        scene.allocateAllDescriptorSets(device.descriptorPool, combinedMeshLayout);
        LOG_MSG("Updating all descriptor set contents for the scene.");
        for (u32 i = 0; i < wd.Frames.size(); ++i)
          scene.updateAllDescriptorSetContents(i);
        LOG_INFO("Scene descriptor sets reallocated and updated.");

        swapChainRebuild = false;
        LOG_MSG("swapChainRebuild set to false.");
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
    LOG_INFO("Main loop finished.");
  }

public:
  int run(SDL_Window *sdl_window) {
    LOG_SCOPE;
    LOG_INFO("Application run started.");
    SetupVulkan();
    LOG_INFO("Vulkan setup completed.");
    SetupVulkanWindow(sdl_window, get_window_size(sdl_window));
    LOG_INFO("Vulkan window setup completed.");

    u32 numMeshesEstimate = 10;
    LOG_MSG("Creating descriptor pool. Estimated meshes: " + std::to_string(numMeshesEstimate) +
            ", Image count: " + std::to_string(wd.Frames.size()));
    EXPECTED_VOID(
        device.createDescriptorPool(static_cast<u32>(wd.Frames.size()) * numMeshesEstimate));

    createPipelines();
    LOG_INFO("Pipelines created.");

    LOG_MSG("Initializing scene with image count: " + std::to_string(wd.Frames.size()));
    scene = Scene(static_cast<u32>(wd.Frames.size()));
    createTexturedCubeScene(static_cast<u32>(wd.Frames.size()));
    createDebugAxesScene(static_cast<u32>(wd.Frames.size()));

    LOG_INFO("Textured cube scene created.");

    LOG_MSG("Allocating initial descriptor sets for the scene.");
    scene.allocateAllDescriptorSets(device.descriptorPool, combinedMeshLayout);
    LOG_MSG("Updating initial descriptor set contents for the scene.");
    for (u32 i = 0; i < wd.Frames.size(); ++i)
      scene.updateAllDescriptorSetContents(i);
    LOG_INFO("Initial scene descriptor sets allocated and updated.");

    LOG_MSG("Setting up ImGui.");
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForVulkan(sdl_window);

    ImGui_ImplVulkan_InitInfo init_info = device.init_info();
    init_info.Instance = instance.get_C_handle();
    init_info.PhysicalDevice = *device.physical();
    init_info.Device = *device.logical();
    init_info.Queue = *device.queue;
    init_info.DescriptorPool = *device.descriptorPool;
    init_info.RenderPass = *wd.RenderPass;
    init_info.MinImageCount = MIN_IMAGE_COUNT;
    init_info.ImageCount = static_cast<uint32_t>(wd.Frames.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.PipelineCache = *pipelineCache;
    init_info.Subpass = 0;
    init_info.CheckVkResultFn = check_vk_result; // Global one
    LOG_MSG("ImGui Vulkan InitInfo populated.");

    ImGui_ImplVulkan_Init(&init_info);
    LOG_INFO("ImGui Vulkan backend initialized.");
    ImGui_ImplVulkan_CreateFontsTexture();
    LOG_INFO("ImGui fonts texture created.");

    mainLoop(sdl_window);
    LOG_INFO("Main loop exited.");

    LOG_MSG("Cleaning up: waiting for logical device idle.");
    device.logical().waitIdle();
    LOG_INFO("Logical device idle. Shutting down ImGui.");
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    LOG_INFO("ImGui shutdown completed.");

    LOG_INFO("Application run finished. Logger will be shut down by App destructor or explicitly "
             "if needed.");
    Logger::Shutdown(); // Explicitly shutdown here if App destructor might not be called
    // reliably on crash
    return 0;
  }
};

export struct SDL_Wrapper {
  SDL_Window *window{nullptr};

  int init() {
    Logger::Init("vulkan_app_log.txt"); // Initialize logger here
    // Logger might not be initialized yet if SDL_Wrapper is created before App
    // So, using std::cerr for these critical early logs.
    // Logger::Log("SDL_Wrapper::init called", "DEBUG"); // If logger is available
    std::cout << "[SDL_Wrapper] init called" << std::endl;

    if constexpr (VK_USE_PLATFORM_WAYLAND_KHR) {
      std::cout << "[SDL_Wrapper] Setting SDL_HINT_VIDEO_DRIVER to wayland" << std::endl;
      SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
      std::cerr << "[SDL_Wrapper] Error: SDL_Init(): " << SDL_GetError() << std::endl;
      LOG_ERROR(std::string("SDL_Init() failed: ") + SDL_GetError());
      return -1;
    }
    std::cout << "[SDL_Wrapper] SDL_Init(SDL_INIT_VIDEO) successful." << std::endl;

    SDL_WindowFlags window_flags =
        (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                          SDL_WINDOW_HIGH_PIXEL_DENSITY); // Start hidden, show after Vulkan setup
    std::cout << "[SDL_Wrapper] Attempting to create SDL window." << std::endl;
    window = SDL_CreateWindow("Dear ImGui SDL3+Vulkan example", 1280, 720, window_flags);
    if (window == nullptr) {
      std::cerr << "[SDL_Wrapper] Error: SDL_CreateWindow(): " << SDL_GetError() << std::endl;
      LOG_ERROR(std::string("SDL_CreateWindow() failed: ") + SDL_GetError());
      return -1;
    }
    std::cout << "[SDL_Wrapper] SDL_CreateWindow successful." << std::endl;
    LOG_INFO("SDL_Wrapper::init successful.");
    return 0;
  }

  void terminate() {
    LOG_SCOPE; // If logger is available and App object still exists
    std::cout << "[SDL_Wrapper] terminate called" << std::endl;
    if (window) {
      SDL_DestroyWindow(window);
      window = nullptr;
      std::cout << "[SDL_Wrapper] SDL_DestroyWindow called." << std::endl;
    }
    SDL_PumpEvents(); // Process any pending events
    SDL_Quit();
    std::cout << "[SDL_Wrapper] SDL_Quit called." << std::endl;
    LOG_INFO("SDL_Wrapper::terminate finished.");
  }
};

// Global check_vk_result is already defined and logs.
// No need to redefine here.
