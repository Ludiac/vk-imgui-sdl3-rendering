module;

#define GLM_ENABLE_EXPERIMENTAL
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "macros.hpp"
#include "primitive_types.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtx/string_cast.hpp>

export module vulkan_app;
export import :SDLWrapper;

import vulkan_hpp;
import std;
import BS.thread_pool;

import :VulkanWindow;
import :VulkanDevice;
import :VulkanInstance;
import :VulkanPipeline;
import :utils;
import :mesh;
import :scene;
import :texture;
import :TextureStore;
import :ModelLoader;
import :SceneBuilder;
import :imgui;
import :TextSystem;
import :UISystem;
import :ui;
import :TextArea;

namespace { // Anonymous namespace for internal linkage
constexpr u32 MIN_IMAGE_COUNT = 2;

std::vector<Vertex> createQuadVertices(const glm::vec3 &center, float width, float height,
                                       const glm::vec3 &normal) {
  float halfW = width / 2.0f;
  float halfH = height / 2.0f;
  return {
      // Bottom-left
      {center + glm::vec3(-halfW, -halfH, 0.0f), normal, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
      // Bottom-right
      {center + glm::vec3(halfW, -halfH, 0.0f), normal, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
      // Top-left
      {center + glm::vec3(-halfW, halfH, 0.0f), normal, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
      // Top-right
      {center + glm::vec3(halfW, halfH, 0.0f), normal, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}};
}

std::vector<u32> createQuadIndices() {
  // std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
  // convertToCCW(indices);
  // return indices;

  return {0, 1, 2, 1, 3, 2}; // Two triangles
}

std::vector<Vertex> createAxisLineVertices(const glm::vec3 &start, const glm::vec3 &end,
                                           const glm::vec3 &normal_placeholder) {
  // Normals and UVs might not be strictly needed for a simple colored line,
  // but the Vertex struct requires them. Tangents also.
  return {{start, normal_placeholder, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
          {end, normal_placeholder, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}};
}
std::vector<u32> createAxisLineIndices() {
  return {0, 1}; // A single line segment
}

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
} // anonymous namespace

export class App {
  VulkanInstance instance;
  VulkanDevice device{instance}; // Default constructor needs instance

  Window wd; // VulkanWindow related data
  bool swapChainRebuild = false;

  // Rendering Resources
  std::vector<VulkanPipeline> graphicsPipelines;
  VulkanPipeline textPipeline;
  VulkanPipeline uiPipeline;
  vk::raii::PipelineCache pipelineCache{nullptr};
  vk::raii::DescriptorSetLayout combinedMeshLayout{nullptr}; // Single layout for meshes

  vk::raii::DescriptorSetLayout sceneLayout{nullptr};       // Layout for scene-wide data
  std::vector<vk::raii::DescriptorSet> sceneDescriptorSets; // One per frame-in-flight

  std::vector<VmaBuffer> sceneLightsUbos; // MODIFIED: Now a vector for per-frame UBOs
  SceneLightsUBO sceneLightsCpuBuffer;

  std::vector<VmaBuffer> shaderTogglesUbos;
  ShaderTogglesUBO shaderTogglesCpuBuffer;

  // Scene and Assets
  Scene scene{0};
  TextureStore textureStore{device, device.queue_};
  std::vector<std::unique_ptr<Mesh>> appOwnedMeshes; // App owns all mesh objects

  Camera camera;

  std::unique_ptr<TextSystem> textSystem;
  std::unique_ptr<UISystem> uiSystem;
  std::vector<Font *> registeredFonts;

  BS::thread_pool<> thread_pool;
  std::vector<LoadedGltfScene> loadedGltfData;
  std::mutex loadedGltfDataMutex;

  // Added for asynchronous font loading
  std::vector<std::future<std::expected<Font *, std::string>>> fontLoadFutures;
  std::mutex fontFuturesMutex;

  float sdf_weight{0.0};
  i32 fontSizeMultiplier{0};
  i32 textAntiAliasingToggle{1};

public:
  std::expected<void, std::string> createDescriptorSetLayouts() NOEXCEPT {
    std::vector<vk::DescriptorSetLayoutBinding> meshDataBindings = {
        {// Binding 0: MVP Uniform Buffer Object (Vertex Shader)
         .binding = 0,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
        {// Binding 1: Material Uniform Buffer Object (Fragment Shader)
         .binding = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {// Binding 2: Base Color Texture (Fragment Shader)
         .binding = 2,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {// Binding 3: Normal Map (Fragment Shader)
         .binding = 3,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {// Binding 4: Metallic/Roughness Map (Fragment Shader)
         .binding = 4,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {.binding = 5,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {.binding = 6,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount =
                                                     static_cast<u32>(meshDataBindings.size()),
                                                 .pBindings = meshDataBindings.data()};

    auto layoutResult = device.logical().createDescriptorSetLayout(layoutInfo);
    if (!layoutResult) {
      std::string errorMsg = "Failed to create combined mesh descriptor set layout: " +
                             vk::to_string(layoutResult.error());

      return std::unexpected(errorMsg);
    }
    combinedMeshLayout = std::move(layoutResult.value());

    std::vector<vk::DescriptorSetLayoutBinding> sceneDataBindings = {
        {// Binding 0: Scene Lights UBO (Fragment Shader)
         .binding = 0,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {// NEW Binding 1: Shader Toggles UBO (Vertex + Fragment)
         .binding = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
    };
    vk::DescriptorSetLayoutCreateInfo sceneLayoutInfo{
        .bindingCount = static_cast<u32>(sceneDataBindings.size()),
        .pBindings = sceneDataBindings.data()};
    auto sceneLayoutResult = device.logical().createDescriptorSetLayout(sceneLayoutInfo);
    if (!sceneLayoutResult) {
      return std::unexpected("Failed to create scene descriptor set layout: " +
                             vk::to_string(sceneLayoutResult.error()));
    }
    sceneLayout = std::move(sceneLayoutResult.value());

    return {};
  }

  void allocateSceneDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(wd.Frames.size(), *sceneLayout);
    vk::DescriptorSetAllocateInfo allocInfo{.descriptorPool = *device.descriptorPool_,
                                            .descriptorSetCount =
                                                static_cast<u32>(wd.Frames.size()),
                                            .pSetLayouts = layouts.data()};
    sceneDescriptorSets = device.logical().allocateDescriptorSets(allocInfo).value();
  }

  void createGraphicsPipelines() { // Simplified pipeline creation
    if (graphicsPipelines.empty()) {
      graphicsPipelines.resize(2);
    }

    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };

    std::array<vk::VertexInputAttributeDescription, 4> attributes = {{
        {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)},
        {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)},
        {2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv)},
        {3, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent)},
    }};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable = true,
        .depthWriteEnable = true,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = false,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = false, // No blending for opaque objects initially
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    VulkanPipeline &mainPipeline = graphicsPipelines[0];

    std::vector<vk::DescriptorSetLayout> layouts = {*combinedMeshLayout, *sceneLayout};
    std::vector<vk::PushConstantRange> noConstantRanges = {};
    EXPECTED_VOID(mainPipeline.createPipelineLayout(device.logical(), layouts, noConstantRanges));

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
                                                      vertexInputInfo, inputAssembly, wd.RenderPass,
                                                      &colorBlendAttachment, &depthStencilState));

    VulkanPipeline &linePipeline = graphicsPipelines[1];
    EXPECTED_VOID(linePipeline.createPipelineLayout(device.logical(), layouts,
                                                    noConstantRanges)); // Reusing same layout

    vk::PipelineInputAssemblyStateCreateInfo lineInputAssembly{
        .topology = vk::PrimitiveTopology::eLineList, .primitiveRestartEnable = false};
    EXPECTED_VOID(linePipeline.createGraphicsPipeline(
        device.logical(), pipelineCache, shaderStages, vertexInputInfo, lineInputAssembly,
        wd.RenderPass, &colorBlendAttachment, &depthStencilState));
  }

  void createTextPipeline(vk::raii::DescriptorSetLayout &textSetLayout,
                          vk::raii::DescriptorSetLayout &uiSetLayout) {
    auto textVertShader = createShaderModuleFromFile(device.logical(), "shaders/text_vert.spv");
    auto textFragShader = createShaderModuleFromFile(device.logical(), "shaders/text_frag.spv");
    if (!textVertShader || !textFragShader) {
      std::println("Error loading text shaders.");
      return;
    }

    // A simple descriptor set layout: just one texture sampler for the font atlas
    std::vector<vk::DescriptorSetLayoutBinding> textBindings = {
        {// Binding 0: Font Atlas Sampler (Fragment Shader)
         .binding = 0,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment}};
    vk::DescriptorSetLayoutCreateInfo textLayoutInfo{.bindingCount = (u32)textBindings.size(),
                                                     .pBindings = textBindings.data()};
    textSetLayout = device.logical().createDescriptorSetLayout(textLayoutInfo).value();

    auto uiVertShader = createShaderModuleFromFile(device.logical(), "shaders/ui_vert.spv");
    auto uiFragShader = createShaderModuleFromFile(device.logical(), "shaders/ui_frag.spv");
    if (!uiVertShader || !uiFragShader) {
      std::println("Error loading ui shaders.");
      return;
    }

    std::vector<vk::DescriptorSetLayoutBinding> instanceBindings = {
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eStorageBufferDynamic,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eVertex}};

    vk::DescriptorSetLayoutCreateInfo instanceLayoutInfo{
        .bindingCount = static_cast<uint32_t>(instanceBindings.size()),
        .pBindings = instanceBindings.data()};

    auto instanceSetLayout = device.logical().createDescriptorSetLayout(instanceLayoutInfo).value();
    // The text pipeline layout will also use a push constant to send per-quad data like color.
    vk::PushConstantRange textPushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(std::array<std::byte, 128>),
    };

    vk::PushConstantRange uiPushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = sizeof(std::array<std::byte, 128>),
    };

    std::vector<vk::DescriptorSetLayout> textLayouts = {*instanceSetLayout, *textSetLayout};
    std::vector<vk::PushConstantRange> constantRanges = {textPushConstantRange};
    EXPECTED_VOID(textPipeline.createPipelineLayout(device.logical(), textLayouts, constantRanges));

    std::vector<vk::DescriptorSetLayout> uiLayouts = {*instanceSetLayout};
    std::vector<vk::PushConstantRange> uiPCRanges = {uiPushConstantRange};
    EXPECTED_VOID(uiPipeline.createPipelineLayout(device.logical(), uiLayouts, uiPCRanges));

    std::vector<vk::PipelineShaderStageCreateInfo> textShaderStages = {
        {.stage = vk::ShaderStageFlagBits::eVertex,
         .module = *textVertShader.value(),
         .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment,
         .module = *textFragShader.value(),
         .pName = "main"}};

    std::vector<vk::PipelineShaderStageCreateInfo> uiShaderStages = {
        {/*vertex*/ vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex,
                                                      .module = *uiVertShader.value(),
                                                      .pName = "main"}},
        {/*fragment*/ vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment,
                                                        .module = *uiFragShader.value(),
                                                        .pName = "main"}}};

    // Vertex input for our simple text quad (pos and uv)
    vk::VertexInputBindingDescription textBindingDesc{0, sizeof(TextQuadVertex),
                                                      vk::VertexInputRate::eVertex};
    std::array<vk::VertexInputAttributeDescription, 2> textAttrDesc = {{
        {0, 0, vk::Format::eR32G32Sfloat, 0},                            // Position (vec2)
        {1, 0, vk::Format::eR32G32Sfloat, offsetof(TextQuadVertex, uv)}, // UV (vec2)
    }};
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &textBindingDesc,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = textAttrDesc.data()};

    // Standard triangle list input assembly
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};

    vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = true,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = false,
        .depthWriteEnable = false, // Don't write to depth buffer
    };

    EXPECTED_VOID(textPipeline.createGraphicsPipeline(
        device.logical(), pipelineCache, textShaderStages, vertexInputInfo, inputAssembly,
        wd.RenderPass, &blendAttachment, &depthStencil));

    EXPECTED_VOID(uiPipeline.createGraphicsPipeline(device.logical(), pipelineCache, uiShaderStages,
                                                    vertexInputInfo, inputAssembly, wd.RenderPass,
                                                    &blendAttachment, &depthStencil));
  }

  [[nodiscard]] std::expected<void, std::string> SetupShaderToggles() {
    vk::DeviceSize bufferSize = sizeof(ShaderTogglesUBO);
    shaderTogglesUbos.resize(wd.Frames.size());

    for (size_t i = 0; i < wd.Frames.size(); i++) {
      vk::BufferCreateInfo bufferInfo{.size = bufferSize,
                                      .usage = vk::BufferUsageFlagBits::eUniformBuffer};
      vma::AllocationCreateInfo allocInfo{
          .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                   vma::AllocationCreateFlagBits::eMapped,
          .usage = vma::MemoryUsage::eAutoPreferHost};

      auto bufferResult = device.createBufferVMA(bufferInfo, allocInfo);
      if (!bufferResult) {
        return std::unexpected("Couldn't create buffer for shader toggles UBO for frame " +
                               std::to_string(i));
      }
      shaderTogglesUbos[i] = std::move(bufferResult.value());
    }
    return {};
  }

  [[nodiscard]] std::expected<void, std::string> SetupSceneLights() {
    // Initialize some default lights on the CPU
    sceneLightsCpuBuffer.lightCount = 2;
    sceneLightsCpuBuffer.lights[0].position = {-20.0f, -20.0f, -20.0f, 1.0f};
    sceneLightsCpuBuffer.lights[0].color = {150.0f, 150.0f, 150.0f, 1.0f}; // White light
    sceneLightsCpuBuffer.lights[1].position = {20.0f, -30.0f, -15.0f, 1.0f};
    sceneLightsCpuBuffer.lights[1].color = {200.0f, 150.0f, 50.0f, 1.0f}; // Warm light

    // Create a GPU buffer for each frame in flight
    vk::DeviceSize bufferSize = sizeof(SceneLightsUBO);
    sceneLightsUbos.resize(wd.Frames.size());

    for (size_t i = 0; i < wd.Frames.size(); i++) {
      vk::BufferCreateInfo bufferInfo{.size = bufferSize,
                                      .usage = vk::BufferUsageFlagBits::eUniformBuffer};
      vma::AllocationCreateInfo allocInfo{
          .flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                   vma::AllocationCreateFlagBits::eMapped,
          .usage = vma::MemoryUsage::eAutoPreferHost};

      auto bufferResult = device.createBufferVMA(bufferInfo, allocInfo);
      if (!bufferResult) {
        return std::unexpected("Couldn't create buffer for light UBO for frame " +
                               std::to_string(i));
      }
      sceneLightsUbos[i] = std::move(bufferResult.value());
    }

    return {};
  }

  void createDebugFontAtlasQuad(u32 currentImageCount) {
    if (registeredFonts.empty())
      return;

    if (graphicsPipelines.empty() || !*graphicsPipelines[0].pipeline) {
      std::println("Error: Main pipeline not initialized for debug quad.");
      return;
    }

    // Create quad geometry
    glm::vec3 center(0.0f, 0.0f, -5.0f);                   // Position in front of camera
    auto font = registeredFonts[0]->atlasData.atlasBitmap; // Assuming this is your float pixel data
    glm::vec3 normal(0.0f, 0.0f, 1.0f);

    auto width = registeredFonts[0]->atlasData.atlasWidth;
    auto height = registeredFonts[0]->atlasData.atlasHeight;

    // --- Pixel Data Normalization ---
    // Assuming atlasBitmap is std::vector<float> and stores R, G, B consecutively
    // For example, if your atlasBitmap has 3 components per pixel (RGB)
    std::vector<unsigned char> normalizedPixels(
        font.size()); // Create a new vector for normalized data

    // Lambda to convert a single float value to unsigned char (0-255)
    auto normalizePixelComponent = [](float value) -> unsigned char {
      // Clamp negative values to 0
      float clampedValue = std::max(0.0f, value);
      // Scale from (0, 1) to (0, 255). Since original range is (-1, 1),
      // we scale the positive part (0, 1) to (0, 255).
      // If your original data spans -1 to 1 and you want to map -1 to 0, 0 to 127.5, and 1 to 255,
      // the scaling would be different. But based on "negative values are to be 0",
      // we treat the effective range for scaling as [0, 1].
      return static_cast<unsigned char>(clampedValue * 255.0f);
    };

    // Apply the lambda to each float in the original font data
    std::transform(font.begin(), font.end(), normalizedPixels.begin(), normalizePixelComponent);
    // --- End Pixel Data Normalization ---

    // Convert RGB (3 components) to RGBA (4 components)
    std::vector<unsigned char> finalPixelData;
    finalPixelData.reserve(width * height * 4); // Reserve space for RGBA
    for (size_t i = 0; i < font.size(); i += 3) {
      finalPixelData.push_back(normalizedPixels[i]);     // R
      finalPixelData.push_back(normalizedPixels[i + 1]); // G
      finalPixelData.push_back(normalizedPixels[i + 2]); // B
      finalPixelData.push_back(255);                     // A (Opaque)
    }
    Material quadMaterial;
    PBRTextures quadTextures = textureStore.getAllDefaultTextures();
    quadTextures.baseColor = textureStore.getTextureFromData({
        .pixels = finalPixelData, // Use the normalized pixel data here
        .width = width,
        .height = height,
        .component = 4, // Assuming 3 components (R, G, B)
        .isSrgb = true,
    });

    auto quadMesh = std::make_unique<Mesh>(
        device, "DebugFontAtlasQuad", createQuadVertices(center, width / 64, height / 64, normal),
        createQuadIndices(), quadMaterial, quadTextures, currentImageCount);

    appOwnedMeshes.emplace_back(std::move(quadMesh));

    scene.createNode({.mesh = appOwnedMeshes.back().get(),
                      .pipeline = &graphicsPipelines[0], // Use main textured pipeline
                      .name = "DebugFontAtlasQuad_Node"},
                     device.descriptorPool_, combinedMeshLayout);
  }

  void createDebugAxesScene(u32 currentImageCount) {
    if (graphicsPipelines.size() < 2 || !*graphicsPipelines[1].pipeline) {
      std::println(
          "Error: TextureStore or Line Pipeline not initialized for createDebugAxesScene.");
      return;
    }

    VulkanPipeline *linePipeline = &graphicsPipelines[1];
    float axisLength = 10000.0f;

    auto createAxis = [&](const std::string &name, glm::vec3 start, glm::vec3 end,
                          std::array<u8, 4> color, glm::vec3 tangent) {
      Material axisMaterial;
      PBRTextures axisTextures = textureStore.getAllDefaultTextures();
      axisTextures.baseColor = textureStore.getColorTexture(color);
      auto axisMesh = std::make_unique<Mesh>(
          device, name, createAxisLineVertices(start, end, tangent), createAxisLineIndices(),
          axisMaterial, axisTextures, currentImageCount);
      appOwnedMeshes.emplace_back(std::move(axisMesh));
      scene.createNode(
          {.mesh = appOwnedMeshes.back().get(), .pipeline = linePipeline, .name = name + "_Node"},
          device.descriptorPool_, combinedMeshLayout);
    };

    createAxis("X_Axis", {-axisLength, 0, 0}, {axisLength, 0, 0}, {255, 0, 0, 255}, {0, 1, 0});
    createAxis("Y_Axis", {0, -axisLength, 0}, {0, axisLength, 0}, {0, 255, 0, 255}, {1, 0, 0});
    createAxis("Z_Axis", {0, 0, -axisLength}, {0, 0, axisLength}, {0, 0, 255, 255}, {1, 0, 0});
  }

  void loadGltfModel(const std::string &filePath, const std::string &baseDir) {
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

    std::lock_guard lock{loadedGltfDataMutex};
    loadedGltfData.emplace_back(*loadedGltfDataResult);
  }

  std::vector<LoadedGltfScene> stealLoadedScenes() {
    std::vector<LoadedGltfScene> out;
    {
      std::lock_guard lock{loadedGltfDataMutex};
      out = std::move(loadedGltfData);
      loadedGltfData.clear();
    }
    return out;
  }

  void instanceGltfModel(const LoadedGltfScene &gltfData, u32 currentImageCount) {
    auto builtMeshesResult =
        populateSceneFromGltf(scene, gltfData, device, textureStore,
                              &graphicsPipelines[0], // Use the main mesh pipeline for GLTF models
                              currentImageCount, combinedMeshLayout);

    if (!builtMeshesResult) {
      std::println("Failed to build engine scene from GLTF data: {}", builtMeshesResult.error());
      return;
    }

    for (auto &mesh_ptr : builtMeshesResult->engineMeshes) {
      appOwnedMeshes.emplace_back(std::move(mesh_ptr));
    }
  }

  void SetupVulkan() {
    EXPECTED_VOID(instance.create());
    EXPECTED_VOID(instance.setupDebugMessenger());
    EXPECTED_VOID(device.pickPhysicalDevice());
    EXPECTED_VOID(device.createLogicalDevice());
    EXPECTED_VOID(createDescriptorSetLayouts());
    auto cacheResult = device.logical().createPipelineCache({});
    if (cacheResult) {
      pipelineCache = std::move(cacheResult.value());
    }
    EXPECTED_VOID(textureStore.createDefaultTextures());
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
    u32 imageIndex = imageIndexx;

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

    // Update UBOs for the current frame
    {
      // 1. Scene Lights UBO
      std::memcpy(sceneLightsUbos[wd.FrameIndex].getMappedData(), &sceneLightsCpuBuffer,
                  sizeof(SceneLightsUBO));

      // 2. Shader Toggles UBO
      std::memcpy(shaderTogglesUbos[wd.FrameIndex].getMappedData(), &shaderTogglesCpuBuffer,
                  sizeof(ShaderTogglesUBO));

      // Update descriptor sets for the current frame
      vk::DescriptorBufferInfo lightUboInfo{.buffer = sceneLightsUbos[wd.FrameIndex].get(),
                                            .offset = 0,
                                            .range = sizeof(SceneLightsUBO)};

      vk::DescriptorBufferInfo togglesUboInfo{.buffer = shaderTogglesUbos[wd.FrameIndex].get(),
                                              .offset = 0,
                                              .range = sizeof(ShaderTogglesUBO)};

      std::array<vk::WriteDescriptorSet, 2> writeInfos = {
          vk::WriteDescriptorSet{
              .dstSet = *sceneDescriptorSets[wd.FrameIndex],
              .dstBinding = 0, // Binding 0 for lights
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo = &lightUboInfo,
          },
          vk::WriteDescriptorSet{
              .dstSet = *sceneDescriptorSets[wd.FrameIndex],
              .dstBinding = 1, // Binding 1 for toggles
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo = &togglesUboInfo,
          }};
      device.logical().updateDescriptorSets(writeInfos, nullptr);
    }

    currentFrame.CommandPool.reset();
    currentFrame.CommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    scene.updateHierarchy(wd.FrameIndex, camera.GetViewMatrix(),
                          camera.GetProjectionMatrix((float)wd.config.swapchainExtent.width /
                                                     (float)wd.config.swapchainExtent.height),
                          deltaTime);

    scene.updateAllDescriptorSetContents(wd.FrameIndex);

    currentFrame.CommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                                  *graphicsPipelines[0].pipelineLayout,
                                                  1, // firstSet = 1
                                                  {*sceneDescriptorSets[wd.FrameIndex]}, {});

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].color = wd.config.ClearValue.color;
    clearValues[1].depthStencil = {1.0f, 0};

    currentFrame.CommandBuffer.beginRenderPass(
        vk::RenderPassBeginInfo{
            .renderPass = *wd.RenderPass,
            .framebuffer = *currentFrame.Framebuffer,
            .renderArea = {.offset = {0, 0}, .extent = wd.config.swapchainExtent},
            .clearValueCount = static_cast<u32>(clearValues.size()),
            .pClearValues = clearValues.data()},
        vk::SubpassContents::eInline);

    currentFrame.CommandBuffer.setViewport(
        0, vk::Viewport{0.0f, 0.0f, static_cast<float>(wd.config.swapchainExtent.width),
                        static_cast<float>(wd.config.swapchainExtent.height), 0.0f, 1.0f});
    currentFrame.CommandBuffer.setScissor(0, vk::Rect2D{{0, 0}, wd.config.swapchainExtent});

    scene.draw(currentFrame.CommandBuffer, wd.FrameIndex);

    static TextArea textArea{300, 200};

    textSystem->beginFrame();
    uiSystem->beginFrame();

    RenderQueue renderQueue;

    {
      // Define colors
      glm::vec4 color1 = {1.f, 1.f, 0.f, 1.f}; // Yellow
      glm::vec4 color2 = {1.f, 0.f, 1.f, 1.f}; // Magenta
      glm::vec4 color3 = {0.f, 1.f, 1.f, 1.f}; // Cyan

      // Define base positions and offsets
      float startX = 100.f;
      float startY = 100.f;

      // Offsets for the different text positions within each font's set
      float offsetX1 = 0.f;
      float offsetY1 = 0.f; // (100, 100)

      float offsetX2 = 600.f;
      float offsetY2 = 200.f; // (700, 300)

      float offsetX3 = 600.f;
      float offsetY3 = 700.f; // (700, 800)

      float offsetX4 = 0.f;
      float offsetY4 = 400.f; // (100, 500)

      float font2YOffset = 100.f; // Offset for the second font relative to the first
      float font3YOffset = 100.f; // Offset for the third font relative to the second

      if (!registeredFonts.empty()) {
        textSystem->queueText(registeredFonts[0], "Vulkan", 15 + fontSizeMultiplier,
                              startX + offsetX1, startY + offsetY1, color1);
        textSystem->queueText(registeredFonts[0], "Vulkan", 30 + fontSizeMultiplier,
                              startX + offsetX4, startY + offsetY4, color1);
        textSystem->queueText(registeredFonts[0], "Vulkan", 60 + fontSizeMultiplier,
                              startX + offsetX2, startY + offsetY2, color1);
        textSystem->queueText(registeredFonts[0], "Vulkan", 120 + fontSizeMultiplier,
                              startX + offsetX3, startY + offsetY3, color1);
      }

      if (registeredFonts.size() > 1) {
        // Apply font2YOffset to all Y coordinates for the second font
        textSystem->queueText(registeredFonts[1], "Vulkan", 15 + fontSizeMultiplier,
                              startX + offsetX1, startY + offsetY1 + font2YOffset, color2);
        textSystem->queueText(registeredFonts[1], "Vulkan", 30 + fontSizeMultiplier,
                              startX + offsetX4, startY + offsetY4 + font2YOffset, color2);
        textSystem->queueText(registeredFonts[1], "Vulkan", 60 + fontSizeMultiplier,
                              startX + offsetX2, startY + offsetY2 + font2YOffset, color2);
        textSystem->queueText(registeredFonts[1], "Vulkan", 120 + fontSizeMultiplier,
                              startX + offsetX3, startY + offsetY3 + font2YOffset, color2);
      }

      if (registeredFonts.size() > 2) {
        // Apply font2YOffset + font3YOffset to all Y coordinates for the third font
        // This makes the third font offset from the second, which is already offset from the first.
        textSystem->queueText(registeredFonts[2], "Vulkan", 15 + fontSizeMultiplier,
                              startX + offsetX1, startY + offsetY1 + font2YOffset + font3YOffset,
                              color3);
        textSystem->queueText(registeredFonts[2], "Vulkan", 30 + fontSizeMultiplier,
                              startX + offsetX4, startY + offsetY4 + font2YOffset + font3YOffset,
                              color3);
        textSystem->queueText(registeredFonts[2], "Vulkan", 60 + fontSizeMultiplier,
                              startX + offsetX2, startY + offsetY2 + font2YOffset + font3YOffset,
                              color3);
        textSystem->queueText(registeredFonts[2], "Vulkan", 120 + fontSizeMultiplier,
                              startX + offsetX3, startY + offsetY3 + font2YOffset + font3YOffset,
                              color3);
      }
    }

    uiSystem->queueSheet(
        {
            .position = {200, 200},
            .size = {300, 300},
            .backgroundColor = {0.f, 1.f, 0.f, 1.0f},
        },
        0.f);

    uiSystem->queueSheet(
        {
            .position = {1000, 100},
            .size = {200, 200},
            .backgroundColor = {1.f, 0.f, 0.f, 1.0f},
        },
        0.f);

    textSystem->prepareBatches(renderQueue, textPipeline, wd.FrameIndex, wd.config.swapchainExtent,
                               sdf_weight, textAntiAliasingToggle);
    uiSystem->prepareBatches(renderQueue, uiPipeline, wd.FrameIndex, wd.config.swapchainExtent);

    processRenderQueue(currentFrame.CommandBuffer, renderQueue);

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
    return {static_cast<u32>(w > 0 ? w : 1), static_cast<u32>(h > 0 ? h : 1)};
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

  void readKeyboard(float dt, SDL_Window *sdl_window) {
    static bool isFullscreen = false;
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
    if (keystate[SDL_SCANCODE_F11]) {
      std::println("pressed f11");
      isFullscreen = !isFullscreen;
      if (isFullscreen) {
        SDL_SetWindowFullscreenMode(sdl_window, nullptr); // Fullscreen desktop
        SDL_SetWindowFullscreen(sdl_window, true);
        SDL_SyncWindow(sdl_window);
      } else {
        SDL_SetWindowFullscreenMode(sdl_window, nullptr); // Exit fullscreen
        SDL_SetWindowFullscreen(sdl_window, false);
        SDL_SyncWindow(sdl_window);
      }
    }
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
      auto newScenes = stealLoadedScenes();

      for (auto &scene : newScenes) {
        instanceGltfModel(scene, wd.Frames.size());
      }

      // Process completed font loading futures
      {
        std::lock_guard lock(fontFuturesMutex);
        // Using a temporary vector to collect completed futures
        std::vector<std::future<std::expected<Font *, std::string>>> readyFutures;
        for (auto it = fontLoadFutures.begin(); it != fontLoadFutures.end();) {
          if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            readyFutures.push_back(std::move(*it));
            it = fontLoadFutures.erase(it);
          } else {
            ++it;
          }
        }

        for (auto &future : readyFutures) {
          auto res = future.get(); // Get the result of the future
          if (res) {
            registeredFonts.emplace_back(*res);
            // std::println("font loaded");
            // createDebugFontAtlasQuad(static_cast<u32>(wd.Frames.size()));
          } else {
            std::println("failed to load font asynchronously: {}", res.error());
          }
        }
      }

      readKeyboard(deltaTime, sdl_window);
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

        EXPECTED_VOID(SetupSceneLights());
        EXPECTED_VOID(SetupShaderToggles()); // NEW: Re-create toggles UBOs
        allocateSceneDescriptorSets();

        scene.setImageCount(static_cast<u32>(wd.Frames.size()), device.descriptorPool_,
                            combinedMeshLayout);
        swapChainRebuild = false;
      }

      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplSDL3_NewFrame();
      ImGui::NewFrame();

      {
        RenderCameraControlMenu(camera);
        RenderVulkanStateWindow(device, wd, 120, deltaTime);
        RenderLightControlMenu(sceneLightsCpuBuffer);
        RenderShaderTogglesMenu(shaderTogglesCpuBuffer);
        RenderTextMenu(fontSizeMultiplier, sdf_weight, textAntiAliasingToggle);
        RenderSceneHierarchyMaterialEditor(scene, wd.FrameIndex);
      }

      ImGui::Render();
      FrameRender(ImGui::GetDrawData(), deltaTime);
      FramePresent();
    }
  }

public:
  int run(SDL_Window *sdl_window) {

    SetupVulkan();

    SetupVulkanWindow(sdl_window, get_window_size(sdl_window));

    u32 numMeshesEstimate = 50;

    EXPECTED_VOID(
        device.createDescriptorPool(static_cast<u32>(wd.Frames.size()) * numMeshesEstimate));

    createGraphicsPipelines();
    vk::raii::DescriptorSetLayout textSetLayout{nullptr};
    vk::raii::DescriptorSetLayout uiSetLayout{nullptr};
    createTextPipeline(textSetLayout, uiSetLayout);
    textSystem = std::make_unique<TextSystem>(device, wd.Frames.size(), device.descriptorPool_);

    // Submit font loading tasks to the thread pool
    {
      std::lock_guard lock(fontFuturesMutex);
      fontLoadFutures.emplace_back(thread_pool.submit_task([this, &textSetLayout]() {
        return textSystem->registerFont("../assets/fonts/ComicSans/ComicSansMS400.ttf", 64,
                                        textSetLayout);
      }));

      fontLoadFutures.emplace_back(thread_pool.submit_task([this, &textSetLayout]() {
        return textSystem->registerFont(
            "../assets/fonts/Inconsolata/InconsolataNerdFontMono-Regular.ttf", 64, textSetLayout);
      }));

      fontLoadFutures.emplace_back(thread_pool.submit_task([this, &textSetLayout]() {
        return textSystem->registerFont("../assets/fonts/Queensides/Queensides-3z7Ey.ttf", 64,
                                        textSetLayout);
      }));
    }

    uiSystem = std::make_unique<UISystem>(device, wd.Frames.size(), device.descriptorPool_);

    EXPECTED_VOID(SetupSceneLights());
    EXPECTED_VOID(SetupShaderToggles()); // NEW: Setup toggles UBO

    scene = Scene(static_cast<u32>(wd.Frames.size()));

    auto future = thread_pool.submit_task([this] {
      // return loadGltfModel("../assets/models/woman/scene.gltf", "../assets/models/woman/");
      // return loadGltfModel("../assets/models/sphinx/scene.gltf", "../assets/models/sphinx/");
      // return loadGltfModel("../assets/models/sphinx2/sphinx2.gltf", "");
      // return loadGltfModel("../assets/models/sarc.glb", "");
    });
    createDebugAxesScene(static_cast<u32>(wd.Frames.size()));

    allocateSceneDescriptorSets();

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
    init_info.ImageCount = static_cast<u32>(wd.Frames.size());
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

    return 0;
  }
};
