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
import :imgui;
import :TextSystem;
import :UISystem;
import :ui;
import :TextArea;
import :TextView;
import :ThreeDEngine; // Import the new 3D engine module

namespace {
constexpr u32 MIN_IMAGE_COUNT = 2;

void check_vk_result(VkResult err) {
  if (err == VK_SUCCESS) {
    return;
  }
  std::string errMsg = "[vulkan] Error: VkResult = " + std::to_string(err);
  std::println("{}", errMsg);
  if (err < 0) {
    std::exit(1);
  }
}

void check_vk_result_hpp(vk::Result err) {
  if (err == vk::Result::eSuccess) {
    return;
  }
  std::string errMsg = "[vulkan] Error: VkResult = " + vk::to_string(err);
  std::print("{}", errMsg); // Keep console print
  std::exit(0);
}
} // anonymous namespace

export class App {
  VulkanInstance instance;
  VulkanDevice device{instance};
  Window wd;
  bool swapChainRebuild = false;

  // 3D Engine
  std::unique_ptr<ThreeDEngine> threeDEngine;

  // 2D Rendering Systems
  VulkanPipeline textPipeline;
  VulkanPipeline uiPipeline;
  vk::raii::PipelineCache pipelineCache{nullptr};
  std::unique_ptr<TextSystem> textSystem;
  std::unique_ptr<UISystem> uiSystem;

  // Text Editor MVC Components
  TextEditor textEditor{"Lorem ipsum vulkan\n Lorem ipsum vulkan Lorem\n ipsum vulkan Lorem ipsum "
                        "vulkan Lorem ipsum\n vulkan"};
  std::unique_ptr<TextView> textView;
  std::vector<Font *> registeredFonts;

  // Async Asset Loading
  BS::thread_pool<> thread_pool;
  std::vector<std::future<std::expected<Font *, std::string>>> fontLoadFutures;
  std::mutex fontFuturesMutex;

  // UI State
  i32 fontSizeMultiplier{0};
  TextToggles textToggles;

private:
  void create2DGraphicsPipelines() {
    auto textVertShader = createShaderModuleFromFile(device.logical(), "shaders/text_vert.spv");
    auto textFragShader = createShaderModuleFromFile(device.logical(), "shaders/text_frag.spv");
    if (!textVertShader || !textFragShader) {
      std::println("Error loading text shaders.");
      return;
    }

    vk::raii::DescriptorSetLayout textSetLayout{nullptr};
    std::vector<vk::DescriptorSetLayoutBinding> textBindings = {
        {.binding = 0,
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

    vk::raii::DescriptorSetLayout instanceSetLayout{nullptr};
    std::vector<vk::DescriptorSetLayoutBinding> instanceBindings = {
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eStorageBufferDynamic,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eVertex}};
    vk::DescriptorSetLayoutCreateInfo instanceLayoutInfo{
        .bindingCount = static_cast<u32>(instanceBindings.size()),
        .pBindings = instanceBindings.data()};
    instanceSetLayout = device.logical().createDescriptorSetLayout(instanceLayoutInfo).value();

    vk::PushConstantRange textPushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex |
                                                              vk::ShaderStageFlagBits::eFragment,
                                                .offset = 0,
                                                .size = sizeof(std::array<std::byte, 128>)};
    std::vector<vk::DescriptorSetLayout> textLayouts = {*instanceSetLayout, *textSetLayout};
    auto textPipelineLayoutResult =
        textPipeline.createPipelineLayout(device.logical(), textLayouts, {textPushConstantRange});
    if (!textPipelineLayoutResult) {
      std::println("{}", textPipelineLayoutResult.error());
      std::exit(1);
    }

    vk::PushConstantRange uiPushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex,
                                              .offset = 0,
                                              .size = sizeof(std::array<std::byte, 128>)};
    std::vector<vk::DescriptorSetLayout> uiLayouts = {*instanceSetLayout};
    auto uiPipelineLayoutResult =
        uiPipeline.createPipelineLayout(device.logical(), uiLayouts, {uiPushConstantRange});
    if (!uiPipelineLayoutResult) {
      std::println("{}", uiPipelineLayoutResult.error());
      std::exit(1);
    }

    std::vector<vk::PipelineShaderStageCreateInfo> textShaderStages = {
        {.stage = vk::ShaderStageFlagBits::eVertex,
         .module = *textVertShader.value(),
         .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment,
         .module = *textFragShader.value(),
         .pName = "main"}};
    std::vector<vk::PipelineShaderStageCreateInfo> uiShaderStages = {
        {.stage = vk::ShaderStageFlagBits::eVertex,
         .module = *uiVertShader.value(),
         .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment,
         .module = *uiFragShader.value(),
         .pName = "main"}};

    vk::VertexInputBindingDescription textBindingDesc{
        .binding = 0, .stride = sizeof(TextQuadVertex), .inputRate = vk::VertexInputRate::eVertex};
    std::array<vk::VertexInputAttributeDescription, 2> textAttrDesc;
    textAttrDesc[0] = {
        .location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = 0};
    textAttrDesc[1] = {.location = 1,
                       .binding = 0,
                       .format = vk::Format::eR32G32Sfloat,
                       .offset = offsetof(TextQuadVertex, uv)};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &textBindingDesc,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = textAttrDesc.data()};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    vk::PipelineDepthStencilStateCreateInfo depthStencil{.depthTestEnable = vk::False,
                                                         .depthWriteEnable = vk::False};

    auto textGraphicsPipelineResult = textPipeline.createGraphicsPipeline(
        device.logical(), pipelineCache, textShaderStages, vertexInputInfo, inputAssembly,
        wd.RenderPass, &blendAttachment, &depthStencil);
    if (!textGraphicsPipelineResult) {
      std::println("{}", textGraphicsPipelineResult.error());
      std::exit(1);
    }
    auto uiGraphicsPipelineResult = uiPipeline.createGraphicsPipeline(
        device.logical(), pipelineCache, uiShaderStages, vertexInputInfo, inputAssembly,
        wd.RenderPass, &blendAttachment, &depthStencil);
    if (!uiGraphicsPipelineResult) {
      std::println("{}", uiGraphicsPipelineResult.error());
      std::exit(1);
    }
  }

  void SetupVulkan() {
    auto instanceCreateResult = instance.create();
    if (!instanceCreateResult) {
      std::println("{}", instanceCreateResult.error());
      std::exit(1);
    }
    auto setupDebugMessengerResult = instance.setupDebugMessenger();
    if (!setupDebugMessengerResult) {
      std::println("{}", setupDebugMessengerResult.error());
      std::exit(1);
    }
    auto pickPhysicalDeviceResult = device.pickPhysicalDevice();
    if (!pickPhysicalDeviceResult) {
      std::println("{}", pickPhysicalDeviceResult.error());
      std::exit(1);
    }
    auto createLogicalDeviceResult = device.createLogicalDevice();
    if (!createLogicalDeviceResult) {
      std::println("{}", createLogicalDeviceResult.error());
      std::exit(1);
    }
    auto cacheResult = device.logical().createPipelineCache({});
    if (cacheResult) {
      pipelineCache = std::move(cacheResult.value());
    }
  }

  void SetupVulkanWindow(SDL_Window *sdl_window, vk::Extent2D extent) {
    VkSurfaceKHR surface_raw_handle;
    if (static_cast<int>(SDL_Vulkan_CreateSurface(sdl_window, instance.get_C_handle(), nullptr,
                                                  &surface_raw_handle)) == 0) {
      std::println("Failed to create Vulkan surface via SDL: {}", SDL_GetError());
      std::exit(EXIT_FAILURE);
    }
    wd.Surface = vk::raii::SurfaceKHR(instance, surface_raw_handle);

    std::vector<vk::Format> requestSurfaceImageFormat = {
        vk::Format::eB8G8R8A8Srgb, vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Unorm,
        vk::Format::eR8G8B8A8Unorm};
    wd.config.SurfaceFormat =
        selectSurfaceFormat(device.physical(), wd.Surface, requestSurfaceImageFormat,
                            vk::ColorSpaceKHR::eSrgbNonlinear);
    std::vector<vk::PresentModeKHR> present_modes = {vk::PresentModeKHR::eFifo};
    wd.config.PresentMode = selectPresentMode(device.physical(), wd.Surface, present_modes);
    wd.config.ClearEnable = true;
    wd.config.ClearValue.color = vk::ClearColorValue(std::array<f32, 4>{0.0, 0.0, 0.0, 1.0});

    createOrResizeWindow(instance, device, wd, extent, MIN_IMAGE_COUNT);
  }

  void FrameRender(ImDrawData *draw_data, f32 deltaTime) {
    if (!*wd.Swapchain) {
      return;
    }

    auto &image_acquired_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    auto &render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;

    auto [acquireRes, imageIndex] =
        device.logical().acquireNextImage2KHR({.swapchain = wd.Swapchain,
                                               .timeout = UINT64_MAX,
                                               .semaphore = image_acquired_semaphore,
                                               .deviceMask = 1});
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

    // Update and draw 3D scene
    if (threeDEngine) {
      threeDEngine->update(wd.FrameIndex, deltaTime, wd.config.swapchainExtent);
    }

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].color = wd.config.ClearValue.color;
    clearValues[1].depthStencil = {.depth = 1.0, .stencil = 0};

    currentFrame.CommandBuffer.beginRenderPass(
        {.renderPass = *wd.RenderPass,
         .framebuffer = *currentFrame.Framebuffer,
         .renderArea = {.offset = {.x = 0, .y = 0}, .extent = wd.config.swapchainExtent},
         .clearValueCount = static_cast<u32>(clearValues.size()),
         .pClearValues = clearValues.data()},
        vk::SubpassContents::eInline);
    currentFrame.CommandBuffer.setViewport(
        0, vk::Viewport{.x = 0.0,
                        .y = 0.0,
                        .width = (f32)wd.config.swapchainExtent.width,
                        .height = (f32)wd.config.swapchainExtent.height,
                        .minDepth = 0.0,
                        .maxDepth = 1.0});
    currentFrame.CommandBuffer.setScissor(
        0, vk::Rect2D{.offset = {.x = 0, .y = 0}, .extent = wd.config.swapchainExtent});

    if (threeDEngine) {
      threeDEngine->draw(currentFrame.CommandBuffer, wd.FrameIndex);
    }

    // Prepare and draw 2D elements
    textSystem->beginFrame();
    uiSystem->beginFrame();
    RenderQueue renderQueue;

    if (textView) {
      textView->setDimensions((float)wd.config.swapchainExtent.width,
                              (float)wd.config.swapchainExtent.height);
      size_t firstLine = textView->getFirstVisibleLine();
      size_t numLines = textView->getVisibleLineCount();
      size_t lastLine = std::min(firstLine + numLines, textEditor.lineCount());
      float currentLineYpos = 100.0;

      if (!registeredFonts.empty()) {
        for (size_t i = firstLine; i < lastLine; ++i) {
          textSystem->queueText(registeredFonts[0], textEditor.getLine(i), 36, 200.0,
                                currentLineYpos, {1.0, 1.0, 1.0, 1.0});
          const double pointSize = 36.0;
          const double fontUnitToPixelScale =
              pointSize * (96.0 / 72.0 * 2) / registeredFonts[0]->atlasData.unitsPerEm;
          const double line_height_px =
              registeredFonts[0]->atlasData.lineHeight * fontUnitToPixelScale;
          currentLineYpos += (line_height_px > 0) ? static_cast<float>(line_height_px) : 38.0;
        }
      }
    }

    uiSystem->queueQuad({.position = {200, 200},
                         .size = {300, 300},
                         .color = {0.0, 1.0, 0.0, 1.0},
                         .z_layer = 0.0});
    uiSystem->queueQuad({.position = {1000, 200},
                         .size = {100, 300},
                         .color = {0.0, 0.0, 1.0, 1.0},
                         .z_layer = 0.0});

    textSystem->prepareBatches(renderQueue, textPipeline, wd.FrameIndex, wd.config.swapchainExtent,
                               textToggles);
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
    }
    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.FrameSemaphores.size();
  }

  static vk::Extent2D get_window_size(SDL_Window *window) {
    int width = 0;
    int height = 0;
    SDL_GetWindowSize(window, &width, &height);
    return {.width = static_cast<u32>(width > 0 ? width : 1),
            .height = static_cast<u32>(height > 0 ? height : 1)};
  }

  void readKeyboard(float deltaTime, SDL_Window *sdl_window) {
    static bool isFullscreen = false;
    const auto *const keystate = SDL_GetKeyboardState(nullptr);
    if (threeDEngine) {
      auto &camera = threeDEngine->getCamera();
      float velocity = camera.MovementSpeed * deltaTime;
      if (keystate[SDL_SCANCODE_W]) {
        camera.Position += camera.Front * velocity;
      }
      if (keystate[SDL_SCANCODE_S]) {
        camera.Position -= camera.Front * velocity;
      }
      if (keystate[SDL_SCANCODE_A]) {
        camera.Position -= camera.Right * velocity;
      }
      if (keystate[SDL_SCANCODE_D]) {
        camera.Position += camera.Right * velocity;
      }
      if (keystate[SDL_SCANCODE_SPACE]) {
        camera.Position += camera.WorldUp * velocity;
      }
      if (keystate[SDL_SCANCODE_LCTRL]) {
        camera.Position -= camera.WorldUp * velocity;
      }
    }
    if (keystate[SDL_SCANCODE_F11]) {
      isFullscreen = !isFullscreen;
      SDL_SetWindowFullscreen(sdl_window, isFullscreen);
    }
  }

  void mainLoop(SDL_Window *sdl_window) {
    ImGuiIO &imguiIO = ImGui::GetIO();
    imguiIO.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    using Clock = std::chrono::high_resolution_clock;
    auto previousTime = Clock::now();
    float deltaTime = 0.0;
    bool done = false;

    while (!done) {
      auto currentTime = Clock::now();
      deltaTime = std::chrono::duration<float>(currentTime - previousTime).count();
      previousTime = currentTime;

      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);
        if (event.type == SDL_EVENT_QUIT ||
            (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
             event.window.windowID == SDL_GetWindowID(sdl_window))) {
          done = true;
        }
        if (event.type == SDL_EVENT_WINDOW_MINIMIZED) {
        }
        if (event.type == SDL_EVENT_WINDOW_RESTORED || event.type == SDL_EVENT_WINDOW_RESIZED ||
            event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
          swapChainRebuild = true;
        }
        if (event.type == SDL_EVENT_MOUSE_WHEEL && textView) {
          textView->scroll(event.wheel.y > 0 ? -3 : 3);
        }
      }

      if (threeDEngine) {
        threeDEngine->processGltfLoads();
      }

      {
        std::lock_guard lock(fontFuturesMutex);
        for (auto it = fontLoadFutures.begin(); it != fontLoadFutures.end();) {
          if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto res = it->get();
            if (res) {
              Font *font = *res;
              registeredFonts.emplace_back(font);
              if (!textView && (font != nullptr)) {
                textView = std::make_unique<TextView>(textEditor, *font);
              }
            } else {
              std::println("failed to load font asynchronously: {}", res.error());
            }
            it = fontLoadFutures.erase(it);
          }
        }
      }

      if (threeDEngine) {
        readKeyboard(deltaTime, sdl_window);
      }

      if ((SDL_GetWindowFlags(sdl_window) & SDL_WINDOW_MINIMIZED) != 0U) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      vk::Extent2D currentExtent = get_window_size(sdl_window);
      if (currentExtent.width == 0 || currentExtent.height == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      if (swapChainRebuild) {
        device.logical().waitIdle();
        createOrResizeWindow(instance, device, wd, currentExtent, MIN_IMAGE_COUNT);
        if (threeDEngine) {
          threeDEngine->onSwapchainRecreated(wd.Frames.size());
        }
        swapChainRebuild = false;
      }

      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplSDL3_NewFrame();
      ImGui::NewFrame();

      if (threeDEngine) {
        RenderCameraControlMenu(threeDEngine->getCamera());
        RenderLightControlMenu(threeDEngine->getLightUbo());
        RenderShaderTogglesMenu(threeDEngine->getShaderToggles());
        RenderSceneHierarchyMaterialEditor(threeDEngine->getScene(), wd.FrameIndex);
      }
      RenderVulkanStateWindow(device, wd, 120, deltaTime);
      RenderTextMenu(fontSizeMultiplier, textToggles);

      ImGui::Render();
      FrameRender(ImGui::GetDrawData(), deltaTime);
      FramePresent();
    }
  }

public:
  int run(SDL_Window *sdl_window) {
    SetupVulkan();
    SetupVulkanWindow(sdl_window, get_window_size(sdl_window));

    u32 numMeshesEstimate = 100;
    auto createDescriptorPoolResult =
        device.createDescriptorPool(static_cast<u32>(wd.Frames.size()) * numMeshesEstimate);
    if (!createDescriptorPoolResult) {
      std::println("{}", createDescriptorPoolResult.error());
      std::exit(1);
    }

    // Initialize Engines
    threeDEngine = std::make_unique<ThreeDEngine>(device, wd.Frames.size(), thread_pool);
    auto threeDEngineInitResult = threeDEngine->initialize(wd.RenderPass, pipelineCache);
    if (!threeDEngineInitResult) {
      std::println("{}", threeDEngineInitResult.error());
      std::exit(1);
    }
    threeDEngine->loadInitialAssets();

    create2DGraphicsPipelines();
    textSystem = std::make_unique<TextSystem>(device, wd.Frames.size(), device.descriptorPool_);
    uiSystem = std::make_unique<UISystem>(device, wd.Frames.size(), device.descriptorPool_);

    // Load 2D assets
    {
      std::lock_guard lock(fontFuturesMutex);
      fontLoadFutures.emplace_back(thread_pool.submit_task([this]() {
        // This is a bit of a hack, we need to get the layout from the pipeline
        // but the pipeline is created after the text system.
        // For now, we assume the layout is known.
        vk::raii::DescriptorSetLayout textSetLayout{nullptr};
        std::vector<vk::DescriptorSetLayoutBinding> textBindings = {
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment}};
        vk::DescriptorSetLayoutCreateInfo textLayoutInfo{.bindingCount = (u32)textBindings.size(),
                                                         .pBindings = textBindings.data()};
        textSetLayout = device.logical().createDescriptorSetLayout(textLayoutInfo).value();
        return textSystem->registerFont(
            "../assets/fonts/Inconsolata/InconsolataNerdFontMono-Regular.ttf", 64, textSetLayout);
      }));
    }

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForVulkan(sdl_window);
    ImGui_ImplVulkan_InitInfo init_info = device.init_info();
    init_info.Instance = instance.get_C_handle();
    init_info.RenderPass = *wd.RenderPass;
    init_info.MinImageCount = MIN_IMAGE_COUNT;
    init_info.ImageCount = static_cast<u32>(wd.Frames.size());
    init_info.PipelineCache = *pipelineCache;
    init_info.CheckVkResultFn = check_vk_result;
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
