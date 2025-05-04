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
import :DDX;
import :VulkanDevice;
import :VulkanInstance;
import :VulkanPipeline;
import :extra;
import :mesh;
import :scene;

namespace {
constexpr uint32_t minImageCount = 2;
}

void check_vk_result(VkResult err) {
  if (err == VK_SUCCESS)
    return;
  std::print("[vulkan] Error: VkResult = ");
  if (err < 0)
    std::abort();
}

void check_vk_result_hpp(vk::Result err) {
  if (err == vk::Result::eSuccess)
    return;
  std::print("[vulkan] Error: VkResult = {}", vk::to_string(err));
  std::abort();
}

export class App {
  VulkanInstance instance;
  VulkanDevice device{instance};
  VulkanPipeline graphicsPipeline;

  std::vector<vk::raii::ShaderModule> shaders;

  Window wd;
  bool swapChainRebuild = false;
  u32 rebuild_counter{0};

  vk::raii::DescriptorSetLayout descriptorSetLayout{nullptr};
  vk::raii::PipelineCache pipelineCache{nullptr};

  std::vector<Mesh> meshes;          // All actual renderable meshes
  std::vector<Transform> transforms; // All transforms (indexed for clarity)

  enum Transforms {
    TX_Shoulder, // Root transform (meshless)
    TX_Stick1,   // First stick (4 units)
    TX_Joint1,   // first rotational joint
    TX_Stick2,   // Second stick (2 units)
    TX_Joint2,   // second rotational joint
    TX_Stick3,   // Third stick (1 unit cube)
    TX_COUNT
  };

  Scene scene;

  Camera camera;

  int frameCap = 120;
  float targetFrameDuration = 1.0f / static_cast<float>(frameCap);

  std::expected<void, std::string> createDescriptorSetLayout() NOEXCEPT {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    };
    if (auto expected = device.logical().createDescriptorSetLayout({
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        expected) {
      descriptorSetLayout = std::move(*expected);
      std::println("Descriptor set layout created successfully!");
      return {};
    } else {
      return std::unexpected("Failed to create descriptor set layout: " +
                             vk::to_string(expected.error()));
    }
  }

  void make_meshes(u32 imageCount) {
    meshes.emplace_back(device, create_cuboid_vertices(0.2f, 4.0f, 0.2f), create_cuboid_indices());
    meshes.emplace_back(device, create_cuboid_vertices(0.2f, 2.0f, 0.2f), create_cuboid_indices());
    meshes.emplace_back(device, create_cuboid_vertices(0.2f, 1.0f, 0.2f), create_cuboid_indices());
  }

  // In App::make_scene()
  void make_scene() {
    transforms[TX_Shoulder] =
        Transform{.translation = {0.0f, 0.0f, 0.0f}, .rotation = {0.0f, 0.0f, 0.0f}};

    transforms[TX_Stick1] = Transform{
        .translation = {0.0f, 2.0f, 0.0f},      // End of first stick
        .rotation_speed = {59.0f, 45.0f, 29.0f} // Z-axis rotation
    };

    transforms[TX_Joint1] = Transform{
        .translation = {0.0f, 2.0f, 0.0f}, // End of first stick
    };

    transforms[TX_Stick2] = Transform{
        .translation = {0.0f, 1.0f, 0.0f},     // End of first stick
        .rotation_speed = {28.0f, 22.3f, 7.0f} // Z-axis rotation
    };

    transforms[TX_Joint2] = Transform{
        .translation = {0.0f, 1.0f, 0.0f}, // Half-length offset
    };

    transforms[TX_Stick3] = Transform{
        .translation = {0.0f, 0.5f, 0.0f},    // Half-length offset
        .rotation_speed = {3.0f, 5.0f, 13.0f} // Z-axis rotation
    };

    // Shoulder (root, meshless)
    auto shoulder = scene.createNode({.transform = &transforms[TX_Shoulder]});

    // Stick 1 - upper arm (4 units)
    auto stick1 = scene.createNode({.mesh = &meshes[0], // First mesh in vector
                                    .transform = &transforms[TX_Stick1],
                                    .parent = shoulder});

    // Joint 1 - controls upper arm rotation
    auto joint1 = scene.createNode({.transform = &transforms[TX_Joint1], .parent = stick1});

    // Stick 2 - forearm (2 units)
    auto stick2 = scene.createNode(
        {.mesh = &meshes[1], .transform = &transforms[TX_Stick2], .parent = joint1});

    // Joint 2 - elbow
    auto joint2 = scene.createNode({.transform = &transforms[TX_Joint2], .parent = stick2});

    // Stick 3 - hand (cube)
    auto stick3 = scene.createNode(
        {.mesh = &meshes[2], .transform = &transforms[TX_Stick3], .parent = joint2});
  }

  std::expected<void, std::string> read_shaders() {
    auto vertexShaderModule = createShaderModuleFromFile(device.logical(), "shaders/vert.spv");
    if (!vertexShaderModule) {
      return std::unexpected(vertexShaderModule.error());
    }
    auto fragmentShaderModule = createShaderModuleFromFile(device.logical(), "shaders/frag.spv");
    if (!fragmentShaderModule) {
      return std::unexpected(fragmentShaderModule.error());
    }
    shaders.emplace_back(std::move(*vertexShaderModule));
    shaders.emplace_back(std::move(*fragmentShaderModule));

    return {};
  }

  void SetupVulkan() {
    EXPECTED_VOID(instance.create());
    if (!NDEBUG)
      EXPECTED_VOID(instance.setupDebugMessenger());
    EXPECTED_VOID(device.pickPhysicalDevice());
    EXPECTED_VOID(device.createLogicalDevice());
    EXPECTED_VOID(createDescriptorSetLayout());
    EXPECTED_VOID(read_shaders());
    transforms.resize(TX_COUNT, Transform{}); // Initialize all transforms
  }

  void SetupVulkanWindow(const vk::raii::SurfaceKHR &surface, vk::Extent2D extent) {
    vk::Bool32 res = device.physical().getSurfaceSupportKHR(device.queueFamily, surface);
    if (res != true) {
      std::print("Error no WSI support on physical device 0\n");
      std::exit(-1);
    }

    std::vector<vk::Format> requestSurfaceImageFormat = {
        vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8Unorm,
        vk::Format::eR8G8B8Unorm};
    const vk::ColorSpaceKHR requestSurfaceColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    wd.config.SurfaceFormat = selectSurfaceFormat(
        device.physical(), wd.Surface, requestSurfaceImageFormat, requestSurfaceColorSpace);

#ifdef APP_USE_UNLIMITED_FRAME_RATE
    std::vector<vk::PresentModeKHR> present_modes = {vk::PresentModeKHR::eMailbox,
                                                     vk::PresentModeKHR::eFifo};
#else
    std::vector<vk::PresentModeKHR> present_modes = {vk::PresentModeKHR::eFifo};
#endif
    wd.config.PresentMode = selectPresentMode(device.physical(), wd.Surface, present_modes);

    IM_ASSERT(minImageCount >= 2);
    createOrResizeWindow(instance, device, wd, extent, minImageCount);
  }

  void CleanupVulkanWindow() {}

  void FrameRender(ImDrawData *draw_data, f32 deltaTime) {
    constexpr uint64_t timeout = std::numeric_limits<uint64_t>::max();

    auto &image_acquired_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    auto &render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;
    auto [acquireRes, imageIndex] = device.logical().acquireNextImage2KHR({
        .swapchain = wd.Swapchain,
        .timeout = timeout,
        .semaphore = image_acquired_semaphore,
        .deviceMask = 1,
    });
    if (acquireRes == vk::Result::eErrorOutOfDateKHR || acquireRes == vk::Result::eSuboptimalKHR)
      swapChainRebuild = true;
    if (acquireRes == vk::Result::eErrorOutOfDateKHR)
      return;
    if (acquireRes != vk::Result::eSuboptimalKHR) {
      if (acquireRes == vk::Result::eSuccess) {
      } else {
        std::println("error with acquiring image", vk::to_string(acquireRes));
        std::exit(1);
      }
    }
    wd.FrameIndex = imageIndex;
    auto &fd = wd.Frames[wd.FrameIndex];

    check_vk_result_hpp(device.logical().waitForFences(*fd.Fence, VK_TRUE, UINT64_MAX));

    device.logical().resetFences(*fd.Fence);

    fd.CommandPool.reset();
    fd.CommandBuffer.begin({
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    fd.CommandBuffer.beginRenderPass(
        {
            .renderPass = wd.RenderPass,
            .framebuffer = fd.Framebuffer,
            .renderArea =
                {
                    .offset = {0, 0},
                    .extent = wd.config.swapchainExtent,
                },
            .clearValueCount = 1,
            .pClearValues = &wd.config.ClearValue,
        },
        vk::SubpassContents::eInline);

    fd.CommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline.pipeline);
    fd.CommandBuffer.setViewport(0,
                                 vk::Viewport{
                                     .x = 0.0f,
                                     .y = 0.0f,
                                     .width = static_cast<float>(wd.config.swapchainExtent.width),
                                     .height = static_cast<float>(wd.config.swapchainExtent.height),
                                     .minDepth = 0.0f,
                                     .maxDepth = 1.0f,
                                 });
    fd.CommandBuffer.setScissor(0, vk::Rect2D{
                                       .offset = {0, 0},
                                       .extent = wd.config.swapchainExtent,
                                   });
    const glm::mat4 viewMatrix = camera.GetViewMatrix();
    const glm::mat4 projMatrix = camera.GetProjectionMatrix((f32)wd.config.swapchainExtent.width /
                                                            (f32)wd.config.swapchainExtent.height);
    scene.update(wd.FrameIndex, viewMatrix, projMatrix, deltaTime);
    scene.draw(fd.CommandBuffer, graphicsPipeline.pipelineLayout, wd.FrameIndex);

    ImGui_ImplVulkan_RenderDrawData(draw_data, *fd.CommandBuffer);

    fd.CommandBuffer.endRenderPass();
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo info{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*image_acquired_semaphore,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*fd.CommandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*render_complete_semaphore,
    };
    fd.CommandBuffer.end();
    device.queue.submit(info, fd.Fence);
  }

  void FramePresent(Window &wd) {
    if (swapChainRebuild)
      return;
    auto &render_complete_semaphore = wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;
    auto rnbe = device.queue.presentKHR({
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*render_complete_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &*wd.Swapchain,
        .pImageIndices = &wd.FrameIndex,
    });
    if (rnbe == vk::Result::eErrorOutOfDateKHR || rnbe == vk::Result::eSuboptimalKHR)
      swapChainRebuild = true;
    if (rnbe == vk::Result::eErrorOutOfDateKHR)
      return;
    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.FrameSemaphores.size();
  }

  vk::Extent2D get_window_size(SDL_Window *window) {
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    return {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
  }

  void ProcessKeyboard(Camera &camera, SDL_Scancode key, float deltaTime) {
    float velocity = camera.MovementSpeed * deltaTime;
    switch (key) {
    case SDL_SCANCODE_W:
      camera.Position += camera.Front * velocity;
      break;
    case SDL_SCANCODE_S:
      camera.Position -= camera.Front * velocity;
      break;
    case SDL_SCANCODE_A:
      camera.Position -= camera.Right * velocity;
      break;
    case SDL_SCANCODE_D:
      camera.Position += camera.Right * velocity;
      break;
    }
  }

  // Update camera based on keyboard state (call each frame)
  void updateCamera(Camera &camera, float deltaTime) {
    const auto keystate = SDL_GetKeyboardState(nullptr);
    if (keystate[SDL_SCANCODE_W])
      ProcessKeyboard(camera, SDL_SCANCODE_W, deltaTime);
    if (keystate[SDL_SCANCODE_S])
      ProcessKeyboard(camera, SDL_SCANCODE_S, deltaTime);
    if (keystate[SDL_SCANCODE_A])
      ProcessKeyboard(camera, SDL_SCANCODE_A, deltaTime);
    if (keystate[SDL_SCANCODE_D])
      ProcessKeyboard(camera, SDL_SCANCODE_D, deltaTime);
  }

  void renderMeshControlsMenu(std::span<Mesh> meshes, f32 framerate) {
    ImGui::Begin("mesh controls");
    for (size_t i = 0; i < meshes.size(); ++i) {
      // Unique header ID using index
      if (ImGui::CollapsingHeader(("Mesh " + std::to_string(i)).c_str())) {
        auto &transform = meshes[i].transform;

        // Add unique suffix to all widget labels using "##"
        std::string meshId = "##Mesh" + std::to_string(i);

        ImGui::SliderFloat3(("Rotation Speed (deg/s)" + meshId).c_str(),
                            &transform.rotation_speed.x, -360.0f, 360.0f, "%.1f");

        ImGui::SliderFloat3(("Rotation (deg)" + meshId).c_str(), &transform.rotation.x, -180.0f,
                            180.0f, "%.1f");

        ImGui::SliderFloat3(("Position" + meshId).c_str(), &transform.translation.x, -5.0f, 5.0f);

        ImGui::SliderFloat3(("Scale" + meshId).c_str(), &transform.scale.x, 0.1f, 5.0f);

        // Unique button IDs
        if (ImGui::Button(("Reset Rotation" + meshId).c_str())) {
          transform.rotation = glm::vec3(45.f);
        }
        if (ImGui::Button(("Reset Rotation speed" + meshId).c_str())) {
          transform.rotation_speed = glm::vec3(0.0f);
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
    ImGui::SliderFloat("Near Plane", &camera.Near, 0.01f, 10.0f);
    ImGui::SliderFloat("Far Plane", &camera.Far, 10.0f, 1000.0f);
    ImGui::SliderFloat("Move Speed", &camera.MovementSpeed, 0.1f, 10.0f);
    ImGui::SliderFloat("Mouse Sens", &camera.MouseSensitivity, 0.01f, 1.0f);
    ImGui::End();
  }

  void RenderVulkanStateWindow(float frameTime) {
    ImGui::Begin("Vulkan State Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // Physical Device Info
    if (ImGui::CollapsingHeader("Physical Device", ImGuiTreeNodeFlags_DefaultOpen)) {
      vk::PhysicalDeviceProperties props = device.physical().getProperties();
      ImGui::Text("GPU: %s", props.deviceName.data());
      ImGui::Text("API Version: %d.%d.%d", VK_VERSION_MAJOR(props.apiVersion),
                  VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
    }

    // Swapchain State
    if (ImGui::CollapsingHeader("Swapchain", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Present Mode: %s", vk::to_string(wd.config.PresentMode).c_str());
      ImGui::Text("Swapchain Images: %zu", wd.Frames.size());
      ImGui::Text("Extent: %dx%d", wd.config.swapchainExtent.width,
                  wd.config.swapchainExtent.height);
      ImGui::Text("Format: %s", vk::to_string(wd.config.SurfaceFormat.format).c_str());
      ImGui::Text("Swapchain rebuild count: %d", rebuild_counter);
    }

    // Frame Timing
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

    // Queue Family Info
    if (ImGui::CollapsingHeader("Queue")) {
      ImGui::Text("Graphics Queue Family: %u", device.queueFamily);
      auto props = device.physical().getQueueFamilyProperties()[device.queueFamily];
      ImGui::Text("Queue Count: %u", props.queueCount);
      ImGui::Text("Timestamp Valid Bits: %u", props.timestampValidBits);
    }

    // Memory Info
    if (ImGui::CollapsingHeader("Memory")) {
      auto memProps = device.physical().getMemoryProperties();
      ImGui::Text("Memory Heaps: %u", memProps.memoryHeapCount);
      ImGui::Text("Memory Types: %u", memProps.memoryTypeCount);
    }

    ImGui::End();
  }

  void mainLoop(SDL_Window *window) {
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

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
        if (event.type == SDL_EVENT_QUIT)
          done = true;
        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
            event.window.windowID == SDL_GetWindowID(window))
          done = true;
      }

      updateCamera(camera, deltaTime);

      if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
        SDL_Delay(10);
        continue;
      }

      auto extent = get_window_size(window);
      if (extent.width > 0 && extent.height > 0 &&
          (swapChainRebuild || wd.config.swapchainExtent != extent)) {
        ImGui_ImplVulkan_SetMinImageCount(minImageCount);
        createOrResizeWindow(instance, device, wd, extent, minImageCount);
        wd.FrameIndex = 0;
        wd.SemaphoreIndex = 0;
        swapChainRebuild = false;
        rebuild_counter++;
        scene.createUniformBuffers(wd.Frames.size());
        scene.updateDescriptorSets(wd.FrameIndex);
      }

      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplSDL3_NewFrame();
      ImGui::NewFrame();

      {
        RenderCameraControlMenu(camera);
        // renderMeshControlsMenu(meshes, io.Framerate);
        RenderVulkanStateWindow(deltaTime);
      }

      ImGui::Render();
      ImDrawData *draw_data = ImGui::GetDrawData();
      FrameRender(draw_data, deltaTime);
      FramePresent(wd);

      // targetFrameDuration = 1.0f / std::max(frameCap, 1);
      // float frameDuration = std::chrono::duration<float>(Clock::now() - currentTime).count();
      // if (frameDuration < targetFrameDuration) {
      //   float sleepTime = targetFrameDuration - frameDuration;
      //   std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
      // }
    }
  }

public:
  int app_run(SDL_Window *window) {
    SetupVulkan();

    VkSurfaceKHR surface_raw_handle;
    if (SDL_Vulkan_CreateSurface(window, instance.get_C_handle(), nullptr, &surface_raw_handle) ==
        0) {
      std::print("Failed to create Vulkan surface.\n {}", SDL_GetError());
      return 1;
    }
    wd.Surface = {instance, surface_raw_handle};

    SetupVulkanWindow(wd.Surface, get_window_size(window));

    EXPECTED_VOID(graphicsPipeline.createPipelineLayout(device.logical(), *descriptorSetLayout));
    EXPECTED_VOID(
        graphicsPipeline.createGraphicsPipeline(device.logical(),
                                                {
                                                    {
                                                        .stage = vk::ShaderStageFlagBits::eVertex,
                                                        .module = shaders[0],
                                                        .pName = "main",
                                                    },
                                                    {
                                                        .stage = vk::ShaderStageFlagBits::eFragment,
                                                        .module = shaders[1],
                                                        .pName = "main",
                                                    },
                                                },
                                                wd.RenderPass));
    make_meshes(wd.Frames.size());
    make_scene();
    EXPECTED_VOID(device.createDescriptorPool(wd.Frames.size() * scene.nodes.size()));
    scene.createUniformBuffers(wd.Frames.size());
    scene.allocateDescriptorSets(device.descriptorPool, descriptorSetLayout, wd.Frames.size());
    for (u32 p = 0; p < wd.Frames.size(); ++p)
      scene.updateDescriptorSets(p);

    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo init_info = device.init_info();
    init_info.Instance = instance.get_C_handle();
    init_info.RenderPass = *wd.RenderPass;
    init_info.MinImageCount = minImageCount;
    init_info.ImageCount = wd.Frames.size();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.PipelineCache = *pipelineCache;
    init_info.Subpass = 0;
    init_info.CheckVkResultFn = check_vk_result;

    ImGui_ImplVulkan_Init(&init_info);

    mainLoop(window);

    device.logical().waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

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
      std::print("Error: SDL_Init(): %s\n", SDL_GetError());
      return -1;
    }

    SDL_WindowFlags window_flags =
        (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY |
                          SDL_WINDOW_HIDDEN);
    window = SDL_CreateWindow("Dear ImGui SDL3+Vulkan example", 1280, 720, window_flags);
    if (window == nullptr) {
      std::print("Error: SDL_CreateWindow(): %s\n", SDL_GetError());
      return -1;
    }
    return 0;
  }

  void terminate() {
    SDL_DestroyWindow(window);
    SDL_PumpEvents();
    SDL_Quit();
  }
};
