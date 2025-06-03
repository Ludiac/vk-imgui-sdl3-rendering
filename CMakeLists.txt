cmake_minimum_required(VERSION 3.30)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
set(CMAKE_CXX_MODULE_STD ON)

project(imgui_sdl3_vulkan_example)

find_package(SDL3 REQUIRED)
find_package(Vulkan REQUIRED)

set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/.cpm_cache")

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
         ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME imgui
  GITHUB_REPOSITORY ocornut/imgui
  GIT_TAG v1.91.8 # Consider using the latest release or a specific commit
  DOWNLOAD_ONLY YES # imgui is added as source files directly
)

CPMAddPackage(
    NAME tinygltf
    GITHUB_REPOSITORY syoyo/tinygltf
    VERSION 2.9.6 # Consider using the latest release or a specific commit
)

CPMAddPackage(
    NAME stb
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG 5c205738c191bcb0abc65c4febfa9bd25ff35234 # This is a specific commit for stb_image.h etc.
    # For header-only libraries like stb, DOWNLOAD_ONLY might be an option
    # if you manually manage include directories. However, if it has a CMakeLists.txt
    # that defines a target, it's better to let CPM handle it.
    # Given stb is typically used as individual headers, often DOWNLOAD_ONLY is fine
    # if you then add stb_SOURCE_DIR to include_directories.
    # For simplicity if it doesn't have a good CMake target, DOWNLOAD_ONLY YES
    # and then target_include_directories(AppModule PRIVATE ${stb_SOURCE_DIR})
    # Let's assume for now it's fine as is, or it provides a target.
)

# Add VulkanMemoryAllocator-Hpp
CPMAddPackage(
  NAME VulkanMemoryAllocatorHpp # Name for CPM to track the package
  GITHUB_REPOSITORY YaaZ/VulkanMemoryAllocator-Hpp
  GIT_TAG v3.2.1 # Use a specific version tag, v3.0.1 is a stable release
  # This package provides an interface library target 'VulkanMemoryAllocator-hpp'
)

add_library(AppModule)
target_compile_features(AppModule PRIVATE cxx_std_23)
target_compile_definitions(AppModule PRIVATE
    VULKAN_HPP_NO_SMART_HANDLE
    VULKAN_HPP_NO_EXCEPTIONS
    VULKAN_HPP_NO_CONSTRUCTORS
    # VULKAN_HPP_STD_MODULE std
    # VULKAN_HPP_ENABLE_STD_MODULE
    VK_USE_PLATFORM_WAYLAND_KHR
    VMA_BUILD_CXX_MODULE
    SDL_MAIN_HANDLED  # Prevent SDL from overriding main()
    # VMA_ENABLE_STD_MODULE
    # APP_USE_UNLIMITED_FRAME_RATE
)

target_sources(AppModule
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${Vulkan_INCLUDE_DIR} src ${VulkanMemoryAllocatorHpp_SOURCE_DIR}
        FILES
        ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
        ${VulkanMemoryAllocatorHpp_SOURCE_DIR}/src/vk_mem_alloc.cppm
        src/tinygltf_module_wrapper.cppm
        src/app.cppm
        src/extra.cppm
        src/VulkanWindow.cppm
        src/VulkanDevice.cppm
        src/VulkanInstance.cppm
        src/VulkanPipeline.cppm
        src/ModelLoader.cppm
        src/types.cppm
        src/memory_types.cppm
        src/mesh.cppm
        src/scene.cppm
        src/SceneBuilder.cppm
        src/texture.cppm
        src/TextureStore.cppm
        src/imgui.cppm
        src/Logger.cppm
)

target_sources(AppModule
    PRIVATE
        src/implementations.cpp
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(AppModule PRIVATE
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${tinygltf_SOURCE_DIR}
  ${SDL3_INCLUDE_DIRS}
  # VulkanMemoryAllocator-hpp is an interface library, so its include directories
  # will be propagated automatically when linking.
  # If VulkanMemoryAllocatorHpp_SOURCE_DIR was needed (e.g. for DOWNLOAD_ONLY):
  ${VulkanMemoryAllocatorHpp_SOURCE_DIR}/VulkanMemoryAllocator/include
  ${VulkanMemoryAllocatorHpp_SOURCE_DIR}/include
)

target_link_libraries(AppModule PRIVATE
    SDL3::SDL3
    Vulkan::Vulkan
    tinygltf
    # GPUOpen::VulkanMemoryAllocator
    # VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
    # VulkanMemoryAllocator-Hpp-Module
)

add_executable(${PROJECT_NAME}
  src/main.cpp
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)

target_link_libraries(${PROJECT_NAME} PRIVATE AppModule)

set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set(VERT_SPV ${SHADER_OUTPUT_DIR}/vert.spv)
set(FRAG_SPV ${SHADER_OUTPUT_DIR}/frag.spv)

# Create the output directory if it doesn't exist
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

add_custom_command(
    OUTPUT ${VERT_SPV} ${FRAG_SPV}
    COMMAND slangc -o ${VERT_SPV}
                   -profile glsl_460
                   -entry main
                   -stage vertex
# -capability spvSparseResidency
                   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vert.slang
    COMMAND slangc -o ${FRAG_SPV}
                   -profile glsl_460
                   -entry main
                   -stage fragment
# -capability spvSparseResidency # Note: This was spvSparseResidency, ensure it's correct for frag if needed
                   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/frag.slang
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vert.slang
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/frag.slang
    COMMENT "Compiling Slang shaders to SPIR-V"
)

add_custom_target(CompileShaders ALL
    DEPENDS ${VERT_SPV} ${FRAG_SPV}
)

add_dependencies(${PROJECT_NAME} CompileShaders)
