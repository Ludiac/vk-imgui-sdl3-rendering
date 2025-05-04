cmake_minimum_required(VERSION 3.30)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
set(CMAKE_CXX_MODULE_STD ON)

project(imgui_sdl3_vulkan_example)

find_package(SDL3 REQUIRED)
find_package(Vulkan REQUIRED)

set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/.cpm_cache")

file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
     ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME imgui
  GITHUB_REPOSITORY ocornut/imgui
  GIT_TAG v1.91.8
  DOWNLOAD_ONLY YES
)

add_library(AppModule)
target_compile_features(AppModule PRIVATE cxx_std_23)
target_compile_definitions(AppModule PRIVATE
    VULKAN_HPP_NO_SMART_HANDLE
    VULKAN_HPP_NO_EXCEPTIONS
    VULKAN_HPP_NO_CONSTRUCTORS
    VK_USE_PLATFORM_WAYLAND_KHR
    SDL_MAIN_HANDLED  # Prevent SDL from overriding main()
    # APP_USE_UNLIMITED_FRAME_RATE
)

target_sources(AppModule
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${Vulkan_INCLUDE_DIR} src
        FILES
        ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
        src/app.cppm
        src/extra.cppm
        src/my_helper_impl.cppm
        src/VulkanDevice.cppm
        src/VulkanInstance.cppm
        src/mesh.cppm
)

target_sources(AppModule
    PRIVATE
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
  ${SDL3_INCLUDE_DIRS} 
)

target_link_libraries(AppModule PRIVATE
    SDL3::SDL3
    Vulkan::Vulkan
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
                   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vert.slang
    COMMAND slangc -o ${FRAG_SPV}
                   -profile glsl_460
                   -entry main
                   -stage fragment
                   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/frag.slang
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vert.slang
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/frag.slang
    COMMENT "Compiling Slang shaders to SPIR-V"
)

add_custom_target(CompileShaders ALL
    DEPENDS ${VERT_SPV} ${FRAG_SPV}
)

add_dependencies(${PROJECT_NAME} CompileShaders)
