module;

#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:VulkanPipeline;

import vulkan_hpp;
import std;
import :DDX;
import :VulkanDevice;
import :extra;

std::expected<std::vector<uint32_t>, std::string>
readSpirvFile(const std::string &filename) NOEXCEPT {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    return std::unexpected("Failed to open file: " + filename);
  }

  std::size_t fileSize = file.tellg();
  std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

  file.seekg(0);
  file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
  file.close();

  return buffer;
}

std::expected<vk::raii::ShaderModule, std::string>
createShaderModule(const vk::raii::Device &device,
                   const std::vector<uint32_t> &spirvCode) NOEXCEPT {
  if (auto expected = device.createShaderModule({
          .codeSize = spirvCode.size() * sizeof(uint32_t),
          .pCode = spirvCode.data(),
      });
      expected) {
    return std::move(*expected);
  } else {
    return std::unexpected("Failed to create shader module: " + vk::to_string(expected.error()));
  }
}

export std::expected<vk::raii::ShaderModule, std::string>
createShaderModuleFromFile(const vk::raii::Device &device, const std::string &filename) NOEXCEPT {
  auto spirvCode = readSpirvFile(filename); // Use the passed filename, not "shaders/vert.spv"
  if (!spirvCode) {
    return std::unexpected(spirvCode.error());
  }

  if (auto expected = device.createShaderModule({
          .codeSize = spirvCode->size() * sizeof(uint32_t),
          .pCode = spirvCode->data(),
      });
      expected) {
    return std::move(*expected);
  } else {
    return std::unexpected("Failed to create shader module: " + vk::to_string(expected.error()));
  }
}

export struct VulkanPipeline {
  vk::raii::PipelineLayout pipelineLayout{nullptr};
  vk::raii::Pipeline pipeline{nullptr};

  std::expected<void, std::string>
  createPipelineLayout(const vk::raii::Device &device,
                       const vk::DescriptorSetLayout &descriptorSetLayout) NOEXCEPT {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    if (auto expected = device.createPipelineLayout(pipelineLayoutInfo); expected) {
      pipelineLayout = std::move(*expected);
    } else {
      return std::unexpected("Failed to create pipeline layout: " +
                             vk::to_string(expected.error()));
    }

    return {};
  }

  std::expected<void, std::string>
  createGraphicsPipeline(const vk::raii::Device &device,
                         std::vector<vk::PipelineShaderStageCreateInfo> shaderStages,
                         const vk::raii::RenderPass &renderPass) NOEXCEPT {
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
        .vertexAttributeDescriptionCount = 4,
        .pVertexAttributeDescriptions = attributes.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = false,
    };

    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{};

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .stageCount = static_cast<u32>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = *pipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
    };

    if (auto expected = device.createGraphicsPipeline(nullptr, pipelineInfo); expected) {
      pipeline = std::move(*expected);
      std::println("Graphics pipeline created successfully!");
      return {};
    } else {
      return std::unexpected("Failed to create graphics pipeline: " +
                             vk::to_string(expected.error()));
    }
  }
};
