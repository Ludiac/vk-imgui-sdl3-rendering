module;

#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:VulkanPipeline;

import :Types;
import vulkan_hpp;
import std;

[[nodiscard]] std::expected<std::vector<u32>, std::string>
readSpirvFile(const std::string &filename) NOEXCEPT {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    return std::unexpected("Failed to open file: " + filename);
  }

  std::size_t fileSize = file.tellg();
  if (fileSize == 0) {
    file.close();
    return std::unexpected("File is empty: " + filename);
  }
  std::vector<u32> buffer(fileSize / sizeof(u32));

  file.seekg(0);
  file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
  file.close();

  return buffer;
}

[[nodiscard]] std::expected<vk::raii::ShaderModule, std::string>
createShaderModule(const vk::raii::Device &device, const std::vector<u32> &spirvCode) NOEXCEPT {
  if (spirvCode.empty()) {
    return std::unexpected("Cannot create shader module from empty SPIR-V code.");
  }
  auto createInfo = vk::ShaderModuleCreateInfo{
      .codeSize = spirvCode.size() * sizeof(u32),
      .pCode = spirvCode.data(),
  };
  auto shaderModuleResult = device.createShaderModule(createInfo);
  if (!shaderModuleResult) {
    return std::unexpected("Failed to create shader module: " +
                           vk::to_string(shaderModuleResult.error()));
  }
  return std::move(shaderModuleResult.value());
}

export [[nodiscard]] std::expected<vk::raii::ShaderModule, std::string>
createShaderModuleFromFile(const vk::raii::Device &device, const std::string &filename) NOEXCEPT {
  auto spirvCodeResult = readSpirvFile(filename);
  if (!spirvCodeResult) {
    return std::unexpected(spirvCodeResult.error());
  }

  return createShaderModule(device, *spirvCodeResult);
}

export struct VulkanPipeline {
  vk::raii::PipelineLayout pipelineLayout{nullptr};
  vk::raii::Pipeline pipeline{nullptr};

  [[nodiscard]] std::expected<void, std::string>
  createPipelineLayout(const vk::raii::Device &device,
                       std::span<vk::DescriptorSetLayout> descriptorSetLayouts) NOEXCEPT {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<u32>(descriptorSetLayouts.size()),
        .pSetLayouts = descriptorSetLayouts.data(),
    };

    auto layoutResult = device.createPipelineLayout(pipelineLayoutInfo);
    if (!layoutResult) {
      return std::unexpected("Failed to create pipeline layout: " +
                             vk::to_string(layoutResult.error()));
    }
    pipelineLayout = std::move(layoutResult.value());
    return {};
  }

  [[nodiscard]] std::expected<void, std::string>
  createGraphicsPipeline(const vk::raii::Device &device,
                         const vk::raii::PipelineCache &pipelineCache,
                         std::vector<vk::PipelineShaderStageCreateInfo> shaderStages,
                         vk::PipelineInputAssemblyStateCreateInfo inputAssembly,
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
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = false,        // Usually false
        .rasterizerDiscardEnable = false, // Usually false
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eClockwise,

        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1, // No MSAA
        .sampleShadingEnable = false,
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

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<u32>(dynamicStates.size()),
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
        .pDepthStencilState = &depthStencilState,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = *pipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
    };

    auto pipelineResult = device.createGraphicsPipeline(pipelineCache, pipelineInfo);

    if (!pipelineResult) {
      return std::unexpected("Failed to create graphics pipeline: " +
                             vk::to_string(pipelineResult.error()));
    }
    pipeline = std::move(pipelineResult.value());
    std::println("Graphics pipeline created successfully!");
    return {};
  }
};
