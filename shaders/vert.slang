struct UniformBufferObject {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
};

// Binding 0, Set 0: MVP Uniform Buffer Object (Vertex Shader)
[[vk::binding(0, 0)]] // Corresponds to Set 0, Binding 0
ConstantBuffer<UniformBufferObject> ubo;

struct VertexInput {
    float3 position  : POSITION;
    float3 normal    : NORMAL;
    float2 uv        : TEXCOORD0;
    float4 tangent   : TANGENT; // Included if used for lighting, not used in current output
};

struct VertexOutput { // This struct will be the input to the Fragment Shader
    float4 position     : SV_Position;
    float3 worldNormal  : NORMAL;
    float2 uv           : TEXCOORD0;
};

VertexOutput main(VertexInput input) {
  VertexOutput output;
  float4 worldPos = mul(ubo.model, float4(input.position, 1.0));
  // float4 viewPos = mul(ubo.view, worldPos); // This line was in the original user prompt but commented out in the thought process. Re-adding as it's standard.
  output.position = mul(ubo.projection, mul(ubo.view, worldPos)); // Combined view and projection
  output.worldNormal = mul((float3x3)ubo.model, input.normal); // More robust normal transformation
  output.uv = input.uv;
  return output;
}
