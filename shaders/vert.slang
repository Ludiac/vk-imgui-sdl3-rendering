struct UniformBufferObject {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;  // Binding 0

struct VertexInput {
    float3 position : POSITION;
    float3 normal : NORMAL;     
    float2 uv : TEXCOORD0;
    float4 tangent : TANGENT;     
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 worldNormal : NORMAL;  
    float2 uv : TEXCOORD0;        
};

VertexOutput main(VertexInput input) {
  VertexOutput output;
  float4 worldPos = mul(ubo.model, float4(input.position, 1.0));
  float4 viewPos = mul(ubo.view, worldPos);
  output.position = mul(ubo.projection, viewPos);
  output.worldNormal = mul(ubo.model, float4(input.normal, 0.0)).xyz;
  output.uv = input.uv;
  return output;
}
