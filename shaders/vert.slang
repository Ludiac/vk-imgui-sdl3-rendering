struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float3x3 normalMatrix;
};

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
    output.position = mul(ubo.projection, mul(ubo.view, worldPos));
    output.worldNormal = mul(ubo.normalMatrix, input.normal);
    output.uv = input.uv;
    return output;
}
