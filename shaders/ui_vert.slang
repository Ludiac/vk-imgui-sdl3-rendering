// This matches our simple UI vertex format
struct VertexInput {
    float2 pos   : POSITION;
    float4 color : COLOR0;
};

// Data sent to the fragment shader
struct VertexOutput {
    float4 position : SV_Position;
    float4 color    : COLOR0;
};

// We only need a projection matrix for UI
struct PushConstants {
    float4x4 projection;
};

[[vk::push_constant]]
PushConstants push_constants;

VertexOutput main(VertexInput input) {
    VertexOutput output;
    output.position = mul(push_constants.projection, float4(input.pos, 0.0, 1.0));
    output.color = input.color;
    return output;
}
