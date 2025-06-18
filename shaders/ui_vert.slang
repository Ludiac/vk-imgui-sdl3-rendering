struct PushConstants {
    float4x4 projection;
};

// Matches the static unit quad vertex format
struct VertexInput {
    float2 pos : POSITION;
    float2 uv  : TEXCOORD0; // Pass UVs for future SDF calculations
};

// Per-instance data coming from the SSBO
struct UIInstanceData {
  float2 screenPos;
  float2 scale;
  float4 color;
  float z_layer;
  // Future params for SDFs: float cornerRadius, float borderWidth, etc.
};

// Output to the fragment shader
struct VertexOutput {
    float4 position : SV_Position;
    float4 color    : COLOR0;
    float2 uv       : TEXCOORD0; // Pass UVs through
};

[[vk::push_constant]]
PushConstants push_constants;

// The SSBO with our instance data.
// We'll bind this at Set 0, Binding 0 for the UI pipeline.
[[vk::binding(0, 0)]]
StructuredBuffer<UIInstanceData> instanceBuffer;

VertexOutput main(VertexInput input, uint instanceID : SV_InstanceID) {
    VertexOutput output;

    // 1. Fetch data for this specific quad instance.
    UIInstanceData instance = instanceBuffer[instanceID];
    
    // 2. Calculate the final vertex position on screen.
    float2 finalPos = instance.screenPos + (input.pos * instance.scale);
    
    // 3. Project into clip space, using the instance's z_layer for depth.
    output.position = mul(push_constants.projection, float4(finalPos, instance.z_layer, 1.0));
    
    // 4. Pass color and UVs to the fragment shader.
    output.color = instance.color;
    output.uv = input.uv; // UVs are crucial for SDFs later
    
    return output;
}
