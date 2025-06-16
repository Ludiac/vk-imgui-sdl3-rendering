struct PushConstants {
    float4x4 projection;
    // Color is no longer here!
};

// This matches the simple TextQuadVertex format
struct VertexInput {
    float2 pos : POSITION;
    float2 uv  : TEXCOORD;
};

// This is the unique data for each character instance
struct TextInstanceData {
  float2 screenPos;
  float2 scale;
  float2 uvTopLeft;
  float2 uvBottomRight;
  float4 color; // NEW: Color is now per-instance
};

// The output sent to the fragment shader
struct VertexOutput {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0; // This will now carry the instance color
};


// BINDINGS
// Set 0, Binding 0: The font atlas texture (used in fragment shader)
// Set 1, Binding 0: The storage buffer with our instance data
[[vk::binding(0, 1)]]
StructuredBuffer<TextInstanceData> instanceBuffer;

[[vk::push_constant]]
PushConstants push_constants;


VertexOutput main(VertexInput input, uint instanceID : SV_InstanceID) {
    VertexOutput output;
    
    // 1. Fetch the data for the current instance (character).
    TextInstanceData instance = instanceBuffer[instanceID];
    
    // 2. Calculate the final vertex position.
    float2 finalPos = instance.screenPos + (input.pos * instance.scale);
    
    // 3. Project into clip space.
    output.position = mul(push_constants.projection, float4(finalPos, 0.0, 1.0));
    
    // 4. Interpolate UVs.
    output.uv = lerp(instance.uvTopLeft, instance.uvBottomRight, input.uv);

    // 5. Pass the INSTANCE color to the fragment shader.
    output.color = instance.color;
    
    return output;
}
