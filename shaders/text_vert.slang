struct TextPushConstants2D {
    float4x4 projection; // 64 bytes
    float sdf_weight;
    uint anti_aliasing_enabled; // 4 bytes. Use uint for boolean to ensure consistent size across platforms.
    float _padding[14];     // 14 floats * 4 bytes/float = 56 bytes
                            // Total size: 64 + 4 + 4 + 56 = 128 bytes
};

// This matches the simple TextQuadVertex format
struct VertexInput {
    float2 pos : POSITION;
    float2 uv  : TEXCOORD;
};

// This is the unique data for each character instance
struct TextInstanceData {
    float2 screenPos;
    float2 size;      // NEW: The original glyph size in the atlas
    float2 uvTopLeft;
    float2 uvBottomRight;
    float4 color;
    float3 _padding;
    float pxRange;
};

// The output sent to the fragment shader
struct VertexOutput {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0; // This will now carry the instance color
    float pxRange   : TEXCOORD1;
};

[[vk::binding(0, 0)]]
StructuredBuffer<TextInstanceData> instanceBuffer;

[[vk::push_constant]]
TextPushConstants2D push_constants;

VertexOutput main(VertexInput input, uint instanceID : SV_InstanceID) {
    VertexOutput output;
    
    TextInstanceData instance = instanceBuffer[instanceID];
    // Calculate vertex position based on actual size
    float2 pos = input.pos * instance.size + instance.screenPos;
    
    // Transform to clip space
    output.position = mul(push_constants.projection, float4(pos, 0.0, 1.0));
    
    // Calculate UV coordinates
    float2 uvRange = instance.uvBottomRight - instance.uvTopLeft;
    output.uv = instance.uvTopLeft + input.uv * uvRange;
    
    output.color = instance.color;
    output.pxRange = instance.pxRange;
    return output;
}
