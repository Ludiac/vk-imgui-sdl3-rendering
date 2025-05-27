struct Material {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float occlusionStrength;
    float3 emissiveFactor;
    float normalScale;
    float heightScale;
};

// Binding 1, Set 0: Material Uniform Buffer Object
[[vk::binding(1, 0)]] // Set 0, Binding 1
ConstantBuffer<Material> material;

[[vk::binding(2, 0)]]
Sampler2D baseColorCombined;

struct FragmentInput { // Corresponds to VertexOutput
    float4 position     : SV_Position;
    float3 worldNormal  : NORMAL;
    float2 uv           : TEXCOORD0;
};

struct FragmentOutput {
    float4 color : SV_Target;
};

FragmentOutput main(FragmentInput input) {
    FragmentOutput output;
    output.color = material.baseColorFactor * 
        baseColorCombined.Sample(input.uv);
    return output;
}
