struct Material {
    float4 baseColorFactor; 
};

[[vk::binding(1, 0)]]
ConstantBuffer<Material> material;

struct FragmentOutput {
    float4 color : SV_Target;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 worldNormal : NORMAL;  
    float2 uv : TEXCOORD0;        
};

FragmentOutput main() {
    FragmentOutput output;
    output.color = material.baseColorFactor; // Direct from uniform
    return output;
}
