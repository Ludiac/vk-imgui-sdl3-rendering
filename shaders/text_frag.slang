struct FragmentInput {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
};

struct FragmentOutput {
    float4 color : SV_Target;
};

// The font atlas sampler
[[vk::binding(0, 1)]]
Sampler2D fontAtlas;

FragmentOutput main(FragmentInput input) {
    FragmentOutput output;
    
    // The font atlas is a single-channel (grayscale) texture.
    // The 'r' component contains the alpha value.
    float alpha = fontAtlas.Sample(input.uv).r;
    
    // The final color is the vertex color, with its alpha multiplied by the texture's alpha.
    output.color = float4(input.color.rgb, input.color.a * alpha);
    
    return output;
}
