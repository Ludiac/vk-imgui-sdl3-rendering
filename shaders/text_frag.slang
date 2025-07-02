struct TextPushConstants2D {
    float4x4 projection; // 64 bytes
    float sdf_weight;
    uint anti_aliasing_enabled; // 4 bytes. Use uint for boolean to ensure consistent size across platforms.
    float _padding[14];     // 14 floats * 4 bytes/float = 56 bytes
                            // Total size: 64 + 4 + 4 + 56 = 128 bytes
};

struct VertexToFragment {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
    float  pxRange  : TEXCOORD1;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[[vk::push_constant]]
TextPushConstants2D push_constants;

[[vk::binding(0, 1)]]
Sampler2D fontAtlas;

// Function to find the median of three values, used for MSDF.
float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

FragmentOutput main(VertexToFragment input) {
    FragmentOutput output;

    // 1. Sample the 4-channel MTSDF texture
    float4 texSample = fontAtlas.Sample(input.uv);

    // 2. Get the signed distance from the MSDF (in RGB) and the true SDF (in Alpha)
    // The texture values are in the [0, 1] range, where 0.5 is the contour edge.
    float sd = median(texSample.r, texSample.g, texSample.b);
    float sdf = texSample.a;

    // 3. Mix the MSDF and SDF values. This uses the MSDF by default but
    // falls back to the true SDF in areas where the MSDF is not well-defined (e.g., inside the glyph).
    // This gives the best of both: sharp corners from MSDF and solid fills from SDF.
    float msd = lerp(sd, sdf, push_constants.sdf_weight);

    // 4. Convert the combined distance field value to screen-space pixel distance.
    float screenPxDistance = input.pxRange * (msd - 0.5);

    float opacity;

    // Anti-aliasing toggling
    if (push_constants.anti_aliasing_enabled == 1) {
        // 5. Calculate the screen-space gradient of the distance field for anti-aliasing.
        // This determines how "blurry" the edge should be.
        float screenPxGradient = fwidth(screenPxDistance);

        // 6. Compute anti-aliased opacity using a smooth step.
        // This creates a smooth transition from transparent to opaque over the width of one pixel.
        opacity = smoothstep(-screenPxGradient, screenPxGradient, screenPxDistance);
    } else {
        // No anti-aliasing: sharp cutoff
        opacity = step(0.0, screenPxDistance); // If screenPxDistance > 0, opacity is 1, else 0.
    }

    // 7. Apply the final color with the computed opacity.
    output.color = float4(input.color.rgb, input.color.a * opacity);

    return output;
}
