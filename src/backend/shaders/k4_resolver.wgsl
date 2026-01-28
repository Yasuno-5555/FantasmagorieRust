// K4: Cinematic Resolver
// Fused kernel combining SDF lighting, temporal history, and tone mapping.

struct K4PushConstants {
    exposure: f32,
    gamma: f32,
    fog_density: f32,
    _pad: f32,
}
var<push_constant> pc: K4PushConstants;

// Bindings must match backend.rs k4_bindings
@group(0) @binding(1) var t_base_color: texture_2d<f32>;
@group(0) @binding(2) var t_sdf: texture_2d<f32>;
@group(0) @binding(3) var t_history: texture_2d<f32>; // Unused for now
@group(0) @binding(4) var s_linear: sampler;
@group(0) @binding(5) var t_output: texture_storage_2d<rgba8unorm, write>;

struct AudioParams {
    bass: f32,
    mid: f32,
    high: f32,
    _pad: f32,
}
@group(0) @binding(6) var<storage, read> audio: AudioParams;

// ACES Tone Mapping
fn aces_film(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Interleaved Gradient Noise for Dithering
fn ign(v: vec2<f32>) -> f32 {
    let f = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(f.z * fract(dot(v, f.xy)));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let coords = vec2<i32>(id.xy);
    let dims = textureDimensions(t_output);
    
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    // Load base color (HDR Linear)
    var color = textureLoad(t_base_color, coords, 0).rgb;
    
    // 1. Exposure adjustment
    color *= pc.exposure;

    // 2. Simple Cinematic Fog
    // Masked by SDF to keep foreground sharp
    let sdf_val = textureLoad(t_sdf, coords, 0).r;
    let fog_factor = 1.0 - exp(-(sdf_val * 0.1) * pc.fog_density);
    let fog_color = vec3<f32>(0.1, 0.1, 0.12);
    color = mix(color, fog_color, clamp(fog_factor, 0.0, 0.8));

    // 3. Audio Reactivity (Bass-driven bloom/glow)
    color += audio.bass * vec3<f32>(0.05, 0.02, 0.08) * max(0.0, 1.0 - sdf_val * 0.05);

    // 4. ACES Film Tone Mapping
    var final_color = aces_film(color);

    // 5. Gamma Correction (Linear -> sRGB approximation)
    final_color = pow(final_color, vec3<f32>(1.0 / pc.gamma));

    // 6. Interleaved Gradient Noise Dithering (fixes banding in dark areas)
    let noise = (ign(vec2<f32>(coords)) - 0.5) / 255.0;
    final_color += noise;

    textureStore(t_output, coords, vec4<f32>(final_color, 1.0));
}
