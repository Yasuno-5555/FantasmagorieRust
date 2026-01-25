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
    let f_dims = vec2<f32>(dims);
    let uv = vec2<f32>(coords) / f_dims;
    
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    // 1. Audio-Driven Chromatic Aberration
    let shift_amount = i32(audio.high * 20.0);
    let r_coord = coords + vec2<i32>(shift_amount, 0);
    let b_coord = coords - vec2<i32>(shift_amount, 0);
    
    let r = textureLoad(t_base_color, r_coord, 0).r;
    let g = textureLoad(t_base_color, coords, 0).g;
    let b = textureLoad(t_base_color, b_coord, 0).b;
    let color_raw = vec3<f32>(r, g, b);
    
    // 2. Load SDF
    let dist = textureLoad(t_sdf, coords, 0).r;

    // 3. Compute Direct Lighting (Bass Driven Pulse)
    let base_decay = 0.05;
    let pulse_decay = base_decay / (1.0 + audio.bass * 2.0);
    let pulse_intensity = 2.0 * (1.0 + audio.bass * 4.0);
    
    let direct_light = exp(-abs(dist) * pulse_decay) * pulse_intensity; 

    // 4. Volumetric Fog (Raymarching towards center)
    var fog_accum = 0.0;
    let light_pos = f_dims * 0.5; // Light at center
    let ray_start = vec2<f32>(coords);
    let ray_dir = normalize(light_pos - ray_start);
    let ray_len = distance(ray_start, light_pos);
    
    // Dithering to prevent banding
    let dither = ign(vec2<f32>(coords));
    
    let steps = 16.0;
    let step_size = ray_len / steps;
    
    for (var i = 0.0; i < steps; i += 1.0) {
        let p = ray_start + ray_dir * (i + dither) * step_size;
        let p_i = vec2<i32>(p);
        
        if (p_i.x < 0 || p_i.x >= i32(dims.x) || p_i.y < 0 || p_i.y >= i32(dims.y)) {
             continue;
        }

        let s_dist = textureLoad(t_sdf, p_i, 0).r;
        let s_light = exp(-abs(s_dist) * (pulse_decay * 1.5)) * pulse_intensity;
        
        // Attenuate based on march distance
        let falloff = exp(-i * step_size * 0.001); 
        fog_accum += s_light * falloff;
    }
    
    let fog_final = (fog_accum / steps) * pc.fog_density * 5.0;

    // 5. Composite
    var final_color = color_raw + vec3<f32>(direct_light * 0.2) + vec3<f32>(fog_final);

    // 6. Tone Mapping & Gamma
    final_color = aces_film(final_color * pc.exposure);
    final_color = pow(final_color, vec3<f32>(1.0 / pc.gamma));

    // 7. Store
    textureStore(t_output, coords, vec4<f32>(final_color, 1.0));
}
