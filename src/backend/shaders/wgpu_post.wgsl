struct CinematicParams {
    exposure: f32,
    ca_strength: f32,
    vignette_intensity: f32,
    bloom_intensity: f32,
    tonemap_mode: u32,
    bloom_mode: u32,
    grain_strength: f32,
    time: f32,
    lut_intensity: f32,
    blur_radius: f32,
    motion_blur_strength: f32,
    debug_mode: u32,
    light_pos: vec2<f32>,
    gi_intensity: f32,
    volumetric_intensity: f32,
    light_color: vec4<f32>,
};

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var s_hdr: sampler;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(3) var<uniform> cinema: CinematicParams;
@group(0) @binding(4) var t_velocity: texture_2d<f32>;
@group(0) @binding(5) var t_reflection: texture_2d<f32>;
@group(0) @binding(6) var t_aux: texture_2d<f32>;   // Normal xy
@group(0) @binding(7) var t_extra: texture_2d<f32>; // Distortion params
@group(0) @binding(8) var t_sdf: texture_2d<f32>;   // Signed Distance Field
@group(0) @binding(9) var t_lut: texture_3d<f32>;   // 3D Color LUT

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
// ... existing vs_main ...

// Helper for LUT sampling
fn sample_lut(color: vec3<f32>) -> vec3<f32> {
    let lut_size = vec3<f32>(textureDimensions(t_lut));
    // ACES/Log to LUT space (0-1)
    // Assuming Standard 0-1 range for now.
    // 3D Texture sampling needs careful UV mapping (0.5/size offset usually handled by sampler/normalized coords)
    // wgpu texture_3d sample takes vec3 normalized coords.
    
    let uvw = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSample(t_lut, s_hdr, uvw).rgb;
}
    var out: VertexOutput;
    // Full screen triangle: (0, -1, 1), (0, 1, 1), (0, 0, -3)? No, standard is:
    // x: -1, 3, -1
    // y: -1, -1, 3
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn aces_approx(v: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn raymarch_shadow(pixel_pos: vec2<f32>, light_pos: vec2<f32>) -> f32 {
    let dir = normalize(light_pos - pixel_pos);
    let max_dist = distance(pixel_pos, light_pos);
    var t = 2.0; // Start offset
    var res = 1.0;
    
    // Size of the world in pixels (assume 1920x1080 for resolution independent scale if needed, or query)
    let size = vec2<f32>(textureDimensions(t_sdf));
    
    for (var i = 0; i < 32; i++) {
        let p = pixel_pos + dir * t;
        let uv = p / size;
        
        // Sample SDF (x channel stores distance)
        let d = textureSampleLevel(t_sdf, s_hdr, uv, 0.0).x;
        
        if (d < 0.1) {
            return 0.0;
        }
        
        // Soft shadow formula: min(d / t * k)
        res = min(res, 8.0 * d / t);
        
        t += max(d, 1.0);
        if (t >= max_dist) { break; }
    }
    return clamp(res, 0.0, 1.0);
}

fn volumetric_lighting(pixel_pos: vec2<f32>, light_posd: vec2<f32>) -> vec3<f32> {
    let dir = normalize(light_posd - pixel_pos);
    let max_dist = distance(pixel_pos, light_posd);
    var t = 0.0;
    var accumulation = 0.0;
    let step_size = 20.0; // Larger steps for performance, maybe dithering needed
    let density = 0.002;
    
    // Dithering start offset
    let dither = fract(sin(dot(pixel_pos, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    t += dither * step_size;

    let size = vec2<f32>(textureDimensions(t_sdf));

    for (var i = 0; i < 16; i++) {
        if (t >= max_dist) { break; }
        
        let p = pixel_pos + dir * t;
        let uv = p / size;
        
        // Check if point is in shadow (simple point lookup in SDF)
        let d = textureSampleLevel(t_sdf, s_hdr, uv, 0.0).x;
        
        if (d > 0.1) {
            // Not occluded (or at least close to surface), accumulate light
            // Distance falloff from light
            let dist_to_light = max_dist - t;
            let falloff = 1.0 / (1.0 + 0.00005 * dist_to_light * dist_to_light);
            accumulation += density * falloff;
        } else {
             // In shadow/solid, absorb light? 
             // For simple god rays, we just don't accumulate
             // Optionally, if we hit solid, we could stop, but god rays pass through "empty" space
        }
        
        t += step_size;
    }
    
    return cinema.light_color.rgb * accumulation * cinema.volumetric_intensity;
}

fn cone_trace_gi(pixel_pos: vec2<f32>, normal: vec2<f32>, rng_seed: vec2<f32>) -> vec3<f32> {
    if (cinema.gi_intensity <= 0.0) { return vec3<f32>(0.0); }
    
    let size = vec2<f32>(textureDimensions(t_sdf));
    var indirect = vec3<f32>(0.0);
    
    // Cone tracing directions: Main reflection + side lobes
    let step_count = 8;
    let max_dist = 400.0;
    
    var t = 10.0; // Start slightly offset
    
    for (var i = 0; i < step_count; i++) {
        let p = pixel_pos + normal * t;
        let uv = p / size;
        
        // Sample SDF to find closest surface
        let d = textureSampleLevel(t_sdf, s_hdr, uv, 0.0).x;
        
        if (d < 2.0) {
            // Hit a surface! Sample radiance
            let radiance = textureSampleLevel(t_hdr, s_hdr, uv, 2.0).rgb; // LoD 2 for blurred/diffuse look
            
            // Weight by distance
            let weight = (1.0 - t / max_dist);
            if (weight > 0.0) {
                indirect += radiance * weight;
            }
            // Bounce only once
            break; 
        }
        
        // Sphere trace step, but clamp max step for cone sampling quality
        t += max(d, 5.0);
        if (t >= max_dist) { break; }
    }
    
    return indirect * cinema.gi_intensity * 0.5; // Scale factor
}

@fragment
fn fs_post(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Lens Distortion & Chromatic Aberration using Texture Coordinates
    let uv_centered = in.uv - 0.5;
    let r2 = dot(uv_centered, uv_centered);
    
    // Distortion (Barrel/Pincushion)
    let k = -0.1 * cinema.vignette_intensity; // Subtle barrel
    let dist_uv = in.uv + uv_centered * (k * r2 + k * r2 * r2);
    
    // Chromatic Aberration (Spectral separation)
    let ca_amount = cinema.ca_strength * r2 * 2.0;
    
    // Check bounds to avoid streaking edges
    if (dist_uv.x < 0.0 || dist_uv.x > 1.0 || dist_uv.y < 0.0 || dist_uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let r_uv = dist_uv - vec2<f32>(ca_amount, 0.0);
    let g_uv = dist_uv;
    let b_uv = dist_uv + vec2<f32>(ca_amount, 0.0);

    // Sample Lighting Result (t_hdr is now the input linear color)
    let r_col = textureSample(t_hdr, s_hdr, r_uv).r;
    let g_col = textureSample(t_hdr, s_hdr, g_uv).g;
    let b_col = textureSample(t_hdr, s_hdr, b_uv).b;
    let scene_color = vec3<f32>(r_col, g_col, b_col);
    
    // Apply Bloom
    let bloom_color = textureSample(t_bloom, s_hdr, dist_uv).rgb;
    
    let combined = scene_color + bloom_color * cinema.bloom_intensity; // Add bloom here
    
    // Exposure
    let exposed = combined * cinema.exposure;
    
    // Tone Mapping
    var tonemapped = exposed;
    if (cinema.tonemap_mode == 1u) {
        tonemapped = aces_approx(exposed);
    }
    
    // LUT Grading
    let lut_color = sample_lut(tonemapped);
    let graded = mix(tonemapped, lut_color, cinema.lut_intensity);
    
    // Film Grain & Vignette
    let noise = fract(sin(dot(in.uv * (cinema.time + 1.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let grainy = graded + (noise - 0.5) * cinema.grain_strength;
    
    // Vignette (Falloff)
    let vign = 1.0 - smoothstep(0.4, 1.4, length(uv_centered) * (1.0 + cinema.vignette_intensity));
    let final_color = grainy * vign;

    // Gamma Correction
    let out_color = pow(final_color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(out_color, 1.0);
}
