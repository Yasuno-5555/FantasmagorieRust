// Fantasmagorie Vulkan WGSL Shader
// Compiled to SPIR-V via Naga for Vulkan backend

struct GlobalUniforms {
    projection: mat4x4<f32>,
    viewport: vec2<f32>,
    time: f32,
    _pad: f32,
};

struct DrawUniforms {
    rect: vec4<f32>,
    radii: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    offset: vec2<f32>,
    scale: f32,
    border_width: f32,
    elevation: f32,
    glow_strength: f32,
    lut_intensity: f32,
    mode: u32,
    is_squircle: u32,
    time: f32,
    _pad: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;
@group(0) @binding(3) var t_lut: texture_2d<f32>; 
@group(0) @binding(4) var t_backdrop: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> instance_data: array<DrawUniforms>;

var<push_constant> pc: DrawUniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) frag_pos: vec2<f32>,
    @location(3) @interpolate(flat) instance_id: u32,
};

@vertex
fn vs_main(model: VertexInput, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    var out: VertexOutput;
    
    var params: DrawUniforms;
    if (pc.mode == 0xFFFFFFFFu) {
        params = instance_data[instance_idx];
    } else {
        params = pc;
    }

    // GPU-driven vertex expansion from unit quad
    // model.position is expected to be in range [0, 1]
    
    let is_shape = params.mode == 2u || params.mode == 6u;
    let pad = select(0.0, 100.0, is_shape && (params.elevation > 0.0 || params.glow_strength > 0.0));
    
    let rect = vec4<f32>(params.rect.xy - vec2<f32>(pad), params.rect.zw + vec2<f32>(pad * 2.0));
    let world_pos = rect.xy + model.position * rect.zw;
    
    out.clip_position = globals.projection * vec4<f32>(world_pos, 0.0, 1.0);
    out.uv = model.uv; // Assuming model.uv is correctly set in unit quad or interpolated
    
    // For text/image modes, we might need specific UVs from params
    if (params.mode == 1u || params.mode == 3u) {
        // Simple UV mapping for full-sprite textures
        // If we need tiled UVs, they should be in params.glow_color or radii
        // For now, assume model.uv is correct [0,1]
    }
    
    out.color = params.border_color;
    out.frag_pos = world_pos;
    out.instance_id = instance_idx;
    
    return out;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    let r_temp = select(r.zw, r.xy, p.x > 0.0);
    let corner_r = select(r_temp.y, r_temp.x, p.y > 0.0);
    let q = abs(p) - b + corner_r;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - corner_r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var params: DrawUniforms;
    if (pc.mode == 0xFFFFFFFFu) {
        params = instance_data[in.instance_id];
    } else {
        params = pc;
    }
    
    let mode = params.mode;
    
    // Linear color from vertex
    let col_lin = vec4<f32>(pow(in.color.rgb, vec3<f32>(2.2)), in.color.a);

    var final_color = vec4<f32>(1.0, 0.0, 1.0, 1.0); // Error Magenta

    if (mode == 0u) { // Solid
        final_color = col_lin;
    } else if (mode == 1u) { // Text
        let dist = textureSample(t_diffuse, s_diffuse, in.uv).r;
        let alpha = smoothstep(0.48, 0.52, dist);
        final_color = vec4<f32>(col_lin.rgb, col_lin.a * alpha);
    } else if (mode == 2u) { // Shape (RoundedRect with Glow/Shadow)
        let center = params.rect.xy + params.rect.zw * 0.5;
        let half_size = params.rect.zw * 0.5;
        let p = in.frag_pos - center;
        
        var d: f32;
        if (params.is_squircle == 1u) {
            let r = max(params.radii.x, 1.0);
            let q = abs(p) - half_size + r;
            let start = max(q, vec2<f32>(0.0));
            let n = 4.0;
            let p_n = pow(start, vec2<f32>(n));
            let len = pow(p_n.x + p_n.y, 1.0 / n);
            d = len + min(max(q.x, q.y), 0.0) - r;
        } else {
            d = sd_rounded_box(p, half_size, params.radii);
        }
        
        let aa = 1.0;
        let alpha = 1.0 - smoothstep(-aa, aa, d);
        
        var color = col_lin;
        
        // Border
        if (params.border_width > 0.0) {
            let border_alpha = 1.0 - smoothstep(-aa, aa, d + params.border_width);
            let b_lin = vec4<f32>(pow(params.border_color.rgb, vec3<f32>(2.2)), params.border_color.a);
            color = mix(b_lin, col_lin, border_alpha);
        }

        // Fresnel Hairline (1px Inner highlight)
        if (alpha > 0.01) {
             let hairline_alpha = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5));
             let highlight = vec4<f32>(1.0, 1.0, 1.0, 0.15); // White, 15%
             color = mix(color, highlight, hairline_alpha * highlight.a);
        }
        
        let main_layer = vec4<f32>(color.rgb, color.a * alpha);
        
        // Glow (Outer Glow)
        var glow_layer = vec4<f32>(0.0);
        if (params.glow_strength > 0.0) {
            let glow_dist = max(d, 0.0);
            let glow_factor = exp(-glow_dist * 0.1) * params.glow_strength; // Simplified glow
            let g_lin = vec4<f32>(pow(params.glow_color.rgb, vec3<f32>(2.2)), params.glow_color.a);
            glow_layer = g_lin * glow_factor;
        }

        // Shadow/Elevation
        var shadow_layer = vec4<f32>(0.0);
        if (params.elevation > 0.0) {
            let sd = sd_rounded_box(p - vec2<f32>(0.0, params.elevation * 0.5), half_size, params.radii);
            let sa = (1.0 - smoothstep(-params.elevation, params.elevation * 3.0, sd)) * 0.4;
            shadow_layer = vec4<f32>(0.0, 0.0, 0.0, sa * col_lin.a);
        }
        
        let mixed_bg = shadow_layer + glow_layer;
        let final_rgb = mix(mixed_bg.rgb, main_layer.rgb, main_layer.a);
        let final_a = max(mixed_bg.a, main_layer.a);
        final_color = vec4<f32>(final_rgb, final_a);
    } else if (mode == 3u) { // Image
        let tex_col = textureSample(t_diffuse, s_diffuse, in.uv);
        let tex_lin = vec4<f32>(pow(tex_col.rgb, vec3<f32>(2.2)), tex_col.a);
        final_color = tex_lin * col_lin;
    } else if (mode == 4u) { // Blur (Glassmorphism)
        let center = params.rect.xy + params.rect.zw * 0.5;
        let half_size = params.rect.zw * 0.5;
        let p = in.frag_pos - center;
        var d: f32;
        if (params.is_squircle == 1u) {
            let r = max(params.radii.x, 1.0);
            let q = abs(p) - half_size + r;
            let start = max(q, vec2<f32>(0.0));
            let n = 4.0;
            let p_n = pow(start, vec2<f32>(n));
            let len = pow(p_n.x + p_n.y, 1.0 / n);
            d = len + min(max(q.x, q.y), 0.0) - r;
        } else {
            d = sd_rounded_box(p, half_size, params.radii);
        }
        
        let alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        let lod = params.border_width; // Passed from Rust
        
        // 1. LOD Backdrop Sampling
        let uv = in.frag_pos / globals.viewport;
        var bg = textureSampleLevel(t_backdrop, s_diffuse, uv, lod);
        
        // 2. Saturation Boost
        let luma = dot(bg.rgb, vec3<f32>(0.299, 0.587, 0.114));
        bg = vec4<f32>(mix(vec3<f32>(luma), bg.rgb, 1.2), bg.a);
        
        // 3. Dithering / Noise
        let noise = (hash(in.frag_pos) - 0.5) / 255.0;
        bg = vec4<f32>(bg.rgb + noise * 4.0, bg.a);
        
        // 4. Tint and Fresnel
        var color = mix(bg, col_lin, col_lin.a * 0.4); // Tint
        
        let hairline_alpha = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5));
        let highlight = vec4<f32>(1.0, 1.0, 1.0, 0.15); // Fresnel highlight
        color = mix(color, highlight, hairline_alpha * highlight.a);
        
        final_color = vec4<f32>(color.rgb, alpha);
    } else if (mode == 5u) { // ImageLUT (Simplified)
        let tex_col = textureSample(t_diffuse, s_diffuse, in.uv);
        let tex_lin = vec4<f32>(pow(tex_col.rgb, vec3<f32>(2.2)), tex_col.a);
        final_color = tex_lin * col_lin;
    } else if (mode == 6u) { // Arc
        let center = params.rect.xy + params.rect.zw * 0.5;
        let p = in.frag_pos - center;
        let r = length(p);
        let inner_r = params.radii.x;
        let thickness = params.radii.y;
        let outer_r = inner_r + thickness;
        
        let d = abs(r - (inner_r + outer_r) * 0.5) - (outer_r - inner_r) * 0.5;
        let alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        final_color = vec4<f32>(col_lin.rgb, col_lin.a * alpha);
    } else if (mode == 7u) { // Plot
        final_color = col_lin;
    } else if (mode == 8u) { // Heatmap
        let val = textureSample(t_diffuse, s_diffuse, in.uv).r;
        let t = clamp((val - params.elevation) / (params.glow_strength - params.elevation + 0.0001), 0.0, 1.0);
        
        let c0 = vec3<f32>(0.267, 0.004, 0.329);
        let c1 = vec3<f32>(0.127, 0.566, 0.550);
        let c2 = vec3<f32>(0.993, 0.906, 0.143);
        
        var result: vec3<f32>;
        if (t < 0.5) {
            result = mix(c0, c1, t * 2.0);
        } else {
            result = mix(c1, c2, (t - 0.5) * 2.0);
        }
        final_color = vec4<f32>(result, 1.0);
    } else if (mode == 9u) { // Aurora
        let uv = in.uv;
        let t = globals.time;
        
        let p = uv * 2.0 - 1.0;
        var color = vec3<f32>(0.0);
        
        for (var i = 1.0; i < 4.0; i += 1.0) {
            let shifted_uv = p + vec2<f32>(
                sin(t * 0.2 + i * 1.5) * 0.5,
                cos(t * 0.3 + i * 2.1) * 0.5
            );
            let dist = length(shifted_uv);
            let wave = sin(dist * 5.0 - t * 2.0) * 0.5 + 0.5;
            
            let hue = fract(t * 0.05 + i * 0.2);
            let c = hsv2rgb(vec3<f32>(hue, 0.7, 0.8));
            color += c * (0.15 / (dist + 0.1)) * wave;
        }
        
        color = clamp(color * 0.5, vec3<f32>(0.05), vec3<f32>(0.6));
        final_color = vec4<f32>(color, col_lin.a);
    } else if (mode == 10u) { // Grid
        let pos = in.frag_pos;
        let g1 = 40.0;
        let f1 = abs(fract(pos / g1 - 0.5) - 0.5) / (fwidth(pos / g1) + 0.001);
        let line1 = 1.0 - min(min(f1.x, f1.y), 1.0);
        final_color = vec4<f32>(col_lin.rgb, col_lin.a * line1);
    }

    // Linear -> sRGB (gamma correction)
    return vec4<f32>(pow(final_color.rgb, vec3<f32>(1.0 / 2.2)), final_color.a);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - vec3<f32>(K.w));
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}
