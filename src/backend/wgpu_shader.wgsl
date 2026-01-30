// Fantasmagorie WGPU Shader
// SDF-based rendering for UI elements
// Modes: 0=Solid, 1=Text, 2=Shape, 3=Image, 4=Blur, 5=Gradient, 6=Arc, 7=Plot, 8=Heatmap, 9=Aurora

struct Uniforms {
    projection: mat4x4<f32>,
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

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var u_texture: texture_2d<f32>;

@group(0) @binding(2)
var u_sampler: sampler;

@group(0) @binding(3)
var u_backdrop_texture: texture_2d<f32>;

struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.projection * vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = vec4<f32>(pow(in.color.rgb, vec3<f32>(2.2)), in.color.a);
    out.world_pos = in.pos;
    return out;
}

// ========== SDF Functions (Improved) ==========

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var radius: vec2<f32>;
    if (p.x > 0.0) {
        if (p.y > 0.0) {
            radius = vec2<f32>(r.z, 0.0); // br
        } else {
            radius = vec2<f32>(r.y, 0.0); // tr
        }
    } else {
        if (p.y > 0.0) {
            radius = vec2<f32>(r.w, 0.0); // bl
        } else {
            radius = vec2<f32>(r.x, 0.0); // tl
        }
    }
    let r_val = radius.x;
    let q = abs(p) - b + r_val;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r_val;
}

fn sd_squircle(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    let start = max(q, vec2<f32>(0.0));
    let n = 4.0;
    let p_n = pow(start, vec2<f32>(n));
    let len = pow(p_n.x + p_n.y, 1.0 / n);
    return len + min(max(q.x, q.y), 0.0) - r;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - vec3<f32>(K.w));
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

// ========== Mode Implementations ==========

fn mode_solid(col: vec4<f32>) -> vec4<f32> {
    return col;
}

fn mode_sdf_text(col: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dist = textureSample(u_texture, u_sampler, uv).r;
    let alpha = smoothstep(0.48, 0.52, dist); // Sharper text
    return vec4<f32>(col.rgb, col.a * alpha);
}

fn mode_shape(col: vec4<f32>, world_pos: vec2<f32>) -> vec4<f32> {
    let center = uniforms.rect.xy + uniforms.rect.zw * 0.5;
    let half_size = uniforms.rect.zw * 0.5;
    let local = world_pos - center;
    
    var d: f32;
    if (uniforms.is_squircle == 1) {
        d = sd_squircle(local, half_size, uniforms.radii.x);
    } else {
        d = sd_rounded_box(local, half_size, uniforms.radii);
    }
    
    let aa = 1.0;
    let alpha = 1.0 - smoothstep(-aa, aa, d);

    var bg = col;
    
    // Border
    if (uniforms.border_width > 0.0) {
        let interior_alpha = 1.0 - smoothstep(-aa, aa, d + uniforms.border_width);
        let border_lin = vec4<f32>(pow(uniforms.border_color.rgb, vec3<f32>(2.2)), uniforms.border_color.a);
        bg = mix(border_lin, col, interior_alpha);
    }
    
    // Fresnel Hairline (1px Inner highlight)
    if (alpha > 0.01) {
        let border_alpha = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5));
        bg = mix(bg, vec4<f32>(1.0, 1.0, 1.0, 0.15), border_alpha);
    }

    let main_layer = vec4<f32>(bg.rgb, bg.a * alpha);
    
    // Glow
    var glow_layer = vec4<f32>(0.0);
    if (uniforms.glow_strength > 0.0) {
        let glow_factor = exp(-max(d, 0.0) * 0.1) * uniforms.glow_strength;
        let glow_lin = vec4<f32>(pow(uniforms.glow_color.rgb, vec3<f32>(2.2)), uniforms.glow_color.a);
        glow_layer = glow_lin * glow_factor;
    }

    // Shadow/Elevation
    var shadow_layer = vec4<f32>(0.0);
    if (uniforms.elevation > 0.0) {
        let d1 = sd_rounded_box(local - vec2<f32>(0.0, uniforms.elevation * 0.25), half_size, uniforms.radii);
        let a1 = (1.0 - smoothstep(-uniforms.elevation * 0.5, uniforms.elevation * 0.5, d1)) * 0.4;
        let d2 = sd_rounded_box(local - vec2<f32>(0.0, uniforms.elevation * 1.5), half_size, uniforms.radii);
        let a2 = (1.0 - smoothstep(-uniforms.elevation * 3.0, uniforms.elevation * 3.0, d2)) * 0.2;
        shadow_layer = vec4<f32>(0.0, 0.0, 0.0, max(a1, a2) * col.a);
    }

    var comp = shadow_layer + glow_layer;
    comp = vec4<f32>(
        main_layer.rgb * main_layer.a + comp.rgb * (1.0 - main_layer.a),
        max(comp.a, main_layer.a)
    );
    return comp;
}

fn mode_image(col: vec4<f32>, uv: vec2<f32>, world_pos: vec2<f32>) -> vec4<f32> {
    let center = uniforms.rect.xy + uniforms.rect.zw * 0.5;
    let half_size = uniforms.rect.zw * 0.5;
    let local = world_pos - center;
    
    var d: f32;
    if (uniforms.is_squircle == 1) {
        d = sd_squircle(local, half_size, uniforms.radii.x);
    } else {
        d = sd_rounded_box(local, half_size, uniforms.radii);
    }
    
    let alpha = 1.0 - smoothstep(-1.0, 1.0, d);
    let tex_col = textureSample(u_texture, u_sampler, uv) * col;
    let tex_lin = pow(tex_col.rgb, vec3<f32>(2.2));
    return vec4<f32>(tex_lin, tex_col.a * alpha);
}

fn mode_blur(col: vec4<f32>, world_pos: vec2<f32>, uv: vec2<f32>) -> vec4<f32> {
    let center = uniforms.rect.xy + uniforms.rect.zw * 0.5;
    let half_size = uniforms.rect.zw * 0.5;
    let local = world_pos - center;
    
    var d: f32;
    if (uniforms.is_squircle == 1) {
        d = sd_squircle(local, half_size, uniforms.radii.x);
    } else {
        d = sd_rounded_box(local, half_size, uniforms.radii);
    }
    
    // Increased base alpha for "Frosted" substance
    let base_alpha = 0.45;
    let edge_aa = 1.0;
    let alpha_mask = 1.0 - smoothstep(-edge_aa, edge_aa, d);
    
    // Backdrop Blur (New: Manual Box Blur via textureLoad)
    // Non-filterable textures can't use textureSample / textureSampleLevel
    let dims = vec2<f32>(textureDimensions(u_backdrop_texture));
    let pixel_coords = vec2<i32>(uv * dims);
    
    var blurred_bg = vec3<f32>(0.0);
    let blur_radius = 4;
    var weight = 0.0;
    
    for (var i = -blur_radius; i <= blur_radius; i++) {
        for (var j = -blur_radius; j <= blur_radius; j++) {
            let offset = vec2<i32>(i * 4, j * 4); // Spread out for more blur
            blurred_bg += textureLoad(u_backdrop_texture, pixel_coords + offset, 0).rgb;
            weight += 1.0;
        }
    }
    blurred_bg /= weight;
    
    // Procedural Frost Grain
    let grain = (hash(world_pos * 0.5 + vec2<f32>(uniforms.time * 0.01)) - 0.5) * 0.04;
    var glass_col = vec4<f32>(mix(blurred_bg, col.rgb, 0.4), col.a) + grain;
    
    // Liquid Fresnel: Catching light on the edges
    // 1. Sharp Hairline (Outer)
    let hairline = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5));
    let highlight_sharp = vec4<f32>(1.0, 1.0, 1.0, 0.4) * hairline;
    
    // 2. Soft Inner Glow (Light refraction)
    let inner_glow = 1.0 - smoothstep(-15.0, 0.0, d);
    let highlight_soft = vec4<f32>(1.0, 1.0, 1.0, 0.15) * inner_glow * alpha_mask;
    
    var final_glass = mix(glass_col, vec4<f32>(1.0), (highlight_sharp.a + highlight_soft.a) * 0.5);
    
    return vec4<f32>(final_glass.rgb, (base_alpha + highlight_sharp.a * 0.2) * alpha_mask);
}

fn mode_arc(col: vec4<f32>, world_pos: vec2<f32>) -> vec4<f32> {
    let center = uniforms.rect.xy + uniforms.rect.zw * 0.5;
    let local = world_pos - center;
    let dist = length(local);
    
    let r = uniforms.radii.x;
    let thickness = uniforms.radii.y;
    let d = abs(dist - r) - thickness * 0.5;
    
    var angle = atan2(local.y, local.x);
    if (angle < 0.0) { angle = angle + 6.283185; }
    
    let start_angle = uniforms.offset.x;
    let end_angle = uniforms.offset.y;
    
    var in_angle = false;
    if (start_angle < end_angle) {
        in_angle = (angle >= start_angle) && (angle <= end_angle);
    } else {
        in_angle = (angle >= start_angle) || (angle <= end_angle);
    }
    
    let aa = 1.0;
    var alpha = 1.0 - smoothstep(-aa, aa, d);
    
    if (!in_angle) {
        let d_angle = min(abs(angle - start_angle), abs(angle - end_angle));
        alpha = alpha * (1.0 - smoothstep(0.0, 0.02, d_angle));
    }
    
    return vec4<f32>(col.rgb, col.a * alpha);
}

fn mode_plot(col: vec4<f32>) -> vec4<f32> {
    return col;
}

fn mode_heatmap(uv: vec2<f32>) -> vec4<f32> {
    let val = textureSample(u_texture, u_sampler, uv).r;
    let t = clamp((val - uniforms.elevation) / (uniforms.glow_strength - uniforms.elevation + 0.0001), 0.0, 1.0);
    
    let c0 = vec3<f32>(0.267, 0.004, 0.329);
    let c1 = vec3<f32>(0.127, 0.566, 0.550);
    let c2 = vec3<f32>(0.993, 0.906, 0.143);
    
    var result: vec3<f32>;
    if (t < 0.5) {
        result = mix(c0, c1, t * 2.0);
    } else {
        result = mix(c1, c2, (t - 0.5) * 2.0);
    }
    return vec4<f32>(result, 1.0);
}

fn mode_aurora(clip_pos: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let t = uniforms.time;
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
    return vec4<f32>(color, 1.0);
}

// ========== Main Fragment Shader ==========

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = vec4<f32>(0.0);
    
    if (uniforms.mode == 0) { final_color = mode_solid(in.color); }
    else if (uniforms.mode == 1) { final_color = mode_sdf_text(in.color, in.uv); }
    else if (uniforms.mode == 2) { final_color = mode_shape(in.color, in.world_pos); }
    else if (uniforms.mode == 3) { final_color = mode_image(in.color, in.uv, in.world_pos); }
    else if (uniforms.mode == 4) { final_color = mode_blur(in.color, in.world_pos, in.uv); }
    else if (uniforms.mode == 5) { final_color = mode_aurora(in.clip_position, in.uv); }
    else if (uniforms.mode == 6) { final_color = mode_arc(in.color, in.world_pos); }
    else if (uniforms.mode == 7) { final_color = mode_plot(in.color); }
    else if (uniforms.mode == 8) { final_color = mode_heatmap(in.uv); }
    else if (uniforms.mode == 9) { final_color = mode_aurora(in.clip_position, in.uv); }
    else { final_color = in.color; }

    // Linear -> sRGB
    return final_color;
}
