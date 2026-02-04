// Fantasmagorie WGPU Shader
// SDF-based rendering for UI elements
// Modes: 0=Solid, 1=Text, 2=Shape, 3=Image, 4=Blur, 5=Gradient, 6=Arc, 7=Plot, 8=Heatmap, 9=Aurora

struct GlobalUniforms {
    projection: mat4x4<f32>,
    time: f32,
    viewport_size: vec2<f32>,
    _pad: f32,
};

struct ShapeInstance {
    rect: vec4<f32>,
    radii: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    params1: vec4<f32>, // border_width, elevation, glow_strength, lut_intensity
    params2: vec4<u32>, // mode, is_squircle, _r1, _r2
};

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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<uniform> glob: GlobalUniforms;
@group(0) @binding(5) var<storage, read> instances: array<ShapeInstance>;

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
    @location(3) @interpolate(flat) iid: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.projection * vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = vec4<f32>(pow(in.color.rgb, vec3<f32>(2.2)), in.color.a);
    out.world_pos = in.pos;
    out.iid = 0xFFFFFFFFu;
    return out;
}

@vertex
fn vs_instanced(
    in: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let inst = instances[instance_index];
    var out: VertexOutput;
    
    // Calculate world position from rect and uv
    let pos = inst.rect.xy + in.uv * inst.rect.zw;
    out.clip_position = glob.projection * vec4<f32>(pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = vec4<f32>(pow(in.color.rgb, vec3<f32>(2.2)), in.color.a);
    out.world_pos = pos;
    out.iid = instance_index;
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
    let width = fwidth(dist);
    let alpha = smoothstep(0.5 - width, 0.5 + width, dist);
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

fn mode_aurora(uv: vec2<f32>) -> vec4<f32> {
    let t = uniforms.time * 0.5;
    let p = uv;
    
    // Lush, flowing aurora logic
    var col = vec3<f32>(0.0);
    let n = 5.0;
    for (var i = 1.0; i < n; i += 1.0) {
        let uv_i = p + vec2<f32>(
            sin(t * 0.1 + i * 0.8) * 0.4,
            cos(t * 0.15 + i * 1.2) * 0.3
        );
        let d = length(uv_i - 0.5);
        let hue = fract(t * 0.02 + i * 0.15);
        let rgb = hsv2rgb(vec3<f32>(hue, 0.8, 0.7));
        col += rgb * (0.1 / (d + 0.15)) * (0.5 + 0.5 * sin(d * 10.0 - t * 2.0));
    }
    
    // Blend with a deep space base
    let base = mix(vec3<f32>(0.02, 0.02, 0.05), vec3<f32>(0.05, 0.02, 0.1), p.y);
    return vec4<f32>(base + col * 0.6, 1.0);
}

fn mode_grid(world_pos: vec2<f32>) -> vec4<f32> {
    let pos = world_pos;
    let t = uniforms.time;
    
    // Smooth derivative-based anti-aliasing
    let grid_size = 50.0;
    let f = abs(fract(pos / grid_size - 0.5) - 0.5);
    let g = f / (fwidth(pos / grid_size) + 0.02);
    let grid = 1.0 - min(min(g.x, g.y), 1.0);
    
    let pulse = sin(t * 2.0 - length(pos) * 0.01) * 0.5 + 0.5;
    // Slightly HDR color (1.2) to trigger a soft cinematic bloom glow
    let col = mix(vec3<f32>(0.02, 0.02, 0.05), vec3<f32>(0.0, 1.2, 1.5), grid * pulse * 0.5);
    return vec4<f32>(col, 1.0);
}

// Placeholder for custom effect injection
// User-provided code should override this if using mode 100
fn custom_effect(color: vec4<f32>, world_pos: vec2<f32>, uv: vec2<f32>, time: f32) -> vec4<f32> {
    return color;
}

// ========== Main Fragment Shader ==========

struct ShapeData {
    rect: vec4<f32>,
    radii: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    border_width: f32,
    elevation: f32,
    glow_strength: f32,
    lut_intensity: f32,
    mode: u32,
    is_squircle: u32,
    time: f32,
    viewport_size: vec2<f32>,
}

fn resolve_shape(in: VertexOutput, d: ShapeData) -> vec4<f32> {
    var final_color = vec4<f32>(0.0);
    let world_pos = in.world_pos;
    let uv = in.uv;
    let col = in.color;
    
    if (d.mode == 0u) { final_color = col; }
    else if (d.mode == 1u) {
        let dist = textureSample(u_texture, u_sampler, uv).r;
        let width = fwidth(dist);
        let alpha = smoothstep(0.5 - width, 0.5 + width, dist);
        final_color = vec4<f32>(col.rgb, col.a * alpha);
    }
    else if (d.mode == 2u) {
        let center = d.rect.xy + d.rect.zw * 0.5;
        let half_size = d.rect.zw * 0.5;
        let local = world_pos - center;
        var dist: f32;
        if (d.is_squircle == 1u) { dist = sd_squircle(local, half_size, d.radii.x); }
        else { dist = sd_rounded_box(local, half_size, d.radii); }
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
        var bg = col;
        if (d.border_width > 0.0) {
            let interior_alpha = 1.0 - smoothstep(-1.0, 1.0, dist + d.border_width);
            let border_lin = vec4<f32>(pow(d.border_color.rgb, vec3<f32>(2.2)), d.border_color.a);
            bg = mix(border_lin, col, interior_alpha);
        }
        let main_layer = vec4<f32>(bg.rgb, bg.a * alpha);
        var glow_layer = vec4<f32>(0.0);
        if (d.glow_strength > 0.0) {
            let glow_factor = exp(-max(dist, 0.0) * 0.1) * d.glow_strength;
            let glow_lin = vec4<f32>(pow(d.glow_color.rgb, vec3<f32>(2.2)), d.glow_color.a);
            glow_layer = glow_lin * glow_factor;
        }
        final_color = vec4<f32>(main_layer.rgb * main_layer.a + glow_layer.rgb * (1.0 - main_layer.a), max(glow_layer.a, main_layer.a));
    }
    else if (d.mode == 3u) {
        let center = d.rect.xy + d.rect.zw * 0.5;
        let half_size = d.rect.zw * 0.5;
        let local = world_pos - center;
        var dist: f32;
        if (d.is_squircle == 1u) { dist = sd_squircle(local, half_size, d.radii.x); }
        else { dist = sd_rounded_box(local, half_size, d.radii); }
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
        let tex_col = textureSample(u_texture, u_sampler, uv) * col;
        final_color = vec4<f32>(pow(tex_col.rgb, vec3<f32>(2.2)), tex_col.a * alpha);
    }
    else if (d.mode == 9u) {
        let t = d.time * 0.5;
        var aurora_col = vec3<f32>(0.0);
        for (var i = 1.0; i < 5.0; i += 1.0) {
            let uv_i = uv + vec2<f32>(sin(t * 0.1 + i * 0.8) * 0.4, cos(t * 0.15 + i * 1.2) * 0.3);
            let dist_i = length(uv_i - 0.5);
            let hue = fract(t * 0.02 + i * 0.15);
            let rgb = hsv2rgb(vec3<f32>(hue, 0.8, 0.7));
            aurora_col += rgb * (0.1 / (dist_i + 0.15)) * (0.5 + 0.5 * sin(dist_i * 10.0 - t * 2.0));
        }
        final_color = vec4<f32>(aurora_col, 1.0);
    }
    else if (d.mode == 12u) {
        let pos = world_pos;
        let t = d.time;
        let grid_size = 50.0;
        let f = abs(fract(pos / grid_size - 0.5) - 0.5);
        let g = f / (fwidth(pos / grid_size) + 0.02);
        let grid = 1.0 - min(min(g.x, g.y), 1.0);
        let pulse = sin(t * 2.0 - length(pos) * 0.01) * 0.5 + 0.5;
        let col_grid = mix(vec3<f32>(0.02, 0.02, 0.05), vec3<f32>(0.0, 1.2, 1.5), grid * pulse * 0.5);
        final_color = vec4<f32>(col_grid, 1.0);
    }
    else { final_color = col; }

    return final_color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var d: ShapeData;
    d.rect = uniforms.rect; d.radii = uniforms.radii; d.border_color = uniforms.border_color;
    d.glow_color = uniforms.glow_color; d.border_width = uniforms.border_width;
    d.elevation = uniforms.elevation; d.glow_strength = uniforms.glow_strength;
    d.lut_intensity = uniforms.lut_intensity; d.mode = uniforms.mode;
    d.is_squircle = uniforms.is_squircle; d.time = uniforms.time;
    d.viewport_size = vec2<f32>(1.0); // Not used yet in simple modes
    return resolve_shape(in, d);
}

@fragment
fn fs_instanced(in: VertexOutput) -> @location(0) vec4<f32> {
    let inst = instances[in.iid];
    var d: ShapeData;
    d.rect = inst.rect; d.radii = inst.radii; d.border_color = inst.border_color;
    d.glow_color = inst.glow_color; d.border_width = inst.params1.x;
    d.elevation = inst.params1.y; d.glow_strength = inst.params1.z;
    d.lut_intensity = inst.params1.w; d.mode = inst.params2.x;
    d.is_squircle = inst.params2.y; d.time = glob.time;
    d.viewport_size = glob.viewport_size;
    return resolve_shape(in, d);
}
