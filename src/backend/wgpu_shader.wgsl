// Fantasmagorie WGSL Shader
// SDF-based rendering for rounded rectangles and text

struct Uniforms {
    projection: mat4x4<f32>,
    rect: vec4<f32>,         // x, y, w, h
    radii: vec4<f32>,        // tl, tr, br, bl
    border_color: vec4<f32>,
    glow_color: vec4<f32>,   // Added
    
    mode: i32,               // 0=solid, 1=text, 2=rounded_rect, 4=blur, 5=mesh, 6=arc, 7=plot
    border_width: f32,
    elevation: f32,
    is_squircle: i32,        // Added
    
    glow_strength: f32,      // Added
    start_angle: f32,        // Added
    end_angle: f32,          // Added
    _padding3: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var font_texture: texture_2d<f32>;

@group(0) @binding(2)
var font_sampler: sampler;

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
    
    // Linear Workflow: Convert sRGB to Linear
    out.color = vec4<f32>(pow(in.color.rgb, vec3<f32>(2.2)), in.color.a);
    
    out.world_pos = in.pos;
    return out;
}

// SDF for rounded rectangle
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var radius = r.x; 
    if (p.x > 0.0) { radius = r.y; }
    if (p.x > 0.0 && p.y > 0.0) { radius = r.z; }
    if (p.x <= 0.0 && p.y > 0.0) { radius = r.w; }

    let q = abs(p) - b + radius;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// Super-Ellipse Squircle (n = 4.0)
fn sd_squircle(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    let start = max(q, vec2<f32>(0.0));
    let n = 4.0;
    let p_n = pow(start, vec2<f32>(n));
    let len = pow(p_n.x + p_n.y, 1.0/n);
    return len + min(max(q.x, q.y), 0.0) - r;
}

// SDF for Arc
// sc is sin/cos of aperture
// r is radius
// w is thickness
fn sd_arc(p: vec2<f32>, start_angle: f32, end_angle: f32, r: f32, thickness: f32) -> f32 {
    // Basic Ring Distance
    let d_ring = abs(length(p) - r) - thickness;
    
    // Angle clipping
    // Atan2 returns -PI to PI.
    // Our angles are e.g. 135 to 405 deg (2.35 to 7.06 rad).
    // Normalize p_angle to same range?
    var angle = atan2(p.y, p.x); // -PI to PI
    if (angle < 0.0) { angle = angle + 6.2831853; } // 0 to 2PI
    
    // Handle wrap-around for multi-turn?
    // Knob usually is < 360 range (e.g. 270 deg).
    // Start=2.35 (135 deg). End=7.06 (405 deg -> 45 deg).
    // 405 is > 2PI.
    // If we map angle to 0..2PI, 405 becomes 45 (0.78).
    // We need logic to check if angle is between start and end.
    
    // Normalized check:
    // If range crosses 0/2PI boundary.
    // Actually, simple Arc logic usually revolves around a central "aperture" vector and a width.
    // Here we have arbitrary start/end.
    
    // Let's rely on simple angle check for now.
    // If end > 2PI, we treat it as > 2PI.
    // We can shift angle by +2PI if needed?
    
    var check_angle = angle;
    if (check_angle < start_angle && check_angle + 6.2831853 <= end_angle) {
        check_angle = check_angle + 6.2831853;
    }
    
    // The previous logic is tricky if start > 2PI etc.
    // Better idea: Midpoint and Half-Aperture.
    let mid = (start_angle + end_angle) * 0.5;
    let half_aper = (end_angle - start_angle) * 0.5;
    
    // Rotate p by -mid
    let c = cos(-mid);
    let s = sin(-mid);
    let p_rot = vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
    
    // Now check against aperture
    // Angle of p_rot should be within [-half_aper, half_aper]
    let rot_angle = atan2(p_rot.y, p_rot.x);
    
    // Distance to sector
    // If inside sector (abs(rot_angle) < half_aper), dist is d_ring.
    // If outside, dist is dist to endpoints.
    
    // Endpoints:
    // P_start is (r * cos(start), r * sin(start))
    // P_end is (r * cos(end), r * sin(end))
    // Distance to line segments? No, distance to point on ring.
    
    // Simplified:
    // d = max(d_ring, abs(angle_diff) ... ) ?
    // Use IQ's sdArc?
    // float sdArc( in vec2 p, in vec2 sc, in float ra, float rb )
    // sc is sin/cos of aperture/2. 
    
    // Setup for IQ's sdArc:
    let sc = vec2<f32>(sin(half_aper), cos(half_aper)); // Note: IQ uses (sin, cos) order in vec2 usually? or (cos, sin)?
    // IQ: vec2 sc = vec2(sin(aperture), cos(aperture));
    // p.x = abs(p.x); 
    // return ((sc.y*p.x > sc.x*p.y) ? length(p-sc*ra) : abs(length(p)-ra)) - rb;
    // Wait, that assumes symmetry around Y axis.
    
    // Our p_rot is symmetric around X axis (angle 0).
    // So we need to swirl it to Y axis? Or adapt formula.
    // IQ's formula assumes aperture is opening upwards (Y+).
    // Our p_rot aligns midpoint to X+ (angle 0).
    // So let's rotate p_rot by -90 deg (-PI/2) to align to Y+?
    
    let p_sym = vec2<f32>(abs(p_rot.y), p_rot.x); // X is "width" (sin component), Y is "height" (cos component).
    // Checking angle against Y axis.
    // If we map X+ to Y+, then p_rot.x becomes p_sym.y, p_rot.y becomes p_sym.x.
    // But we need symmetry, so abs(x).
    
    // IQ sdArc reference:
    // float sdArc( in vec2 p, in vec2 sc, in float ra, float rb )
    // {
    //     p.x = abs(p.x);
    //     return ((sc.y*p.x>sc.x*p.y) ? length(p-sc*ra) : abs(length(p)-ra)) - rb;
    // }
    // sc is (sin(a), cos(a)). p is aligned to Y axis.
    
    // Our setup:
    // p_rot aligned to X axis.
    // We want to treat it as Y aligned.
    // Swap x/y.
    // p_aligned = vec2(abs(p_rot.y), p_rot.x);
    
    let sc_vec = vec2<f32>(sin(half_aper), cos(half_aper));
    let ra = r;
    let rb = thickness;
    
    let p_calc = vec2<f32>(abs(p_rot.y), p_rot.x);
    
    let dist_raw = select(
        abs(length(p_calc) - ra),
        length(p_calc - sc_vec * ra),
        sc_vec.y * p_calc.x > sc_vec.x * p_calc.y
    );
    
    return dist_raw - rb;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = vec4<f32>(0.0);

    if (uniforms.mode == 0) {
        // Solid color (Linear)
        final_color = in.color;
    }
    else if (uniforms.mode == 1) {
        // SDF text
        let dist = textureSample(font_texture, font_sampler, in.uv).r;
        let alpha = smoothstep(0.4, 0.6, dist);
        final_color = vec4<f32>(in.color.rgb, in.color.a * alpha);
    }
    else if (uniforms.mode == 2) {
        // Shape Rendering (Rounded Rect / Squircle)
        let center = uniforms.rect.xy + uniforms.rect.zw * 0.5;
        let half_size = uniforms.rect.zw * 0.5;
        let local = in.world_pos - center;
        
        var d: f32;
        if (uniforms.is_squircle == 1) {
            d = sd_squircle(local, half_size, uniforms.radii.x);
        } else {
            d = sd_rounded_box(local, half_size, uniforms.radii);
        }
        
        let aa = 1.0;
        let alpha = 1.0 - smoothstep(-aa, aa, d);

        var bg = in.color;
        
        // Border
        if (uniforms.border_width > 0.0) {
            let interior_alpha = 1.0 - smoothstep(-aa, aa, d + uniforms.border_width);
            let border_col_lin = vec4<f32>(pow(uniforms.border_color.rgb, vec3<f32>(2.2)), uniforms.border_color.a);
            bg = mix(border_col_lin, in.color, interior_alpha);
        }

        // 1px Hairline (Inner Stroke)
        if (alpha > 0.01) {
             let border_alpha = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5));
             let hairline = vec4<f32>(1.0, 1.0, 1.0, 0.15); 
             bg = mix(bg, hairline, border_alpha);
        }

        let main_layer = vec4<f32>(bg.rgb, bg.a * alpha);
        
        // Glow (Outer)
        var glow_layer = vec4<f32>(0.0);
        if (uniforms.glow_strength > 0.0) {
             let glow_factor = exp(-max(d, 0.0) * 0.1) * uniforms.glow_strength;
             let glow_col_lin = vec4<f32>(pow(uniforms.glow_color.rgb, vec3<f32>(2.2)), uniforms.glow_color.a);
             glow_layer = glow_col_lin * glow_factor;
        }

        // Shadow
        var shadow_layer = vec4<f32>(0.0);
        if (uniforms.elevation > 0.0) {
            let offset1 = vec2<f32>(0.0, uniforms.elevation * 0.25);
            let d1 = sd_rounded_box(local - offset1, half_size, uniforms.radii);
            let a1 = (1.0 - smoothstep(-uniforms.elevation*0.5, uniforms.elevation*0.5, d1)) * 0.4;
            
            let offset2 = vec2<f32>(0.0, uniforms.elevation * 1.5);
            let d2 = sd_rounded_box(local - offset2, half_size, uniforms.radii);
            let a2 = (1.0 - smoothstep(-uniforms.elevation*3.0, uniforms.elevation*3.0, d2)) * 0.2;
            
            let shadow_alpha = max(a1, a2) * in.color.a;
            shadow_layer = vec4<f32>(0.0, 0.0, 0.0, shadow_alpha);
        }

        // Composite
        var comp = shadow_layer;
        comp = comp + glow_layer; 
        comp = vec4<f32>(
            main_layer.rgb * main_layer.a + comp.rgb * (1.0 - main_layer.a),
            max(comp.a, main_layer.a)
        );
        
        final_color = comp;
    }
    else if (uniforms.mode == 5) {
        // Mesh Gradient (Aurora)
        // reuse padding/elevation for time? 
        // We need a time uniform. Let's use 'elevation' as time for Mode 5.
        // And rect.zw for resolution.
        
        let t = uniforms.elevation; 
        let uv = in.clip_position.xy / uniforms.rect.zw; // gl_FragCoord equivalent? 
        // in.clip_position is screen space coordinates? Yes.
        
        let p1 = vec2<f32>(0.5 + 0.3*sin(t*0.5), 0.5 + 0.3*cos(t*0.3));
        let p2 = vec2<f32>(0.2 + 0.4*sin(t*0.7 + 1.0), 0.8 + 0.2*cos(t*0.5 + 2.0));
        let p3 = vec2<f32>(0.8 + 0.2*sin(t*0.4 + 4.0), 0.2 + 0.5*cos(t*0.6 + 3.0));
        
        let aspect = uniforms.rect.z / uniforms.rect.w;
        let uv_aspect = vec2<f32>(uv.x * aspect, uv.y);
        let p1_aspect = vec2<f32>(p1.x * aspect, p1.y);
        let p2_aspect = vec2<f32>(p2.x * aspect, p2.y);
        let p3_aspect = vec2<f32>(p3.x * aspect, p3.y);

        let d1 = length(uv_aspect - p1_aspect);
        let d2 = length(uv_aspect - p2_aspect);
        let d3 = length(uv_aspect - p3_aspect);
        
        let c1 = vec3<f32>(0.1, 0.0, 0.3); 
        let c2 = vec3<f32>(0.0, 0.2, 0.4); 
        let c3 = vec3<f32>(0.2, 0.0, 0.1); 
        
        let w1 = 1.0 / (d1 * d1 + 0.01);
        let w2 = 1.0 / (d2 * d2 + 0.01);
        let w3 = 1.0 / (d3 * d3 + 0.01);
        
        var aurora = (c1*w1 + c2*w2 + c3*w3) / (w1 + w2 + w3);
        
        // Simple noise
        let noise = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 0.05;
        aurora += noise;
        
        final_color = vec4<f32>(aurora, 1.0);
    }
    else if (uniforms.mode == 6) {
        // SDF Arc
        let center = uniforms.rect.xy + uniforms.rect.zw * 0.5;
        // radii.x = radius, radii.y = thickness
        let r = uniforms.radii.x;
        let thickness = uniforms.radii.y;
        
        let local = in.world_pos - center;
        
        // Invert Y because screen coords Y is down, but math usually Y up?
        // But our projection maps 0..H to -1..1 (or similar).
        // Let's assume standard pixel coords for logic.
        // atan2(y, x). If Y is down, Y+ is down. 
        // 90 deg is Down.
        // This matches our Knob logic (0=Right, 90=Down).
        // So no Y inversion needed.
        
        let d = sd_arc(local, uniforms.start_angle, uniforms.end_angle, r, thickness);
        
        let aa = 1.0;
        let alpha = 1.0 - smoothstep(-aa, aa, d);
        
        final_color = vec4<f32>(in.color.rgb, in.color.a * alpha);
    }
    else if (uniforms.mode == 7) {
        // Plot mode (Vertex color passthrough)
        final_color = in.color;
    }
    
    // Linear -> sRGB conversion at end of pipeline (or let swapchain handle it?)
    // If output format is sRGB, hardware does it.
    // Assuming simple output:
    // return vec4<f32>(pow(final_color.rgb, vec3<f32>(1.0/2.2)), final_color.a);
    // But WGPU surface format usually handles sRGB if configured correctly.
    // Let's assume surface is sRGB-aware or we do explicit correction. 
    // OpenglBackend explicitly disabled GL_FRAMEBUFFER_SRGB and did manual power.
    // Here we'll do manual output correction to match.
    return vec4<f32>(pow(final_color.rgb, vec3<f32>(1.0/2.2)), final_color.a);
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var font_texture: texture_2d<f32>;

@group(0) @binding(2)
var font_sampler: sampler;

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
    out.color = in.color;
    out.world_pos = in.pos;
    return out;
}

// SDF for rounded rectangle with per-corner radius
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    // Select radius per quadrant
    var radius = r.x; // tl
    if (p.x > 0.0) { radius = r.y; } // tr
    if (p.x > 0.0 && p.y > 0.0) { radius = r.z; } // br
    if (p.x <= 0.0 && p.y > 0.0) { radius = r.w; } // bl

    let q = abs(p) - b + radius;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (uniforms.mode == 0) {
        // Solid color
        return in.color;
    }
    else if (uniforms.mode == 1) {
        // SDF text
        let dist = textureSample(font_texture, font_sampler, in.uv).r;
        let alpha = smoothstep(0.4, 0.6, dist);
        return vec4<f32>(in.color.rgb, in.color.a * alpha);
    }
    else if (uniforms.mode == 2) {
        // Rounded rectangle with SDF
        let center = uniforms.rect.xy + uniforms.rect.zw * 0.5;
        let half_size = uniforms.rect.zw * 0.5;
        let local = in.world_pos - center;
        
        let d = sd_rounded_box(local, half_size, uniforms.radii);
        
        // Anti-aliasing
        let aa = 1.0;
        let alpha = 1.0 - smoothstep(-aa, aa, d);

        // Border and interior
        var bg = in.color;
        if (uniforms.border_width > 0.0) {
            let interior_alpha = 1.0 - smoothstep(-aa, aa, d + uniforms.border_width);
            bg = mix(uniforms.border_color, in.color, interior_alpha);
        }

        // Shadow
        if (uniforms.elevation > 0.0) {
            let shadow_offset = vec2<f32>(0.0, uniforms.elevation * 0.5);
            let shadow_d = sd_rounded_box(local - shadow_offset, half_size, uniforms.radii);
            let shadow_alpha = (1.0 - smoothstep(-uniforms.elevation, uniforms.elevation * 2.0, shadow_d)) * 0.3;
            
            if (d > 0.0) {
                return vec4<f32>(0.0, 0.0, 0.0, shadow_alpha);
            }
        }

        return vec4<f32>(bg.rgb, bg.a * alpha);
    }
    else if (uniforms.mode == 7) {
        return in.color;
    }
    
    return in.color;
}
