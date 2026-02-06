struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
    life: f32,
    size: f32,
    _pad: vec2<f32>, 
};

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

struct ParticleControl {
    count: u32,
    emit_rate: f32,
    gravity: vec2<f32>,
    delta_time: f32,
    seed: f32,
    drag_coefficient: f32,
    _pad0: u32,
    jitter: vec2<f32>,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> cinema: CinematicParams;
@group(0) @binding(2) var t_sdf: texture_2d<f32>;
@group(0) @binding(3) var s_sdf: sampler;
@group(0) @binding(4) var<uniform> control: ParticleControl;
@group(0) @binding(5) var t_velocity: texture_2d<f32>;
@group(0) @binding(6) var t_depth: texture_depth_2d; // Depth for Soft Particles

// ... Hash functions (unchanged) ...
fn hash(p: f32) -> f32 {
    var p3 = fract(vec3<f32>(p) * vec3<f32>(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn random2(p: f32) -> vec2<f32> {
    return vec2<f32>(hash(p), hash(p + 42.1337));
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= control.count) {
        return;
    }

    var p = particles[index];

    // Life update
    p.life -= control.delta_time;

    // Emitter Logic (Respawn if dead)
    if (p.life <= 0.0) {
        let rnd = random2(f32(index) * 0.1 + control.seed);
        p.position = cinema.light_pos + (rnd - 0.5) * 50.0;
        
        let angle = hash(f32(index) * 0.13 + control.seed) * 6.28318;
        let speed = hash(f32(index) * 0.17 + control.seed) * 200.0 + 50.0;
        p.velocity = vec2<f32>(cos(angle), sin(angle)) * speed;
        
        p.life = hash(f32(index) * 0.19 + control.seed) * 3.0 + 1.0;
        p.color = vec4<f32>(1.0, 0.5, 0.1, 1.0); // Start Color (Orange/Gold)
        p.size = hash(f32(index) * 0.23 + control.seed) * 8.0 + 3.0; // Slightly larger for soft effect
        
        particles[index] = p;
        return;
    }

    // Interaction Force (Flow Field)
    let dims = vec2<f32>(textureDimensions(t_velocity));
    let uv = p.position / dims; // Assuming Position matches screen coords 1:1
    if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
        // Sample velocity
        // Velocity texture usually stores (vx, vy) in R,G.
        // Assuming format is Rgba16Float.
        let scene_vel = textureSampleLevel(t_velocity, s_sdf, uv, 0.0).xy;
        
        // Advection: pull particle towards flow
        // Force = (TargetVel - CurrentVel) * Drag
        // Or simple addition: velocity += scene_vel * drag * dt
        
        // Let's use simple addition for "wind" effect
        p.velocity += scene_vel * control.drag_coefficient * 100.0 * control.delta_time;
        // 100.0 boost because scene velocity might be small (per frame pixels?)
    }

    // Physics
    p.velocity += control.gravity * control.delta_time;
    let next_pos = p.position + p.velocity * control.delta_time;

    // SDF Collision
    let sdf_dims = vec2<f32>(textureDimensions(t_sdf));
    let sdf_uv = next_pos / sdf_dims;
    
    if (sdf_uv.x > 0.0 && sdf_uv.x < 1.0 && sdf_uv.y > 0.0 && sdf_uv.y < 1.0) {
        let dist = textureSampleLevel(t_sdf, s_sdf, sdf_uv, 0.0).r * sdf_dims.x;
        
        if (dist < 5.0) {
             let eps = vec2<f32>(1.0 / sdf_dims.x, 0.0);
             let d_c = dist / sdf_dims.x;
             let d_r = textureSampleLevel(t_sdf, s_sdf, sdf_uv + eps.xy, 0.0).r;
             let d_u = textureSampleLevel(t_sdf, s_sdf, sdf_uv + eps.yx, 0.0).r;
             
             let grad_x = d_r - d_c;
             let grad_y = d_u - d_c;
             let normal = normalize(vec2<f32>(grad_x, grad_y));
             
             p.velocity = reflect(p.velocity, -normal) * 0.6; // Bounciness
             p.position += normal * (5.0 - dist);
        } else {
             p.position = next_pos;
        }
    } else {
        p.position = next_pos;
    }
    
    // Color Ramp over Life
    // Start: Gold (1.0, 0.8, 0.2), End: Purple/Smoke (0.1, 0.0, 0.2)
    // Normalized life 1.0 (birth) -> 0.0 (death)
    // We want life to map to progress 0..1
    // Actually p.life keeps decreasing. Max life varies.
    // Easier: Just decay color towards end color.
    // Let's make it burn out: Yellow -> Red -> Smoke
    
    // Simple gradient based on current life value?
    // Since we don't store max_life, we can approximation or just tint based on remaining life.
    if (p.life > 1.5) {
        p.color = mix(vec4<f32>(1.0, 0.2, 0.0, 1.0), vec4<f32>(1.0, 0.8, 0.2, 1.0), clamp((p.life - 1.5), 0.0, 1.0));
    } else {
        p.color = mix(vec4<f32>(0.2, 0.2, 0.2, 0.0), vec4<f32>(1.0, 0.2, 0.0, 1.0), clamp(p.life / 1.5, 0.0, 1.0));
    }
    
    // Fade out alpha
    p.color.a = clamp(p.life, 0.0, 1.0);

    particles[index] = p;
}


// --- Rendering ---

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) screen_uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_in: u32,
    @builtin(instance_index) instance_in: u32
) -> VertexOutput {
    let p = particles[instance_in];
    
    let x = f32(vertex_in & 1u) * 2.0 - 1.0;
    let y = f32(vertex_in / 2u) * 2.0 - 1.0;
    let local_pos = vec2<f32>(x, y);
    
    let dims = vec2<f32>(textureDimensions(t_sdf));
    let world_pos = p.position + local_pos * p.size;
    
    // Apply Jitter
    // jitter is in pixels. ndc is 2.0 wide.
    let jitter_ndc = (control.jitter / dims) * 2.0;
    
    let ndc_pos = (world_pos / dims) * 2.0 - 1.0 + jitter_ndc;
    
    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_pos.x, -ndc_pos.y, 0.0, 1.0);
    out.color = p.color;
    out.uv = local_pos;
    
    // Pass screen UV for depth sampling
    // NDC -1..1 -> 0..1
    out.screen_uv = vec2<f32>(ndc_pos.x * 0.5 + 0.5, -ndc_pos.y * 0.5 + 0.5); // Y flipped for texture sampling
    // Wait, WGPU clip Y is -1..1 (up) or down?
    // t_depth sampling requires 0..1 (top-down or bottom-up depending on backend).
    // In WGPU/Metal, texture coords 0,0 is usually Top-Left.
    // clip_positon Y is Down (NDC -1 is bottom).
    // Let's standardise: screen_uv.y = 1.0 - (ndc.y * 0.5 + 0.5) if NDC Y is Up.
    // WGPU NDC: Y is Up. Texture UV: Y is Down (0 is top).
    // So 1.0 - ... is correct.
    out.screen_uv = vec2<f32>(ndc_pos.x * 0.5 + 0.5, 1.0 - (ndc_pos.y * 0.5 + 0.5));

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Soft Circle Shape
    let r = length(in.uv);
    if (r > 1.0) { discard; }
    let shape_alpha = 1.0 - smoothstep(0.5, 1.0, r);
    
    // Soft Particles (Depth Fade)
    // Sample Scene Depth
    // NOTE: WGPU Depth is usually 0.0 (near) to 1.0 (far), or reverse depending on projection.
    // Fantasmagorie uses default Perspective/Ortho.
    // Does 2D renderer write depth? Yes, Geometry pass writes depth.
    let scene_depth = textureSample(t_depth, s_sdf, in.screen_uv);
    
    // Calculate particle linear depth? 
    // In 2D, everything is z=0 or sorted by layer (stored in Z?).
    // Geometry pass writes Z values.
    // Particles are usually drawn "on top" but if they are 3D volumetric, they respect Z.
    // Here particles are at Z=0?
    // Let's assume particles are "in front" if Z is not written.
    // But for "Soft Particles", we usually compare particle Z vs Scene Z.
    // If we don't write Particle Z, we can't compare easily unless we pass it.
    // But wait, the user asked for Soft Particles (Depth Buffer fade).
    // This implies fading when Particle Z is close to Scene Z (e.g. ground).
    // In 2D top-down, Z is usually "Height" or "Layer".
    // If ground is Z=0 and particle is Z=0, no fade?
    // Actually, maybe User means "Adhesion" style soft particles where they fade if they are "behind" geometry?
    // Or maybe just fading out if they overlap geometry?
    // In 2D SDF context, "geometry" is drawn to SDF.
    // But Depth Buffer contains 3D/Layer depth?
    // If the scene is purely 2D, Depth Buffer might just be flat?
    // Ah, `GeometryNode` draws sprites with Z depth?
    // Let's assume standard behavior:
    // Fade = clamp((SceneDepth - ParticleDepth) * Scale, 0, 1)
    
    // If particles are drawn without depth write/test (which they are, LoadOp::Load),
    // we need to know intrinsic particle depth.
    // For 2D, let's assume particles are "floating" at specific Z or just use a fake Z?
    // Or maybe we treat SDF distance as "depth" for volumetric effect?
    // But request said "Depth Buffer fade".
    
    // Let's implement generic soft particle fade.
    let part_depth = in.clip_position.z; // Frag depth
    // scene_depth is likely raw 0..1.
    // In standard Z-buffer, larger is further (0..1).
    // Fade if SceneDepth is close behind ParticleDepth.
    // Diff = SceneDepth - PartDepth.
    // If Diff is small positive, fade.
    
    let depth_diff = scene_depth - part_depth;
    let soft_fade = clamp(depth_diff * 100.0, 0.0, 1.0); // 100.0 is softness scale
    
    return in.color * shape_alpha * soft_fade * 1.5;
}
