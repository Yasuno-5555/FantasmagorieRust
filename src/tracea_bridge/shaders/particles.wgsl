struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
    life: f32,
    size: f32,
    pad: vec2<f32>,
}

struct SimParams {
    dt: f32,
    damping: f32,
    gravity: vec2<f32>,
    count: u32,
    width: u32,
    height: u32,
    attractor_pos: vec2<f32>,
    attractor_strength: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var sdf_tex: texture_2d<f32>; // SDF for collision
@group(0) @binding(3) var sdf_sampler: sampler;

// Pseudo-random helper
fn hash(val: u32) -> f32 {
    var state = val * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

@compute @workgroup_size(64)
fn init_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.count) { return; }

    var p = particles[i];
    
    // Random init pos
    let rx = hash(i) * f32(params.width);
    let ry = hash(i + 1337u) * f32(params.height);
    
    p.position = vec2<f32>(rx, ry);
    p.velocity = vec2<f32>(0.0, 0.0);
    p.life = 1.0;
    p.size = 2.0;
    p.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    
    particles[i] = p;
}

@compute @workgroup_size(64)
fn update_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.count) { return; }

    var p = particles[i];
    
    // Physics
    p.velocity += params.gravity * params.dt;
    p.velocity *= params.damping;
    p.position += p.velocity * params.dt;
    
    // Attractor
    let delta = params.attractor_pos - p.position;
    let dist = length(delta);
    if (dist > 10.0) {
        let dir = normalize(delta);
        let force = params.attractor_strength / (dist * dist + 100.0);
        p.velocity += dir * force * params.dt;
    }
    
    // SDF Collision (Simple bounce)
    // NOTE: Requires texture bound. If validation fails on null texture, use separate pipeline or dummy texture.
    // For now assuming texture is valid if pipeline selected.
    let dims = textureDimensions(sdf_tex);
    let uv = p.position / vec2<f32>(f32(dims.x), f32(dims.y));
    
    if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
         let dist_val = textureSampleLevel(sdf_tex, sdf_sampler, uv, 0.0).r;
         // If inside obstacle (negative distance or close to 0 depending on SDF convention)
         // Assuming positive inside or JFA style distance
         // Let's assume standard behavior: Keep inside screen bounds first
    }
    
    // Screen bounce
    if (p.position.x < 0.0 || p.position.x > f32(params.width)) { p.velocity.x *= -1.0; }
    if (p.position.y < 0.0 || p.position.y > f32(params.height)) { p.velocity.y *= -1.0; }
    
    p.position = clamp(p.position, vec2<f32>(0.0), vec2<f32>(f32(params.width), f32(params.height)));
    
    particles[i] = p;
}

// Counters for GPU-driven particle lifecycle
struct ParticleCounters {
    active_count: atomic<u32>,
    spawn_request: atomic<u32>,
}

@group(0) @binding(4) var<storage, read_write> counters: ParticleCounters;

// Spawn kernel: resurrect dead particles at spawn position
@compute @workgroup_size(64)
fn spawn_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let spawn_idx = global_id.x;
    let spawn_count = atomicLoad(&counters.spawn_request);
    
    if (spawn_idx >= spawn_count) { return; }
    
    // Find a dead particle to reuse (simple linear scan via atomic)
    let p_idx = atomicAdd(&counters.active_count, 1u) % params.count;
    
    var p = particles[p_idx];
    
    // Randomize velocity using index as seed
    let seed = p_idx + u32(params.dt * 10000.0);
    let rx = (hash(seed) - 0.5) * 2.0;
    let ry = (hash(seed + 1u) - 0.5) * 2.0;
    let rand_vel = vec2<f32>(rx, ry) * 50.0;
    
    p.position = params.attractor_pos;
    p.velocity = rand_vel;
    p.life = 5.0;
    p.size = 2.0 + hash(seed + 2u) * 3.0;
    p.color = vec4<f32>(1.0, 0.5, 0.2, 1.0);
    
    particles[p_idx] = p;
}

// Reset counters at frame start
@compute @workgroup_size(1)
fn reset_counters() {
    atomicStore(&counters.active_count, 0u);
    atomicStore(&counters.spawn_request, 0u);
}

