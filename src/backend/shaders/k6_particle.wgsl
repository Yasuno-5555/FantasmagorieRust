// K6: Particle Life-Cycle (WGSL)
// GPU-driven particle simulation and spawning.

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    life: f32,
    color: vec4<f32>,
};

struct ParticleUniforms {
    dt: f32,
    num_particles: u32,
    spawn_pos: vec2<f32>,
    spawn_count: u32,
    time: f32,
};

@group(0) @binding(0) var<uniform> uniforms: ParticleUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> active_counter: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> pool_counter: atomic<u32>;

// Update Kernel
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.num_particles) {
        return;
    }

    if (particles[idx].life <= 0.0) {
        return;
    }

    // Physics
    particles[idx].pos += particles[idx].vel * uniforms.dt;
    particles[idx].life -= uniforms.dt;

    // Simple fade
    particles[idx].color.a = clamp(particles[idx].life, 0.0, 1.0);

    // Collision (Conceptual)
    // if (particles[idx].pos.y > 1.0) { particles[idx].vel.y *= -0.8; }

    if (particles[idx].life > 0.0) {
        atomicAdd(&active_counter, 1u);
    }
}

// Spawn Kernel
@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.spawn_count) {
        return;
    }

    // Try to acquire an index from the "dead" pool or simple atomic increment
    let p_idx = atomicAdd(&pool_counter, 1u) % uniforms.num_particles;
    
    // Low-quality hash for velocity
    let seed = f32(p_idx) + uniforms.time;
    let rx = sin(seed * 12.9898) * 43758.5453;
    let ry = cos(seed * 78.233) * 43758.5453;
    let rand_vel = vec2<f32>(fract(rx) - 0.5, fract(ry) - 0.5) * 2.0;

    particles[p_idx].pos = uniforms.spawn_pos;
    particles[p_idx].vel = rand_vel * 0.1;
    particles[p_idx].life = 5.0; // 5 seconds
    particles[p_idx].color = vec4<f32>(1.0, 0.5, 0.2, 1.0);
}
