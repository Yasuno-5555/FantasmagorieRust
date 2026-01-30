//! K6: Particle Life-Cycle
//!
//! GPU-driven particle simulation with spatial hash collision.

use tracea::runtime::manager::KernelArg; use tracea::doctor::BackendKind as DeviceBackend;

pub struct ParticleKernel {
    pub backend: DeviceBackend,
}

impl ParticleKernel {
    pub fn new(backend: DeviceBackend) -> Self {
        Self { backend }
    }

    pub fn generate_source(&self) -> String {
        match self.backend {
            DeviceBackend::Cuda => self.cuda_source(),
            DeviceBackend::Vulkan => self.vulkan_source(),
            _ => String::new(),
        }
    }

    fn cuda_source(&self) -> String {
        r#"
struct Particle {
    float2 pos;
    float2 vel;
    float life;
    float4 color;
};

// K6: Particle Update & Collision
extern "C" __global__ void k6_particle_update(
    Particle* particles,
    int* spatial_hash,
    unsigned int* active_counter,
    int num_particles,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    if (particles[idx].life <= 0.0f) return;

    // 1. Update Physics
    particles[idx].pos += particles[idx].vel * dt;
    particles[idx].life -= dt;

    // 2. Spatial Hash Collision (Simplified)
    // int cell = hash_pos(particles[idx].pos);
    // resolve_collision(particles[idx], cell, spatial_hash);

    // 3. Mark for Indirect Draw if alive
    if (particles[idx].life > 0.0f) {
        atomicAdd(active_counter, 1);
    }
}

// GPU-driven Spawning
extern "C" __global__ void k6_particle_spawn(
    Particle* particles,
    unsigned int* pool_counter,
    float2 spawn_pos,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Acquire from pool
    unsigned int p_idx = atomicAdd(pool_counter, 1);
    // initialize particles[p_idx]
}
"#.to_string()
    }

    fn vulkan_source(&self) -> String {
        r#"#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec2 pos;
    vec2 vel;
    float life;
    vec4 color;
};

layout(set = 0, binding = 0) buffer Particles { Particle particles[]; };
layout(set = 0, binding = 1) buffer ActiveCounter { uint active_counter; };

layout(push_constant) uniform Params {
    int num_particles;
    float dt;
} p;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= p.num_particles) return;

    if (particles[idx].life <= 0.0) return;

    particles[idx].pos += particles[idx].vel * p.dt;
    particles[idx].life -= p.dt;

    if (particles[idx].life > 0.0) {
        atomicAdd(active_counter, 1);
    }
}
"#.to_string()
    }
}
