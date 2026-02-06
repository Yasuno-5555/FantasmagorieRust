#include <metal_stdlib>
using namespace metal;

struct SPHParticle {
    float2 position;
    float2 velocity;
    float density;
    float pressure;
    float2 _pad;
};

struct SPHParams {
    uint count;
    uint grid_width;
    uint grid_height;
    float cell_size;
    
    float h;
    float dt;
    float rest_density;
    float stiff;
    float viscosity;
    float2 gravity;
    float _pad;
};

// Utils
int get_cell_index(float2 pos, float cell_size, uint width, uint height) {
    int x = int(pos.x / cell_size);
    int y = int(pos.y / cell_size);
    if (x < 0 || x >= int(width) || y < 0 || y >= int(height)) return -1;
    return y * int(width) + x;
}

// Kernels

// Poly6 Kernel for Density
float poly6(float r2, float h2, float h) {
    if (r2 > h2) return 0.0;
    float term = h2 - r2;
    // Coeff: 315 / (64 * PI * h^9)
    // Simplify for performance, we care about relative values mostly or use classic SPH constants
    return 1.0 * term * term * term; 
}

// Spiky Gradient Kernel for Pressure
float2 spiky_grad(float2 r, float dist, float h) {
    if (dist > h || dist <= 0.001) return float2(0.0);
    float term = h - dist;
    // Coeff: -45 / (PI * h^6)
    float scalar = term * term;
    return normalize(r) * scalar;
}

// 1. Init
kernel void init_sph(
    device SPHParticle *particles [[buffer(0)]],
    constant SPHParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    
    // Grid-like init
    uint row_size = 100; // sqrt(10000) approx
    float spacing = params.h * 0.8;
    
    float x = (gid % row_size) * spacing + 500.0;
    float y = (gid / row_size) * spacing + 500.0;
    
    particles[gid].position = float2(x, y);
    particles[gid].velocity = float2(0.0, 0.0);
    particles[gid].density = 0.0;
    particles[gid].pressure = 0.0;
}

// 2. Clear Grid
kernel void clear_grid(
    device atomic_int *grid_head [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    // We assume buffer size covers grid
    atomic_store_explicit(&grid_head[gid], -1, memory_order_relaxed);
}

// 3. Build Grid
kernel void build_grid(
    device SPHParticle *particles [[buffer(0)]],
    constant SPHParams &params [[buffer(1)]],
    device atomic_int *grid_head [[buffer(2)]],
    device int *particle_next [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    
    float2 pos = particles[gid].position;
    int cell_idx = get_cell_index(pos, params.cell_size, params.grid_width, params.grid_height);
    
    if (cell_idx != -1) {
        // Atomic exchange: head[cell] = gid, return old head
        int next = atomic_exchange_explicit(&grid_head[cell_idx], int(gid), memory_order_relaxed);
        particle_next[gid] = next;
    } else {
        particle_next[gid] = -1;
    }
}

// 4. Compute Density & Pressure
kernel void compute_density(
    device SPHParticle *particles [[buffer(0)]],
    constant SPHParams &params [[buffer(1)]],
    device atomic_int *grid_head [[buffer(2)]],
    device int *particle_next [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    
    float2 pos = particles[gid].position;
    float h = params.h;
    float h2 = h * h;
    float density = 0.0;
    
    int cx = int(pos.x / params.cell_size);
    int cy = int(pos.y / params.cell_size);
    
    // Neighbor Search (3x3)
    for (int y = cy - 1; y <= cy + 1; y++) {
        for (int x = cx - 1; x <= cx + 1; x++) {
             if (x < 0 || x >= int(params.grid_width) || y < 0 || y >= int(params.grid_height)) continue;
             int cell_idx = y * int(params.grid_width) + x;
             
             // Traverse linked list
             int neighbor_id = atomic_load_explicit(&grid_head[cell_idx], memory_order_relaxed);
             while (neighbor_id != -1) {
                 float2 n_pos = particles[neighbor_id].position;
                 float dist2 = length_squared(pos - n_pos);
                 
                 if (dist2 < h2) {
                     density += poly6(dist2, h2, h);
                 }
                 
                 // WatchDog for infinite loops (shouldn't happen if acyclic)
                 neighbor_id = particle_next[neighbor_id];
             }
        }
    }
    
    // Update particle
    density = max(density, params.rest_density); // Avoid div by zero / neg pressure
    particles[gid].density = density;
    // P = k * (rho - rho0)
    particles[gid].pressure = params.stiff * (density - params.rest_density);
}

// 5. Force & Integrate
kernel void compute_forces(
    device SPHParticle *particles [[buffer(0)]],
    constant SPHParams &params [[buffer(1)]],
    device atomic_int *grid_head [[buffer(2)]],
    device int *particle_next [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    
    SPHParticle p = particles[gid];
    float2 force = float2(0.0);
    float h = params.h;
    float h2 = h * h;
    
    int cx = int(p.position.x / params.cell_size);
    int cy = int(p.position.y / params.cell_size);
    
    // Neighbor Search (3x3)
    for (int y = cy - 1; y <= cy + 1; y++) {
        for (int x = cx - 1; x <= cx + 1; x++) {
             if (x < 0 || x >= int(params.grid_width) || y < 0 || y >= int(params.grid_height)) continue;
             int cell_idx = y * int(params.grid_width) + x;
             
             int neighbor_id = atomic_load_explicit(&grid_head[cell_idx], memory_order_relaxed);
             while (neighbor_id != -1) {
                 if (uint(neighbor_id) != gid) {
                     SPHParticle n = particles[neighbor_id];
                     float2 diff = p.position - n.position;
                     float dist = length(diff);
                     
                     if (dist < h && dist > 0.001) {
                         // Pressure Force: Fp = -grad(P)
                         // Simple symetric force: (Pi + Pj) * grad(W)
                         // Here simplified: (Pi + Pj)/(2*rho) * grad
                         // We ignore rho division scaling for simplicity in this demo, absorbing into 'stiff'
                         
                         float p_term = (p.pressure + n.pressure) * 0.5;
                         force -= spiky_grad(diff, dist, h) * p_term;
                         
                         // Viscosity would go here
                     }
                 }
                 neighbor_id = particle_next[neighbor_id];
             }
        }
    }
    
    // Gravity
    force += params.gravity * p.density; // F = ma, mass ~ density 
    
    // Integrate
    float2 acc = force / p.density;
    p.velocity += acc * params.dt;
    p.position += p.velocity * params.dt;
    
    // Boundaries
    if (p.position.x < 0.0) { p.position.x = 0.0; p.velocity.x *= -0.5; }
    if (p.position.x > float(params.grid_width * params.cell_size)) {
         p.position.x = float(params.grid_width * params.cell_size); p.velocity.x *= -0.5; 
    }
    if (p.position.y < 0.0) { p.position.y = 0.0; p.velocity.y *= -0.5; }
    if (p.position.y > float(params.grid_height * params.cell_size)) {
         p.position.y = float(params.grid_height * params.cell_size); p.velocity.y *= -0.5; 
    }
    
    particles[gid] = p;
}
