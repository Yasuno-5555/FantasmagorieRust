#include <metal_stdlib>
using namespace metal;

struct Particle {
    float2 position;
    float2 velocity;
    float4 color;
    float life;
    float size;
    float2 _pad;
};

struct SimParams {
    float dt;
    float damping;
    float2 gravity;
    uint count;
    uint width;
    uint height;
    float2 attractor_pos;
    float attractor_strength;
};

// Hash function for random seed
uint hash(uint s) {
    s ^= 2747636419u;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    return s;
}

float random(uint seed) {
    return float(hash(seed)) / 4294967295.0;
}

kernel void init_particles(
    device Particle *particles [[buffer(0)]],
    constant SimParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    
    uint seed = gid + 12345;
    
    // Random position within bounds
    float x = random(seed) * float(params.width);
    float y = random(seed * 2) * float(params.height);
    
    particles[gid].position = float2(x, y);
    particles[gid].velocity = float2(0.0, 0.0);
    particles[gid].color = float4(1.0, 1.0, 1.0, 1.0);
    particles[gid].life = 1.0;
    particles[gid].size = 2.0;
}

kernel void update_particles(
    device Particle *particles [[buffer(0)]],
    constant SimParams &params [[buffer(1)]],
    texture2d<float, access::sample> sdf [[texture(0)]], // Optional SDF
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    
    Particle p = particles[gid];
    
    // Physics update
    float2 acc = params.gravity;
    
    // Attractor force
    float2 diff = params.attractor_pos - p.position;
    float dist = length(diff);
    if (dist > 1.0) {
        float2 dir = diff / dist;
        acc += dir * (params.attractor_strength / (dist * 0.1));
    }
    
    p.velocity += acc * params.dt;
    p.velocity *= params.damping;
    p.position += p.velocity * params.dt;
    
    // SDF collision (if texture bound)
    if (!is_null_texture(sdf)) {
        // Sample SDF at current position
        float2 uv = p.position / float2(params.width, params.height);
        
        // Simple bounds check
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
            float dist_val = sdf.sample(s, uv).r;
            
            // If distance < 0 (inside obstacle) or close, repel
            // Assuming signed distance: negative inside
            // Or unsigned distance from JFA? JFA is usually unsigned unless signed kernel used.
            // Let's assume unsigned distance to nearest obstacle.
            // If very close to 0, it means obstacle.
            
            // For now just basic ground plane bounce
            if (p.position.y < 0.0) {
                p.position.y = 0.0;
                p.velocity.y *= -0.8;
            }
        }
    } else {
        // Basic screen bounce
        if (p.position.x < 0.0 || p.position.x > float(params.width)) {
            p.velocity.x *= -1.0;
            p.position.x = clamp(p.position.x, 0.0, float(params.width));
        }
        if (p.position.y < 0.0 || p.position.y > float(params.height)) {
            p.velocity.y *= -0.8;
            p.position.y = clamp(p.position.y, 0.0, float(params.height));
        }
    }
    
    // Color update based on velocity
    float speed = length(p.velocity);
    p.color = float4(mix(float3(0.1, 0.2, 0.8), float3(1.0, 0.5, 0.2), min(speed * 0.01, 1.0)), 1.0);
    
    particles[gid] = p;
}
