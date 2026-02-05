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

struct CinematicParams {
    float exposure;
    float ca_strength;
    float vignette_intensity;
    float bloom_intensity;
    uint tonemap_mode;
    uint bloom_mode;
    float grain_strength;
    float time;
    float lut_intensity;
    float blur_radius;
    float motion_blur_strength;
    uint debug_mode;
    float2 light_pos;
    float gi_intensity;
    float volumetric_intensity;
    float4 light_color;
};

struct ParticleControl {
    uint count;
    float emit_rate;
    float2 gravity;
    float delta_time;
    float seed;
    float drag_coefficient;
    uint _pad;
};

// Utils
float hash(float p) {
    float3 p3 = fract(float3(p) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float2 random2(float p) {
    return float2(hash(p), hash(p + 42.1337));
}

// Compute
kernel void update_particles(
    device Particle* particles [[buffer(0)]],
    constant CinematicParams& cinema [[buffer(1)]],
    texture2d<float> sdf [[texture(2)]],
    sampler s [[sampler(3)]],
    constant ParticleControl& control [[buffer(4)]],
    texture2d<float> velocity_tex [[texture(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= control.count) return;
    
    Particle p = particles[id];
    
    // Life
    p.life -= control.delta_time;
    
    // Respawn
    if (p.life <= 0.0) {
        float rnd_seed = float(id) * 0.1 + control.seed;
        float2 rnd = random2(rnd_seed);
        p.position = cinema.light_pos + (rnd - 0.5) * 50.0;
        
        float angle = hash(rnd_seed + 0.13) * 6.28318;
        float speed = hash(rnd_seed + 0.17) * 200.0 + 50.0;
        p.velocity = float2(cos(angle), sin(angle)) * speed;
        
        p.life = hash(rnd_seed + 0.19) * 3.0 + 1.0;
        p.color = float4(1.0, 0.5, 0.1, 1.0);
        p.size = hash(rnd_seed + 0.23) * 8.0 + 3.0;
        
        particles[id] = p;
        return;
    }
    
    // Flow Field
    float2 dims = float2(sdf.get_width(), sdf.get_height());
    float2 uv = p.position / dims;
    
    if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
        float2 scene_vel = velocity_tex.sample(s, uv).xy;
        p.velocity += scene_vel * control.drag_coefficient * 100.0 * control.delta_time;
    }
    
    // Physics
    p.velocity += control.gravity * control.delta_time;
    float2 next_pos = p.position + p.velocity * control.delta_time;
    
    // Collision
    float2 sdf_uv = next_pos / dims;
    
    if (sdf_uv.x > 0.0 && sdf_uv.x < 1.0 && sdf_uv.y > 0.0 && sdf_uv.y < 1.0) {
        float dist = sdf.sample(s, sdf_uv).r * dims.x; // approx pixels
        
        if (dist < 5.0) {
            float2 eps = float2(1.0/dims.x, 0.0);
            float d_c = dist / dims.x;
            float d_r = sdf.sample(s, sdf_uv + eps.xy).r;
            float d_u = sdf.sample(s, sdf_uv + eps.yx).r;
            
            float2 normal = normalize(float2(d_r - d_c, d_u - d_c));
            
            p.velocity = reflect(p.velocity, -normal) * 0.6;
            p.position += normal * (5.0 - dist);
        } else {
             p.position = next_pos;
        }
    } else {
        p.position = next_pos;
    }
    
    // Color Ramp
    if (p.life > 1.5) {
        p.color = mix(float4(1.0, 0.2, 0.0, 1.0), float4(1.0, 0.8, 0.2, 1.0), clamp((p.life - 1.5), 0.0, 1.0));
    } else {
        p.color = mix(float4(0.2, 0.2, 0.2, 0.0), float4(1.0, 0.2, 0.0, 1.0), clamp(p.life / 1.5, 0.0, 1.0));
    }
    p.color.a = clamp(p.life, 0.0, 1.0);
    
    particles[id] = p;
}

// Render
struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 uv;
    float2 screen_uv;
};

vertex VertexOut vs_particles(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const Particle* particles [[buffer(0)]],
    texture2d<float> sdf [[texture(2)]] // Reusing binding 2 for SDF/Dims
) {
    Particle p = particles[instance_id];
    
    float x = float(vertex_id & 1) * 2.0 - 1.0;
    float y = float(vertex_id / 2) * 2.0 - 1.0;
    float2 local_pos = float2(x, y);
    
    float2 dims = float2(sdf.get_width(), sdf.get_height());
    float2 world_pos = p.position + local_pos * p.size;
    float2 ndc = (world_pos / dims) * 2.0 - 1.0;
    
    VertexOut out;
    out.position = float4(ndc.x, -ndc.y, 0.0, 1.0);
    out.color = p.color;
    out.uv = local_pos;
    
    // Screen UV for depth
    // NDC -1..1 -> 0..1
    out.screen_uv = float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5)); // Flip Y used here
    
    return out;
}

fragment float4 fs_particles(
    VertexOut in [[stage_in]],
    texture2d<float> depth_tex [[texture(6)]],
    sampler s [[sampler(3)]]
) {
    float r = length(in.uv);
    if (r > 1.0) discard_fragment();
    float shape_alpha = 1.0 - smoothstep(0.5, 1.0, r);
    
    // Soft Particles
    float scene_depth = depth_tex.sample(s, in.screen_uv).r;
    float part_depth = in.position.z; // Frag depth
    
    // This depth comparison depends on coordinate system.
    // Metal depth is 0..1. 
    // Assuming standard Z-buffer.
    // If scene_depth is background (1.0) and particle is 0.0, massive diff.
    // If scene_depth is obstacle (0.5), diff is 0.5.
    
    // We want fade if scene_depth is *just behind* particle?
    // Or if particle intersects scene.
    // Actually in 2D, we might not have meaningful Z difference if everything is Z=0.
    // But let's assume valid depth buffer usage.
    
    float depth_diff = abs(scene_depth - part_depth);
    float soft_fade = clamp(depth_diff * 100.0, 0.0, 1.0);
    
    return in.color * shape_alpha * soft_fade * 1.5;
}
