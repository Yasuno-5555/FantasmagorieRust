// Tracea Indirect Dispatch Kernel (from K13)
// GPU-driven command generation for sprites, particles, and compute

struct DrawIndirectCommand {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

struct DispatchIndirectCommand {
    x: u32,
    y: u32,
    z: u32,
};

struct Counters {
    visible_sprites: u32,
    active_particles: u32,
    sdf_updates: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> counters: Counters;
@group(0) @binding(1) var<storage, read_write> draw_cmds: array<DrawIndirectCommand>;
@group(0) @binding(2) var<storage, read_write> dispatch_cmds: array<DispatchIndirectCommand>;

@compute @workgroup_size(1)
fn main() {
    let visible_sprites = counters.visible_sprites;
    let active_particles = counters.active_particles;
    let sdf_updates = counters.sdf_updates;

    // Sprite Draw Command
    draw_cmds[0].vertex_count = 6u;
    draw_cmds[0].instance_count = visible_sprites;
    draw_cmds[0].first_vertex = 0u;
    draw_cmds[0].first_instance = 0u;

    // Particle Draw Command
    draw_cmds[1].vertex_count = 6u;
    draw_cmds[1].instance_count = active_particles;
    draw_cmds[1].first_vertex = 0u;
    draw_cmds[1].first_instance = 0u;

    // Visibility Dispatch
    dispatch_cmds[0].x = (visible_sprites + 63u) / 64u;
    dispatch_cmds[0].y = 1u;
    dispatch_cmds[0].z = 1u;

    // SDF JFA Dispatch
    dispatch_cmds[1].x = (sdf_updates + 255u) / 256u;
    dispatch_cmds[1].y = 1u;
    dispatch_cmds[1].z = 1u;
}
