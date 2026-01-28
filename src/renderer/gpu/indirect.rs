//! K13: Indirect Dispatch Kernel (The Tower)
//!
//! Generates Indirect Dispatch and Draw commands on the GPU.

use crate::tracea::core::config::PipelineConfig;
use crate::tracea::runtime::manager::KernelArg; use crate::tracea::doctor::BackendKind as DeviceBackend;

pub struct IndirectDispatchKernel {
    pub backend: DeviceBackend,
}

impl IndirectDispatchKernel {
    pub fn new(backend: DeviceBackend) -> Self {
        Self { backend }
    }

    pub fn generate_source(&self) -> String {
        match self.backend {
            DeviceBackend::Cuda => self.cuda_source(),
            DeviceBackend::Rocm => self.cuda_source(), // ROCm uses HIP (CUDA-ish)
            DeviceBackend::Metal => self.metal_source(),
            DeviceBackend::Vulkan => String::new(), // TODO: Vulkan indirect
            DeviceBackend::Cpu => String::new(),
        }
    }

    fn cuda_source(&self) -> String {
        r#"
extern "C" __global__ void k13_indirect_dispatch(
    unsigned int* dispatch_cmds,
    unsigned int* draw_cmds,
    unsigned int* counters
) {
    // K1: Sprites
    unsigned int visible_sprites = counters[0];
    draw_cmds[0] = 6;               // vertexCount
    draw_cmds[1] = visible_sprites; // instanceCount
    draw_cmds[2] = 0;               // firstVertex
    draw_cmds[3] = 0;               // firstInstance

    // K6: Particles
    unsigned int particles = counters[1];
    draw_cmds[4] = 6;
    draw_cmds[5] = particles;
    draw_cmds[6] = 0;
    draw_cmds[7] = 0;

    // Dispatches
    dispatch_cmds[0] = (visible_sprites + 63) / 64; // K8 or similar
    dispatch_cmds[3] = (particles + 255) / 256;     // K6
}
"#.to_string()
    }

    fn metal_source(&self) -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

struct DispatchCommand {
    uint3 x, y, z;
};

struct DrawCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

kernel void k13_indirect_dispatch(
    device uint* dispatch_cmds [[buffer(0)]],
    device uint* draw_cmds [[buffer(1)]],
    device uint* counters [[buffer(2)]]
) {
    uint visible_sprites = counters[0];
    uint particles = counters[1];

    // Draw Commands
    draw_cmds[0] = 6;
    draw_cmds[1] = visible_sprites;
    // ...
}
"#.to_string()
    }
}
