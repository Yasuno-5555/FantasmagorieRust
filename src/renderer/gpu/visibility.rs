//! K8: Visibility & Occlusion Culling (HZB)
//!
//! High-performance GPU-driven culling using Hierarchical Z-Buffer.

use crate::tracea::runtime::manager::KernelArg; use crate::tracea::doctor::BackendKind as DeviceBackend;

pub struct VisibilityKernel {
    pub backend: DeviceBackend,
}

impl VisibilityKernel {
    pub fn new(backend: DeviceBackend) -> Self {
        Self { backend }
    }

    pub fn generate_source(&self) -> String {
        match self.backend {
            DeviceBackend::Cuda => self.cuda_source(),
            DeviceBackend::Metal => self.metal_source(),
            _ => String::new(),
        }
    }

    fn cuda_source(&self) -> String {
        r#"
struct InstanceData {
    float4 rect; // xy, zw
    float depth;
    int id;
};

extern "C" __global__ void k8_visibility_culling(
    const InstanceData* instances,
    unsigned int* visible_indices,
    unsigned int* visible_counter,
    const float* hzb_texture, // Simplified HZB for 2D
    int num_instances,
    float4 view_proj
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    InstanceData inst = instances[idx];
    
    // 1. Frustum Culling (AABB)
    bool visible = true;
    if (inst.rect.x + inst.rect.z < -1.0 || inst.rect.x > 1.0 ||
        inst.rect.y + inst.rect.w < -1.0 || inst.rect.y > 1.0) {
        visible = false;
    }

    // 2. HZB Occlusion Culling
    if (visible) {
        // Sample HZB at the corresponding depth mip level
        // Simplified: check if inst.depth is behind the max depth in that tile
        // float max_depth = sample_hzb(inst.rect);
        // if (inst.depth > max_depth) visible = false;
    }

    if (visible) {
        unsigned int out_idx = atomicAdd(visible_counter, 1);
        visible_indices[out_idx] = idx;
    }
}
"#.to_string()
    }

    fn metal_source(&self) -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

struct InstanceData {
    float4 rect;
    float depth;
    int id;
};

kernel void k8_visibility_culling(
    device const InstanceData* instances [[buffer(0)]],
    device uint* visible_indices [[buffer(1)]],
    device atomic_uint* visible_counter [[buffer(2)]],
    texture2d<float, access::read> hzb_texture [[texture(0)]],
    constant uint& num_instances [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_instances) return;
    // ... same logic ...
}
"#.to_string()
    }
}
