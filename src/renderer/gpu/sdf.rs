//! K5: SDF Lighting (Jump Flooding Algorithm)
//!
//! Rapid 2D distance field generation for dynamic lighting and reflections.

use crate::tracea::runtime::manager::KernelArg; use crate::tracea::doctor::BackendKind as DeviceBackend;

pub struct SDFKernel {
    pub backend: DeviceBackend,
}

impl SDFKernel {
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
// Pass N of Jump Flooding Algorithm
extern "C" __global__ void k5_jfa_pass(
    const float2* input_seeds,
    float2* output_seeds,
    int width, int height,
    int step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float2 best_seed = input_seeds[y * width + x];
    float best_dist = 1e10f;
    
    if (best_seed.x >= 0) {
        float dx = best_seed.x - x;
        float dy = best_seed.y - y;
        best_dist = dx*dx + dy*dy;
    }

    // Check 8 neighbors at 'step' distance
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx * step;
            int ny = y + dy * step;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float2 s = input_seeds[ny * width + nx];
                if (s.x >= 0) {
                    float ds_x = s.x - x;
                    float ds_y = s.y - y;
                    float d = ds_x*ds_x + ds_y*ds_y;
                    if (d < best_dist) {
                        best_dist = d;
                        best_seed = s;
                    }
                }
            }
        }
    }

    output_seeds[y * width + x] = best_seed;
}

extern "C" __global__ void k5_sdf_final(
    const float2* seeds,
    float* sdf_buffer,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float2 s = seeds[y * width + x];
    float dist = sqrtf((s.x - x) * (s.x - x) + (s.y - y) * (s.y - y));
    sdf_buffer[y * width + x] = dist;
}
"#.to_string()
    }

    fn vulkan_source(&self) -> String {
        r#"#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputSeeds { vec2 input_seeds[]; };
layout(set = 0, binding = 1) buffer OutputSeeds { vec2 output_seeds[]; };

layout(push_constant) uniform Params {
    int width, height, step;
} p;

void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= p.width || y >= p.height) return;

    vec2 best_seed = input_seeds[y * p.width + x];
    float best_dist = 1e10;
    
    if (best_seed.x >= 0.0) {
        float dx = best_seed.x - float(x);
        float dy = best_seed.y - float(y);
        best_dist = dx*dx + dy*dy;
    }

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx * p.step;
            int ny = y + dy * p.step;
            
            if (nx >= 0 && nx < p.width && ny >= 0 && ny < p.height) {
                vec2 s = input_seeds[ny * p.width + nx];
                if (s.x >= 0.0) {
                    float ds_x = s.x - float(x);
                    float ds_y = s.y - float(y);
                    float d = ds_x*ds_x + ds_y*ds_y;
                    if (d < best_dist) {
                        best_dist = d;
                        best_seed = s;
                    }
                }
            }
        }
    }

    output_seeds[y * p.width + x] = best_seed;
}
"#.to_string()
    }
}
