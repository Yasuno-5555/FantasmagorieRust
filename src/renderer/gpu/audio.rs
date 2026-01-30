//! K12: Audio Reactive (GPU FFT)
//!
//! Real-time audio spectrum analysis on the GPU.

use tracea::runtime::manager::KernelArg; use tracea::doctor::BackendKind as DeviceBackend;

pub struct AudioKernel {
    pub backend: DeviceBackend,
}

impl AudioKernel {
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
// Simplified GPU FFT (Cooley-Tukey)
extern "C" __global__ void k12_audio_fft_pass(
    float2* data,
    int n,
    int step
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n / 2) return;

    int section_size = step * 2;
    int section = i / step;
    int offset = i % step;
    int i0 = section * section_size + offset;
    int i1 = i0 + step;

    float2 v0 = data[i0];
    float2 v1 = data[i1];

    // Butterfly operation
    float angle = -2.0f * 3.14159f * (float)offset / (float)section_size;
    float2 w = make_float2(cosf(angle), sinf(angle));
    
    // Complex multiply: w * v1
    float2 wv1 = make_float2(
        w.x * v1.x - w.y * v1.y,
        w.x * v1.y + w.y * v1.x
    );

    data[i0] = make_float2(v0.x + wv1.x, v0.y + wv1.y);
    data[i1] = make_float2(v0.x - wv1.x, v0.y - wv1.y);
}
"#.to_string()
    }

    fn vulkan_source(&self) -> String {
        r#"#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data { vec2 data[]; };

layout(push_constant) uniform Params {
    int n, step;
} p;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.n / 2) return;

    int section_size = p.step * 2;
    int section = i / p.step;
    int offset = i % p.step;
    int i0 = section * section_size + offset;
    int i1 = i0 + p.step;

    vec2 v0 = data[i0];
    vec2 v1 = data[i1];

    // Butterfly operation
    float angle = -2.0 * 3.14159 * float(offset) / float(section_size);
    vec2 w = vec2(cos(angle), sin(angle));
    
    // Complex multiply: w * v1
    vec2 wv1 = vec2(
        w.x * v1.x - w.y * v1.y,
        w.x * v1.y + w.y * v1.x
    );

    data[i0] = v0 + wv1;
    data[i1] = v0 - wv1;
}
"#.to_string()
    }
}
