//! K12: Audio Reactive (GPU FFT)
//!
//! Real-time audio spectrum analysis on the GPU.

use crate::tracea::runtime::manager::KernelArg; use crate::tracea::doctor::BackendKind as DeviceBackend;

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
}
