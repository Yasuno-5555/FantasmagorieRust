use crate::tracea::emitter::traits::{Emitter, UnifiedOpIR};
use crate::tracea::emitter::cuda::CUDAEmitter;
use crate::tracea::emitter::rocm::ROCMEmitter;
use crate::tracea::emitter::metal::MetalEmitter;
use crate::tracea::runtime::manager::DeviceBackend;

pub struct UniversalEmitter {
    pub backend: DeviceBackend,
}

impl UniversalEmitter {
    pub fn new(backend: DeviceBackend) -> Self {
        Self { backend }
    }

    pub fn generate(&self, ir: UnifiedOpIR) -> String {
        if let crate::tracea::emitter::traits::UnifiedOpType::Elementwise { .. } = ir.op_type {
            return crate::tracea::emitter::elementwise::generate_elementwise(&ir);
        }
        if let crate::tracea::emitter::traits::UnifiedOpType::Conv2d { .. } = ir.op_type {
            #[cfg(feature = "vulkan")]
            if self.backend == DeviceBackend::Vulkan {
                return crate::tracea::emitter::vulkan::generate_vulkan_conv(&ir);
            }
            return crate::tracea::emitter::conv::generate_conv(&ir);
        }
        if let crate::tracea::emitter::traits::UnifiedOpType::FusedAttention { .. } = ir.op_type {
            return crate::tracea::emitter::attention::generate_attention(&ir, self.backend);
        }
        if let crate::tracea::emitter::traits::UnifiedOpType::Gemm { .. } = ir.op_type {
            return crate::tracea::emitter::gemm::generate_gemm(&ir, self.backend);
        }
        if let crate::tracea::emitter::traits::UnifiedOpType::MatrixCore { .. } = ir.op_type {
            #[cfg(feature = "vulkan")]
            if self.backend == DeviceBackend::Vulkan {
                return crate::tracea::emitter::vulkan::generate_vulkan_mma(&ir);
            }
            // CUDA/Metal fallbacks to Gemm or specialized emitters
        }

        match self.backend {
            DeviceBackend::Cuda => {
                let emitter = CUDAEmitter::new();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Rocm => {
                let emitter = ROCMEmitter::detect();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Metal => {
                let emitter = MetalEmitter::detect();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Cpu => {
                "/* CPU implementation is static */".to_string()
            }
            #[cfg(feature = "vulkan")]
            DeviceBackend::Vulkan => {
                // Vulkan specific ops handled above, fallback to generic GLSL
                "/* Vulkan generic fallback */".to_string()
            }
            #[cfg(feature = "wgpu")]
            DeviceBackend::Wgpu => {
                "/* WGPU generic fallback */".to_string()
            }
        }
    }
}
