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
            return crate::tracea::emitter::conv::generate_conv(&ir);
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
            DeviceBackend::Vulkan => {
                "/* Vulkan (WGSL/SPIR-V) implementation placeholder */".to_string()
            }
        }
    }
}
