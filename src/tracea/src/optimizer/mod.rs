use crate::PipelineConfig;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

use crate::runtime::manager::DeviceBackend;
use crate::core::backend::{Device, CudaArch, CpuArch};
pub use problem::{ProblemDescriptor, LayerType, HeroConfig, ArchHint, HeroScope, Layout, Fa2Variant, AsmParams, GpuAsmParams, Shape};

pub mod heroscope;
pub mod model;
pub mod problem;
pub mod policy;
pub mod tuner;
pub mod history;
pub mod evolution;
pub mod cache;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub name: String,
    pub backend: DeviceBackend,
    pub shared_memory_per_block: usize,
    pub max_registers_per_thread: u32,
    pub registers_per_sm: u32,
    pub max_registers_per_block: u32,
    pub max_warps_per_sm: u32, 
    pub wavefront_size: u32,  
    pub max_blocks_per_sm: u32,
    pub shared_memory_per_sm: usize,
    pub has_specialized_units: bool,
    pub compute_capability: Option<(u32, u32)>,
    pub supported_intrinsic_shapes: Vec<crate::core::config::IntrinsicShape>,
    pub max_threadgroup_memory: usize, 
    pub preferred_tile_shape: [usize; 3], 
    pub simd_width: usize, 
}

impl HardwareProfile {
    pub fn dummy() -> Self {
        Self {
            name: "Generic Profile".to_string(),
            backend: DeviceBackend::Cpu,
            shared_memory_per_block: 48 * 1024,
            max_registers_per_thread: 255,
            registers_per_sm: 65536,
            max_registers_per_block: 65536,
            max_warps_per_sm: 32,
            wavefront_size: 32,
            max_blocks_per_sm: 16,
            shared_memory_per_sm: 65536,
            has_specialized_units: false,
            compute_capability: None,
            supported_intrinsic_shapes: vec![],
            max_threadgroup_memory: 0,
            preferred_tile_shape: [64, 64, 32],
            simd_width: 32,
        }
    }
    
    pub fn rtx3070() -> Self { Self::dummy() }
    pub fn apple_m1() -> Self { Self::dummy() }
    pub fn mi250() -> Self { Self::dummy() }
    pub fn a100() -> Self { Self::dummy() }

    pub fn to_device_profile(&self) -> crate::core::device::DeviceProfile {
        crate::core::device::DeviceProfile {
            backend: match self.backend {
                DeviceBackend::Cuda => crate::core::device::BackendType::Cuda,
                DeviceBackend::Metal => crate::core::device::BackendType::Metal,
                DeviceBackend::Rocm => crate::core::device::BackendType::Rocm,
                _ => crate::core::device::BackendType::Cpu,
            },
            name: self.name.clone(),
            max_threads_per_block: 1024,
            simd_width: self.simd_width,
            local_memory_size: self.shared_memory_per_block,
            has_tensor_cores: self.has_specialized_units,
            has_fp16_storage: true,
            texture_alignment: 256,
        }
    }
    
    pub fn check_feasibility(&self, _config: &PipelineConfig, _problem: &ProblemDescriptor) -> Result<(), PruningReason> {
        Ok(())
    }
    pub fn to_device(&self) -> Device {
         match self.backend {
            DeviceBackend::Cuda => Device::Cuda(CudaArch::Unknown),
            DeviceBackend::Metal => Device::Metal,
            _ => Device::Cpu(CpuArch::Scalar),
        }
    }
    pub fn estimate_occupancy(&self, _config: &PipelineConfig) -> f32 { 1.0 }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PruningReason {
    SharedMemoryOverflow,
    RegisterPressureTooHigh,
    LowOccupancy(u32),
    InvalidTileSize,
    UnsupportedIntrinsic,
    ForbiddenZone(&'static str),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TuningStats {
    pub total_trials: usize,
    pub pruned_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationGoal {
    MaximizeTFLOPS,
    MinimizeLatency,
    Balanced { tflops_weight: f32 },
}

#[derive(Debug)]
pub struct AutoTuner {
    pub gpu: HardwareProfile,
    pub device: Device,
    pub hardware_id: String,
    pub stats: TuningStats,
    pub runtime: Option<std::sync::Weak<crate::runtime::manager::RuntimeManager>>,
}

impl AutoTuner {
    pub fn new(gpu: HardwareProfile) -> Self {
        let device = gpu.to_device();
        
        Self {
            gpu,
            device,
            hardware_id: "generic".to_string(),
            stats: TuningStats::default(),
            runtime: None,
        }
    }

    pub fn with_runtime(mut self, runtime: std::sync::Arc<crate::runtime::manager::RuntimeManager>) -> Self {
        self.runtime = Some(std::sync::Arc::downgrade(&runtime));
        self
    }

    pub fn optimize<B: benchmark::MicroBenchmark>(&mut self, benchmark: &B, iterations: usize, goal: OptimizationGoal, _extra: Vec<f32>) -> PipelineConfig {
        self.optimize_v2(benchmark, benchmark.problem(), iterations, goal)
    }
    
    pub fn optimize_v2<B: benchmark::MicroBenchmark>(&mut self, _benchmark: &B, problem: &ProblemDescriptor, _iterations: usize, _goal: OptimizationGoal) -> PipelineConfig {
        use crate::policy::standard::StandardPolicyEngine;
        let engine = StandardPolicyEngine::new();
        let device_profile = self.gpu.to_device_profile();
        
        let candidates = match problem.layer_type {
            LayerType::Conv2d { .. } => engine.propose_conv_configs(&device_profile),
            _ => engine.propose_gemm_configs(&device_profile),
        };
        
        candidates.first().cloned().unwrap_or_else(|| PipelineConfig::new(2, 64, 64, 32))
    }
    
    pub fn optimize_conv<B: benchmark::Conv2dBenchmark>(&mut self, _benchmark: &B, _iterations: usize, _goal: OptimizationGoal) -> crate::optimizer::benchmark::ConvConfig {
        crate::optimizer::benchmark::ConvConfig {
            base: PipelineConfig::new(2, 64, 64, 32),
            use_tensor_core: true,
            use_nhwc: true,
            magic_strategy: crate::core::config::MagicNumberStrategy::Standard,
        }
    }
}

pub mod benchmark {
    use super::*;
    pub trait MicroBenchmark {
        fn m(&self) -> u32 { 0 }
        fn n(&self) -> u32 { 0 }
        fn k(&self) -> u32 { 0 }
        fn problem(&self) -> &ProblemDescriptor {
            Box::leak(Box::new(ProblemDescriptor::new_gemm(0, 0, 0)))
        }
        fn device_info(&self) -> EnvironmentInfo { EnvironmentInfo::dummy() }
        fn validate_config(&self, _config: &PipelineConfig) -> bool { true }
        fn measure(&self, _config: &PipelineConfig) -> BenchmarkResult { BenchmarkResult::dummy() }
        fn observe_hardware(&self, _config: &PipelineConfig) -> Option<crate::optimizer::model::HardwareObservation> { None }
    }

    #[derive(Debug, Clone)]
    pub struct EnvironmentInfo {
        pub backend: crate::runtime::manager::DeviceBackend,
        pub api_version: String,
        pub driver_version: String,
        pub arch: String,
    }
    impl EnvironmentInfo {
        pub fn dummy() -> Self {
            Self {
                backend: crate::runtime::manager::DeviceBackend::Cpu,
                api_version: "stub".to_string(),
                driver_version: "stub".to_string(),
                arch: "stub".to_string(),
            }
        }
    }

    pub struct BenchmarkResult {
        pub tflops: f32,
        pub mean_tflops: f32,
        pub std_dev: f32,
        pub latency_ms: f32,
        pub observation: Option<crate::optimizer::model::HardwareObservation>,
    }
    impl BenchmarkResult {
        pub fn dummy() -> Self {
            Self { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 0.0, observation: None }
        }
    }

    pub struct NVRTCBenchmark {}
    impl NVRTCBenchmark {
        pub fn new(_runtime: Arc<crate::runtime::manager::RuntimeManager>, _m: u32, _n: u32, _k: u32) -> Self { Self {} }
    }
    impl MicroBenchmark for NVRTCBenchmark {}

    pub struct NVRTCConvBenchmark {}
    impl NVRTCConvBenchmark {
        pub fn new(_runtime: Arc<crate::runtime::manager::RuntimeManager>, _problem: Conv2dProblem) -> Self { Self {} }
    }
    impl MicroBenchmark for NVRTCConvBenchmark {
        // Conv-specific might be needed but MicroBenchmark covers basic
    }

    pub trait Conv2dBenchmark {
        fn problem(&self) -> &Conv2dProblem;
        fn measure(&self, config: &ConvConfig) -> BenchmarkResult;
    }
    impl Conv2dBenchmark for NVRTCConvBenchmark {
        fn problem(&self) -> &Conv2dProblem { 
             // Leaked box for simplicity in stub
             Box::leak(Box::new(ProblemDescriptor::new_gemm(0, 0, 0)))
        }
        fn measure(&self, _config: &ConvConfig) -> BenchmarkResult { BenchmarkResult::dummy() }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConvConfig {
        pub base: PipelineConfig,
        pub use_tensor_core: bool,
        pub use_nhwc: bool,
        pub magic_strategy: crate::core::config::MagicNumberStrategy,
    }

    pub type Conv2dProblem = crate::optimizer::problem::ProblemDescriptor;

    pub struct FlashAttentionProblem;
    impl FlashAttentionProblem {
        pub fn new(_b: usize, _h: usize, _s: usize, _d: usize, _causal: bool) -> Self { Self }
    }
    pub struct FlashAttentionBenchmark;
    impl FlashAttentionBenchmark {
        pub fn new(_runtime: std::sync::Arc<crate::runtime::manager::RuntimeManager>, _prob: FlashAttentionProblem) -> Self { Self }
    }
    impl MicroBenchmark for FlashAttentionBenchmark {
        fn measure(&self, _cfg: &PipelineConfig) -> BenchmarkResult { BenchmarkResult::dummy() }
    }
}

pub struct ConvBenchmarkAdapter<'a, B: benchmark::MicroBenchmark> {
    pub inner: &'a B,
    pub magic_strategy: crate::core::config::MagicNumberStrategy,
}
impl<'a, B: benchmark::MicroBenchmark> benchmark::MicroBenchmark for ConvBenchmarkAdapter<'a, B> {}
