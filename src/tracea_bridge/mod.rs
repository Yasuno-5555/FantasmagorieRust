//! Tracea Bridge - Integration layer between Fantasmagorie HAL and Tracea GPU kernels
//!
//! This module provides optimized GPU compute implementations for:
//! - Bloom/Blur (Phase 1)
//! - JFA/SDF (Phase 2)  
//! - Audio FFT (Phase 3)
//! - Particle Simulation (Phase 4)
//! - SPH Physics (Phase 5)
//! - Neural Style Transfer (Phase 6)
//! - Resolve/Post-Process (K4)
//! - Visibility/Culling (K8)
//! - Audio Aggregation (K12)
//! - Indirect Dispatch (K13)

pub mod context;
pub mod blur;
pub mod jfa;
pub mod fft;
pub mod particles;
pub mod sph;
pub mod neural;
pub mod interop;
pub mod resolve;
pub mod visibility;
pub mod audio;
pub mod indirect;

pub use context::TraceaContext;
pub use blur::TraceaBlurKernel;
pub use jfa::TraceaJFAKernel;
pub use fft::TraceaFFTKernel;
pub use particles::TraceaParticleKernel;
pub use sph::TraceaSPHKernel;
pub use neural::TraceaNeuralKernel;
pub use resolve::TraceaResolveKernel;
pub use visibility::TraceaVisibilityKernel;
pub use audio::TraceaAudioKernel;
pub use indirect::TraceaIndirectKernel;

