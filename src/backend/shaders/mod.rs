//! Shader module - Cross-platform shader definitions
//! 
//! Architecture:
//! - GLSL 450 as source (written in WGSL-compatible style)
//! - build.rs compiles to SPIR-V (shaderc) and cross-compiles (naga)
//! - Runtime loads pre-compiled binaries

pub mod types;

use types::*;

/// Embedded shader sources (GLSL)
pub mod sources {
    pub const VERTEX: &str = include_str!("vertex.glsl");
    pub const FRAGMENT: &str = include_str!("fragment.glsl");
    pub const BLUR: &str = include_str!("blur.glsl");
}

/// Shader compilation target
#[derive(Clone, Copy, Debug)]
pub enum ShaderTarget {
    /// SPIR-V binary (Vulkan)
    SpirV,
    /// WGSL text (wgpu)
    Wgsl,
    /// HLSL text (DX12)
    Hlsl,
    /// MSL text (Metal)
    Msl,
    /// GLSL text (OpenGL) - passthrough
    Glsl,
}

/// Pre-compiled shader data
pub struct CompiledShaders {
    pub vertex_spirv: Vec<u32>,
    pub fragment_spirv: Vec<u32>,
    pub blur_spirv: Vec<u32>,
}

// Re-export types
pub use types::{GlobalUniforms, DrawUniforms, ShaderMode, RenderPassType};
pub use types::{create_projection, ProjectionParams};
