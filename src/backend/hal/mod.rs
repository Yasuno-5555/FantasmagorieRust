//! Hardware Abstraction Layer (HAL) for GPU Backends
//! 
//! This module defines the common traits and types that all Fantasmagorie backends
//! must implement to ensure structural parity and modularity.

use std::sync::Arc;

/// Common GPU Resource types (Backend-agnostic descriptors)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    Vertex,
    Index,
    Uniform,
    Storage,
    CopySrc,
    CopyDst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    R8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    Depth32Float,
}

#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub label: Option<&'static str>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub usage: TextureUsage,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TextureUsage: u32 {
        const COPY_SRC          = 0x1;
        const COPY_DST          = 0x2;
        const TEXTURE_BINDING   = 0x4;
        const STORAGE_BINDING   = 0x8;
        const RENDER_ATTACHMENT = 0x10;
    }
}

/// Core Resource Provider Trait
pub trait GpuResourceProvider: Send + Sync {
    type Buffer;
    type Texture;
    type TextureView;
    type Sampler;

    fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String>;
    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String>;
    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String>;
    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String>;
    
    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]);
    fn write_texture(&self, texture: &Self::Texture, data: &[u8], width: u32, height: u32);
}

/// Core Pipeline Provider Trait
pub trait GpuPipelineProvider: Send + Sync {
    type RenderPipeline;
    type ComputePipeline;
    type BindGroupLayout;

    fn create_render_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
        layout: Option<&Self::BindGroupLayout>,
    ) -> Result<Self::RenderPipeline, String>;

    fn create_compute_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
        layout: Option<&Self::BindGroupLayout>,
    ) -> Result<Self::ComputePipeline, String>;
}

/// Command Submission Trait
pub trait GpuCommandEncoder: Send + Sync {
    type CommandBuffer;
    
    fn begin_frame(&mut self) -> Result<(), String>;
    fn end_frame(&mut self) -> Result<Self::CommandBuffer, String>;
    fn submit(&self, cmd: Self::CommandBuffer);
}
