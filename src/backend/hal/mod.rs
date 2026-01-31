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

/// GpuExecutor - The unified "Muscle" that provides resources, pipelines, and executes primitive GPU operations.
/// All logic for "how to render a frame" is moved to the Orchestrator.
pub trait GpuExecutor: Send + Sync {
    type Buffer: Send + Sync;
    type Texture: Send + Sync;
    type TextureView: Send + Sync;
    type Sampler: Send + Sync;
    type RenderPipeline: Send + Sync;
    type ComputePipeline: Send + Sync;
    type BindGroupLayout: Send + Sync;
    type BindGroup: Send + Sync;

    // --- Resource Management ---
    fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String>;
    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String>;
    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String>;
    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String>;
    
    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]);
    fn write_texture(&self, texture: &Self::Texture, data: &[u8], width: u32, height: u32);
    
    fn destroy_buffer(&self, buffer: Self::Buffer);
    fn destroy_texture(&self, texture: Self::Texture);
    fn destroy_bind_group(&self, bind_group: Self::BindGroup);

    // --- Pipeline Management ---
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

    // --- Command Execution ---
    /// Start a new rendering frame/command list
    fn begin_execute(&self) -> Result<(), String>;

    /// Finish recording and submit all commands
    fn end_execute(&self) -> Result<(), String>;

    /// Perform a draw call with provided uniforms and vertices
    fn draw(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        vertex_count: u32,
        uniform_data: &[u8],
    ) -> Result<(), String>;

    /// Dispatch a compute kernel
    fn dispatch(
        &self,
        pipeline: &Self::ComputePipeline,
        bind_group: Option<&Self::BindGroup>,
        groups: [u32; 3],
        push_constants: &[u8],
    ) -> Result<(), String>;

    /// Copy texture content
    fn copy_texture(
        &self,
        src: &Self::Texture,
        dst: &Self::Texture,
    ) -> Result<(), String>;

    /// Generate mipmaps for a texture
    fn generate_mipmaps(&self, texture: &Self::Texture) -> Result<(), String>;

    fn create_bind_group(
        &self,
        layout: &Self::BindGroupLayout,
        buffers: &[&Self::Buffer],
        textures: &[&Self::TextureView],
        samplers: &[&Self::Sampler],
    ) -> Result<Self::BindGroup, String>;

    // --- Standard Resource Access ---
    /// Get the standard font atlas texture view
    fn get_font_view(&self) -> &Self::TextureView;

    /// Get the current backdrop texture view
    fn get_backdrop_view(&self) -> &Self::TextureView;

    /// Get the default bind group layout (for standard draw commands)
    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout;

    /// Get the default render pipeline
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline;

    /// Get the default sampler
    fn get_default_sampler(&self) -> &Self::Sampler;

    /// Get or create a custom render pipeline from shader source (GLSL/WGSL)
    fn get_custom_render_pipeline(
        &self,
        shader_source: &str,
    ) -> Result<Self::RenderPipeline, String>;

    // --- Global Operations ---
    /// Perform final resolve/composition pass
    fn resolve(&mut self) -> Result<(), String>;

    /// Present the frame
    fn present(&self) -> Result<(), String>;

    /// Check if the backend requires a Y-flip in projection (UI space to NDC)
    fn y_flip(&self) -> bool;
}

/// A single step in the rendering process
#[derive(Debug, Clone)]
pub enum FantaRenderTask {
    /// Draw a set of commands (Geometry)
    DrawGeometry {
        commands: Vec<crate::draw::DrawCommand>,
    },
    /// Capture the current buffer for backdrop
    CaptureBackdrop,
    /// Apply a blur or effect to a texture
    ComputeEffect {
        effect_name: String,
        params: Vec<f32>,
    },
    /// Final composition and tone-mapping
    Resolve,
}
