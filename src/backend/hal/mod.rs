//! Hardware Abstraction Layer (HAL) for GPU Backends
//! 
//! This module defines the common traits and types that all Fantasmagorie backends
//! must implement to ensure structural parity and modularity.


/// Common GPU Resource types (Backend-agnostic descriptors)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferUsage(pub u32);

#[allow(non_upper_case_globals)]
impl BufferUsage {
    pub const Vertex: Self = Self(1 << 0);
    pub const Index: Self = Self(1 << 1);
    pub const Uniform: Self = Self(1 << 2);
    pub const Storage: Self = Self(1 << 3);
    pub const CopySrc: Self = Self(1 << 4);
    pub const CopyDst: Self = Self(1 << 5);
    pub const Indirect: Self = Self(1 << 6);
    
    pub fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    R8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    Depth32Float,
    Rgba16Float,
    Rg16Float,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct TextureDescriptor {
    pub label: Option<&'static str>,
    pub width: u32,
    pub height: u32,
    pub depth: u32, // Default 1 for 2D
    pub format: TextureFormat,
    pub usage: TextureUsage,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TextureUsage: u32 {
        const COPY_SRC          = 0x1;
        const COPY_DST          = 0x2;
        const TEXTURE_BINDING   = 0x4;
        const STORAGE_BINDING   = 0x8;
        const RENDER_ATTACHMENT = 0x10;
    }
}

pub enum BindingResource<'a, E: GpuExecutor + ?Sized> {
    Buffer(&'a E::Buffer),
    Texture(&'a E::TextureView),
    Sampler(&'a E::Sampler),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UpscaleParams {
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub reset_history: bool,
}

pub struct BindGroupEntry<'a, E: GpuExecutor + ?Sized> {
    pub binding: u32,
    pub resource: BindingResource<'a, E>,
}

/// GpuExecutor - The unified "Muscle" that provides resources, pipelines, and executes primitive GPU operations.
/// All logic for "how to render a frame" is moved to the Orchestrator.
pub trait GpuExecutor: Send + Sync {
    type Buffer: Send + Sync;
    type Texture: Send + Sync + Clone + std::fmt::Debug;
    type TextureView: Send + Sync + Clone + std::fmt::Debug;
    type Sampler: Send + Sync + Clone + std::fmt::Debug;
    type RenderPipeline: Send + Sync + Clone + std::fmt::Debug;
    type ComputePipeline: Send + Sync + Clone + std::fmt::Debug;
    type BindGroupLayout: Send + Sync + Clone + std::fmt::Debug;
    type BindGroup: Send + Sync + Clone + std::fmt::Debug;

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

    // Old create_compute_pipeline removed

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

    /// Perform an instanced draw call
    fn draw_instanced(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        instance_buffer: &Self::Buffer,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), String>;

    /// Perform an instanced draw call using an indirect buffer
    fn draw_instanced_indirect(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        instance_buffer: &Self::Buffer,
        indirect_buffer: &Self::Buffer,
        indirect_offset: u64,
    ) -> Result<(), String>;

    /// Perform an instanced draw call with MRT (Main + Aux/Normal)
    fn draw_instanced_gbuffer(
        &mut self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        instance_buffer: &Self::Buffer,
        vertex_count: u32,
        instance_count: u32,
        aux_view: &Self::TextureView,
        velocity_view: &Self::TextureView,
        depth_view: &Self::TextureView,
    ) -> Result<(), String>;

    /// Perform an indirect instanced draw call with MRT
    fn draw_instanced_gbuffer_indirect(
        &mut self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        instance_buffer: &Self::Buffer,
        indirect_buffer: &Self::Buffer,
        indirect_offset: u64,
        aux_view: &Self::TextureView,
        velocity_view: &Self::TextureView,
        depth_view: &Self::TextureView,
    ) -> Result<(), String>;

    /// Draw particles (Additive/Alpha, LoadOp::Load)
    fn draw_particles(
        &mut self,
        pipeline: &Self::RenderPipeline,
        bind_group: &Self::BindGroup,
        particle_count: u32,
    ) -> Result<(), String>;

    fn draw_ssr(
        &mut self,
        hdr_view: &Self::TextureView,
        depth_view: &Self::TextureView,
        aux_view: &Self::TextureView,
        velocity_view: &Self::TextureView,
        output_texture: &Self::Texture,
    ) -> Result<(), String>;

    /// Dispatch a compute kernel
    fn dispatch(
        &self,
        pipeline: &Self::ComputePipeline,
        bind_group: Option<&Self::BindGroup>,
        groups: [u32; 3],
        push_constants: &[u8],
    ) -> Result<(), String>;

    /// Dispatch a compute kernel with indirect parameters
    fn dispatch_indirect(
        &self,
        pipeline: &Self::ComputePipeline,
        bind_group: Option<&Self::BindGroup>,
        indirect_buffer: &Self::Buffer,
        indirect_offset: u64,
    ) -> Result<(), String>;

    /// Dispatch Tracea Visibility Kernel (GPU Culling)
    fn dispatch_visibility(
        &self,
        projection: [[f32; 4]; 4],
        num_instances: u32,
        instances: &Self::Buffer,
        hzb: &Self::TextureView,
        visible_indices: &Self::Buffer,
        visible_counter: &Self::Buffer,
    ) -> Result<(), String>;

    /// Dispatch Tracea Indirect Kernel (Command Generation)
    fn dispatch_indirect_command(
        &self,
        counter_buffer: &Self::Buffer,
        draw_commands: &Self::Buffer,
    ) -> Result<(), String>;


    /// Copy texture content
    fn copy_texture(
        &self,
        src: &Self::Texture,
        dst: &Self::Texture,
    ) -> Result<(), String>;

    /// Generate mipmaps for a texture
    fn generate_mipmaps(&self, texture: &Self::Texture) -> Result<(), String>;

    /// Copy current framebuffer content to a texture
    fn copy_framebuffer_to_texture(&self, dst: &Self::Texture) -> Result<(), String>;

    /// Apply motion blur effect
    fn draw_motion_blur(
        &self,
        dst_view: &Self::TextureView,
        src_view: &Self::TextureView,
        vel_view: &Self::TextureView,
        strength: f32,
    ) -> Result<(), String>;

    /// Acquire a transient texture from the backend's pool
    fn acquire_transient_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String>;
    
    /// Dispatch optimized Tracea blur (if supported and enabled)
    fn dispatch_tracea_blur(&self, _input: &Self::Texture, _output: &Self::Texture, _sigma: f32) -> Result<bool, String> {
        Ok(false)
    }

    /// Check if Tracea particles are supported
    fn supports_tracea_particles(&self) -> bool { false }

    /// Get HZB (Hierarchical Z-Buffer) view for occlusion culling
    fn get_hzb_view(&self) -> &Self::TextureView;
    
    /// Check if native tilemap rendering is supported
    fn supports_tilemap(&self) -> bool { false }

    /// Check if indirect draw / GPU-driven rendering is supported
    fn supports_indirect_draw(&self) -> bool { false }

    /// Draw a tilemap using native support
    fn draw_tilemap(
        &mut self,
        _params: crate::backend::shaders::types::TilemapParams,
        _data: &[u32],
        _texture_view: &Self::TextureView,
        _global_buffer: &Self::Buffer,
        _aux_view: Option<&Self::TextureView>,
        _velocity_view: Option<&Self::TextureView>,
        _depth_view: Option<&Self::TextureView>,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Draw a skinned mesh
    fn draw_skinned(
        &mut self,
        _vertex_buffer: &Self::Buffer,
        _index_buffer: &Self::Buffer,
        _index_count: u32,
        _bone_matrices_buffer: &Self::Buffer,
        _texture_view: &Self::TextureView,
        _global_buffer: &Self::Buffer,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Get the particle buffer managed by Tracea (if any)
    fn get_tracea_particle_buffer(&self) -> Option<Self::Buffer> { None }

    /// Dispatch Tracea particle simulation
    fn dispatch_tracea_particles(&self, _dt: f32, _attractor: [f32; 2], _sdf_texture: Option<&Self::Texture>) -> Result<bool, String> {
        Ok(false)
    }

    fn update_audio_data(&mut self, _data: &[f32]);
    fn update_audio_pcm(&mut self, _samples: &[f32]);

    /// Release a transient texture back to the backend's pool
    fn release_transient_texture(&self, _texture: Self::Texture, desc: &TextureDescriptor);

    fn create_bind_group(
        &self,
        layout: &Self::BindGroupLayout,
        entries: &[BindGroupEntry<Self>],
    ) -> Result<Self::BindGroup, String>;

    /// Get the standard font atlas texture view
    fn get_font_view(&self) -> &Self::TextureView;

    /// Get the current backdrop texture view
    fn get_backdrop_view(&self) -> &Self::TextureView;

    /// Get the primary HDR texture (if managed by backend)
    fn get_hdr_texture(&self) -> Option<Self::Texture>;

    /// Get the backdrop texture (if managed by backend)
    fn get_backdrop_texture(&self) -> Option<Self::Texture>;

    fn get_extra_texture(&self) -> Option<Self::Texture>;
    fn get_aux_texture(&self) -> Option<Self::Texture>;
    fn get_velocity_texture(&self) -> Option<Self::Texture>;
    fn get_depth_texture(&self) -> Option<Self::Texture>;

    /// Get the default bind group layout    /// Get default bind group layout
    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout;
    /// Get instanced bind group layout
    fn get_instanced_bind_group_layout(&self) -> &Self::BindGroupLayout;
    /// Get culling compute bind group layout
    fn get_culling_bind_group_layout(&self) -> &Self::BindGroupLayout;

    /// Get default sampler pipeline
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline;

    /// Get the instanced render pipeline (for batched shapes)
    fn get_instanced_render_pipeline(&self) -> &Self::RenderPipeline;
    fn get_instanced_gbuffer_render_pipeline(&self) -> &Self::RenderPipeline;
    fn get_culling_pipeline(&self) -> &Self::ComputePipeline;
    /// Get a dummy storage buffer (e.g. for empty instance lists)
    fn get_dummy_storage_buffer(&self) -> &Self::Buffer;

    fn set_reflection_texture(&mut self, _texture: &Self::TextureView) -> Result<(), String> {
        Ok(())
    }
    fn set_velocity_view(&mut self, _view: &Self::TextureView) -> Result<(), String> {
        Ok(())
    }
    fn set_sdf_view(&mut self, _view: &Self::TextureView) -> Result<(), String> {
        Ok(())
    }
    fn set_lut_view(&mut self, _view: &Self::TextureView) -> Result<(), String> {
        Ok(())
    }
    fn set_hdr_view(&mut self, _view: &Self::TextureView) -> Result<(), String> {
        Ok(())
    }

    fn get_lut_texture(&self) -> Option<Self::Texture>;


    fn draw_lighting_pass(&mut self, output_view: &Self::TextureView) -> Result<(), String>;
    fn draw_post_process_pass(&mut self, input_view: &Self::TextureView, output_view: Option<&Self::TextureView>) -> Result<(), String>;
    fn draw_fxaa_pass(&mut self, input_view: &Self::TextureView) -> Result<(), String>;
    fn upscale(&mut self, input: &Self::TextureView, output: &Self::TextureView, params: UpscaleParams) -> Result<(), String>;

    fn draw_bloom_pass(&mut self, _input_view: &Self::TextureView) -> Result<(), String> {
        Ok(())
    }

    fn draw_dof_pass(&mut self, _hdr_view: &Self::TextureView, _depth_view: &Self::TextureView, _output_view: &Self::TextureView) -> Result<(), String> {
        Err("DoF not implemented for this backend".to_string())
    }

    fn draw_flare_pass(&mut self, _hdr_view: &Self::TextureView, _output_view: &Self::TextureView) -> Result<(), String> {
        Err("Lens Flare not implemented for this backend".to_string())
    }

    /// Get the default sampler
    fn get_default_sampler(&self) -> &Self::Sampler;

    /// Get or create a custom render pipeline from shader source (GLSL/WGSL)
    fn get_custom_render_pipeline(
        &self,
        shader_source: &str,
    ) -> Result<Self::RenderPipeline, String>;

    /// Get or create a custom render pipeline from shader source (GLSL/WGSL)
    fn create_compute_pipeline(&self, shader_name: &str, shader_source: &str, entry_point: Option<&str>) -> Result<Self::ComputePipeline, String>;
    
    fn get_compute_pipeline_layout(&self, pipeline: &Self::ComputePipeline, index: u32) -> Result<Self::BindGroupLayout, String>;
    fn get_render_pipeline_layout(&self, pipeline: &Self::RenderPipeline, index: u32) -> Result<Self::BindGroupLayout, String>;

    // --- Global Operations ---
    /// Perform final resolve/composition pass
    fn resolve(&mut self) -> Result<(), String>;

    /// Present the frame
    fn present(&self) -> Result<(), String>;

    /// Check if the backend requires a Y-flip in projection (UI space to NDC)
    fn y_flip(&self) -> bool;

    fn get_cinematic_buffer(&self) -> &Self::Buffer;
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
