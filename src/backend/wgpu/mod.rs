use std::sync::{Arc, Mutex};
use crate::draw::DrawList;
use crate::backend::GraphicsBackend;
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use wgpu::util::DeviceExt;
use crate::renderer::graph::TransientPool;

/// Vertex format for WGPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x2,  // pos
        1 => Float32x2,  // uv
        2 => Float32x4,  // color
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Vertex format for Skinned Meshes
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub bone_indices: [u32; 4],
    pub bone_weights: [f32; 4],
}

impl SkinnedVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x2,  // pos
        1 => Float32x2,  // uv
        2 => Float32x4,  // color
        3 => Uint32x4,   // bone_indices
        4 => Float32x4,  // bone_weights
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SkinnedVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}


pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    
    pub main_pipeline: Arc<wgpu::RenderPipeline>,
    pub instanced_pipeline: Arc<wgpu::RenderPipeline>,
    pub instanced_gbuffer_pipeline: Arc<wgpu::RenderPipeline>,
    pub culling_pipeline: Arc<wgpu::ComputePipeline>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub instanced_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub culling_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub pipeline_layout: wgpu::PipelineLayout,
    
    pub sampler: Arc<wgpu::Sampler>,
    pub font_texture: Arc<wgpu::Texture>,
    pub font_view: Arc<wgpu::TextureView>,
    pub backdrop_view: Arc<wgpu::TextureView>,
    
    pub hdr_texture: Arc<wgpu::Texture>,
    pub hdr_view: Arc<wgpu::TextureView>,
    pub aux_texture: Arc<wgpu::Texture>,
    pub aux_view: Arc<wgpu::TextureView>,
    pub extra_texture: Arc<wgpu::Texture>,
    pub extra_view: Arc<wgpu::TextureView>,
    pub reflection_view: Option<Arc<wgpu::TextureView>>,
    pub velocity_view: Option<Arc<wgpu::TextureView>>,
    pub velocity_texture: Option<Arc<wgpu::Texture>>,
    pub depth_texture: Arc<wgpu::Texture>,
    pub depth_view: Arc<wgpu::TextureView>,
    pub backdrop_texture: Arc<wgpu::Texture>,
    pub lut_texture: Option<Arc<wgpu::Texture>>,
    pub lut_view: Option<Arc<wgpu::TextureView>>,

    pub ldr_texture: Arc<wgpu::Texture>,
    pub ldr_view: Arc<wgpu::TextureView>,
    
    pub blit_pipeline: wgpu::RenderPipeline,
    pub blit_bind_group_layout: wgpu::BindGroupLayout,
    
    pub upscale_pipeline: wgpu::RenderPipeline,
    pub upscale_bind_group_layout: wgpu::BindGroupLayout,
    
    pub k4_pipeline: Arc<wgpu::RenderPipeline>, // Resolve/Post-process
    pub k4_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub lighting_pipeline: wgpu::RenderPipeline,
    pub lighting_bind_group_layout: wgpu::BindGroupLayout,
    
    pub post_pipeline: wgpu::RenderPipeline,
    pub post_bind_group_layout: wgpu::BindGroupLayout,
    
    pub dof_pipeline: Arc<wgpu::RenderPipeline>,
    pub dof_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub flare_pipeline: Arc<wgpu::RenderPipeline>,
    pub flare_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub fxaa_pipeline: Arc<wgpu::RenderPipeline>,
    pub fxaa_bind_group_layout: wgpu::BindGroupLayout,

    // Profiler
    pub query_set: Option<wgpu::QuerySet>,
    pub resolve_buffer: Option<wgpu::Buffer>,
    pub readback_buffer: Option<wgpu::Buffer>,
    
    // Bloom
    pub bright_pipeline: Arc<wgpu::RenderPipeline>,
    pub blur_pipeline: Arc<wgpu::RenderPipeline>,
    pub bloom_textures: Vec<wgpu::Texture>,
    pub bloom_views: Vec<Arc<wgpu::TextureView>>,
    pub bloom_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub dummy_velocity_view: Arc<wgpu::TextureView>,

    // Motion Blur
    pub motion_blur_pipeline: Arc<wgpu::RenderPipeline>,
    pub motion_blur_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    // SSR
    pub ssr_pipeline: Arc<wgpu::RenderPipeline>,
    pub ssr_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub ssr_history_texture: Option<Arc<wgpu::Texture>>,
    pub ssr_history_view: Option<Arc<wgpu::TextureView>>,
    pub sdf_view: Option<Arc<wgpu::TextureView>>,

    pub blur_uniform_buffer: wgpu::Buffer,
    pub cinematic_buffer: wgpu::Buffer,
    pub dummy_storage_buffer: wgpu::Buffer, // Fallback for instanced bindings
    
    // Tilemap
    pub tilemap_pipeline: Arc<wgpu::RenderPipeline>,
    pub tilemap_gbuffer_pipeline: Arc<wgpu::RenderPipeline>,
    pub tilemap_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pub skinned_pipeline: Arc<wgpu::RenderPipeline>,
    pub skinned_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pub current_cinematic: Mutex<crate::backend::shaders::types::CinematicParams>,

    pub current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
    pub current_texture: Mutex<Option<wgpu::SurfaceTexture>>,
    pub current_view: Mutex<Option<Arc<wgpu::TextureView>>>,
    
    pub start_time: std::time::Instant,
    
    // Resource Management
    pub transient_pool: Mutex<TransientPool<WgpuBackend>>,
    
    // Screenshot
    pub screenshot_requested: Mutex<Option<String>>,
    pub pipeline_cache: Mutex<std::collections::HashMap<String, Arc<wgpu::RenderPipeline>>>,

    pub shader_reload_rx: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    pub _shader_watcher: Option<Box<dyn std::any::Any + Send + Sync>>,

    // Tracea Integration
    pub tracea_context: crate::tracea_bridge::TraceaContext,
    pub tracea_particle_cache: Mutex<Option<crate::tracea_bridge::TraceaParticleKernel>>,
    pub tracea_fft_kernel: Mutex<Option<crate::tracea_bridge::TraceaFFTKernel>>,
    pub tracea_visibility_kernel: Mutex<Option<crate::tracea_bridge::TraceaVisibilityKernel>>,
    pub tracea_indirect_kernel: Mutex<Option<crate::tracea_bridge::TraceaIndirectKernel>>,
    pub audio_data: Vec<f32>,
    
    // Resolution & Jitter
    pub frame_index: u32,
    pub internal_width: u32,
    pub internal_height: u32,
    pub resolution_scale: f32,
}

impl GpuExecutor for WgpuBackend {
    type Buffer = wgpu::Buffer;
    type Texture = Arc<wgpu::Texture>;
    type TextureView = Arc<wgpu::TextureView>;
    type Sampler = Arc<wgpu::Sampler>;
    type RenderPipeline = Arc<wgpu::RenderPipeline>;
    type ComputePipeline = Arc<wgpu::ComputePipeline>;
    type BindGroupLayout = Arc<wgpu::BindGroupLayout>;
    type BindGroup = Arc<wgpu::BindGroup>;

    fn dispatch_tracea_blur(&self, _input: &Self::Texture, _output: &Self::Texture, _sigma: f32) -> Result<bool, String> {
        Ok(false)
    }

    fn supports_tracea_particles(&self) -> bool { true }
    
    fn get_tracea_particle_buffer(&self) -> Option<Self::Buffer> { 
        // TODO: Return GPU buffer for shared rendering. 
        // Currently returning None implies CPU fallback or separate render path.
        // Needs refactoring Self::Buffer to Arc<wgpu::Buffer> to support sharing.
        None 
    }
    
    fn dispatch_tracea_particles(&self, dt: f32, attractor: [f32; 2], sdf_texture: Option<&Self::Texture>) -> Result<bool, String> {
        let mut cache = self.tracea_particle_cache.lock().unwrap();
        // Init if missing
        if cache.is_none() {
             if let Ok(kernel) = crate::tracea_bridge::TraceaParticleKernel::new(&self.tracea_context, 100_000) {
                 *cache = Some(kernel);
             }
        }
        
        if let Some(kernel) = cache.as_ref() {
            // Need a view for texture. Self::Texture is Arc<wgpu::Texture>.
            // We need to create a view.
            let view = sdf_texture.map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
            // We need a sampler. WGPU backend doesn't store a default sampler accessible here easily?
            // Actually usually we have one. Let's create a temp one or store in backend.
            // For now create temp linear sampler.
            let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            
            // Dispatch
            // Note: `kernel.update_wgpu` expects `Option<&wgpu::TextureView>`.
            // Our view is local, we pass reference.
            let _ = kernel.update_wgpu(&self.tracea_context, dt, attractor, view.as_ref(), Some(&sampler));
            return Ok(true);
        }
        Ok(false)
    }

    fn update_audio_data(&mut self, spectrum: &[f32]) {
        self.audio_data = spectrum.to_vec();
    }

    fn update_audio_pcm(&mut self, samples: &[f32]) {
        if !self.tracea_context.is_ready() { return; }
        
        // Lazy Init
        let mut kernel_lock = self.tracea_fft_kernel.lock().unwrap();
        if kernel_lock.is_none() {
             let size = samples.len().max(1024).next_power_of_two();
             if let Ok(k) = crate::tracea_bridge::TraceaFFTKernel::new(&self.tracea_context, size) {
                 *kernel_lock = Some(k);
             }
        }
        
        if let Some(kernel) = kernel_lock.as_ref() {
            // Re-init if size changes? For now assume fixed block size or best effort.
            if kernel.fft_size() == samples.len() {
                 if let Ok(spectrum) = kernel.compute_spectrum(&self.tracea_context, samples) {
                      self.audio_data = spectrum;
                 }
            } else if samples.len() > 0 {
                // Recreate if size mismatch (expensive but necessary)
                let size = samples.len().next_power_of_two();
                if let Ok(k) = crate::tracea_bridge::TraceaFFTKernel::new(&self.tracea_context, size) {
                    *kernel_lock = Some(k);
                     if let Ok(spectrum) = kernel_lock.as_ref().unwrap().compute_spectrum(&self.tracea_context, samples) {
                          self.audio_data = spectrum;
                     }
                }
            }
        }
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String> {
        let mut wgpu_usage = wgpu::BufferUsages::empty();
        if usage.contains(BufferUsage::Vertex) { wgpu_usage |= wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST; }
        if usage.contains(BufferUsage::Index) { wgpu_usage |= wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST; }
        if usage.contains(BufferUsage::Storage) { wgpu_usage |= wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC; }
        if usage.contains(BufferUsage::Uniform) { wgpu_usage |= wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST; }
        if usage.contains(BufferUsage::CopySrc) { wgpu_usage |= wgpu::BufferUsages::COPY_SRC; }
        if usage.contains(BufferUsage::CopyDst) { wgpu_usage |= wgpu::BufferUsages::COPY_DST; }
        if usage.contains(BufferUsage::Indirect) { wgpu_usage |= wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST; }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu_usage,
            mapped_at_creation: false,
        }))
    }

    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let format = match desc.format {
            TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
            TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            TextureFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        };
        let mut usage = wgpu::TextureUsages::empty();
        if desc.usage.contains(TextureUsage::COPY_SRC) { usage |= wgpu::TextureUsages::COPY_SRC; }
        if desc.usage.contains(TextureUsage::COPY_DST) { usage |= wgpu::TextureUsages::COPY_DST; }
        if desc.usage.contains(TextureUsage::TEXTURE_BINDING) { usage |= wgpu::TextureUsages::TEXTURE_BINDING; }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) { usage |= wgpu::TextureUsages::STORAGE_BINDING; }
        if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) { usage |= wgpu::TextureUsages::RENDER_ATTACHMENT; }

        let size = wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: desc.depth,
        };
        let dimension = if desc.depth > 1 { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 };

        Ok(Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        })))
    }

    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String> {
        Ok(Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default())))
    }

    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String> {
        Ok(Arc::new(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        })))
    }

    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    fn write_texture(&self, texture: &Self::Texture, data: &[u8], width: u32, height: u32) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }

    fn destroy_buffer(&self, _buffer: Self::Buffer) {}
    fn destroy_texture(&self, _texture: Self::Texture) {}
    fn destroy_bind_group(&self, _bind_group: Self::BindGroup) {}

    fn create_render_pipeline(&self, label: &str, wgsl_source: &str, layout: Option<&Self::BindGroupLayout>) -> Result<Self::RenderPipeline, String> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });
        
        let pipeline_layout_resource: Option<wgpu::PipelineLayout>;
        let layout_ref = if let Some(l) = layout {
            pipeline_layout_resource = Some(self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[l],
                push_constant_ranges: &[],
            }));
            pipeline_layout_resource.as_ref()
        } else {
            None
        };

        let mut vertex_buffers = Vec::new();
        if !(label.contains("Particle") || label.contains("Resolve") || label.contains("Blur") || label.contains("Bright") || label.contains("SSR")) {
            vertex_buffers.push(Vertex::desc());
        }

        Ok(Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: layout_ref,
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &vertex_buffers },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })))
    }

    fn create_compute_pipeline(&self, label: &str, wgsl_source: &str, entry_point: Option<&str>) -> Result<Self::ComputePipeline, String> {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_source)),
        };
        let module = self.device.create_shader_module(desc);

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None, // Auto layout
            module: &module,
            entry_point: entry_point.unwrap_or("main"),
        });

        Ok(Arc::new(pipeline))
    }

    fn get_compute_pipeline_layout(&self, pipeline: &Self::ComputePipeline, index: u32) -> Result<Self::BindGroupLayout, String> {
        Ok(Arc::new(pipeline.get_bind_group_layout(index)))
    }

    fn get_render_pipeline_layout(&self, pipeline: &Self::RenderPipeline, index: u32) -> Result<Self::BindGroupLayout, String> {
        Ok(Arc::new(pipeline.get_bind_group_layout(index)))
    }

    fn begin_execute(&self) -> Result<(), String> {
        let output = self.surface.get_current_texture().map_err(|e| e.to_string())?;
        let view = Arc::new(output.texture.create_view(&wgpu::TextureViewDescriptor::default()));
        *self.current_texture.lock().unwrap() = Some(output);
        *self.current_view.lock().unwrap() = Some(view);
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") });
        
        // Clear main targets (internal resolution)
        {
            let color_attachments = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.aux_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.extra_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
            ];
            
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Frame Main Clear"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        // Clear bloom targets (downsampled)
        for (i, v) in self.bloom_views.iter().enumerate() {
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Bloom Clear {}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &**v,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        
        // Clear velocity target
        if let Some(vel) = &self.velocity_view {
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Velocity Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &**vel,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        *self.current_encoder.lock().unwrap() = Some(encoder);
        Ok(())
    }

    fn end_execute(&self) -> Result<(), String> {
        println!("DEBUG: [PASS] end_execute starting");
        
        let encoder = self.current_encoder.lock().unwrap().take().ok_or("No active encoder")?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn get_hzb_view(&self) -> &Self::TextureView {
        &self.depth_view
    }

    fn supports_indirect_draw(&self) -> bool {
        true
    }

    fn dispatch(&self, pipeline: &Self::ComputePipeline, bind_group: Option<&Self::BindGroup>, groups: [u32; 3], _push_constants: &[u8]) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Dispatch"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { cpass.set_bind_group(0, bg, &[]); }
        cpass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }

    fn dispatch_indirect(&self, pipeline: &Self::ComputePipeline, bind_group: Option<&Self::BindGroup>, indirect_buffer: &Self::Buffer, indirect_offset: u64) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Indirect Compute Dispatch"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { cpass.set_bind_group(0, bg, &[]); }
        cpass.dispatch_workgroups_indirect(indirect_buffer, indirect_offset);
        Ok(())
    }

    fn draw(&self, pipeline: &Self::RenderPipeline, bind_group: Option<&Self::BindGroup>, vertex_buffer: &Self::Buffer, vertex_count: u32, _uniform_data: &[u8]) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { rpass.set_bind_group(0, bg, &[]); }
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..vertex_count, 0..1);
        Ok(())
    }

    fn draw_instanced(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        _instance_buffer: &Self::Buffer,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Instanced Geometry Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { rpass.set_bind_group(0, bg, &[]); }
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..vertex_count, 0..instance_count);
        Ok(())
    }

    fn dispatch_indirect_command(&self, counter_buffer: &Self::Buffer, draw_commands: &Self::Buffer) -> Result<(), String> {
        let mut kernel_lock = self.tracea_indirect_kernel.lock().unwrap();
        if kernel_lock.is_none() {
            *kernel_lock = Some(crate::tracea_bridge::TraceaIndirectKernel::new_wgpu(&self.tracea_context)?);
        }
        
        if let Some(kernel) = kernel_lock.as_ref() {
            kernel.dispatch(&self.tracea_context, counter_buffer)?;
            // Copy generated commands to the target buffer
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Indirect Copy") });
            encoder.copy_buffer_to_buffer(kernel.draw_commands(), 0, draw_commands, 0, 16); // 16 bytes for 1 command
            self.queue.submit(Some(encoder.finish()));
            return Ok(());
        }
        Err("Indirect kernel not available".into())
    }

    fn dispatch_visibility(
        &self,
        projection: [[f32; 4]; 4],
        num_instances: u32,
        instances: &Self::Buffer,
        hzb: &Self::TextureView,
        visible_indices: &Self::Buffer,
        visible_counter: &Self::Buffer,
    ) -> Result<(), String> {
        let mut kernel_lock = self.tracea_visibility_kernel.lock().unwrap();
        if kernel_lock.is_none() {
            *kernel_lock = Some(crate::tracea_bridge::TraceaVisibilityKernel::new_wgpu(&self.tracea_context)?);
        }
        
        if let Some(kernel) = kernel_lock.as_ref() {
            let uniforms = crate::tracea_bridge::visibility::CullingUniforms {
                view_proj: projection,
                num_instances,
                hzb_mip_levels: 1, 
                _pad: [0, 0],
            };
            kernel.dispatch(&self.tracea_context, &uniforms, instances, hzb, visible_indices, visible_counter)?;
            return Ok(());
        }
        Err("Visibility kernel not available".into())
    }

    fn draw_instanced_indirect(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        _instance_buffer: &Self::Buffer,
        indirect_buffer: &Self::Buffer,
        indirect_offset: u64,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Instanced Indirect Geometry Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { rpass.set_bind_group(0, bg, &[]); }
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw_indirect(indirect_buffer, indirect_offset);
        Ok(())
    }

    fn draw_instanced_gbuffer(
        &mut self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        _instance_buffer: &Self::Buffer,
        vertex_count: u32,
        instance_count: u32,
        aux_view: &Self::TextureView,
        velocity_view: &Self::TextureView,
        depth_view: &Self::TextureView,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G-Buffer Draw"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: aux_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: velocity_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.extra_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { rpass.set_bind_group(0, bg, &[]); }
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..vertex_count, 0..instance_count);
        Ok(())
    }

    fn draw_instanced_gbuffer_indirect(
        &mut self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        _instance_buffer: &Self::Buffer,
        indirect_buffer: &Self::Buffer,
        indirect_offset: u64,
        aux_view: &Self::TextureView,
        velocity_view: &Self::TextureView,
        depth_view: &Self::TextureView,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Instanced GBuffer Indirect Geometry Draw"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: aux_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: velocity_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.extra_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { rpass.set_bind_group(0, bg, &[]); }
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw_indirect(indirect_buffer, indirect_offset);
        Ok(())
    }

    fn supports_tilemap(&self) -> bool { true }

    fn draw_tilemap(
        &mut self,
        params: crate::backend::shaders::types::TilemapParams,
        data: &[u32],
        texture_view: &Self::TextureView,
        global_buffer: &Self::Buffer,
        aux_view: Option<&Self::TextureView>,
        velocity_view: Option<&Self::TextureView>,
        depth_view: Option<&Self::TextureView>,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        // 1. Create Buffers
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tilemap Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let data_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tilemap Data"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Vertex Buffer (Unit Quad)
        let quad_verts = crate::renderer::nodes::geometry::unit_quad();
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tilemap Quad Vertex"),
            contents: &quad_verts,
            usage: wgpu::BufferUsages::VERTEX,
        });

        // 2. Create Bind Group 1 (Tilemap Specific)
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile BG1"),
            layout: &self.tilemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
            ],
        });

        // 3. Create Bind Group 0 (Shared)
        let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile BG0"),
            layout: &self.instanced_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: global_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.dummy_storage_buffer.as_entire_binding() }, // Placeholder for Instance Buffer
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(texture_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.backdrop_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        // 4. Determine Pipeline & Attachments
        let use_gbuffer = aux_view.is_some() && velocity_view.is_some() && depth_view.is_some();
        let pipeline = if use_gbuffer { &self.tilemap_gbuffer_pipeline } else { &self.tilemap_pipeline };

        // 5. Begin Render Pass
        {
            let color_attachments = if use_gbuffer {
                 vec![
                    Some(wgpu::RenderPassColorAttachment { view: &self.hdr_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                    Some(wgpu::RenderPassColorAttachment { view: aux_view.unwrap(), resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                    Some(wgpu::RenderPassColorAttachment { view: velocity_view.unwrap(), resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                    Some(wgpu::RenderPassColorAttachment { view: &self.extra_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                 ]
            } else {
                 vec![
                    Some(wgpu::RenderPassColorAttachment { view: &self.hdr_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                 ]
            };

            let depth_att = if use_gbuffer {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view.unwrap(),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                })
            } else {
                None
            };
            
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tilemap Draw"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_att,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &bg0, &[]);
            rpass.set_bind_group(1, &bg1, &[]);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..6, 0..data.len() as u32);
        }

        Ok(())
    }

    fn draw_skinned(
        &mut self,
        vertex_buffer: &Self::Buffer,
        index_buffer: &Self::Buffer,
        index_count: u32,
        bone_matrices_buffer: &Self::Buffer,
        texture_view: &Self::TextureView,
        global_buffer: &Self::Buffer,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skinned Bind Group"),
            layout: &self.skinned_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: global_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(texture_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: bone_matrices_buffer.as_entire_binding() },
            ],
        });

        // Use G-buffer attachments if we want skinning to participate in lighting
        // Same logic as tilemap: check if we have G-buffer targets
        
        // Safety check for views (velocity is optional but others are mandatory for deferred)
        // Note: aux_view and extra_view are Arc<TextureView>, so always present.
        
        let velocity_view = self.velocity_view.as_ref().unwrap_or(&self.dummy_velocity_view);

        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment { view: &self.hdr_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: &self.aux_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: velocity_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: &self.extra_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
        ];
        
        // Depth is always present in WgpuBackend
        let depth_att = Some(wgpu::RenderPassDepthStencilAttachment {
            view: &self.depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Skinned Mesh Draw"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_att,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.skinned_pipeline);
        rpass.set_bind_group(0, &bg, &[]);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..index_count, 0, 0..1);

        Ok(())
    }

    fn draw_particles(
        &mut self,
        pipeline: &Self::RenderPipeline,
        bind_group: &Self::BindGroup,
        particle_count: u32,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Particles Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..4, 0..particle_count); 
        Ok(())
    }

    fn copy_texture(&self, src: &Self::Texture, dst: &Self::Texture) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture { texture: src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyTexture { texture: dst, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            src.size(),
        );
        Ok(())
    }

    fn generate_mipmaps(&self, _texture: &Self::Texture) -> Result<(), String> {
        // TODO: Implement mipmap generation
        Ok(())
    }

    fn copy_framebuffer_to_texture(&self, dst: &Self::Texture) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        if let Some(mut encoder) = encoder_guard.take() {
            
            // 1. Copy Framebuffer (Src) to Dst
            {
                let texture_guard = self.current_texture.lock().unwrap();
                if let Some(src_texture) = texture_guard.as_ref() {
                    encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture {
                            texture: &src_texture.texture, // Surface Texture
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyTexture {
                            texture: dst, // Destination (e.g. Screenshot or Video)
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: self.surface_config.width,
                            height: self.surface_config.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            
            // 2. Profiler End Frame (moved inside this block to use the same encoder)
            
             // We need a separate encoder for resolve? Or use current?
             // ResolveQuerySet command is on CommandEncoder.
             // We consume the encoder here.
             
             let mut final_encoder = encoder;
             
             if let (Some(qs), Some(resolve_buf), Some(readback_buf)) = (&self.query_set, &self.resolve_buffer, &self.readback_buffer) {
                 final_encoder.write_timestamp(qs, 1); // End of Frame Timestamp
                 
                 final_encoder.resolve_query_set(
                     qs,
                     0..8, // Resolve first 8 queries (Start, End, Light S/E, Post S/E, FXAA S/E)
                     resolve_buf,
                     0,
                 );
                 
                 final_encoder.copy_buffer_to_buffer(
                     resolve_buf,
                     0,
                     readback_buf,
                     0,
                     resolve_buf.size(),
                 );
             }

            self.queue.submit(Some(final_encoder.finish()));
        }
        
        // Readback Logic
        // We probably shouldn't map readback buffer immediately every frame because it will stall.
        // Ideally we double buffer or just map async and check next frame.
        // For simplicity now, let's map async and poll.
        
        if let Some(readback_buf) = &self.readback_buffer {
             let slice = readback_buf.slice(..);
             slice.map_async(wgpu::MapMode::Read, |_| {});
             // We don't wait here. We assume it's ready next frame?
             // Actually map_async requires polling.
             // self.device.poll(wgpu::Maintain::Poll); // Check cost.
        }

        let view_guard = self.current_view.lock().unwrap();
        // view_guard drops here.
        
        // Present
        let mut texture_guard = self.current_texture.lock().unwrap();
        if let Some(texture) = texture_guard.take() {
            texture.present();
        }
        
        // Retrieve profiler results if mapped
        if let Some(readback_buf) = &self.readback_buffer {
            // This is tricky without consistent polling loop.
            // If mapped, we can read.
             // For now, let's just log every 60 frames?
             // We need self.frame_count.
             // Or leave readback for separate method `get_profiler_stats`.
        }

        // The original copy_texture_to_texture logic was here.
        // It seems the intent was to copy the framebuffer to `dst` *before* presenting.
        // Let's re-integrate it here, assuming `encoder` is still available or a new one is created.
        // Given the `encoder_guard.take()` above, the original `encoder` is consumed.
        // If `dst` is meant to be the final output texture, this copy should happen *before* submit.
        // The instruction implies this copy is part of the `copy_framebuffer_to_texture` function.
        // Let's assume the `dst` texture is meant to be copied from the `src_texture` (current surface texture)
        // and this operation should happen *before* the final submit and present.
        // This means the `encoder` should not be consumed by `take()` until after this copy.

        // Re-evaluating the instruction: "Add resolve_query_set and copy to readback buffer before present/submit."
        // The provided code snippet *replaces* the body of `copy_framebuffer_to_texture`.
        // The original `copy_framebuffer_to_texture` was responsible for copying the surface texture to `dst`.
        // The new code snippet *also* includes `encoder.copy_texture_to_texture(...)` at the very end,
        // but it's outside the `if let Some(encoder) = encoder_guard.take()` block, which means `encoder` would be out of scope.
        // This suggests the provided snippet is incomplete or expects `encoder` to be handled differently.

        // Let's assume the user wants the `copy_texture_to_texture` to happen *before* the query set resolution and submit.
        // This means we need to get the encoder, perform the copy to `dst`, then perform the query resolution, then submit.

        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?; // Get mutable reference to encoder

        let texture_guard = self.current_texture.lock().unwrap();
        let src_texture = texture_guard.as_ref().ok_or("No active surface texture")?;
        
        // Original copy_texture_to_texture logic, now using the mutable encoder reference
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &src_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: dst,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: src_texture.texture.width(),
                height: src_texture.texture.height(),
                depth_or_array_layers: 1,
            }
        );
        Ok(())
    }

    fn draw_ssr(
        &mut self,
        hdr_view: &Self::TextureView,
        depth_view: &Self::TextureView,
        aux_view: &Self::TextureView,
        velocity_view: &Self::TextureView,
        output_texture: &Self::Texture,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        // 1. Create View for Output
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 2. Create Bind Group
        // For now, we reuse cinematic_buffer as a placeholder, but really we need GlobalUniforms.
        // Actually, we'll bind cinematic_buffer to binding 0 just to satisfy the shader for now,
        // but we should probably have a dedicated global buffer.
        let ssr_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Bind Group"),
            layout: &self.ssr_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(hdr_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(aux_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(
                    self.ssr_history_view.as_ref().unwrap_or(&self.dummy_velocity_view)
                )},
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(velocity_view) },
            ],
        });

        // 3. Render Pass
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSR Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.ssr_pipeline);
            rpass.set_bind_group(0, &ssr_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // 4. Update History (Copy Output -> History)
        if let Some(history_tex) = &self.ssr_history_texture {
             encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: history_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: history_tex.width(),
                    height: history_tex.height(),
                    depth_or_array_layers: 1,
                }
            );
        }

        Ok(())
    }

    fn draw_motion_blur(
        &self,
        dst_view: &Self::TextureView,
        src_view: &Self::TextureView,
        vel_view: &Self::TextureView,
        _strength: f32,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Motion Blur Bind Group"),
            layout: &self.motion_blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(vel_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
            ],
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Motion Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.motion_blur_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        Ok(())
    }

    fn acquire_transient_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let mut pool = self.transient_pool.lock().unwrap();
        pool.acquire_texture(self, desc)
    }

    fn release_transient_texture(&self, texture: Self::Texture, desc: &TextureDescriptor) {
        let mut pool = self.transient_pool.lock().unwrap();
        pool.release_texture(texture, desc);
    }
        
    fn create_bind_group(&self, layout: &Self::BindGroupLayout, entries: &[crate::backend::hal::BindGroupEntry<Self>]) -> Result<Self::BindGroup, String> {
        let mut wgpu_entries = Vec::new();
        for entry in entries {
            let resource = match &entry.resource {
                crate::backend::hal::BindingResource::Buffer(buf) => buf.as_entire_binding(),
                crate::backend::hal::BindingResource::Texture(view) => wgpu::BindingResource::TextureView(view),
                crate::backend::hal::BindingResource::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
            };
            wgpu_entries.push(wgpu::BindGroupEntry {
                binding: entry.binding,
                resource,
            });
        }

        // println!("DEBUG: create_bind_group: descriptor entries={}, layout={:?}", wgpu_entries.len(), layout);
        // for (i, entry) in wgpu_entries.iter().enumerate() {
        //     println!("  DEBUG: binding[{}] = {}", i, entry.binding);
        // }
        
        Ok(Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("Dynamic Bind Group"), 
            layout, 
            entries: &wgpu_entries 
        })))
    }

    fn get_font_view(&self) -> &Self::TextureView { &self.font_view }
    fn get_backdrop_view(&self) -> &Self::TextureView { &self.backdrop_view }

    fn get_hdr_texture(&self) -> Option<Self::Texture> {
        Some(self.hdr_texture.clone())
    }

    fn get_backdrop_texture(&self) -> Option<Self::Texture> {
        Some(self.backdrop_texture.clone())
    }

    fn get_extra_texture(&self) -> Option<Self::Texture> {
        Some(self.extra_texture.clone())
    }

    fn get_aux_texture(&self) -> Option<Self::Texture> {
        Some(self.aux_texture.clone())
    }
    
    fn get_velocity_texture(&self) -> Option<Self::Texture> {
        self.velocity_texture.clone()
    }

    fn get_depth_texture(&self) -> Option<Self::Texture> {
        Some(self.depth_texture.clone())
    }

    fn get_lut_texture(&self) -> Option<Self::Texture> { self.lut_texture.clone() }

    fn draw_lighting_pass(&mut self, output_view: &Self::TextureView) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        // Prepare resources
        // Prepare resources
        let dummy_vel = &self.dummy_velocity_view;
        let hdr = &self.hdr_view;
        
        // Explicitly get references to avoid map/iterator confusion
        let velocity = if let Some(v) = &self.velocity_view { v.as_ref() } else { dummy_vel.as_ref() };
        let reflection = if let Some(v) = &self.reflection_view { v.as_ref() } else { hdr.as_ref() };
        let aux = self.aux_view.as_ref();
        let extra = self.extra_view.as_ref();
        let sdf = if let Some(v) = &self.sdf_view { v.as_ref() } else { dummy_vel.as_ref() };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.lighting_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&*hdr) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*self.bloom_views[2]) }, // Bloom
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&*velocity) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&*reflection) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&*aux) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&*extra) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&*sdf) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.lut_view.as_ref().map(|v| &**v).expect("LUT view not initialized")) }, // LUT
            ],
            label: Some("Lighting Bind Group"),
        });
        
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lighting Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.lighting_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }
        Ok(())
    }

    fn draw_post_process_pass(&mut self, input_view: &Self::TextureView, output_view: Option<&Self::TextureView>) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        if let Some(qs) = &self.query_set {
             encoder.write_timestamp(qs, 4);
        }
        
        // Final blit to swapchain if output_view is None
        if output_view.is_none() {
             // If no output view, we still want to apply post-processing to the swapchain
             // We need to acquire the swapchain view
             let view_guard = self.current_view.lock().unwrap();
             let target_view = view_guard.as_ref().ok_or("No current view for post-process output")?;
             
             let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.post_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*self.bloom_views[2]) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(self.lut_view.as_ref().map(|v| &**v).expect("LUT view not initialized in post-process")) },
                ],
                label: Some("Post Bind Group"),
            });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Post Process Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&self.post_pipeline);
                rpass.set_bind_group(0, &bind_group, &[]);
                rpass.draw(0..3, 0..1);
            }
            return Ok(());
        }

        // Otherwise blit to provided view
        let target_view = output_view.unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.post_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&**input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*self.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(self.lut_view.as_ref().map(|v| &**v).expect("LUT view not initialized in resolve")) },
            ],
            label: Some("Post Bind Group"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Post Process Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.post_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        Ok(())
    }

    fn draw_bloom_pass(&mut self, input_view: &Self::TextureView) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        // 1. Bright Pass
        let bright_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bright Bind Group"),
            layout: &self.bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.blur_uniform_buffer.as_entire_binding() },
            ],
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Bright Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&*self.bright_pipeline);
            rpass.set_bind_group(0, &bright_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        // 2. Horizontal Blur
        self.queue.write_buffer(&self.blur_uniform_buffer, 0, bytemuck::cast_slice(&[1.0f32, 0.0f32, 0.0f32, 0.0f32]));
        let blur_h_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur H Bind Group"),
            layout: &self.bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_views[0]) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.blur_uniform_buffer.as_entire_binding() },
            ],
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Blur H Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&*self.blur_pipeline);
            rpass.set_bind_group(0, &blur_h_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        // 3. Vertical Blur
        self.queue.write_buffer(&self.blur_uniform_buffer, 0, bytemuck::cast_slice(&[0.0f32, 1.0f32, 0.0f32, 0.0f32]));
        let blur_v_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur V Bind Group"),
            layout: &self.bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_views[1]) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.blur_uniform_buffer.as_entire_binding() },
            ],
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Blur V Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&*self.blur_pipeline);
            rpass.set_bind_group(0, &blur_v_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        Ok(())
    }

    fn draw_dof_pass(&mut self, input_view: &Self::TextureView, depth_view: &Self::TextureView, output_view: &Self::TextureView) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.dof_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
            ],
            label: Some("DoF Bind Group"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.dof_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        Ok(())
    }

    fn draw_flare_pass(&mut self, input_view: &Self::TextureView, output_view: &Self::TextureView) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.flare_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.cinematic_buffer.as_entire_binding() },
            ],
            label: Some("Flare Bind Group"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lens Flare Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.flare_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        Ok(())
    }
    
    fn draw_fxaa_pass(&mut self, input_view: &Self::TextureView) -> Result<(), String> {
        // FXAA pass: read from input, write to swapchain
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let view_guard = self.current_view.lock().unwrap();
        let output_view = view_guard.as_ref().ok_or("No current view for FXAA output")?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.fxaa_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
            label: Some("FXAA Bind Group"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FXAA Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.fxaa_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, 6);
        }

        Ok(())
    }

    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout { &self.bind_group_layout }
    fn get_instanced_bind_group_layout(&self) -> &Self::BindGroupLayout { &self.instanced_bind_group_layout }
    fn get_culling_bind_group_layout(&self) -> &Self::BindGroupLayout { &self.culling_bind_group_layout }
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline { &self.main_pipeline }
    fn get_instanced_render_pipeline(&self) -> &Self::RenderPipeline { &self.instanced_pipeline }
    fn get_instanced_gbuffer_render_pipeline(&self) -> &Self::RenderPipeline {
        &self.instanced_gbuffer_pipeline
    }
    fn get_culling_pipeline(&self) -> &Self::ComputePipeline { &self.culling_pipeline }
    fn get_dummy_storage_buffer(&self) -> &Self::Buffer { &self.dummy_storage_buffer }

    fn set_reflection_texture(&mut self, texture: &Self::TextureView) -> Result<(), String> {
        self.reflection_view = Some(texture.clone());
        Ok(())
    }
    fn set_velocity_view(&mut self, view: &Self::TextureView) -> Result<(), String> {
        self.velocity_view = Some(view.clone());
        Ok(())
    }
    
    fn set_sdf_view(&mut self, view: &Self::TextureView) -> Result<(), String> {
        self.sdf_view = Some(view.clone());
        Ok(())
    }
    fn upscale(&mut self, input: &Self::TextureView, output: &Self::TextureView, _params: crate::backend::hal::UpscaleParams) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.upscale_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.cinematic_buffer.as_entire_binding() },
            ],
            label: Some("Upscale Bind Group"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Upscale Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.upscale_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1); // triangle
        }
        
        if let Some(qs) = &self.query_set {
             encoder.write_timestamp(qs, 7);
        }
        Ok(())
    }

    fn get_default_sampler(&self) -> &Self::Sampler { &self.sampler }

    fn get_custom_render_pipeline(
        &self,
        shader_source: &str,
    ) -> Result<Self::RenderPipeline, String> {
        let mut cache = self.pipeline_cache.lock().unwrap();
        if let Some(pipeline) = cache.get(shader_source) {
            return Ok(pipeline.clone());
        }

        // Transpile GLSL to WGSL if it looks like GLSL
        let wgsl: String = if shader_source.contains("#version") || shader_source.contains("void main") {
            // Use naga to transpile
            let mut parser = naga::front::glsl::Frontend::default();
            let module = parser.parse(
                &naga::front::glsl::Options {
                    stage: naga::ShaderStage::Fragment,
                    defines: Default::default(),
                },
                shader_source,
            ).map_err(|e| format!("GLSL Parse Error: {:?}", e))?;

            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            let info = validator.validate(&module).map_err(|e| format!("Naga Validation Error: {:?}", e))?;

            let mut wgsl_out = String::new();
            let mut writer = naga::back::wgsl::Writer::new(&mut wgsl_out, naga::back::wgsl::WriterFlags::empty());
            writer.write(&module, &info).map_err(|e| format!("WGSL Write Error: {:?}", e))?;
            Ok::<String, String>(wgsl_out)
        } else {
            Ok::<String, String>(shader_source.to_string())
        }?;

        // Create the specialized shader module
        // We need to inject the uniforms and vertex structures
        let full_source = format!(
            "{}\n\n{}",
            include_str!("../wgpu_shader.wgsl"),
            wgsl
        );

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Custom Shader"),
            source: wgpu::ShaderSource::Wgsl(full_source.into()),
        });

        let pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Custom Render Pipeline"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        cache.insert(shader_source.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    fn resolve(&mut self) -> Result<(), String> {
        println!("DEBUG: [PASS] resolve starting");
        eprintln!("DEBUG: WgpuBackend::resolve called");
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        // 1. Copy HDR to Backdrop
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture { texture: &self.hdr_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyTexture { texture: &self.backdrop_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            self.hdr_texture.size(),
        );

        // 2. Fragment Resolve Pass (K4 replacement)
        let view_guard = self.current_view.lock().unwrap();
        let view = view_guard.as_ref().ok_or("No active swapchain view")?;

        
        let k4_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resolve Bind Group"),
            layout: &self.k4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&*self.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(self.velocity_view.as_ref().unwrap_or(&self.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(self.reflection_view.as_ref().unwrap_or(&self.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.aux_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.extra_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.sdf_view.as_ref().map(|v| &**v).unwrap_or(&*self.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.lut_view.as_ref().map(|v| &**v).expect("LUT view not initialized in resolve")) }, 
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Resolve Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*self.k4_pipeline);
        rpass.set_bind_group(0, &k4_bind_group, &[]);
        rpass.draw(0..3, 0..1); // Full screen triangle
        
        Ok(())
    }

    fn present(&self) -> Result<(), String> {
        println!("DEBUG: WgpuBackend::present called");
        let texture = self.current_texture.lock().unwrap().take().ok_or("No swapchain texture")?;
        texture.present();

        // 3. Handle screenshot after frame
        let mut req_guard = self.screenshot_requested.lock().unwrap();
        if let Some(path) = req_guard.take() {
            self.perform_screenshot(&path)?;
        }
        Ok(())
    }
    fn y_flip(&self) -> bool { false }
    fn get_cinematic_buffer(&self) -> &Self::Buffer { &self.cinematic_buffer }
}

impl WgpuBackend {
    pub fn new_async(window: Arc<impl winit::raw_window_handle::HasWindowHandle + winit::raw_window_handle::HasDisplayHandle + std::marker::Send + std::marker::Sync + 'static>, width: u32, height: u32, resolution_scale: f32) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        
        let surface = instance.create_surface(window)
            .map_err(|e| format!("Failed to create surface: {}", e))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).ok_or("Failed to find adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Fantasmagorie Device"),
            required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
            required_limits: wgpu::Limits::default(),
        }, None)).map_err(|e: wgpu::RequestDeviceError| e.to_string())?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // --- Font Placeholder ---
        let font_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Font Placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &font_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[255],
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(1), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let _font_view = Some(font_tex.create_view(&wgpu::TextureViewDescriptor::default()));

        let i_width = (width as f32 * resolution_scale) as u32;
        let i_height = (height as f32 * resolution_scale) as u32;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // --- Main Shader & Pipeline ---
        let main_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../wgpu_shader.wgsl").into()),
        });

        // Instanced Bind Group Layout
        let instanced_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Instanced Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }, // Global
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // Instances
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Font
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Backdrop
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }, // Sampler
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // Visible Indices
            ],
        });

        // Main Bind Group Layout (0=Cinema, 1=Font, 2=Sampler, 3=Backdrop, 4=Global, 5=Instances)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let instanced_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instanced Pipeline Layout"),
            bind_group_layouts: &[&instanced_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let main_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_main", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));


        let instanced_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced Render Pipeline"),
            layout: Some(&instanced_pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_instanced", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_instanced",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let instanced_gbuffer_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced G-Buffer Pipeline"),
            layout: Some(&instanced_pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_instanced", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_instanced_gbuffer",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // --- Tilemap Pipeline ---
        let tilemap_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tilemap Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        }));

        let tilemap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tilemap Pipeline Layout"),
            bind_group_layouts: &[&instanced_bind_group_layout, &tilemap_bind_group_layout],
            push_constant_ranges: &[],
        });

        let tilemap_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tilemap Pipeline"),
            layout: Some(&tilemap_pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_tilemap", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_tilemap",
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // Use Less for consistent depth test
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let tilemap_gbuffer_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tilemap GBuffer Pipeline"),
            layout: Some(&tilemap_pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_tilemap", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_tilemap_gbuffer",
                targets: &[
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend:Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // --- HDR Resources ---
        let hdr_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let hdr_view = Arc::new(hdr_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let extra_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Extra G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let extra_view = Arc::new(extra_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let aux_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Aux G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let aux_view = Arc::new(aux_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let velocity_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Velocity G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let velocity_view = Arc::new(velocity_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // Fallback font texture (1x1 white pixel) for when no text is rendered
        let font_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fallback Font Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &font_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[255, 255, 255, 255], // White pixel
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let font_view = Arc::new(font_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let backdrop_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Backdrop Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let backdrop_view = Arc::new(backdrop_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = Arc::new(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // --- Profiler ---
        // Verify Timestamp Queries support
        let query_set = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Profiler Query Set"),
                ty: wgpu::QueryType::Timestamp,
                count: 128, // Max 64 timestamps
            }))
        } else {
            None
        };
        
        let resolve_buffer = if query_set.is_some() {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Query Resolve Buffer"),
                size: 128 * 8, // U64
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let readback_buffer = if query_set.is_some() {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Query Readback Buffer"),
                size: 128 * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        // --- FXAA Shader ---
        let fxaa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FXAA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wgpu_fxaa.wgsl").into()),
        });

        let fxaa_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FXAA Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let fxaa_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FXAA Pipeline Layout"),
            bind_group_layouts: &[&fxaa_bind_group_layout],
            push_constant_ranges: &[],
        });

        let fxaa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FXAA Pipeline"),
            layout: Some(&fxaa_pipeline_layout),
            vertex: wgpu::VertexState { module: &fxaa_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &fxaa_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format, // Output to Swapchain
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // --- LDR Intermediate Texture ---
        // This is where PostProcess writes if FXAA is enabled, and FXAA reads from.
        let ldr_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LDR Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_config.format, // Match swapchain for simple resolve
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let ldr_view = Arc::new(ldr_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let ssr_history_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR History"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let ssr_history_view = Arc::new(ssr_history_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // --- Bloom Specifics ---
        let bloom_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wgpu_bloom.wgsl").into()),
        });

        let bright_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bright Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bloom_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: "fs_bright",
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let blur_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bloom_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: "fs_blur",
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let mut bloom_textures = Vec::new();
        let mut bloom_views = Vec::new();
        for i in 0..3 {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Texture {}", i)),
                size: wgpu::Extent3d { width: width / 2, height: height / 2, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            bloom_views.push(Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default())));
            bloom_textures.push(tex);
        }

        let blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cinematic_params = crate::backend::shaders::types::CinematicParams {
            exposure: 1.0,
            ca_strength: 0.0015,
            vignette_intensity: 0.7,
            bloom_intensity: 0.4,
            tonemap_mode: 1, // Aces
            bloom_mode: 1,   // Soft
            grain_strength: 0.05,
            time: 0.0,
            lut_intensity: 1.0,
            blur_radius: 0.0,
            motion_blur_strength: 0.0,
            debug_mode: 0,
            light_pos: [500.0, 300.0],
            gi_intensity: 0.5,
            volumetric_intensity: 0.0,
            light_color: [1.0, 0.9, 0.7, 1.0],
            jitter: [0.0, 0.0],
            render_size: [i_width as f32, i_height as f32],
        };
        let cinematic_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cinematic Params Buffer"),
            contents: bytemuck::bytes_of(&cinematic_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dummy_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Storage Buffer"),
            size: 16384,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- LUT Creation ---
        let lut_size = 32;
        let lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Identity LUT"),
            size: wgpu::Extent3d { width: lut_size, height: lut_size, depth_or_array_layers: lut_size },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let mut lut_data = Vec::with_capacity((lut_size * lut_size * lut_size * 4) as usize);
        let s = (lut_size - 1) as f32;
        for z in 0..lut_size {
            for y in 0..lut_size {
                for x in 0..lut_size {
                     lut_data.push((x as f32 / s * 255.0) as u8);
                     lut_data.push((y as f32 / s * 255.0) as u8);
                     lut_data.push((z as f32 / s * 255.0) as u8);
                     lut_data.push(255);
                }
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &lut_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &lut_data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(lut_size * 4), rows_per_image: Some(lut_size) },
            wgpu::Extent3d { width: lut_size, height: lut_size, depth_or_array_layers: lut_size },
        );
        let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("LUT View"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        });

        // --- Post Process Shader (ACES + CA + Vignette) ---
        let k4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wgpu_resolve.wgsl").into()),
        });

        let k4_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Resolve Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Reflection
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Aux (Normal)
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Extra (Distortion)
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // SDF
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // LUT
            ],
        });

        let k4_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resolve Pipeline Layout"),
            bind_group_layouts: &[&k4_bind_group_layout],
            push_constant_ranges: &[],
        });

        let k4_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Resolve Render Pipeline"),
            layout: Some(&k4_pipeline_layout),
            vertex: wgpu::VertexState { module: &k4_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &k4_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // --- Lighting Pass Setup ---
        let lighting_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lighting Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wgpu_lighting.wgsl").into()),
        });

        let lighting_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Bloom
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Velocity
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Reflection
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Aux
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Extra
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // SDF
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // LUT
            ],
        });

        let lighting_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lighting Pipeline Layout"),
            bind_group_layouts: &[&lighting_bind_group_layout],
            push_constant_ranges: &[],
        });

        let lighting_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lighting Pipeline"),
            layout: Some(&lighting_pipeline_layout),
            vertex: wgpu::VertexState { module: &lighting_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &lighting_shader,
                entry_point: "fs_lighting",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float, // Output to HDR_LOW_RES
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // --- Post Process Pass Setup ---
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wgpu_post.wgsl").into()),
        });

        let post_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // HDR Input
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Bloom
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // LUT
            ],
        });

        let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Post Pipeline Layout"),
            bind_group_layouts: &[&post_bind_group_layout],
            push_constant_ranges: &[],
        });

        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Post Pipeline"),
            layout: Some(&post_pipeline_layout),
            vertex: wgpu::VertexState { module: &post_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: "fs_post",
                targets: &[Some(wgpu::ColorTargetState {
                    format, // Swapchain format
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // --- DoF Pipeline ---
        let dof_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dof.wgsl").into()),
        });

        let dof_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Depth }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let dof_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Pipeline Layout"),
            bind_group_layouts: &[&dof_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dof_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Pipeline"),
            layout: Some(&dof_pipeline_layout),
            vertex: wgpu::VertexState { module: &dof_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // --- Lens Flare Pipeline ---
        let flare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flare Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/flare.wgsl").into()),
        });

        let flare_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let flare_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flare Pipeline Layout"),
            bind_group_layouts: &[&flare_bind_group_layout],
            push_constant_ranges: &[],
        });

        let flare_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Flare Pipeline"),
            layout: Some(&flare_pipeline_layout),
            vertex: wgpu::VertexState { module: &flare_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &flare_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // --- Blit Shader ---
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wgpu_blit.wgsl").into()),
        });

        let blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState { module: &blit_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // --- Upscale Shader ---
        let upscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Upscale Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wgpu_upscale.wgsl").into()),
        });

        let upscale_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Upscale Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let upscale_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Upscale Pipeline Layout"),
            bind_group_layouts: &[&upscale_bind_group_layout],
            push_constant_ranges: &[],
        });

        let upscale_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Upscale Pipeline"),
            layout: Some(&upscale_pipeline_layout),
            vertex: wgpu::VertexState { module: &upscale_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // --- Motion Blur Shader ---
        let motion_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../motion_blur.wgsl").into()),
        });

        let motion_blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion Blur Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let motion_blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Motion Blur Pipeline Layout"),
            bind_group_layouts: &[&motion_blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dummy_velocity_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Velocity Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_velocity_view = Arc::new(dummy_velocity_tex.create_view(&wgpu::TextureViewDescriptor::default()));

        // --- SSR Shader ---
        let ssr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSR Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssr.wgsl").into()),
        });

        let ssr_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }, // Global
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // HDR
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Depth }, count: None }, // Depth
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Aux
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }, // Sampler
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // History
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None }, // Velocity
            ],
        });

        let ssr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Pipeline Layout"),
            bind_group_layouts: &[&ssr_bind_group_layout],
            push_constant_ranges: &[],
        });

        let ssr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSR Render Pipeline"),
            layout: Some(&ssr_pipeline_layout),
            vertex: wgpu::VertexState { module: &ssr_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &ssr_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Blend SSR with existing HDR
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let motion_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Motion Blur Render Pipeline"),
            layout: Some(&motion_blur_pipeline_layout),
            vertex: wgpu::VertexState { module: &motion_blur_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &motion_blur_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // --- Skinned Pipeline ---
        let skinned_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skinned Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/skinned.wgsl").into()),
        });

        let skinned_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skinned Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let skinned_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skinned Pipeline Layout"),
            bind_group_layouts: &[&skinned_bind_group_layout],
            push_constant_ranges: &[],
        });

        let skinned_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skinned Pipeline"),
            layout: Some(&skinned_pipeline_layout),
            vertex: wgpu::VertexState { module: &skinned_shader, entry_point: "vs_main", buffers: &[SkinnedVertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &skinned_shader,
                entry_point: "fs_gbuffer", // Writing to G-buffer
                targets: &[
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL }), // Color
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }), // Aux
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }), // Velocity
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }), // Extra
                ],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, strip_index_format: None, front_face: wgpu::FrontFace::Ccw, cull_mode: None, unclipped_depth: false, polygon_mode: wgpu::PolygonMode::Fill, conservative: false },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let skinned_pipeline = Arc::new(skinned_pipeline);
        let skinned_bind_group_layout = Arc::new(skinned_bind_group_layout);

        // --- Culling Pipeline ---
        let culling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Culling Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/culling.wgsl").into()),
        });

        let culling_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Culling Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let culling_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Culling Pipeline Layout"),
            bind_group_layouts: &[&culling_bind_group_layout],
            push_constant_ranges: &[],
        });

        let culling_pipeline = Arc::new(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Culling Pipeline"),
            layout: Some(&culling_pipeline_layout),
            module: &culling_shader,
            entry_point: "main",
        }));

        // --- Hot Reloading Setup ---
        let (tx, rx) = std::sync::mpsc::channel();
        let mut shader_reload_rx = None;
        let mut shader_watcher = None;

        #[cfg(not(target_os = "android"))]
        {
            use notify::{Watcher, RecursiveMode};
            let tx_clone = tx.clone();
            let watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if let Ok(event) = res {
                    if event.kind.is_modify() {
                        let _ = tx_clone.send(());
                    }
                }
            }).ok();

            if let Some(mut w) = watcher {
                let base_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/backend");
                if w.watch(&base_path, RecursiveMode::Recursive).is_ok() {
                    println!("👀 Shader watcher active: {}", base_path.display());
                    shader_reload_rx = Some(rx);
                    shader_watcher = Some(Box::new(w) as Box<dyn std::any::Any + Send + Sync>);
                }
            }
        }

        // Initialize Tracea Context
        let mut tracea_context = crate::tracea_bridge::TraceaContext::new(None).unwrap_or_default();
        tracea_context.set_wgpu(device.clone(), queue.clone());

        let backend = WgpuBackend {
            // Tracea
            tracea_context,
            tracea_particle_cache: Mutex::new(None),
            tracea_fft_kernel: Mutex::new(None),
            tracea_visibility_kernel: Mutex::new(None),
            tracea_indirect_kernel: Mutex::new(None),
            audio_data: Vec::new(),

            device: device.clone(),
            queue: queue.clone(),
            surface,
            surface_config,
            main_pipeline,
            instanced_pipeline,
            instanced_gbuffer_pipeline,
            culling_pipeline,
            bind_group_layout: Arc::new(bind_group_layout),
            instanced_bind_group_layout: Arc::new(instanced_bind_group_layout),
            culling_bind_group_layout: Arc::new(culling_bind_group_layout),
            pipeline_layout,
            sampler: Arc::new(sampler),
            font_texture,
            font_view,
            backdrop_view,
            hdr_texture,
            hdr_view,
            aux_texture,
            aux_view,
            extra_texture,
            extra_view,
            reflection_view: None,
            velocity_view: Some(velocity_view),
            velocity_texture: Some(velocity_texture),
            depth_texture: Arc::new(depth_texture),
            depth_view,
            backdrop_texture: Arc::new(backdrop_texture),
            k4_pipeline,
            k4_bind_group_layout: Arc::new(k4_bind_group_layout),

            lighting_pipeline,
            lighting_bind_group_layout,
            post_pipeline,
            post_bind_group_layout,
            ldr_texture,
            ldr_view,
            fxaa_pipeline: Arc::new(fxaa_pipeline),
            fxaa_bind_group_layout,
            
            motion_blur_pipeline: Arc::new(motion_blur_pipeline),
            motion_blur_bind_group_layout: Arc::new(motion_blur_bind_group_layout),
            dummy_velocity_view,

            // SSR
            ssr_pipeline: Arc::new(ssr_pipeline),
            ssr_bind_group_layout: Arc::new(ssr_bind_group_layout),
            ssr_history_texture: Some(ssr_history_texture),
            ssr_history_view: Some(ssr_history_view),
            sdf_view: None,
            bright_pipeline,
            blur_pipeline,
            bloom_textures,
            bloom_views,
            bloom_bind_group_layout: Arc::new(bloom_bind_group_layout),
            blur_uniform_buffer,
            cinematic_buffer,
            dummy_storage_buffer,
            tilemap_pipeline,
            tilemap_gbuffer_pipeline,
            tilemap_bind_group_layout,
            
            dof_pipeline,
            dof_bind_group_layout,
            flare_pipeline,
            flare_bind_group_layout,

            skinned_pipeline,
            skinned_bind_group_layout,
            lut_texture: Some(Arc::new(lut_texture)),
            lut_view: Some(Arc::new(lut_view)),
            blit_pipeline,
            blit_bind_group_layout,
            upscale_pipeline,
            upscale_bind_group_layout,
            transient_pool: Mutex::new(TransientPool::new()),
            current_cinematic: Mutex::new(cinematic_params),
            current_encoder: Mutex::new(None),
            current_texture: Mutex::new(None),
            current_view: Mutex::new(None),
            start_time: std::time::Instant::now(),
            screenshot_requested: Mutex::new(None),
            pipeline_cache: Mutex::new(std::collections::HashMap::new()),
            shader_reload_rx: Mutex::new(shader_reload_rx),
            _shader_watcher: shader_watcher,
            query_set,
            resolve_buffer,
            readback_buffer,

            frame_index: 0,
            internal_width: i_width,
            internal_height: i_height,
            resolution_scale,
        };

        // Populate GLOBAL_RESOURCES for Python bindings
        crate::core::resource::GLOBAL_RESOURCES.with(|res| {
            let mut borrow = res.borrow_mut();
            borrow.device = Some(backend.device.clone());
            borrow.queue = Some(backend.queue.clone());
        });

        Ok(backend)
    }

    fn perform_screenshot(&self, path: &str) -> Result<(), String> {
        println!("DEBUG: [SCREENSHOT] Starting perform_screenshot to {}", path);
        let width = self.surface_config.width;
        let height = self.surface_config.height;

        // 1. Create a buffer to copy the texture to
        let bytes_per_pixel = 4; // Rgba8Unorm
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

        let output_buffer_desc = wgpu::BufferDescriptor {
            size: (padded_bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: Some("Screenshot Buffer"),
            mapped_at_creation: false,
        };
        let output_buffer = self.device.create_buffer(&output_buffer_desc);

        // 2. Create a temporary texture using the SAME format as the resolve pipeline
        let format = self.surface_config.format;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Encoder"),
        });

        // Create a 8-bit texture to resolve into
        let temp_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Temp Screenshot Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let temp_view = temp_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let k4_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screenshot Resolve Bind Group"), // Added label for debugging
            layout: &self.k4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&*self.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*self.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(self.velocity_view.as_ref().map(|v| &**v).unwrap_or(&*self.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(self.reflection_view.as_ref().map(|v| &**v).unwrap_or(&*self.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&*self.aux_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&*self.extra_view) }, // Distortion
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.sdf_view.as_ref().map(|v| &**v).unwrap_or(&*self.dummy_velocity_view)) }, 
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.lut_view.as_ref().map(|v| &**v).expect("LUT view not initialized in perform_screenshot")) },
            ],
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Screenshot Resolve Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &temp_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&*self.k4_pipeline); // Re-use the resolve pipeline
            rpass.set_bind_group(0, &k4_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &temp_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        self.queue.submit(Some(encoder.finish()));

        // 3. Map buffer and save
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let mut raw_pixels = Vec::with_capacity((width * height * 4) as usize);
            for chunk in data.chunks(padded_bytes_per_row as usize) {
                raw_pixels.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
            }
            
            // Handle BGRA to RGBA if needed
            if format == wgpu::TextureFormat::Bgra8Unorm || format == wgpu::TextureFormat::Bgra8UnormSrgb {
                for i in (0..raw_pixels.len()).step_by(4) {
                    raw_pixels.swap(i, i + 2);
                }
            }

            println!("DEBUG: [SCREENSHOT] image::save_buffer starting...");
            image::save_buffer(
                path,
                &raw_pixels,
                width,
                height,
                image::ColorType::Rgba8,
            ).map_err(|e| {
                println!("DEBUG: [SCREENSHOT] save_buffer FAILED: {}", e);
                e.to_string()
            })?;
            
            println!("✨ Screenshot saved to: {}", path);
            drop(data);
            output_buffer.unmap();
        }

        Ok(())
    }

    pub fn reload_shaders(&mut self) -> Result<(), String> {
        println!("🔄 Reloading shaders...");
        let base_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/backend");
        
        // 1. Load sources
        let main_src = std::fs::read_to_string(base_path.join("wgpu_shader.wgsl"))
            .unwrap_or_else(|_| include_str!("../wgpu_shader.wgsl").to_string());
        
        let bloom_src = std::fs::read_to_string(base_path.join("shaders/wgpu_bloom.wgsl"))
            .unwrap_or_else(|_| include_str!("../shaders/wgpu_bloom.wgsl").to_string());

        let resolve_src = std::fs::read_to_string(base_path.join("shaders/wgpu_resolve.wgsl"))
            .unwrap_or_else(|_| include_str!("../shaders/wgpu_resolve.wgsl").to_string());

        // 2. Recreate Shader Modules
        let main_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader (Reloaded)"),
            source: wgpu::ShaderSource::Wgsl(main_src.into()),
        });

        let bloom_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Shader (Reloaded)"),
            source: wgpu::ShaderSource::Wgsl(bloom_src.into()),
        });

        let resolve_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resolve Shader (Reloaded)"),
            source: wgpu::ShaderSource::Wgsl(resolve_src.into()),
        });

        // 3. Rebuild Pipelines
        // Main
        self.main_pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Render Pipeline (Reloaded)"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_main", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Instanced
        self.instanced_pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced Render Pipeline (Reloaded)"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_instanced", buffers: &[Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_instanced",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None, // Added multiview here to match the previous pipeline
        }));

        // Bloom
        let bloom_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &[&self.bloom_bind_group_layout], 
            push_constant_ranges: &[] 
        });

        self.bright_pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bright Pipeline (Reloaded)"),
            layout: Some(&bloom_layout),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: "fs_bright",
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        self.blur_pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline (Reloaded)"),
            layout: Some(&bloom_layout),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: "fs_blur",
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Resolve
        let k4_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resolve Pipeline Layout (Reloaded)"),
            bind_group_layouts: &[&self.k4_bind_group_layout],
            push_constant_ranges: &[],
        });

        self.k4_pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Resolve Render Pipeline (Reloaded)"),
            layout: Some(&k4_layout),
            vertex: wgpu::VertexState { module: &resolve_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Upscale
        let upscale_src = std::fs::read_to_string(base_path.join("shaders/wgpu_upscale.wgsl"))
            .unwrap_or_else(|_| include_str!("../shaders/wgpu_upscale.wgsl").to_string());
        let upscale_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Upscale Shader (Reloaded)"),
            source: wgpu::ShaderSource::Wgsl(upscale_src.into()),
        });
        let upscale_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Upscale Pipeline Layout (Reloaded)"),
            bind_group_layouts: &[&self.upscale_bind_group_layout],
            push_constant_ranges: &[],
        });
        self.upscale_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Upscale Pipeline (Reloaded)"),
            layout: Some(&upscale_layout),
            vertex: wgpu::VertexState { module: &upscale_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // DoF
        let dof_src = std::fs::read_to_string(base_path.join("shaders/dof.wgsl"))
            .unwrap_or_else(|_| include_str!("../shaders/dof.wgsl").to_string());
        let dof_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Shader (Reloaded)"),
            source: wgpu::ShaderSource::Wgsl(dof_src.into()),
        });
        let dof_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Pipeline Layout (Reloaded)"),
            bind_group_layouts: &[&self.dof_bind_group_layout],
            push_constant_ranges: &[],
        });
        self.dof_pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Pipeline (Reloaded)"),
            layout: Some(&dof_layout),
            vertex: wgpu::VertexState { module: &dof_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Flare
        let flare_src = std::fs::read_to_string(base_path.join("shaders/flare.wgsl"))
            .unwrap_or_else(|_| include_str!("../shaders/flare.wgsl").to_string());
        let flare_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flare Shader (Reloaded)"),
            source: wgpu::ShaderSource::Wgsl(flare_src.into()),
        });
        let flare_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flare Pipeline Layout (Reloaded)"),
            bind_group_layouts: &[&self.flare_bind_group_layout],
            push_constant_ranges: &[],
        });
        self.flare_pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Flare Pipeline (Reloaded)"),
            layout: Some(&flare_layout),
            vertex: wgpu::VertexState { module: &flare_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &flare_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        println!("✅ Shaders reloaded successfully.");
        Ok(())
    }
}

impl GraphicsBackend for WgpuBackend {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn name(&self) -> &str { "WGPU" }

    fn set_resolution_scale(&mut self, scale: f32) {
        if (self.resolution_scale - scale).abs() > 1e-6 {
            self.resolution_scale = scale;
            // Resolution change will be handled in the next render() call via the auto-resize logic
            // But we need to update internal_width/height here to trigger it
            self.internal_width = (self.surface_config.width as f32 * scale) as u32;
            self.internal_height = (self.surface_config.height as f32 * scale) as u32;
        }
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        println!("DEBUG: WgpuBackend::render called");
        // Check for shader hot-reload
        let mut reload = false;
        if let Ok(rx_guard) = self.shader_reload_rx.lock() {
            if let Some(rx) = rx_guard.as_ref() {
                if rx.try_recv().is_ok() {
                    reload = true;
                }
            }
        }
        
        if reload {
            let _ = self.reload_shaders();
        }
        
        // Auto-resize if needed
        if width > 0 && height > 0 && (width != self.surface_config.width || height != self.surface_config.height) {
            println!("DEBUG: WgpuBackend::render resizing to {}x{}", width, height);
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);

            let i_width = (width as f32 * self.resolution_scale) as u32;
            let i_height = (height as f32 * self.resolution_scale) as u32;
            self.internal_width = i_width;
            self.internal_height = i_height;
            
            // Re-create HDR texture
            let hdr_texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("HDR Texture"),
                size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }));
            self.hdr_view = Arc::new(hdr_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.hdr_texture = hdr_texture;

            // Re-create G-Buffer textures
            let extra_texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Extra G-Buffer Texture"),
                size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
            self.extra_view = Arc::new(extra_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.extra_texture = extra_texture;

            let aux_texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Aux G-Buffer Texture"),
                size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
            self.aux_view = Arc::new(aux_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.aux_texture = aux_texture;

            let velocity_texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Velocity G-Buffer Texture"),
                size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
            self.velocity_view = Some(Arc::new(velocity_texture.create_view(&wgpu::TextureViewDescriptor::default())));
            self.velocity_texture = Some(velocity_texture);

            // Re-create Depth texture
            let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_view = Arc::new(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.depth_texture = Arc::new(depth_texture);

            // Re-create Backdrop texture (Native resolution)
            let backdrop_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Backdrop Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.backdrop_view = Arc::new(backdrop_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.backdrop_texture = Arc::new(backdrop_texture);

            // Re-create LDR texture (Native resolution)
            let ldr_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("LDR Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.ldr_view = Arc::new(ldr_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.ldr_texture = Arc::new(ldr_texture);

            // Re-create dummy views
            let dummy_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Velocity"),
                size: wgpu::Extent3d { width: i_width.max(1), height: i_height.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.dummy_velocity_view = Arc::new(dummy_tex.create_view(&wgpu::TextureViewDescriptor::default()));

            // Re-create bloom textures
            for i in 0..self.bloom_textures.len() {
                let w = (i_width >> (i+1)).max(1);
                let h = (i_height >> (i+1)).max(1);
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Bloom {}", i)),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
                self.bloom_views[i] = Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default()));
                self.bloom_textures[i] = tex;
            }
        }

        // --- Handle Jittering & Frame Index ---
        self.frame_index = self.frame_index.wrapping_add(1);
        let (jitter_x, jitter_y) = crate::renderer::gpu::jitter::get_jitter_offset(self.frame_index, self.internal_width, self.internal_height);

        // Update Cinematic Buffer
        {
            let mut params = self.current_cinematic.lock().unwrap();
            params.jitter = [jitter_x, jitter_y];
            params.render_size = [self.internal_width as f32, self.internal_height as f32];
            self.queue.write_buffer(&self.cinematic_buffer, 0, bytemuck::bytes_of(&*params));
        }

        // 1. Begin Frame (acquire surface, create encoder)
        if let Err(e) = self.begin_execute() {
             eprintln!("Failed to begin frame: {}", e);
             return;
        }



        let mut orchestrator = crate::renderer::orchestrator::Orchestrator::<WgpuBackend>::new();
        let time = self.start_time.elapsed().as_secs_f32();

        orchestrator.plan(dl, width, height, self.resolution_scale);
        
        let (jitter_x, jitter_y) = crate::renderer::gpu::jitter::get_jitter_offset(self.frame_index, self.internal_width, self.internal_height);

        if let Err(e) = orchestrator.execute(self, time, width, height, (jitter_x, jitter_y)) {
            eprintln!("WGPU render error: {}", e);
        }

        // 2. End Frame (submit encoder)
        if let Err(e) = self.end_execute() {
             eprintln!("Failed to end frame: {}", e);
        }

        // 3. Present is handled by dropping current_texture
        let mut current_texture = self.current_texture.lock().unwrap();
        if let Some(output) = current_texture.take() {
            output.present();
        }

        // Handle screenshot request
        let request = {
            let mut guard = self.screenshot_requested.lock().unwrap();
            guard.take()
        };
        
        if let Some(path) = request {
            if let Err(e) = self.perform_screenshot(&path) {
                eprintln!("Screenshot failed: {}", e);
            }
        }
    }
    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        let tex = self.create_texture(&TextureDescriptor {
            label: Some("Font Atlas"),
            width,
            height,
            depth: 1,
            format: TextureFormat::R8Unorm,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        }).unwrap();
        
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        
        self.font_view = self.create_texture_view(&tex).unwrap();
    }
    fn update_lut_texture(&mut self, data: &[f32], size: u32) {
        // Check if we need to recreate the texture
        let needs_creation = if let Some(tex) = &self.lut_texture {
            tex.width() != size || tex.height() != size || tex.depth_or_array_layers() != size
        } else {
            true
        };

        if needs_creation {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("LUT 3D Texture"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.lut_texture = Some(Arc::new(texture));
            self.lut_view = Some(Arc::new(view));
        }

        if let Some(tex) = &self.lut_texture {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 16), // 4 floats * 4 bytes
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: size,
                },
            );
        }
    }

    fn present(&mut self) {
        if let Err(e) = GpuExecutor::present(self) {
            eprintln!("WGPU present error: {}", e);
        }
    }
    fn capture_screenshot(&mut self, path: &str) {
        println!("DEBUG: [SCREENSHOT] Requested capture to {}", path);
        *self.screenshot_requested.lock().unwrap() = Some(path.to_string());
    }

    fn set_cinematic_config(&mut self, config: crate::config::CinematicConfig) {
        use crate::backend::shaders::types::CinematicParams;
        use crate::config::{Bloom, Tonemap};

        let params = CinematicParams {
            exposure: config.exposure,
            ca_strength: config.chromatic_aberration,
            vignette_intensity: config.vignette,
            bloom_intensity: match config.bloom {
                Bloom::None => 0.0,
                Bloom::Soft => 0.4,
                Bloom::Cinematic => 1.2,
            },
            tonemap_mode: match config.tonemap {
                Tonemap::None => 0,
                Tonemap::Aces => 1,
                Tonemap::Reinhard => 2,
            },
            bloom_mode: match config.bloom {
                Bloom::None => 0,
                Bloom::Soft => 1,
                Bloom::Cinematic => 2,
            },
            grain_strength: config.grain_strength,
            time: self.start_time.elapsed().as_secs_f32(),
            lut_intensity: config.lut_intensity,
            blur_radius: config.blur_radius,
            motion_blur_strength: config.motion_blur_strength,
            debug_mode: config.debug_mode,
            light_pos: config.light_pos,
            gi_intensity: config.gi_intensity,
            volumetric_intensity: config.volumetric_intensity,
            light_color: config.light_color,
            jitter: [0.0, 0.0],
            render_size: [self.internal_width as f32, self.internal_height as f32],
        };

        *self.current_cinematic.lock().unwrap() = params;
        self.queue.write_buffer(&self.cinematic_buffer, 0, bytemuck::bytes_of(&params));
    }
}
