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

pub mod resources;
pub mod pipelines;
pub mod tracea;
pub mod profiler;
pub mod passes;

use self::resources::ResourceManager;
use self::pipelines::PipelineManager;
use self::tracea::TraceaManager;
use self::profiler::WgpuProfiler;

pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    
    pub resources: ResourceManager,
    pub pipelines: Mutex<PipelineManager>,
    pub tracea: TraceaManager,
    pub profiler: WgpuProfiler,

    pub current_cinematic: Mutex<crate::backend::shaders::types::CinematicParams>,

    pub current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
    pub current_texture: Mutex<Option<wgpu::SurfaceTexture>>,
    pub current_view: Mutex<Option<Arc<wgpu::TextureView>>>,
    
    pub start_time: std::time::Instant,
    
    // Screenshot
    pub screenshot_requested: Mutex<Option<String>>,

    pub shader_reload_rx: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    pub _shader_watcher: Option<Box<dyn std::any::Any + Send + Sync>>,

    pub audio_data: Vec<f32>,
    
    // Resolution & Jitter
    pub frame_index: u32,
    pub internal_width: u32,
    pub internal_height: u32,
    pub resolution_scale: f32,
    pub config: crate::config::EngineConfig,
}


impl GpuExecutor for WgpuBackend {
    type Buffer = Arc<wgpu::Buffer>;
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
        self.tracea.dispatch_particles(&self.device, dt, attractor, sdf_texture.map(|t| t.as_ref()))
    }

    fn update_audio_data(&mut self, spectrum: &[f32]) {
        self.audio_data = spectrum.to_vec();
    }

    fn update_audio_pcm(&mut self, samples: &[f32]) {
        self.tracea.update_audio_pcm(samples);
        self.audio_data = self.tracea.audio_data.lock().unwrap().clone();
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
        Ok(Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu_usage,
            mapped_at_creation: false,
        })))
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
        // Check for shader reload events
        if let Some(rx) = self.shader_reload_rx.lock().unwrap().as_ref() {
            if rx.try_recv().is_ok() {
                // Clear any pending events
                while rx.try_recv().is_ok() {}

                // Reload pipelines. We need a way to get &mut self or handle it safely.
                // Since this is &self, we might need internal mutability or just cast.
                // In this case, self.pipelines might need to be wrapped in a Mutex or we use unsafe.
                // Given the architecture, let's check if we can make it mut or use a Mutex.
                println!("DEBUG: [WATCHER] Shader change detected! Triggering reload...");
                
                // For now, let's assume we can trigger a reload.
                // We'll need to update WgpuBackend struct to wrap pipelines in a Mutex if not already.
                let mut pipelines = self.pipelines.lock().unwrap();
                pipelines.reload(&self.device, wgpu::TextureFormat::Rgba16Float, self.surface_config.format);
            }
        }

        let output = self.surface.get_current_texture().map_err(|e| e.to_string())?;
        let view = Arc::new(output.texture.create_view(&wgpu::TextureViewDescriptor::default()));
        *self.current_texture.lock().unwrap() = Some(output);
        *self.current_view.lock().unwrap() = Some(view);
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") });
        
        // Clear main targets (internal resolution)
        {
            let color_attachments = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.resources.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.resources.aux_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.resources.extra_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                }),
            ];
            
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Frame Main Clear"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.resources.depth_view,
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
        for (i, v) in self.resources.bloom_views.iter().enumerate() {
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
        if let Some(vel) = &self.resources.velocity_view {
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
                view: &self.resources.hdr_view,
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
                view: &self.resources.hdr_view,
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
        self.tracea.dispatch_indirect_command(&self.device, &self.queue, counter_buffer, draw_commands)
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
        self.tracea.dispatch_visibility(projection, num_instances, instances, hzb, visible_indices, visible_counter)
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
                view: &self.resources.hdr_view,
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
        extra_view: &Self::TextureView,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G-Buffer Draw"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.resources.hdr_view,
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
                    view: extra_view,
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
        extra_view: &Self::TextureView,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Instanced GBuffer Indirect Geometry Draw"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &*self.resources.hdr_view,
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
                    view: extra_view,
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
        passes::draw_tilemap(self, params, data, texture_view, global_buffer, aux_view.map(|v| &**v), velocity_view.map(|v| &**v), depth_view.map(|v| &**v))
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

        let pipelines = self.pipelines.lock().unwrap();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skinned Bind Group"),
            layout: &pipelines.skinned_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bone_matrices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(texture_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.resources.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: bone_matrices_buffer.as_entire_binding() },
            ],
        });

        // Use G-buffer attachments if we want skinning to participate in lighting
        // Same logic as tilemap: check if we have G-buffer targets
        
        // Safety check for views (velocity is optional but others are mandatory for deferred)
        // Note: aux_view and extra_view are Arc<TextureView>, so always present.
        
        let velocity_view = self.resources.velocity_view.as_ref().unwrap_or(&self.resources.dummy_velocity_view);

        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment { view: &self.resources.hdr_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: &self.resources.aux_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: velocity_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: &self.resources.extra_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
        ];
        
        // Depth is always present in WgpuBackend
        let depth_att = Some(wgpu::RenderPassDepthStencilAttachment {
            view: &self.resources.depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        });

        let pipelines = self.pipelines.lock().unwrap();
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Skinned Mesh Draw"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_att,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&pipelines.skinned_pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..index_count, 0, 0..1);
        }

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
                view: &self.resources.hdr_view,
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
                        src_texture.texture.size(),
                    );
                }
            }
            
            // 2. Profiler End Frame
            self.profiler.resolve(&mut encoder);

            self.queue.submit(Some(encoder.finish()));
        }
        
        self.profiler.map_readback();

        // Present
        let mut texture_guard = self.current_texture.lock().unwrap();
        if let Some(texture) = texture_guard.take() {
            texture.present();
        }
        
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
        passes::draw_ssr(self, hdr_view, depth_view, aux_view, velocity_view, output_texture)
    }

    fn draw_motion_blur(
        &self,
        dst_view: &Self::TextureView,
        src_view: &Self::TextureView,
        vel_view: &Self::TextureView,
        _strength: f32,
    ) -> Result<(), String> {
        passes::draw_motion_blur(self, dst_view, src_view, vel_view)
    }

    fn acquire_transient_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let mut pool = self.resources.transient_pool.lock().unwrap();
        pool.acquire_texture(self, desc)
    }

    fn release_transient_texture(&self, texture: Self::Texture, desc: &TextureDescriptor) {
        let mut pool = self.resources.transient_pool.lock().unwrap();
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


    fn get_hdr_texture(&self) -> Option<Self::Texture> {
        Some(self.resources.hdr_texture.clone())
    }

    fn get_backdrop_texture(&self) -> Option<Self::Texture> {
        Some(self.resources.backdrop_texture.clone())
    }

    fn get_extra_texture(&self) -> Option<Self::Texture> {
        Some(self.resources.extra_texture.clone())
    }

    fn get_aux_texture(&self) -> Option<Self::Texture> {
        Some(self.resources.aux_texture.clone())
    }
    
    fn get_velocity_texture(&self) -> Option<Self::Texture> {
        self.resources.velocity_texture.clone()
    }

    fn get_depth_texture(&self) -> Option<Self::Texture> { Some(self.resources.depth_texture.clone()) }
    fn get_extra_texture(&self) -> Option<Self::Texture> { Some(self.resources.extra_texture.clone()) }
    fn get_taa_history_texture(&self) -> Option<Self::Texture> { self.resources.taa_history_texture.clone() }

    fn get_lut_texture(&self) -> Option<Self::Texture> { self.resources.lut_texture.clone() }

    fn draw_lighting_pass(&mut self, output_view: &Self::TextureView) -> Result<(), String> {
        passes::draw_lighting_pass(self, output_view)
    }

    fn draw_post_process_pass(&mut self, input_view: &Self::TextureView, output_view: Option<&Self::TextureView>) -> Result<(), String> {
        passes::draw_post_process_pass(self, input_view, output_view.map(|v| &**v))
    }

    fn draw_bloom_pass(&mut self, input_view: &Self::TextureView) -> Result<(), String> {
        passes::draw_bloom_pass(self, input_view)
    }

    fn draw_dof_pass(&mut self, input_view: &Self::TextureView, depth_view: &Self::TextureView, output_view: &Self::TextureView) -> Result<(), String> {
        passes::draw_dof_pass(self, input_view, depth_view, output_view)
    }

    fn draw_flare_pass(&mut self, input_view: &Self::TextureView, output_view: &Self::TextureView) -> Result<(), String> {
        passes::draw_flare_pass(self, input_view, output_view)
    }
    
    fn draw_fxaa_pass(&mut self, input_view: &Self::TextureView) -> Result<(), String> {
        passes::draw_fxaa_pass(self, input_view)
    }

    fn get_hzb_view(&self) -> Self::TextureView { self.resources.depth_view.clone() }
    fn get_font_view(&self) -> Self::TextureView { self.resources.font_view.clone() }
    fn get_backdrop_view(&self) -> Self::TextureView { self.resources.backdrop_view.clone() }

    fn get_default_bind_group_layout(&self) -> Self::BindGroupLayout { self.pipelines.lock().unwrap().bind_group_layout.clone() }
    fn get_instanced_bind_group_layout(&self) -> Self::BindGroupLayout { self.pipelines.lock().unwrap().instanced_bind_group_layout.clone() }
    fn get_culling_bind_group_layout(&self) -> Self::BindGroupLayout { self.pipelines.lock().unwrap().culling_bind_group_layout.clone() }
    fn get_default_render_pipeline(&self) -> Self::RenderPipeline { self.pipelines.lock().unwrap().main_pipeline.clone() }
    fn get_instanced_render_pipeline(&self) -> Self::RenderPipeline { self.pipelines.lock().unwrap().instanced_pipeline.clone() }
    fn get_instanced_gbuffer_render_pipeline(&self) -> Self::RenderPipeline {
        self.pipelines.lock().unwrap().instanced_gbuffer_pipeline.clone()
    }
    fn get_culling_pipeline(&self) -> Self::ComputePipeline { self.pipelines.lock().unwrap().culling_pipeline.clone() }
    fn get_dummy_storage_buffer(&self) -> Self::Buffer { self.resources.dummy_storage_buffer.clone() }

    fn set_reflection_texture(&mut self, texture: &Self::TextureView) -> Result<(), String> {
        self.resources.reflection_view = Some(texture.clone());
        Ok(())
    }
    fn set_velocity_view(&mut self, view: &Self::TextureView) -> Result<(), String> {
        self.resources.velocity_view = Some(view.clone());
        Ok(())
    }
    
    fn set_sdf_view(&mut self, view: &Self::TextureView) -> Result<(), String> {
        self.resources.sdf_view = Some(view.clone());
        Ok(())
    }
    fn upscale(&mut self, input: &Self::TextureView, output: &Self::TextureView, params: crate::backend::hal::UpscaleParams) -> Result<(), String> {
        passes::upscale(self, input, output, params)
    }
    
    fn draw_taa_pass(&mut self, current_view: &Self::TextureView, history_view: &Self::TextureView, velocity_view: &Self::TextureView, output_view: &Self::TextureView) -> Result<(), String> {
        passes::draw_taa_pass(self, current_view, history_view, velocity_view, output_view)
    }

    fn get_default_sampler(&self) -> Self::Sampler { self.resources.sampler.clone() }

    fn get_custom_render_pipeline(
        &self,
        shader_source: &str,
    ) -> Result<Self::RenderPipeline, String> {
        let pipelines = self.pipelines.lock().unwrap();
        let mut cache = pipelines.cache.lock().unwrap();
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
            layout: Some(&pipelines.pipeline_layout),
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
            wgpu::ImageCopyTexture { texture: &self.resources.hdr_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyTexture { texture: &self.resources.backdrop_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            self.resources.hdr_texture.size(),
        );

        // 2. Fragment Resolve Pass (K4 replacement)
        let view_guard = self.current_view.lock().unwrap();
        let view = view_guard.as_ref().ok_or("No active swapchain view")?;

        let pipelines = self.pipelines.lock().unwrap();
        
        let k4_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resolve Bind Group"),
            layout: &pipelines.k4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&*self.resources.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.resources.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.resources.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: self.resources.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(self.resources.velocity_view.as_ref().unwrap_or(&self.resources.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(self.resources.ssr_history_view.as_ref().unwrap_or(&self.resources.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.resources.aux_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.resources.extra_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.resources.sdf_view.as_ref().map(|v| &**v).unwrap_or(&*self.resources.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.resources.lut_view.as_ref().map(|v| &**v).unwrap_or(&self.resources.dummy_lut_view)) }, 
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
        rpass.set_pipeline(&*pipelines.k4_pipeline);
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
    fn get_cinematic_buffer(&self) -> &Self::Buffer { &self.resources.cinematic_buffer }
}

impl WgpuBackend {
    pub fn new_async(window: Arc<impl winit::raw_window_handle::HasWindowHandle + winit::raw_window_handle::HasDisplayHandle + std::marker::Send + std::marker::Sync + 'static>, width: u32, height: u32, resolution_scale: f32) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        
        let surface = instance.create_surface(window)
            .map_err(|e| format!("Failed to create surface: {}", e))?;

        let adapter = crate::core::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).ok_or("No suitable GPU adapter found")?;

        let (device, queue) = crate::core::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Fantasmagorie Device"),
            required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE | wgpu::Features::TIMESTAMP_QUERY,
            required_limits: wgpu::Limits::default(),
        }, None)).map_err(|e: wgpu::RequestDeviceError| e.to_string())?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
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

        let i_width = (width as f32 * resolution_scale) as u32;
        let i_height = (height as f32 * resolution_scale) as u32;

        let resources = ResourceManager::new(&device, &queue, &surface_config, i_width, i_height);
        let pipelines = PipelineManager::new(&device, wgpu::TextureFormat::Rgba16Float, surface_config.format);

        let cinematic_params = crate::backend::shaders::types::CinematicParams {
            exposure: 1.0,
            ca_strength: 0.0015,
            vignette_intensity: 0.7,
            bloom_intensity: 0.4,
            tonemap_mode: 1,
            bloom_mode: 1,
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
            shadow_softness: 8.0,
            _pad1: 0.0,
            _pad2: [0.0, 0.0],
        };

        let mut backend = Self {
            device: device.clone(),
            queue: queue.clone(),
            surface,
            surface_config,
            resources,
            pipelines: Mutex::new(pipelines),
            tracea: {
                let mut tm = TraceaManager::new();
                tm.context.set_wgpu(device.clone(), queue.clone());
                tm
            },
            profiler: WgpuProfiler::new(&device),
            current_cinematic: Mutex::new(cinematic_params),
            current_encoder: Mutex::new(None),
            current_texture: Mutex::new(None),
            current_view: Mutex::new(None),
            start_time: std::time::Instant::now(),
            screenshot_requested: Mutex::new(None),
            shader_reload_rx: Mutex::new(None),
            _shader_watcher: None,
            audio_data: Vec::new(),
            frame_index: 0,
            internal_width: i_width,
            internal_height: i_height,
            resolution_scale,
            config: crate::config::EngineConfig::lite(),
        };

        // Initialize Shader Watcher
        if let Err(e) = backend.setup_shader_watcher() {
            eprintln!("WARNING: Failed to setup shader watcher: {}", e);
        }

        // Populate GLOBAL_RESOURCES for Python bindings
        crate::core::resource::GLOBAL_RESOURCES.with(|res| {
            let mut borrow = res.borrow_mut();
            borrow.device = Some(backend.device.clone());
            borrow.queue = Some(backend.queue.clone());
        });

        Ok(backend)
    }

    pub fn setup_shader_watcher(&mut self) -> Result<(), String> {
        use notify::{Watcher, RecursiveMode};
        
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                if event.kind.is_modify() {
                    let _ = tx.send(());
                }
            }
        }).map_err(|e| e.to_string())?;

        // Watch shaders directory
        let mut shader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        shader_path.push("src/backend/shaders");
        
        if shader_path.exists() {
            watcher.watch(&shader_path, RecursiveMode::NonRecursive).map_err(|e| e.to_string())?;
            println!("DEBUG: [WATCHER] Watching shaders at {:?}", shader_path);
        } else {
            // Try parent if we are in a different CWD
            shader_path = std::path::PathBuf::from("src/backend/shaders");
            if shader_path.exists() {
                 watcher.watch(&shader_path, RecursiveMode::NonRecursive).map_err(|e| e.to_string())?;
                 println!("DEBUG: [WATCHER] Watching shaders at {:?}", shader_path);
            }
        }

        *self.shader_reload_rx.lock().unwrap() = Some(rx);
        self._shader_watcher = Some(Box::new(watcher));
        Ok(())
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

        let pipelines = self.pipelines.lock().unwrap();
        let k4_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screenshot Resolve Bind Group"), // Added label for debugging
            layout: &pipelines.k4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&*self.resources.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.resources.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*self.resources.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: self.resources.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(self.resources.velocity_view.as_ref().map(|v| &**v).unwrap_or(&*self.resources.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(self.resources.reflection_view.as_ref().map(|v| &**v).unwrap_or(&*self.resources.dummy_velocity_view)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&*self.resources.aux_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&*self.resources.extra_view) }, // Distortion
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.resources.sdf_view.as_ref().map(|v| &**v).unwrap_or(&*self.resources.dummy_velocity_view)) }, 
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(self.resources.lut_view.as_ref().map(|v| &**v).unwrap_or(&self.resources.dummy_lut_view)) },
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
            rpass.set_pipeline(&*pipelines.k4_pipeline); // Re-use the resolve pipeline
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

    pub fn reload_shaders(&self) -> Result<(), String> {
        println!("🔄 Reloading shaders...");
        let mut pipelines = self.pipelines.lock().unwrap();
        pipelines.reload(&self.device, wgpu::TextureFormat::Rgba16Float, self.surface_config.format);
        Ok(())
    }

    pub fn set_debug_mode(&self, mode: crate::renderer::graph::DebugDisplayMode) {
        let mut params = self.current_cinematic.lock().unwrap();
        params.debug_mode = mode as u32;
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
            
            self.resources.resize(&self.device, &self.surface_config, i_width, i_height);

        }

        // --- Handle Jittering & Frame Index ---
        self.frame_index = self.frame_index.wrapping_add(1);
        let (jitter_x, jitter_y) = crate::renderer::gpu::jitter::get_jitter_offset(self.frame_index, self.internal_width, self.internal_height);

        // Update Cinematic Buffer
        {
            let mut params = self.current_cinematic.lock().unwrap();
            params.jitter = [jitter_x, jitter_y];
            params.render_size = [self.internal_width as f32, self.internal_height as f32];
            self.queue.write_buffer(&self.resources.cinematic_buffer, 0, bytemuck::bytes_of(&*params));
        }

        // 1. Begin Frame (acquire surface, create encoder)
        if let Err(e) = self.begin_execute() {
             eprintln!("Failed to begin frame: {}", e);
             return;
        }

        let mut orchestrator = crate::renderer::orchestrator::Orchestrator::<WgpuBackend>::new();
        let time = self.start_time.elapsed().as_secs_f32();

        orchestrator.plan(dl, width, height, self.resolution_scale, self.config.profile);
        
        // Update orchestrator debug mode from cinematic params
        {
            let params = self.current_cinematic.lock().unwrap();
            let mode = match params.debug_mode {
                0 => crate::renderer::graph::DebugDisplayMode::None,
                1 => crate::renderer::graph::DebugDisplayMode::Albedo,
                2 => crate::renderer::graph::DebugDisplayMode::Normal,
                3 => crate::renderer::graph::DebugDisplayMode::Velocity,
                4 => crate::renderer::graph::DebugDisplayMode::Depth,
                5 => crate::renderer::graph::DebugDisplayMode::Emissive,
                6 => crate::renderer::graph::DebugDisplayMode::SDF,
                _ => crate::renderer::graph::DebugDisplayMode::None,
            };
            orchestrator.set_debug_mode(mode);
        }

        if let Err(e) = orchestrator.execute(self, time, width, height, (jitter_x, jitter_y)) {
            eprintln!("WGPU render error: {}", e);
        }

        // 2. Resolve Pass (Composite HDR to Swapchain)
        if let Err(e) = self.resolve() {
            eprintln!("WGPU resolve error: {}", e);
        }

        // 3. End Frame (submit encoder)
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
        
        self.resources.font_view = self.create_texture_view(&tex).unwrap();
    }
    fn update_lut_texture(&mut self, data: &[f32], size: u32) {
        // Check if we need to recreate the texture
        let needs_creation = if let Some(tex) = &self.resources.lut_texture {
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
            self.resources.lut_texture = Some(Arc::new(texture));
            self.resources.lut_view = Some(Arc::new(view));
        }

        if let Some(tex) = &self.resources.lut_texture {
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
    fn set_config(&mut self, config: crate::config::EngineConfig) {
        self.config = config;
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
            shadow_softness: 8.0,
            _pad1: 0.0,
            _pad2: [0.0, 0.0],
        };

        *self.current_cinematic.lock().unwrap() = params;
        self.queue.write_buffer(&self.resources.cinematic_buffer, 0, bytemuck::bytes_of(&params));
    }
}
