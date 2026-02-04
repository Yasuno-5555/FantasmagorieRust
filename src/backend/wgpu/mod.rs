use std::sync::{Arc, Mutex};
use crate::draw::DrawList;
use crate::backend::GraphicsBackend;
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use crate::backend::shaders::types::create_projection;
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

pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    
    pub main_pipeline: Arc<wgpu::RenderPipeline>,
    pub instanced_pipeline: Arc<wgpu::RenderPipeline>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    
    pub sampler: wgpu::Sampler,
    pub font_view: Option<wgpu::TextureView>,
    pub backdrop_view: Arc<wgpu::TextureView>,
    
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub backdrop_texture: wgpu::Texture,
    
    pub k4_pipeline: Arc<wgpu::RenderPipeline>, // Resolve/Post-process
    pub k4_bind_group_layout: wgpu::BindGroupLayout,
    
    // Bloom
    pub bright_pipeline: Arc<wgpu::RenderPipeline>,
    pub blur_pipeline: Arc<wgpu::RenderPipeline>,
    pub bloom_textures: Vec<wgpu::Texture>,
    pub bloom_views: Vec<wgpu::TextureView>,
    pub bloom_bind_group_layout: wgpu::BindGroupLayout,
    pub blur_uniform_buffer: wgpu::Buffer,
    pub cinematic_buffer: wgpu::Buffer,
    pub dummy_storage_buffer: wgpu::Buffer, // Fallback for instanced bindings
    pub current_cinematic: Mutex<crate::backend::shaders::types::CinematicParams>,

    pub current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
    pub current_texture: Mutex<Option<wgpu::SurfaceTexture>>,
    pub current_view: Mutex<Option<wgpu::TextureView>>,
    
    pub start_time: std::time::Instant,
    
    // Resource Management
    pub transient_pool: Mutex<TransientPool<WgpuBackend>>,
    
    // Screenshot
    pub screenshot_requested: Mutex<Option<String>>,
    pub pipeline_cache: Mutex<std::collections::HashMap<String, Arc<wgpu::RenderPipeline>>>,

    pub shader_reload_rx: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    pub _shader_watcher: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl GpuExecutor for WgpuBackend {
    type Buffer = wgpu::Buffer;
    type Texture = wgpu::Texture;
    type TextureView = wgpu::TextureView;
    type Sampler = wgpu::Sampler;
    type RenderPipeline = Arc<wgpu::RenderPipeline>;
    type ComputePipeline = wgpu::ComputePipeline;
    type BindGroupLayout = wgpu::BindGroupLayout;
    type BindGroup = wgpu::BindGroup;

    fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String> {
        let wgpu_usage = match usage {
            BufferUsage::Vertex => wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Index => wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Storage => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            BufferUsage::CopySrc => wgpu::BufferUsages::COPY_SRC,
            BufferUsage::CopyDst => wgpu::BufferUsages::COPY_DST,
        };
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
        };
        let mut usage = wgpu::TextureUsages::empty();
        if desc.usage.contains(TextureUsage::COPY_SRC) { usage |= wgpu::TextureUsages::COPY_SRC; }
        if desc.usage.contains(TextureUsage::COPY_DST) { usage |= wgpu::TextureUsages::COPY_DST; }
        if desc.usage.contains(TextureUsage::TEXTURE_BINDING) { usage |= wgpu::TextureUsages::TEXTURE_BINDING; }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) { usage |= wgpu::TextureUsages::STORAGE_BINDING; }
        if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) { usage |= wgpu::TextureUsages::RENDER_ATTACHMENT; }

        Ok(self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label,
            size: wgpu::Extent3d { width: desc.width, height: desc.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        }))
    }

    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String> {
        Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }

    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String> {
        Ok(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }))
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
        
        let layouts: Vec<&wgpu::BindGroupLayout> = if let Some(l) = layout { vec![l] } else { vec![] };
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            push_constant_ranges: &[],
        });

        Ok(Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
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
        })))
    }

    fn create_compute_pipeline(&self, label: &str, wgsl_source: &str, layout: Option<&Self::BindGroupLayout>) -> Result<Self::ComputePipeline, String> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });
        
        let layouts: Vec<&wgpu::BindGroupLayout> = if let Some(l) = layout { vec![l] } else { vec![] };
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            push_constant_ranges: &[],
        });

        Ok(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        }))
    }

    fn begin_execute(&self) -> Result<(), String> {
        let output = self.surface.get_current_texture().map_err(|e| e.to_string())?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        *self.current_texture.lock().unwrap() = Some(output);
        *self.current_view.lock().unwrap() = Some(view);
        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") });
        *self.current_encoder.lock().unwrap() = Some(encoder);
        Ok(())
    }

    fn end_execute(&self) -> Result<(), String> {
        let encoder = self.current_encoder.lock().unwrap().take().ok_or("No active encoder")?;
        self.queue.submit(Some(encoder.finish()));
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
        // We use @builtin(instance_index) in shader, so we just say 0..instance_count
        rpass.draw(0..vertex_count, 0..instance_count);
        Ok(())
    }

    fn dispatch(&self, pipeline: &Self::ComputePipeline, bind_group: Option<&Self::BindGroup>, groups: [u32; 3], _push_constants: &[u8]) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Task"), timestamp_writes: None });
        cpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group { cpass.set_bind_group(0, bg, &[]); }
        cpass.dispatch_workgroups(groups[0], groups[1], groups[2]);
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
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        // We need to access the current surface texture
        // Note: This requires us to store the surface texture in a generic-accessible way
        // For now, let's assume we can grab it if we stored it (which we do in Render)
        
        let texture_guard = self.current_texture.lock().unwrap();
        let src_texture = texture_guard.as_ref().ok_or("No active surface texture")?;
        
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

    fn acquire_transient_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let mut pool = self.transient_pool.lock().unwrap();
        pool.acquire_texture(self, desc)
    }

    fn release_transient_texture(&self, texture: Self::Texture, desc: &TextureDescriptor) {
        let mut pool = self.transient_pool.lock().unwrap();
        pool.release_texture(texture, desc);
    }

    fn create_bind_group(&self, layout: &Self::BindGroupLayout, buffers: &[&Self::Buffer], textures: &[&Self::TextureView], samplers: &[&Self::Sampler]) -> Result<Self::BindGroup, String> {
        let mut entries = Vec::new();
        
        // Binding 0: Uniform (Main)
        // If we have only 1 buffer, it's the main uniform buffer for Text/Primitives
        let buf0 = if buffers.len() == 1 { buffers[0] } else { &self.cinematic_buffer }; // Fallback to something
        entries.push(wgpu::BindGroupEntry { binding: 0, resource: buf0.as_entire_binding() });

        // Binding 1: Texture
        let tex1 = textures.first().map(|&t| t).unwrap_or(self.font_view.as_ref().unwrap());
        entries.push(wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(tex1) });

        // Binding 2: Sampler
        let samp2 = samplers.first().map(|&s| s).unwrap_or(&self.sampler);
        entries.push(wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(samp2) });

        // Binding 3: Backdrop
        if let Some(tex) = textures.get(1) {
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(tex) });
        } else {
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.backdrop_view) });
        }

        // Binding 4: Global Uniforms
        // Binding 5: Instance Data
        // If we have 2 buffers, they are Global and Instances
        let (buf4, buf5) = if buffers.len() >= 2 {
            (buffers[0], buffers[1])
        } else {
            (&self.cinematic_buffer, &self.dummy_storage_buffer)
        };
        entries.push(wgpu::BindGroupEntry { binding: 4, resource: buf4.as_entire_binding() });
        entries.push(wgpu::BindGroupEntry { binding: 5, resource: buf5.as_entire_binding() });

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Dynamic Bind Group"), layout, entries: &entries }))
    }

    fn get_font_view(&self) -> &Self::TextureView { self.font_view.as_ref().expect("Font view missing") }
    fn get_backdrop_view(&self) -> &Self::TextureView { &*self.backdrop_view }
    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout { &self.bind_group_layout }
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline { &self.main_pipeline }
    fn get_instanced_render_pipeline(&self) -> &Self::RenderPipeline { &self.instanced_pipeline }
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
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        // 1. Copy HDR to Backdrop
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture { texture: &self.hdr_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyTexture { texture: &self.backdrop_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            self.hdr_texture.size(),
        );

        // 2. Bloom Passes
        // Bright Pass
        let bright_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bright Bind Group"),
            layout: &self.bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
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

        // Horizontal Blur
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

        // Vertical Blur
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

        // 3. Fragment Resolve Pass (K4 replacement)
        let view_guard = self.current_view.lock().unwrap();
        let view = view_guard.as_ref().ok_or("No active swapchain view")?;

        let k4_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resolve Bind Group"),
            layout: &self.k4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Resolve Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
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
}

impl WgpuBackend {
    pub fn new_async(window: Arc<impl winit::raw_window_handle::HasWindowHandle + winit::raw_window_handle::HasDisplayHandle + std::marker::Send + std::marker::Sync + 'static>, width: u32, height: u32) -> Result<Self, String> {
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
            required_features: wgpu::Features::empty(),
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
        let font_view = Some(font_tex.create_view(&wgpu::TextureViewDescriptor::default()));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Main Shader & Pipeline ---
        let main_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../wgpu_shader.wgsl").into()),
        });

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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let instanced_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced Render Pipeline"),
            layout: Some(&pipeline_layout),
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
            multiview: None,
        }));

        // --- HDR Resources ---
        let hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let hdr_view = hdr_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            bloom_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
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
            lut_intensity: 0.0,
            _pad: [0.0; 3],
        };
        let cinematic_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cinematic Params Buffer"),
            contents: bytemuck::bytes_of(&cinematic_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dummy_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Storage Buffer"),
            size: 64, // Minimal size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        let backend = WgpuBackend {
            device,
            queue,
            surface,
            surface_config,
            main_pipeline,
            instanced_pipeline,
            bind_group_layout,
            pipeline_layout,
            sampler,
            font_view,
            backdrop_view,
            hdr_texture,
            hdr_view,
            backdrop_texture,
            k4_pipeline,
            k4_bind_group_layout,
            bright_pipeline,
            blur_pipeline,
            bloom_textures,
            bloom_views,
            bloom_bind_group_layout,
            blur_uniform_buffer,
            cinematic_buffer,
            dummy_storage_buffer,
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
            label: None,
            layout: &self.k4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cinematic_buffer.as_entire_binding() },
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

            image::save_buffer(
                path,
                &raw_pixels,
                width,
                height,
                image::ColorType::Rgba8,
            ).map_err(|e| e.to_string())?;
            
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

        println!("✅ Shaders reloaded successfully.");
        Ok(())
    }
}

impl GraphicsBackend for WgpuBackend {
    fn name(&self) -> &str { "WGPU" }
    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
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

        let orchestrator = crate::renderer::orchestrator::RenderOrchestrator::new();
        // Plan and execute via RenderGraph
        let time = self.start_time.elapsed().as_secs_f32();

        // Update cinematic time for dynamic effects (like grain)
        {
            let mut params = self.current_cinematic.lock().unwrap();
            params.time = time;
            self.queue.write_buffer(&self.cinematic_buffer, 0, bytemuck::bytes_of(&*params));
        }
        let mut graph = orchestrator.plan(dl);
        if let Err(e) = orchestrator.execute(self, &mut graph, time, width, height) {
            if !e.contains("Outdated") && !e.contains("Lost") {
                eprintln!("Render execution failed: {}", e);
            }
        }
    }
    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        let tex = self.create_texture(&TextureDescriptor {
            label: Some("Font Atlas"),
            width,
            height,
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
        
        self.font_view = Some(self.create_texture_view(&tex).unwrap());
    }
    fn present(&mut self) {
        GpuExecutor::present(self).unwrap();
    }
    fn capture_screenshot(&mut self, path: &str) {
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
            _pad: [0.0; 3],
        };

        *self.current_cinematic.lock().unwrap() = params;
        self.queue.write_buffer(&self.cinematic_buffer, 0, bytemuck::bytes_of(&params));
    }
}
