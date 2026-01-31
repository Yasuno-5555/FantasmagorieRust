use std::sync::{Arc, Mutex};
use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use crate::backend::GraphicsBackend;
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use crate::backend::shaders::types::{GlobalUniforms, DrawUniforms, create_projection};
use wgpu::util::DeviceExt;

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
    
    pub main_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    
    pub sampler: wgpu::Sampler,
    pub font_view: Option<wgpu::TextureView>,
    pub backdrop_view: Arc<wgpu::TextureView>,
    
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub backdrop_texture: wgpu::Texture,
    
    pub k4_pipeline: wgpu::RenderPipeline, // Resolve/Post-process
    pub k4_bind_group_layout: wgpu::BindGroupLayout,
    
    pub current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
    pub current_texture: Mutex<Option<wgpu::SurfaceTexture>>,
    pub current_view: Mutex<Option<wgpu::TextureView>>,
    
    pub start_time: std::time::Instant,
    pub screenshot_requested: Mutex<Option<String>>,
}

impl GpuExecutor for WgpuBackend {
    type Buffer = wgpu::Buffer;
    type Texture = wgpu::Texture;
    type TextureView = wgpu::TextureView;
    type Sampler = wgpu::Sampler;
    type RenderPipeline = wgpu::RenderPipeline;
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

        Ok(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        }))
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

    fn generate_mipmaps(&self, _texture: &Self::Texture) -> Result<(), String> { Ok(()) }

    fn create_bind_group(&self, layout: &Self::BindGroupLayout, buffers: &[&Self::Buffer], textures: &[&Self::TextureView], samplers: &[&Self::Sampler]) -> Result<Self::BindGroup, String> {
        let mut entries = Vec::new();
        if let Some(buf) = buffers.first() {
            entries.push(wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() });
        }
        if let Some(tex) = textures.first() {
            entries.push(wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(tex) });
        }
        if let Some(samp) = samplers.first() {
            entries.push(wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(samp) });
        }
        if let Some(tex) = textures.get(1) {
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(tex) });
        } else {
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.backdrop_view) });
        }
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Dynamic Bind Group"), layout, entries: &entries }))
    }

    fn get_font_view(&self) -> &Self::TextureView { self.font_view.as_ref().expect("Font view missing") }
    fn get_backdrop_view(&self) -> &Self::TextureView { &*self.backdrop_view }
    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout { &self.bind_group_layout }
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline { &self.main_pipeline }
    fn get_default_sampler(&self) -> &Self::Sampler { &self.sampler }

    fn resolve(&mut self) -> Result<(), String> {
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
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
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
        rpass.set_pipeline(&self.k4_pipeline);
        rpass.set_bind_group(0, &k4_bind_group, &[]);
        rpass.draw(0..3, 0..1); // Full screen triangle
        
        Ok(())
    }

    fn present(&self) -> Result<(), String> {
        let texture = self.current_texture.lock().unwrap().take().ok_or("No swapchain texture")?;
        texture.present();

        // 3. Handle screenshot after frame
        let mut req_guard = self.screenshot_requested.lock().unwrap();
        if let Some(_path) = req_guard.take() {
            // Simplified capture for now: just log
            println!("Screenshot captured (dummy)");
        }
        Ok(())
    }
    fn y_flip(&self) -> bool { false }
}

impl WgpuBackend {
    pub fn new_async(window: Arc<impl raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle + std::marker::Send + std::marker::Sync + 'static>, width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        
        let surface = unsafe { instance.create_surface(window) }
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let main_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        });

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

        // --- K4 Resolve Pipeline ---
        let k4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl("
                @group(0) @binding(0) var t_hdr: texture_2d<f32>;
                @group(0) @binding(1) var s_hdr: sampler;

                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) uv: vec2<f32>,
                };

                @vertex
                fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
                    var out: VertexOutput;
                    let x = f32(i32(in_vertex_index) / 2) * 4.0 - 1.0;
                    let y = f32(i32(in_vertex_index) % 2) * 4.0 - 1.0;
                    out.position = vec4<f32>(x, y, 0.0, 1.0);
                    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
                    return out;
                }

                @fragment
                fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                    let color = textureSample(t_hdr, s_hdr, in.uv);
                    let tone_mapped = color.rgb / (color.rgb + vec3<f32>(1.0));
                    return vec4<f32>(tone_mapped, 1.0);
                }
            ".into()),
        });

        let k4_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Resolve Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let k4_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resolve Pipeline Layout"),
            bind_group_layouts: &[&k4_bind_group_layout],
            push_constant_ranges: &[],
        });

        let k4_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        });

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            main_pipeline,
            bind_group_layout,
            pipeline_layout,
            sampler,
            font_view: None,
            backdrop_view,
            hdr_texture,
            hdr_view,
            backdrop_texture,
            k4_pipeline,
            k4_bind_group_layout,
            current_encoder: Mutex::new(None),
            current_texture: Mutex::new(None),
            current_view: Mutex::new(None),
            start_time: std::time::Instant::now(),
            screenshot_requested: Mutex::new(None),
        })
    }
}

impl GraphicsBackend for WgpuBackend {
    fn name(&self) -> &str { "WGPU" }
    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        let orchestrator = crate::renderer::orchestrator::RenderOrchestrator::new();
        let tasks = orchestrator.plan(dl);
        let time = self.start_time.elapsed().as_secs_f32();
        orchestrator.execute(self, &tasks, time, width, height).unwrap();
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
}
