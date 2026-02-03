use std::sync::{Arc, Mutex};
use winit::raw_window_handle::{HasWindowHandle, HasDisplayHandle};
use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use crate::backend::GraphicsBackend;
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use crate::backend::shaders::types::{GlobalUniforms, DrawUniforms, PostProcessUniforms, BlendUniforms, create_projection};
use crate::renderer::graph::{RenderGraph, RenderContext};
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
    
    pub main_pipeline: Arc<wgpu::RenderPipeline>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    
    pub sampler: wgpu::Sampler,
    pub font_view: Option<wgpu::TextureView>,
    pub backdrop_view: Arc<wgpu::TextureView>,
    
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: Arc<wgpu::TextureView>,
    pub backdrop_texture: wgpu::Texture,
    
    pub k4_pipeline: Arc<wgpu::RenderPipeline>, // Resolve/Post-process
    pub k4_bind_group_layout: wgpu::BindGroupLayout,
    
    pub blend_pipeline: Arc<wgpu::RenderPipeline>,
    pub blend_bind_group_layout: wgpu::BindGroupLayout,

    pub extract_pipeline: Arc<wgpu::RenderPipeline>,
    pub blur_pipeline: Arc<wgpu::RenderPipeline>,
    pub grading_pipeline: Arc<wgpu::RenderPipeline>,
    pub hdr_grading_pipeline: Arc<wgpu::RenderPipeline>,
    pub post_process_layout: wgpu::BindGroupLayout,
    pub grading_layout: wgpu::BindGroupLayout,
    
    pub lut_view: Arc<wgpu::TextureView>,
    
    pub current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
    pub current_texture: Mutex<Option<wgpu::SurfaceTexture>>,
    pub current_view: Mutex<Option<Arc<wgpu::TextureView>>>,
    pub forced_render_target: Mutex<Option<Arc<wgpu::TextureView>>>, // For RenderGraph
    pub start_time: std::time::Instant,
    pub screenshot_requested: Mutex<Option<String>>,
    pub pipeline_cache: Mutex<std::collections::HashMap<String, Arc<wgpu::RenderPipeline>>>,
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

    fn create_buffer(&self, mut size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String> {
        let wgpu_usage = match usage {
            BufferUsage::Vertex => wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Index => wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Uniform => {
                // Uniform buffers must be a multiple of 256 bytes on some backends (DX12/Vulkan)
                size = (size + 255) & !255;
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            },
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
        let forced_target = self.forced_render_target.lock().unwrap().clone();
        if let Some(target) = forced_target {
            *self.current_view.lock().unwrap() = Some(target);
            *self.current_texture.lock().unwrap() = None;
        } else {
            let output = self.surface.get_current_texture().map_err(|e| e.to_string())?;
            let view = Arc::new(output.texture.create_view(&wgpu::TextureViewDescriptor::default()));
            *self.current_texture.lock().unwrap() = Some(output);
            *self.current_view.lock().unwrap() = Some(view);
        }
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
        let target = self.forced_render_target.lock().unwrap()
            .clone()
            .or_else(|| self.current_view.lock().unwrap().clone())
            .unwrap_or(self.hdr_view.clone());

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.as_ref(),
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
        let hdr_view = self.hdr_view.clone();
        self.resolve_from(&hdr_view)
    }

    fn present(&self) -> Result<(), String> {
        let mut tex_guard = self.current_texture.lock().unwrap();
        if let Some(texture) = tex_guard.take() {
            texture.present();
        } else {
            if self.forced_render_target.lock().unwrap().is_some() {
                // Rendering to offscreen target, no presentation needed
                return Ok(());
            }
            return Err("No swapchain texture".to_string());
        }

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
    pub fn new_async<W>(window: &W, width: u32, height: u32) -> Result<Self, String> 
    where W: winit::raw_window_handle::HasWindowHandle + winit::raw_window_handle::HasDisplayHandle + std::marker::Send + std::marker::Sync + 'static 
    {
        let instance = wgpu::Instance::default();
        
        // SAFETY: We guarantee that the window outlives the surface because the caller (main) holds the Arc.
        let surface = unsafe { 
            let s = instance.create_surface(window)
                .map_err(|e| format!("Failed to create surface: {}", e))?;
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(s)
        };

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
        let format = if caps.formats.contains(&wgpu::TextureFormat::Bgra8Unorm) {
             wgpu::TextureFormat::Bgra8Unorm
        } else if caps.formats.contains(&wgpu::TextureFormat::Bgra8UnormSrgb) {
             wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            caps.formats[0]
        };

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
        let hdr_view = Arc::new(hdr_texture.create_view(&wgpu::TextureViewDescriptor::default()));

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

        // --- Blend Pipeline ---
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blend Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blend.wgsl").into()),
        });

        let blend_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blend Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let blend_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blend Pipeline Layout"),
            bind_group_layouts: &[&blend_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blend_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blend Render Pipeline"),
            layout: Some(&blend_pipeline_layout),
            vertex: wgpu::VertexState { module: &blend_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &blend_shader,
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

        // --- Post Process Pipelines (Extract & Blur) ---
        let post_process_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Process Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/post_process.wgsl").into()),
        });

        let post_process_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Process Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let grading_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grading Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let post_process_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Post Process Pipeline Layout"),
            bind_group_layouts: &[&post_process_layout],
            push_constant_ranges: &[],
        });

        let grading_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grading Pipeline Layout"),
            bind_group_layouts: &[&grading_layout],
            push_constant_ranges: &[],
        });

        let extract_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Extract Brightness Pipeline"),
            layout: Some(&post_process_pipeline_layout),
            vertex: wgpu::VertexState { module: &post_process_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &post_process_shader,
                entry_point: "fs_extract",
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

        let blur_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
            layout: Some(&post_process_pipeline_layout),
            vertex: wgpu::VertexState { module: &post_process_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &post_process_shader,
                entry_point: "fs_blur",
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

        let grading_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grading Pipeline"),
            layout: Some(&grading_pipeline_layout),
            vertex: wgpu::VertexState { module: &post_process_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &post_process_shader,
                entry_point: "fs_grade",
                targets: &[Some(wgpu::ColorTargetState { 
                    format: format, // Use surface format
                    blend: Some(wgpu::BlendState::REPLACE), 
                    write_mask: wgpu::ColorWrites::ALL 
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let hdr_grading_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HDR Grading Pipeline"),
            layout: Some(&grading_pipeline_layout),
            vertex: wgpu::VertexState { module: &post_process_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &post_process_shader,
                entry_point: "fs_grade",
                targets: &[Some(wgpu::ColorTargetState { 
                    format: wgpu::TextureFormat::Rgba16Float, 
                    blend: Some(wgpu::BlendState::REPLACE), 
                    write_mask: wgpu::ColorWrites::ALL 
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // --- Create Default 3D LUT (Identity) ---
        let lut_size = 32;
        let mut lut_data = Vec::with_capacity(lut_size * lut_size * lut_size * 4);
        for z in 0..lut_size {
            for y in 0..lut_size {
                for x in 0..lut_size {
                    lut_data.push((x as f32 / (lut_size - 1) as f32 * 255.0) as u8);
                    lut_data.push((y as f32 / (lut_size - 1) as f32 * 255.0) as u8);
                    lut_data.push((z as f32 / (lut_size - 1) as f32 * 255.0) as u8);
                    lut_data.push(255);
                }
            }
        }
        
        let lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Identity LUT"),
            size: wgpu::Extent3d { width: lut_size as u32, height: lut_size as u32, depth_or_array_layers: lut_size as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let bytes_per_row = (4 * lut_size as u32 + 255) & !255;
        let mut padded_lut_data = Vec::with_capacity(bytes_per_row as usize * lut_size as usize * lut_size as usize);
        for z in 0..lut_size {
            for y in 0..lut_size {
                let start = (z * lut_size * lut_size + y * lut_size) * 4;
                let end = start + (lut_size * 4);
                padded_lut_data.extend_from_slice(&lut_data[start..end]);
                padded_lut_data.resize(padded_lut_data.len() + (bytes_per_row as usize - (lut_size * 4)), 0);
            }
        }

        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &lut_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &padded_lut_data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(lut_size as u32) },
            wgpu::Extent3d { width: lut_size as u32, height: lut_size as u32, depth_or_array_layers: lut_size as u32 },
        );
        let lut_view = Arc::new(lut_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let backend = WgpuBackend {
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
            blend_pipeline,
            blend_bind_group_layout,
            extract_pipeline,
            blur_pipeline,
            grading_pipeline,
            hdr_grading_pipeline,
            post_process_layout,
            grading_layout,
            lut_view,
            current_encoder: Mutex::new(None),
            current_texture: Mutex::new(None),
            current_view: Mutex::new(None),
            forced_render_target: Mutex::new(None),
            start_time: std::time::Instant::now(),
            screenshot_requested: Mutex::new(None),
            pipeline_cache: Mutex::new(std::collections::HashMap::new()),
        };

        // Populate GLOBAL_RESOURCES for Python bindings
        crate::core::resource::GLOBAL_RESOURCES.with(|res| {
            let mut borrow = res.borrow_mut();
            borrow.device = Some(backend.device.clone());
            borrow.queue = Some(backend.queue.clone());
        });

        Ok(backend)
    }

    pub fn set_render_target(&self, view: Option<Arc<wgpu::TextureView>>) {
        *self.forced_render_target.lock().unwrap() = view;
    }

    pub fn resolve_from(&self, input_view: &wgpu::TextureView) -> Result<(), String> {
        let target = self.current_view.lock().unwrap().clone()
            .ok_or("No active swapchain view for resolution")?;
        self.color_grade(input_view, target.as_ref())
    }

    pub fn extract_brightness(&self, input: &wgpu::TextureView, threshold: f32) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        let uniforms = PostProcessUniforms {
            threshold,
            _pad_to_vec2: 0.0,
            direction: [0.0; 2],
            intensity: 0.0,
            _pad_to_array: [0.0; 3],
            _pad: [0.0; 248],
        }; 
        
        let u_buf = wgpu::util::DeviceExt::create_buffer_init(self.device.as_ref(), &wgpu::util::BufferInitDescriptor {
            label: Some("Extract Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extract Bind Group"),
            layout: &self.post_process_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: u_buf.as_entire_binding() },
            ],
        });

        let target = self.forced_render_target.lock().unwrap()
            .clone()
            .or_else(|| self.current_view.lock().unwrap().clone())
            .ok_or("No active render target")?;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Extract Brightness Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.as_ref(),
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.extract_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        
        Ok(())
    }

    pub fn blur(&self, input: &wgpu::TextureView, direction: [f32; 2]) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        let uniforms = PostProcessUniforms {
            threshold: 0.0,
            _pad_to_vec2: 0.0,
            direction,
            intensity: 0.0,
            _pad_to_array: [0.0; 3],
            _pad: [0.0; 248],
        };

        let u_buf = wgpu::util::DeviceExt::create_buffer_init(self.device.as_ref(), &wgpu::util::BufferInitDescriptor {
            label: Some("Blur Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur Bind Group"),
            layout: &self.post_process_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: u_buf.as_entire_binding() },
            ],
        });

        let target = self.forced_render_target.lock().unwrap()
            .clone()
            .or_else(|| self.current_view.lock().unwrap().clone())
            .ok_or("No active render target")?;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blur Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.as_ref(),
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.blur_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        
        Ok(())
    }

    pub fn execute_shader_slot(&self, target: &wgpu::TextureView, input: Option<&wgpu::TextureView>, slot: &crate::renderer::layer::ShaderSlot, time: f64) -> Result<(), String> {
        let cache_key = format!("{}:{}", slot.source, slot.entry_point);
        let mut cache = self.pipeline_cache.lock().unwrap();
        
        let pipeline = if let Some(p) = cache.get(&cache_key) {
            p.clone()
        } else {
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Custom Shader Slot"),
                source: wgpu::ShaderSource::Wgsl(slot.source.clone().into()),
            });

            let pipeline = Arc::new(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Custom Shader Pipeline"),
                layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Custom Shader Layout"),
                    bind_group_layouts: &[&self.post_process_layout],
                    push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module: &self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Default Post Process Vertex"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/post_process.wgsl").into()),
                    }),
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: &slot.entry_point,
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
            
            cache.insert(cache_key.clone(), pipeline.clone());
            pipeline
        };

        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        // Pack parameters: [time, (3 pads), p0, p1, ..., p15] -> 20 floats (80 bytes)
        // Pad to 256 bytes for hardware alignment requirements
        let mut uniforms = [0.0f32; 256]; // Keep raw array here for flexible slot params
        uniforms[0] = time as f32;
        
        // Sort keys to be deterministic
        let mut keys: Vec<_> = slot.parameters.keys().collect();
        keys.sort();
        
        for (i, key) in keys.iter().enumerate().take(16) {
            uniforms[i+4] = *slot.parameters.get(*key).unwrap();
        }

        let u_buf = wgpu::util::DeviceExt::create_buffer_init(self.device.as_ref(), &wgpu::util::BufferInitDescriptor {
            label: Some("Shader Slot Uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let input_view = input.unwrap_or(&self.hdr_view);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shader Slot Bind Group"),
            layout: &self.post_process_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: u_buf.as_entire_binding() },
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Shader Slot Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        
        Ok(())
    }

    pub fn composite(&self, bg: &wgpu::TextureView, fg: &wgpu::TextureView, opacity: f32, mode: u32) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        let uniforms = BlendUniforms {
            opacity,
            mode,
            _pad: [0.0; 254],
        };

        let u_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blend Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend Bind Group"),
            layout: &self.blend_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(bg) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(fg) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: u_buf.as_entire_binding() },
            ],
        });

        let target = self.forced_render_target.lock().unwrap()
            .clone()
            .or_else(|| self.current_view.lock().unwrap().clone())
            .ok_or("No active render target for composite")?;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Composition Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.as_ref(),
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.blend_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        
        Ok(())
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
            
            println!("? Screenshot saved to: {}", path);
            drop(data);
            output_buffer.unmap();
        }

        Ok(())
    }
}

impl GraphicsBackend for WgpuBackend {
    fn name(&self) -> &str { "WGPU" }
    fn render(&mut self, dl: &DrawList, width: u32, height: u32, time: f64) {
        // ... (legacy implementaion)
        // I'll keep it for now but call render_composition for a single layer mock later
        let mut comp = crate::renderer::layer::Composition::new();
        let mut layer = crate::renderer::layer::Layer::new("Default");
        layer.source = crate::renderer::layer::LayerSource::DrawList(dl.clone());
        comp.add_layer(layer);
        self.render_composition(&comp, width, height, time);
    }

    fn render_composition(&mut self, composition: &crate::renderer::layer::Composition, width: u32, height: u32, time: f64) {
        use std::collections::HashMap;
        use std::sync::Arc;
        
        // 1. Build Render Graph 
        let mut graph = RenderGraph::<WgpuBackend>::new();
        
        // Final Output Texture (the surface) is external, but we work in HDR internally
        graph.create_texture("AccumColor", width, height, wgpu::TextureFormat::Rgba16Float);

        // Pass 0: Clear AccumColor
        graph.add_pass("Clear", vec![], vec!["AccumColor".to_string()], Box::new(move |ctx, backend| {
            let view = ctx.resources.get("AccumColor").unwrap();
            let _ = backend.clear_target(&**view, wgpu::Color::BLACK);
        }));
        
        let mut bg_tex = "AccumColor".to_string();

        for (i, layer) in composition.layers.iter().enumerate() {
            if !layer.visible { continue; }
            
            let layer_name = format!("Layer_{}", i);
            let layer_tex = format!("{}_Tex", layer_name);
            
            graph.create_texture(&layer_tex, width, height, wgpu::TextureFormat::Rgba16Float);

            // Pass A: Draw Layer to its texture
            let out_tex = layer_tex.clone();
            match &layer.source {
                crate::renderer::layer::LayerSource::DrawList(dl) => {
                    let dl = dl.clone();
                    let out_tex_in = out_tex.clone();
                    graph.add_pass(format!("{}_Draw", layer_name), vec![], vec![out_tex_in.clone()], Box::new(move |ctx, backend| {
                        let view = ctx.resources.get(&out_tex_in).unwrap().clone();
                        backend.set_render_target(Some(view));
                        
                        let orchestrator = crate::renderer::orchestrator::RenderOrchestrator::new();
                        let tasks = orchestrator.plan(&dl);
                        let _ = orchestrator.execute(backend, &tasks, time as f32, width, height);
                        
                        backend.set_render_target(None);
                    }));
                }
                crate::renderer::layer::LayerSource::Shader(slot) => {
                    let slot_in = slot.clone();
                    let bg_tex_shader = "AccumColor".to_string(); // In Phase 3, shader slots can read backdrop
                    let out_tex_in = out_tex.clone();
                    graph.add_pass(format!("{}_Draw_Shader", layer_name), vec![bg_tex_shader.clone()], vec![out_tex_in.clone()], Box::new(move |ctx, backend| {
                        let view = ctx.resources.get(&out_tex_in).unwrap();
                        let bg = ctx.resources.get(&bg_tex_shader).unwrap();
                        let _ = backend.execute_shader_slot(&**view, Some(&**bg), &slot_in, ctx.time);
                    }));
                }
            }

            // Pass B: Composite Layer onto bg_tex
            let fg_tex_in = layer_tex.clone();
            let opacity_prop = layer.opacity.clone();
            let mode = match layer.blend_mode {
                crate::renderer::layer::BlendMode::Alpha => 0,
                crate::renderer::layer::BlendMode::Additive => 1,
                crate::renderer::layer::BlendMode::Multiply => 2,
                crate::renderer::layer::BlendMode::Screen => 0, // Fallback
            };

            let blend_out = format!("{}_AccumOut", layer_name);
            graph.create_texture(blend_out.clone(), width, height, wgpu::TextureFormat::Rgba16Float);
            
            let bg_tex_in = bg_tex.clone();
            let fg_tex_in = out_tex.clone();
            let blend_out_final = blend_out.clone();
            graph.add_pass(format!("{}_Blend", layer_name), vec![bg_tex_in.clone(), fg_tex_in.clone()], vec![blend_out_final.clone()], Box::new(move |ctx, backend| {
                let bg = ctx.resources.get(&bg_tex_in).unwrap();
                let fg = ctx.resources.get(&fg_tex_in).unwrap();
                let target = ctx.resources.get(&blend_out_final).unwrap();
                
                let current_opacity = opacity_prop.evaluate(ctx.time);
                
                backend.set_render_target(Some(target.clone()));
                let _ = backend.composite(&**bg, &**fg, current_opacity, mode);
                backend.set_render_target(None);
            }));
            
            bg_tex = blend_out;
        }

        // 1. Extract Brightness
        graph.create_texture("BrightTex", width, height, wgpu::TextureFormat::Rgba16Float);
        graph.create_texture("BlurX", width, height, wgpu::TextureFormat::Rgba16Float);
        graph.create_texture("BloomFinal", width, height, wgpu::TextureFormat::Rgba16Float);
        
        let final_src_copy = bg_tex.clone();
        graph.add_pass("Bloom_Extract", vec![final_src_copy.clone()], vec!["BrightTex".to_string()], Box::new(move |ctx, backend| {
            let input = ctx.resources.get(&final_src_copy).unwrap();
            let target = ctx.resources.get("BrightTex").unwrap();
            backend.set_render_target(Some(target.clone()));
            let threshold = 0.8 + (ctx.time * 0.5).sin() as f32 * 0.1; 
            let _ = backend.extract_brightness(&**input, threshold);
            backend.set_render_target(None);
        }));

        // 2. Blur X
        graph.add_pass("Bloom_BlurX", vec!["BrightTex".to_string()], vec!["BlurX".to_string()], Box::new(move |ctx, backend| {
            let input = ctx.resources.get("BrightTex").unwrap();
            let target = ctx.resources.get("BlurX").unwrap();
            backend.set_render_target(Some(target.clone()));
            let _ = backend.blur(&**input, [1.0, 0.0]);
            backend.set_render_target(None);
        }));

        // 3. Blur Y
        graph.add_pass("Bloom_BlurY", vec!["BlurX".to_string()], vec!["BloomFinal".to_string()], Box::new(move |ctx, backend| {
            let input = ctx.resources.get("BlurX").unwrap();
            let target = ctx.resources.get("BloomFinal").unwrap();
            backend.set_render_target(Some(target.clone()));
            let _ = backend.blur(&**input, [0.0, 1.0]);
            backend.set_render_target(None);
        }));

        // Final Combine Pass: AccumColor + BloomFinal -> BloomAccum
        graph.create_texture("BloomAccum", width, height, wgpu::TextureFormat::Rgba16Float);
        let final_src = bg_tex.clone();
        graph.add_pass("Bloom_Combine", vec![final_src.clone(), "BloomFinal".to_string()], vec!["BloomAccum".to_string()], Box::new(move |ctx, backend| {
            let scene = ctx.resources.get(&final_src).unwrap();
            let bloom = ctx.resources.get("BloomFinal").unwrap();
            let target = ctx.resources.get("BloomAccum").unwrap();
            backend.set_render_target(Some(target.clone()));
            let _ = backend.composite(&**scene, &**bloom, 1.0, 1); // Additive combine
            backend.set_render_target(None);
        }));

        // 3. Execution (Internal allocation + Sequence)
        let mut context_resources = HashMap::new();
        for (id, desc) in &graph.resource_info {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(id),
                size: wgpu::Extent3d { width: desc.width, height: desc.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: desc.format,
                usage: desc.usage,
                view_formats: &[],
            });
            let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            context_resources.insert(id.clone(), view);
        }

        let _ = self.begin_execute();
        
        let device_arc = self.device.clone();
        let queue_arc = self.queue.clone();
        
        // Create a dummy encoder for Context, but our passes will use the backend's internal one
        let mut dummy_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Dummy") });

        for pass in graph.passes.drain(..) {
            let mut ctx = RenderContext {
                device: &device_arc,
                queue: &queue_arc,
                encoder: &mut dummy_encoder,
                resources: &context_resources,
                time,
            };
            (pass.execute)(&mut ctx, self);
        }

        // 4. Final Resolve & Present
        let final_view_id = "BloomAccum".to_string();
        let final_view = context_resources.get(&final_view_id).expect("Failed to get BloomAccum view").clone();
        
        // Ensure state is clean before final resolve
        *self.forced_render_target.lock().unwrap() = None;

        let _ = self.resolve_from(final_view.as_ref());
        
        // Also resolve to hdr_view for screenshots
        let hdr_view = self.hdr_view.clone();
        let _ = self.color_grade_ext(final_view.as_ref(), &hdr_view, true);
        let _ = self.end_execute();
        let _ = self.present();
    }

    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        let tex = <Self as GpuExecutor>::create_texture(self, &crate::backend::hal::TextureDescriptor {
            label: Some("Font Atlas"),
            width,
            height,
            format: crate::backend::hal::TextureFormat::R8Unorm,
            usage: crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::COPY_DST,
        }).expect("Failed to create font texture");
        
        let bytes_per_row = (width + 255) & !255;
        let mut padded_data = Vec::with_capacity(bytes_per_row as usize * height as usize);
        for y in 0..height {
            let start = (y * width) as usize;
            let end = start + width as usize;
            padded_data.extend_from_slice(&data[start..end]);
            padded_data.resize(padded_data.len() + (bytes_per_row as usize - width as usize), 0);
        }

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        
        self.font_view = Some(<Self as GpuExecutor>::create_texture_view(self, &tex).expect("Failed to create font view"));
    }

    fn present(&mut self) {
        let _ = GpuExecutor::present(self);
    }

    fn capture_screenshot(&mut self, path: &str) {
        *self.screenshot_requested.lock().unwrap() = Some(path.to_string());
    }
}

impl WgpuBackend {
    pub fn clear_target(&self, view: &wgpu::TextureView, color: wgpu::Color) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Clear Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(color), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        Ok(())
    }

    pub fn color_grade(&self, input: &wgpu::TextureView, target: &wgpu::TextureView) -> Result<(), String> {
        self.color_grade_ext(input, target, false)
    }

    pub fn color_grade_ext(&self, input: &wgpu::TextureView, target: &wgpu::TextureView, is_hdr_target: bool) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let uniforms = PostProcessUniforms {
            threshold: 0.0,
            _pad_to_vec2: 0.0,
            direction: [0.0; 2],
            intensity: 1.0, // Default full intensity
            _pad_to_array: [0.0; 3],
            _pad: [0.0; 248],
        };
        
        let u_buf = wgpu::util::DeviceExt::create_buffer_init(self.device.as_ref(), &wgpu::util::BufferInitDescriptor {
            label: Some("Grading Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grading Bind Group"),
            layout: &self.grading_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.lut_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Grading Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if is_hdr_target {
            rpass.set_pipeline(&self.hdr_grading_pipeline);
        } else {
            rpass.set_pipeline(&self.grading_pipeline);
        }
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        
        Ok(())
    }
}
