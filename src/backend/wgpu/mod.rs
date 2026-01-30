use std::sync::Arc;
use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use crate::backend::GraphicsBackend;
use crate::backend::hal::{GpuResourceProvider, GpuPipelineProvider};

pub mod resource_provider;
pub mod pipeline_provider;
pub mod compute;

use resource_provider::WgpuResourceProvider;
use pipeline_provider::WgpuPipelineProvider;
use compute::ComputePipelines;

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

/// Uniform data for shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub projection: [[f32; 4]; 4],
    pub rect: [f32; 4],
    pub radii: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub offset: [f32; 2],
    pub scale: f32,
    pub border_width: f32,
    pub elevation: f32,
    pub glow_strength: f32,
    pub lut_intensity: f32,
    pub mode: u32,
    pub is_squircle: u32,
    pub time: f32,
    pub _pad: f32,
    pub _pad2: f32,
}

pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    
    pub resources: WgpuResourceProvider,
    pub pipelines: WgpuPipelineProvider,
    pub compute: ComputePipelines,
    
    pub main_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    
    pub uniform_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_capacity: usize,

    pub font_texture: Option<wgpu::Texture>,
    pub font_bind_group: Option<wgpu::BindGroup>,
    pub sampler: wgpu::Sampler,
    
    // Cinematic Textures
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub sdf_texture: wgpu::Texture,
    pub sdf_view: wgpu::TextureView,
    pub jfa_textures: [wgpu::Texture; 2],
    pub jfa_views: [wgpu::TextureView; 2],
    pub backdrop_texture: wgpu::Texture,
    pub backdrop_view: wgpu::TextureView,
    
    // Performance Control
    pub exposure: f32,
    pub gamma: f32,
    pub fog_density: f32,
    
    pub audio_params_buffer: wgpu::Buffer,
    pub dummy_history_view: wgpu::TextureView,
    
    pub start_time: std::time::Instant,
    pub screenshot_requested: Option<String>,
}

impl std::fmt::Debug for WgpuBackend {
     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend")
            .field("device", &self.device)
            .field("queue", &self.queue)
            .finish_non_exhaustive()
    }
}

impl WgpuBackend {
    pub async fn new_async(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Fantasmagorie WGPU Device"),
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 128,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| **f == wgpu::TextureFormat::Rgba8Unorm)
            .copied()
            .or_else(|| surface_caps.formats.iter().find(|f| f.is_srgb()).copied())
            .unwrap_or(surface_caps.formats[0]);

        let mut surface_config = surface.get_default_config(&adapter, width, height).unwrap();
        surface_config.format = surface_format;
        surface_config.usage |= wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC;
        surface.configure(&device, &surface_config);
        let surface_format = surface_config.format;

        let resources = WgpuResourceProvider::new(device.clone(), queue.clone());
        let pipelines = WgpuPipelineProvider::new(device.clone(), surface_format);
        let compute = ComputePipelines::new(&device);

        // Bind group layout (same as before)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main Bind Group Layout"),
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fantasmagorie Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../wgpu_shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let main_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
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
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_capacity = 1024;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (vertex_capacity * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Font Sampler"),
            ..Default::default()
        });

        // Initialize with dummy font (same as before)
        let dummy_data = [255u8];
        let font_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Font"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &font_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &dummy_data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(1), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let font_view = font_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- HDR Texture (Linear, High Precision) ---
        let hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Render Target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let hdr_view = hdr_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- SDF Texture (RG32Float) ---
        let sdf_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Final SDF Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sdf_view = sdf_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- JFA Ping-Pong Textures ---
        let jfa_textures = [
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("JFA Ping"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }),
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("JFA Pong"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }),
        ];
        let jfa_views = [
            jfa_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            jfa_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // --- Backdrop Texture (for Blur) ---
        let backdrop_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Backdrop Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let backdrop_view = backdrop_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let font_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Font Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&font_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        let audio_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Audio Params Buffer"),
            size: 16, // AudioParams is 16 bytes (3 floats + pad)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_history = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy History"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_history_view = dummy_history.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            resources,
            pipelines,
            compute,
            main_pipeline,
            bind_group_layout,
            uniform_buffer,
            vertex_buffer,
            vertex_capacity,
            font_texture: Some(font_texture),
            font_bind_group: Some(font_bind_group),
            sampler,
            hdr_texture,
            hdr_view,
            sdf_texture,
            sdf_view,
            jfa_textures,
            jfa_views,
            backdrop_texture,
            backdrop_view,
            audio_params_buffer,
            dummy_history_view,
            exposure: 1.0,
            gamma: 2.2,
            fog_density: 0.0,
            start_time: std::time::Instant::now(),
            screenshot_requested: None,
        })
    }

    fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);
        [
            [2.0 / (right - left), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (top - bottom), 0.0, 0.0],
            [0.0, 0.0, -2.0 / (far - near), 0.0],
            [tx, ty, tz, 1.0],
        ]
    }

    fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> Vec<Vertex> {
        let c = [color.r, color.g, color.b, color.a];
        vec![
            Vertex { pos: [pos.x, pos.y], uv: [0.0, 0.0], color: c },
            Vertex { pos: [pos.x + size.x, pos.y], uv: [1.0, 0.0], color: c },
            Vertex { pos: [pos.x + size.x, pos.y + size.y], uv: [1.0, 1.0], color: c },
            Vertex { pos: [pos.x, pos.y], uv: [0.0, 0.0], color: c },
            Vertex { pos: [pos.x + size.x, pos.y + size.y], uv: [1.0, 1.0], color: c },
            Vertex { pos: [pos.x, pos.y + size.y], uv: [0.0, 1.0], color: c },
        ]
    }

    fn quad_vertices_uv(pos: Vec2, size: Vec2, uv: [f32; 4], color: ColorF) -> Vec<Vertex> {
        let c = [color.r, color.g, color.b, color.a];
        vec![
            Vertex { pos: [pos.x, pos.y], uv: [uv[0], uv[1]], color: c },
            Vertex { pos: [pos.x + size.x, pos.y], uv: [uv[2], uv[1]], color: c },
            Vertex { pos: [pos.x + size.x, pos.y + size.y], uv: [uv[2], uv[3]], color: c },
            Vertex { pos: [pos.x, pos.y], uv: [uv[0], uv[1]], color: c },
            Vertex { pos: [pos.x + size.x, pos.y + size.y], uv: [uv[2], uv[3]], color: c },
            Vertex { pos: [pos.x, pos.y + size.y], uv: [uv[0], uv[3]], color: c },
        ]
    }
}

impl GraphicsBackend for WgpuBackend {
    fn name(&self) -> &str { "WGPU (Unified)" }

    fn submit(&mut self, _packets: &[crate::renderer::packet::DrawPacket]) {
        // In the unified backend, submit is the primary way to draw.
        // For now, we will use it to prepare commands that will be executed in render.
        // Or actually, submit should probably do its own queue submission.
        println!("⚠️ WGPU submit() called with {} packets - mapping to high-perf path", _packets.len());
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        if width != self.surface_config.width || height != self.surface_config.height {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        // Resource Tracking
        struct PreparedDraw {
            bind_group: wgpu::BindGroup,
            v_buf: wgpu::Buffer,
            #[allow(dead_code)]
            u_buf: wgpu::Buffer,
            vertex_count: u32,
        }
        let mut prepared = Vec::new();

        for cmd in dl.commands() {
            match cmd {
                DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, .. } => {
                    let uniforms = Uniforms {
                        projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                        rect: [pos.x, pos.y, size.x, size.y],
                        radii: *radii,
                        border_color: [border_color.r, border_color.g, border_color.b, border_color.a],
                        glow_color: [glow_color.r, glow_color.g, glow_color.b, glow_color.a],
                        offset: [0.0; 2],
                        scale: 1.0,
                        border_width: *border_width,
                        elevation: *elevation,
                        glow_strength: *glow_strength,
                        lut_intensity: 0.0,
                        mode: 2, // Shape
                        is_squircle: if *is_squircle { 1 } else { 0 },
                        time: self.start_time.elapsed().as_secs_f32(),
                        _pad: 0.0,
                        _pad2: 0.0,
                    };
                    let verts = Self::quad_vertices(*pos, *size, *color);
                    
                    use wgpu::util::DeviceExt;
                    let u_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Temp Uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Temp Vertices"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    
                    let font_view = self.font_texture.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor::default());
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Temp Bind Group"),
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&font_view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        ],
                    });

                    prepared.push(PreparedDraw { bind_group, v_buf, u_buf, vertex_count: verts.len() as u32 });
                }
                DrawCommand::Text { pos, size, uv, color } => {
                     let uniforms = Uniforms {
                        projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                        rect: [pos.x, pos.y, size.x, size.y],
                        radii: [0.0; 4],
                        border_color: [color.r, color.g, color.b, color.a],
                        glow_color: [0.0; 4],
                        offset: [0.0; 2],
                        scale: 1.0,
                        border_width: 0.0,
                        elevation: 0.0,
                        glow_strength: 0.0,
                        lut_intensity: 0.0,
                        mode: 1, // Text
                        is_squircle: 0,
                        time: 0.0,
                        _pad: 0.0,
                        _pad2: 0.0,
                    };
                    let verts = Self::quad_vertices_uv(*pos, *size, *uv, *color);
                    
                    use wgpu::util::DeviceExt;
                    let u_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Text Uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Text Vertices"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    
                    let font_view = self.font_texture.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor::default());
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Text Bind Group"),
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&font_view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        ],
                    });
                    prepared.push(PreparedDraw { bind_group, v_buf, u_buf, vertex_count: verts.len() as u32 });
                }
                DrawCommand::Aurora { pos, size } => {
                     let uniforms = Uniforms {
                        projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                        rect: [pos.x, pos.y, size.x, size.y],
                        radii: [0.0; 4],
                        border_color: [0.0; 4],
                        glow_color: [0.0; 4],
                        offset: [0.0; 2],
                        scale: 1.0,
                        border_width: 0.0,
                        elevation: 0.0,
                        glow_strength: 0.0,
                        lut_intensity: 0.0,
                        mode: 9, // Aurora
                        is_squircle: 0,
                        time: self.start_time.elapsed().as_secs_f32(),
                        _pad: 0.0,
                        _pad2: 0.0,
                    };
                    let verts = Self::quad_vertices(*pos, *size, ColorF::white());
                    
                    use wgpu::util::DeviceExt;
                    let u_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Aurora Uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Aurora Vertices"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    
                    let font_view = self.font_texture.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor::default());
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Aurora Bind Group"),
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&font_view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        ],
                    });
                    prepared.push(PreparedDraw { bind_group, v_buf, u_buf, vertex_count: verts.len() as u32 });
                }
                _ => {}
            }
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.main_pipeline);
            for p in &prepared {
                render_pass.set_bind_group(0, &p.bind_group, &[]);
                render_pass.set_vertex_buffer(0, p.v_buf.slice(..));
                render_pass.draw(0..p.vertex_count, 0..1);
            }
        }

        // --- K5: Jump Flooding Algorithm (SDF Generation) ---
        // Basic implementation: 8 steps for 1024x1024
        let steps = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1];
        let mut ping_pong = 0;
        
        for &step in &steps {
            if step > width.max(height) as i32 { continue; }
            
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("JFA Pass Bind Group"),
                layout: &self.compute.k5_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.jfa_views[ping_pong]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.jfa_views[ping_pong]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.jfa_views[1 - ping_pong]) },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("JFA Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute.k5_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Push Constants: width, height, jfa_step, ping_pong_idx, intensity, decay, radius, _pad
            let jfa_pc = [width, height, step as u32, ping_pong as u32, 100u32, 100u32, 100u32, 0u32]; 
            // Wait, JFAUniforms uses broad types, let's just cast carefully
            let mut pc_data = [0u32; 8];
            pc_data[0] = width;
            pc_data[1] = height;
            pc_data[2] = step as u32;
            pc_data[3] = ping_pong as u32;
            // intensity, decay, radius are f32 in JFAUniforms but let's use bits
            pc_data[4] = (1.0f32).to_bits(); // intensity
            pc_data[5] = (0.95f32).to_bits(); // decay
            pc_data[6] = (50.0f32).to_bits(); // radius
            
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&pc_data));
            compute_pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
            
            ping_pong = 1 - ping_pong;
        }

        // Copy final JFA result to sdf_texture
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.jfa_textures[ping_pong],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.sdf_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        // --- K4: Cinematic Resolve Compute Pass ---
        {
            let k4_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("K4 Bind Group"),
                layout: &self.compute.k4_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.sdf_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.dummy_history_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 6, resource: self.audio_params_buffer.as_entire_binding() },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("K4 Resolver Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute.k4_pipeline);
            compute_pass.set_bind_group(0, &k4_bind_group, &[]);
            
            let k4_pc = [self.exposure, self.gamma, self.fog_density, 0.0];
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&k4_pc));
            
            compute_pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Handle deferred screenshot
        if let Some(path) = self.screenshot_requested.take() {
            // 1. Create buffer
            let width = self.surface_config.width;
            let height = self.surface_config.height;
            let bytes_per_pixel = 4;
            let bytes_per_row = if (width * bytes_per_pixel) % 256 != 0 {
                ((width * bytes_per_pixel) / 256 + 1) * 256
            } else {
                width * bytes_per_pixel
            };
            let size = (bytes_per_row * height) as u64;

            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screenshot Buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // 2. Refresh current texture for copy
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Screenshot Copy Encoder") });
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                }
            );
            self.queue.submit(std::iter::once(encoder.finish()));

            // 3. Map and save
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);

            if let Ok(Ok(())) = rx.recv() {
                let data = buffer_slice.get_mapped_range();
                let mut image_data = Vec::with_capacity((width * height * 4) as usize);
                for row in 0..height {
                    let start = (row * bytes_per_row) as usize;
                    let end = start + (width * 4) as usize;
                    image_data.extend_from_slice(&data[start..end]);
                }

                // Convert BGRA to RGBA if needed
                let format = self.surface_config.format;
                if format == wgpu::TextureFormat::Bgra8Unorm || format == wgpu::TextureFormat::Bgra8UnormSrgb {
                    for chunk in image_data.chunks_exact_mut(4) {
                        let tmp = chunk[0];
                        chunk[0] = chunk[2];
                        chunk[2] = tmp;
                    }
                }

                if let Err(e) = image::save_buffer(
                    &path,
                    &image_data,
                    width,
                    height,
                    image::ColorType::Rgba8,
                ) {
                    eprintln!("Failed to save deferred screenshot: {}", e);
                } else {
                    println!("Deferred screenshot saved to {}", path);
                }
            }
        }

        output.present();
    }

    fn update_audio_data(&mut self, spectrum: &[f32]) {
        // AudioParams is 16 bytes: bass, mid, high, _pad (all f32)
        // We calculate these from the spectrum data
        if spectrum.len() < 3 { return; }
        
        let bass = spectrum[0..spectrum.len()/3].iter().fold(0.0, |a, &b| a + b) / (spectrum.len() as f32 / 3.0);
        let mid = spectrum[spectrum.len()/3..2*spectrum.len()/3].iter().fold(0.0, |a, &b| a + b) / (spectrum.len() as f32 / 3.0);
        let high = spectrum[2*spectrum.len()/3..].iter().fold(0.0, |a, &b| a + b) / (spectrum.len() as f32 / 3.0);
        
        let params = [bass, mid, high, 0.0f32];
        self.queue.write_buffer(&self.audio_params_buffer, 0, bytemuck::cast_slice(&params));
    }

    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        let mut recreate = false;
        if let Some(tex) = &self.font_texture {
            let size = tex.size();
            if size.width != width || size.height != height {
                recreate = true;
            }
        } else {
            recreate = true;
        }

        if recreate {
            let font_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Font Atlas"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.font_texture = Some(font_texture);
            
            // Note: font_bind_group is recreated per-draw in the current render implementation,
            // but we update the cached one anyway if we ever switch to a more efficient path.
            let font_view = self.font_texture.as_ref().unwrap().create_view(&wgpu::TextureViewDescriptor::default());
            let font_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Font Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&font_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            self.font_bind_group = Some(font_bind_group);
        }

        if let Some(tex) = &self.font_texture {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
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
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}
