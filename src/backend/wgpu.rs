//! WGPU backend - Cross-platform GPU rendering via WebGPU API
//!
//! Supports: Windows (DX12/Vulkan), macOS/iOS (Metal), Linux (Vulkan), Web (WebGPU)

use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use std::sync::Arc;

/// Vertex format for WGPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x2,  // pos
        1 => Float32x2,  // uv
        2 => Float32x4,  // color
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// WGPU-based rendering backend
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    vertex_capacity: usize,

    // Font texture
    #[allow(dead_code)]
    font_texture: Option<wgpu::Texture>,
    font_bind_group: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
}

/// Uniform data for shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    projection: [[f32; 4]; 4],
    rect: [f32; 4],  // x, y, w, h
    radii: [f32; 4], // tl, tr, br, bl
    border_color: [f32; 4],
    glow_color: [f32; 4], // Added

    mode: i32,
    border_width: f32,
    elevation: f32,
    is_squircle: i32, // Added

    glow_strength: f32, // Added
    start_angle: f32,   // Added for Arc
    end_angle: f32,     // Added for Arc
    _pad3: f32,
}

impl WgpuBackend {
    /// Create a new WGPU backend
    pub async fn new_async(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find suitable GPU adapter")?;

        println!("🎮 WGPU Adapter: {}", adapter.get_info().name);

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Fantasmagorie WGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fantasmagorie Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgpu_shader.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                // Uniforms
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
                // Texture
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
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
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
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create vertex buffer (initial capacity)
        let vertex_capacity = 1024;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (vertex_capacity * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Font Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        println!("   ✅ WGPU backend initialized");

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            vertex_buffer,
            vertex_capacity,
            font_texture: None,
            font_bind_group: None,
            sampler,
        })
    }

    /// Create orthographic projection matrix
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

    /// Generate quad vertices
    fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> [Vertex; 6] {
        Self::quad_vertices_uv(pos, size, [0.0, 0.0, 1.0, 1.0], color)
    }

    fn quad_vertices_uv(pos: Vec2, size: Vec2, uv: [f32; 4], color: ColorF) -> [Vertex; 6] {
        let (x0, y0) = (pos.x, pos.y);
        let (x1, y1) = (pos.x + size.x, pos.y + size.y);
        let c = [color.r, color.g, color.b, color.a];
        let (u0, v0, u1, v1) = (uv[0], uv[1], uv[2], uv[3]);

        [
            Vertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color: c,
            },
            Vertex {
                pos: [x0, y1],
                uv: [u0, v1],
                color: c,
            },
            Vertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color: c,
            },
            Vertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color: c,
            },
            Vertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color: c,
            },
            Vertex {
                pos: [x1, y0],
                uv: [u1, v0],
                color: c,
            },
        ]
    }

    /// Update font texture from FontManager
    fn update_font_texture(&mut self) {
        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            if fm.texture_dirty {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Font Texture"),
                    size: wgpu::Extent3d {
                        width: fm.atlas.width as u32,
                        height: fm.atlas.height as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &fm.atlas.texture_data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(fm.atlas.width as u32),
                        rows_per_image: Some(fm.atlas.height as u32),
                    },
                    wgpu::Extent3d {
                        width: fm.atlas.width as u32,
                        height: fm.atlas.height as u32,
                        depth_or_array_layers: 1,
                    },
                );

                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Font Bind Group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                self.font_texture = Some(texture);
                self.font_bind_group = Some(bind_group);
                fm.texture_dirty = false;
            }
        });
    }

    /// Render to a surface
    pub fn render_to_surface(
        &mut self,
        dl: &DrawList,
        surface: &wgpu::Surface,
        width: u32,
        height: u32,
    ) {
        use wgpu::util::DeviceExt;
        self.update_font_texture();

        let output = match surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Prepare Resources
        struct PreparedDraw {
            u_buf: wgpu::Buffer,
            v_buf: wgpu::Buffer,
            bind_group: wgpu::BindGroup,
            vertex_count: u32,
        }
        let mut prepared = Vec::new();

        // Create font view once if texture exists
        let font_view = self
            .font_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        // Helper to prepare a draw
        let mut prepare = |uniforms: Uniforms, vertices: &[Vertex], label: &str| {
            let u_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} Uniforms", label)),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let v_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} Vertices", label)),
                    contents: bytemuck::cast_slice(vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

            // Use font view if available, otherwise we might need a dummy?
            // For now, if no font, we can't really draw text but we can draw other things if we use a dummy view.
            // But our bind group layout EXPECTS a texture.
            if font_view.is_none() {
                return;
            }

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Bind Group", label)),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(font_view.as_ref().unwrap()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            prepared.push(PreparedDraw {
                u_buf,
                v_buf,
                bind_group,
                vertex_count: vertices.len() as u32,
            });
        };

        // 1. Aurora
        let aurora_uniforms = Uniforms {
            projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
            rect: [0.0, 0.0, width as f32, height as f32],
            radii: [0.0; 4],
            border_color: [0.0; 4],
            glow_color: [0.0; 4],
            mode: 5,
            border_width: 0.0,
            elevation: 0.0,
            is_squircle: 0,
            glow_strength: 0.0,
            start_angle: 0.0,
            end_angle: 0.0,
            _pad3: 0.0,
        };
        let aurora_verts = Self::quad_vertices(
            Vec2::ZERO,
            Vec2::new(width as f32, height as f32),
            ColorF::white(),
        );
        prepare(aurora_uniforms, &aurora_verts, "Aurora");

        // 2. Commands
        for cmd in dl.commands() {
            match cmd {
                DrawCommand::RoundedRect {
                    pos,
                    size,
                    radii,
                    color,
                    elevation,
                    is_squircle,
                    border_width,
                    border_color,
                    wobble: _,
                    glow_strength,
                    glow_color,
                } => {
                    let uniforms = Uniforms {
                        projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                        rect: [pos.x, pos.y, size.x, size.y],
                        radii: *radii,
                        border_color: [
                            border_color.r,
                            border_color.g,
                            border_color.b,
                            border_color.a,
                        ],
                        glow_color: [glow_color.r, glow_color.g, glow_color.b, glow_color.a],
                        mode: 2,
                        border_width: *border_width,
                        elevation: *elevation,
                        is_squircle: if *is_squircle { 1 } else { 0 },
                        glow_strength: *glow_strength,
                        start_angle: 0.0,
                        end_angle: 0.0,
                        _pad3: 0.0,
                    };
                    let verts = Self::quad_vertices(*pos, *size, *color);
                    prepare(uniforms, &verts, "RoundedRect");
                }
                DrawCommand::Text {
                    pos,
                    size,
                    uv,
                    color,
                } => {
                    let uniforms = Uniforms {
                        projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                        rect: [pos.x, pos.y, size.x, size.y],
                        radii: [0.0; 4],
                        border_color: [0.0; 4],
                        glow_color: [0.0; 4],
                        mode: 1,
                        border_width: 0.0,
                        elevation: 0.0,
                        is_squircle: 0,
                        glow_strength: 0.0,
                        start_angle: 0.0,
                        end_angle: 0.0,
                        _pad3: 0.0,
                    };
                    let verts = Self::quad_vertices_uv(*pos, *size, *uv, *color);
                    prepare(uniforms, &verts, "Text");
                }
                DrawCommand::Arc {
                    center,
                    radius,
                    start_angle,
                    end_angle,
                    thickness,
                    color,
                } => {
                    let s = *radius * 2.0 + *thickness * 2.0;
                    let pos = Vec2::new(center.x - s * 0.5, center.y - s * 0.5);
                    let uniforms = Uniforms {
                        projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                        rect: [pos.x, pos.y, s, s],
                        radii: [*radius, *thickness, 0.0, 0.0],
                        border_color: [0.0; 4],
                        glow_color: [color.r, color.g, color.b, color.a],
                        mode: 6,
                        border_width: 0.0,
                        elevation: 0.0,
                        is_squircle: 0,
                        glow_strength: 0.0,
                        start_angle: *start_angle,
                        end_angle: *end_angle,
                        _pad3: 0.0,
                    };
                    let verts = Self::quad_vertices(pos, Vec2::new(s, s), *color);
                    prepare(uniforms, &verts, "Arc");
                }
                DrawCommand::Plot {
                    points,
                    color,
                    fill_color,
                    thickness,
                    baseline,
                } => {
                    if points.len() < 2 {
                        continue;
                    }
                    let uniforms = Uniforms {
                        projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                        rect: [0.0, 0.0, width as f32, height as f32],
                        radii: [0.0; 4],
                        border_color: [fill_color.r, fill_color.g, fill_color.b, fill_color.a],
                        glow_color: [color.r, color.g, color.b, color.a],
                        mode: 7,
                        border_width: *thickness,
                        elevation: 0.0,
                        is_squircle: 0,
                        glow_strength: 0.0,
                        start_angle: 0.0,
                        end_angle: 0.0,
                        _pad3: 0.0,
                    };
                    let mut verts = Vec::with_capacity(points.len() * 12);
                    for i in 0..points.len() - 1 {
                        let p0 = points[i];
                        let p1 = points[i + 1];
                        let b0 = Vec2::new(p0.x, *baseline);
                        let b1 = Vec2::new(p1.x, *baseline);
                        let c = [fill_color.r, fill_color.g, fill_color.b, fill_color.a];
                        verts.push(Vertex {
                            pos: [p0.x, p0.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [p1.x, p1.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [b1.x, b1.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [p0.x, p0.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [b1.x, b1.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [b0.x, b0.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                    }
                    let half_t = *thickness * 0.5;
                    for i in 0..points.len() - 1 {
                        let p0 = points[i];
                        let p1 = points[i + 1];
                        let dir = (p1 - p0).normalized();
                        let normal = Vec2::new(-dir.y, dir.x) * half_t;
                        let v0 = p0 + normal;
                        let v1 = p0 - normal;
                        let v2 = p1 + normal;
                        let v3 = p1 - normal;
                        let c = [color.r, color.g, color.b, color.a];
                        verts.push(Vertex {
                            pos: [v0.x, v0.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [v1.x, v1.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [v2.x, v2.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [v1.x, v1.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [v3.x, v3.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        verts.push(Vertex {
                            pos: [v2.x, v2.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                    }
                    prepare(uniforms, &verts, "Plot");
                }
                _ => {}
            }
        }

        // 3. Execution Pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            for p in &prepared {
                render_pass.set_bind_group(0, &p.bind_group, &[]);
                render_pass.set_vertex_buffer(0, p.v_buf.slice(..));
                render_pass.draw(0..p.vertex_count, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

impl super::Backend for WgpuBackend {
    fn name(&self) -> &str {
        "WGPU"
    }
    fn render(&mut self, _dl: &DrawList, _width: u32, _height: u32) {
        eprintln!("WGPU: Use render_to_surface() for proper rendering");
    }
}
