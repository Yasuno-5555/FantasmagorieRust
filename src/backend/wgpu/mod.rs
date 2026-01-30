use std::sync::{Arc, Mutex};
use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use crate::backend::GraphicsBackend;
use image; // Ensure image crate is available
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
pub struct UniformsWgpu {
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
    pub font_view: Option<wgpu::TextureView>,
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
    pub backdrop_view: Arc<wgpu::TextureView>,
    
    // Performance Control
    pub exposure: f32,
    pub gamma: f32,
    pub fog_density: f32,
    
    pub audio_params_buffer: wgpu::Buffer,
    pub dummy_history_texture: wgpu::Texture,
    pub dummy_history_view: wgpu::TextureView,
    
    pub start_time: std::time::Instant,
    pub screenshot_requested: Mutex<Option<String>>,
    
    pub current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
    pub current_texture: Mutex<Option<wgpu::SurfaceTexture>>,
    pub current_view: Mutex<Option<wgpu::TextureView>>,
}

impl crate::backend::hal::GpuResourceProvider for WgpuBackend {
    type Buffer = wgpu::Buffer;
    type Texture = wgpu::Texture;
    type TextureView = wgpu::TextureView;
    type Sampler = wgpu::Sampler;

    fn create_buffer(&self, size: u64, usage: crate::backend::hal::BufferUsage, label: &str) -> Result<Self::Buffer, String> {
        self.resources.create_buffer(size, usage, label)
    }

    fn create_texture(&self, desc: &crate::backend::hal::TextureDescriptor) -> Result<Self::Texture, String> {
        self.resources.create_texture(desc)
    }

    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String> {
        self.resources.create_texture_view(texture)
    }

    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String> {
        self.resources.create_sampler(label)
    }

    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]) {
        self.resources.write_buffer(buffer, offset, data)
    }

    fn write_texture(&self, texture: &Self::Texture, data: &[u8], width: u32, height: u32) {
        self.resources.write_texture(texture, data, width, height)
    }
}

impl crate::backend::hal::GpuPipelineProvider for WgpuBackend {
    type RenderPipeline = wgpu::RenderPipeline;
    type ComputePipeline = wgpu::ComputePipeline;
    type BindGroupLayout = wgpu::BindGroupLayout;
    type BindGroup = wgpu::BindGroup;

    fn create_render_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
        layout: Option<&Self::BindGroupLayout>,
    ) -> Result<Self::RenderPipeline, String> {
        self.pipelines.create_render_pipeline(label, wgsl_source, layout)
    }

    fn create_compute_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
        layout: Option<&Self::BindGroupLayout>,
    ) -> Result<Self::ComputePipeline, String> {
        self.pipelines.create_compute_pipeline(label, wgsl_source, layout)
    }
}

impl crate::backend::hal::GpuExecutor for WgpuBackend {
    fn begin_execute(&self) -> Result<(), String> {
        let output = self.surface.get_current_texture().map_err(|e| e.to_string())?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        *self.current_texture.lock().unwrap() = Some(output);
        *self.current_view.lock().unwrap() = Some(view);

        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fantasmagorie Frame Encoder") });
        *self.current_encoder.lock().unwrap() = Some(encoder);
        Ok(())
    }

    fn end_execute(&self) -> Result<(), String> {
        let encoder = self.current_encoder.lock().unwrap().take().ok_or("No active encoder")?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn draw(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        vertex_count: u32,
        _uniform_data: &[u8],
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        // We use a temporary pass for each draw for now.
        // In a real implementation, we'd batch these.
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // We must Load to preserve previous draws in the batch
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(pipeline);
        if let Some(bg) = bind_group {
            rpass.set_bind_group(0, bg, &[]);
        }
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..vertex_count, 0..1);
        Ok(())
    }

    fn dispatch(
        &self,
        pipeline: &Self::ComputePipeline,
        _bind_group: Option<&Self::BindGroupLayout>,
        groups: [u32; 3],
        _push_constants: &[u8],
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
            label: Some("Compute Task"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }

    fn copy_texture(
        &self,
        src: &Self::Texture,
        dst: &Self::Texture,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        encoder.copy_texture_to_texture(
            src.as_image_copy(),
            dst.as_image_copy(),
            src.size(),
        );
        Ok(())
    }

    fn generate_mipmaps(&self, _texture: &Self::Texture) -> Result<(), String> {
        Ok(())
    }

    fn create_bind_group(
        &self,
        layout: &Self::BindGroupLayout,
        buffers: &[&Self::Buffer],
        textures: &[&Self::TextureView],
        samplers: &[&Self::Sampler],
    ) -> Result<Self::BindGroup, String> {
        let mut entries = Vec::new();
        
        // Binding 0: Uniform Buffer (First buffer)
        if let Some(buf) = buffers.first() {
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            });
        }

        // Binding 1: Main Texture (First texture, usually Font or Atlas)
        if let Some(tex) = textures.first() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(tex),
            });
        }

        // Binding 2: Main Sampler (First sampler)
        if let Some(samp) = samplers.first() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(samp),
            });
        }

        // Binding 3: Secondary Texture (Second texture or Fallback Backdrop)
        let secondary_tex = textures.get(1).map(|&t| t).unwrap_or(&self.backdrop_view);
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(secondary_tex),
        });

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamic Bind Group"),
            layout,
            entries: &entries,
        }))
    }

    fn get_font_view(&self) -> &Self::TextureView {
        self.font_view.as_ref().expect("Font view not initialized")
    }

    fn get_backdrop_view(&self) -> &Self::TextureView {
        &self.backdrop_view
    }

    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout {
        &self.bind_group_layout
    }

    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline {
        &self.main_pipeline
    }

    fn get_default_sampler(&self) -> &Self::Sampler {
        &self.sampler
    }

    fn resolve(&mut self) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
        
        let view_guard = self.current_view.lock().unwrap();
        let view = view_guard.as_ref().ok_or("No active swapchain view")?;

        // K4: Cinematic Resolve Compute Pass
        {
            let k4_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("K4 Bind Group"),
                layout: &self.compute.k4_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hdr_view) },
                    // Placeholder: SDF view not implemented here yet (requires JFA). using hdr as dummy/fallback?
                    // Actually we have self.sdf_view. But is it populated?
                    // JFA was removed from Orchestrator logic.
                    // For now, assume SDF is 0 (use hdr_view or sdf_view empty).
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.sdf_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.dummy_history_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 6, resource: self.audio_params_buffer.as_entire_binding() },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("K4 Resolver Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute.k4_pipeline);
            compute_pass.set_bind_group(0, &k4_bind_group, &[]);
            
            // Push constants? K4 uses them (exposure, gamma...).
            let k4_pc = [self.exposure, self.gamma, self.fog_density, 0.0];
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&k4_pc));
            
            let width = self.surface_config.width;
            let height = self.surface_config.height;
            compute_pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }
        Ok(())
    }

    fn present(&self) -> Result<(), String> {
        // Screenshot Capture Logic
        if let Some(path) = self.screenshot_requested.lock().unwrap().take() {
            let width = self.surface_config.width;
            let height = self.surface_config.height;
            let bytes_per_pixel = 4; // Rgba8 or Bgra8
            let bytes_per_row = ((width * bytes_per_pixel) + 255) & !255;
            let buffer_size = (bytes_per_row * height) as u64;

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Screenshot Encoder") });
            
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screenshot Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy from current swapchain texture
            let tex_guard = self.current_texture.lock().unwrap();
            if let Some(texture) = tex_guard.as_ref() {
                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(bytes_per_row),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                );
            }
            self.queue.submit(Some(encoder.finish()));

            let slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                tx.send(res).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = slice.get_mapped_range();
            let mut image_data = Vec::with_capacity((width * height * 4) as usize);
            
            // Unpad rows and Swap BGR to RGB if needed (assuming Bgra8Unorm surface)
            for y in 0..height {
                let row_start = (y * bytes_per_row) as usize;
                let row_end = row_start + (width * 4) as usize;
                let row = &data[row_start..row_end];
                // Check surface format
                if self.surface_config.format == wgpu::TextureFormat::Bgra8Unorm || self.surface_config.format == wgpu::TextureFormat::Bgra8UnormSrgb {
                    for chunk in row.chunks(4) {
                        image_data.push(chunk[2]); // R
                        image_data.push(chunk[1]); // G
                        image_data.push(chunk[0]); // B
                        image_data.push(chunk[3]); // A
                    }
                } else {
                    image_data.extend_from_slice(row);
                }
            }
            
            // Save using image crate
            image::save_buffer(&path, &image_data, width, height, image::ColorType::Rgba8).map_err(|e| e.to_string())?;
            println!("Screenshot saved to {}", path);
        }

        let mut tex_guard = self.current_texture.lock().unwrap();
        let texture = tex_guard.take().ok_or("No active texture to present")?;
        texture.present();
        Ok(())
    }
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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
            size: std::mem::size_of::<UniformsWgpu>() as u64,
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

        // --- Sampler ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Main Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let backdrop_view = Arc::new(backdrop_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let font_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Font Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&font_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&font_view) }, // Dummy for Binding 3
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
            device: device,
            queue: queue,
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
            font_view: Some(font_view), // Storing the view
            font_bind_group: Some(font_bind_group),
            sampler,
            hdr_texture,
            hdr_view,
            sdf_texture,
            sdf_view,
            jfa_textures,
            jfa_views,
            backdrop_texture,
            backdrop_view: backdrop_view,
            exposure: 1.0,
            gamma: 2.2,
            fog_density: 0.05,
            audio_params_buffer,
            dummy_history_texture: dummy_history,
            dummy_history_view,
            start_time: std::time::Instant::now(),
            screenshot_requested: Mutex::new(None),
            current_encoder: Mutex::new(None),
            current_texture: Mutex::new(None),
            current_view: Mutex::new(None),
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

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        if width != self.surface_config.width || height != self.surface_config.height {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
        }

        let orchestrator = crate::renderer::orchestrator::RenderOrchestrator::new();
        let tasks = orchestrator.plan(dl);
        let time = self.start_time.elapsed().as_secs_f32();

        if let Err(e) = orchestrator.execute(self, &tasks, time, width, height) {
            eprintln!("WGPU Render Error: {}", e);
        }
    }

    fn update_audio_data(&mut self, spectrum: &[f32]) {
        if spectrum.len() < 3 { return; }
        let bass = spectrum[0..spectrum.len()/3].iter().fold(0.0, |a, &b| a + b) / (spectrum.len() as f32 / 3.0);
        let mid = spectrum[spectrum.len()/3..2*spectrum.len()/3].iter().fold(0.0, |a, &b| a + b) / (spectrum.len() as f32 / 3.0);
        let high = spectrum[2*spectrum.len()/3..].iter().fold(0.0, |a, &b| a + b) / (spectrum.len() as f32 / 3.0);
        let params = [bass, mid, high, 0.0f32];
        self.queue.write_buffer(&self.audio_params_buffer, 0, bytemuck::cast_slice(&params));
    }

    fn capture_screenshot(&mut self, path: &str) {
        *self.screenshot_requested.lock().unwrap() = Some(path.to_string());
    }

    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        let needs_resize = if let Some(tex) = &self.font_texture {
            tex.width() != width || tex.height() != height
        } else {
            true
        };

        if needs_resize {
            let new_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Font Atlas"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let new_view = new_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.font_texture = Some(new_texture);
            self.font_view = Some(new_view);
            
            // Recreate bind group to avoid stale texture references
            let entries = [
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(self.font_view.as_ref().unwrap()) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.backdrop_view) },
            ];
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                 label: Some("Dynamic Bind Group"),
                 layout: &self.bind_group_layout,
                 entries: &entries,
            });
            self.font_bind_group = Some(bg);
        }

        if let Some(tex) = &self.font_texture {
             self.queue.write_texture(
                wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(width), rows_per_image: Some(height) },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
        }
    }
}
