use crate::backend::shaders::types::DrawUniforms;
use cocoa::base::id;
use objc::{msg_send, sel, sel_impl, rc::autoreleasepool};
use block::ConcreteBlock;
use metal::*;
use metal::foreign_types::ForeignType;
use std::sync::{Arc, Mutex};
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor};
use crate::renderer::graph::TransientPool;
use crate::core::{Vec2, ColorF};
use crate::draw::DrawList;
use crate::backend::GraphicsBackend;

pub mod resource_provider;
pub mod builtin;
mod metalfx;
pub mod pipeline_provider;

use pipeline_provider::{MetalPipelineProvider, MetalBindGroup, MetalBindGroupLayout};
use resource_provider::MetalResourceProvider;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Clone, Copy)]
pub struct MetalIdWrapper<T>(pub T);
unsafe impl<T> Send for MetalIdWrapper<T> {}
unsafe impl<T> Sync for MetalIdWrapper<T> {}

#[derive(Clone, Copy)]
pub struct MetalLayerWrapper(pub id);
unsafe impl Send for MetalLayerWrapper {}
unsafe impl Sync for MetalLayerWrapper {}

pub struct MetalBackend {
    pub device: Device,
    command_queue: CommandQueue,
    resources: MetalResourceProvider,
    pipelines: MetalPipelineProvider,
    
    // Tracea Integration
    pub tracea_context: crate::tracea_bridge::TraceaContext,
    pub tracea_blur_cache: Mutex<Option<crate::tracea_bridge::TraceaBlurKernel>>,
    pub tracea_particle_cache: Mutex<Option<crate::tracea_bridge::TraceaParticleKernel>>,
    pub tracea_fft_kernel: Mutex<Option<crate::tracea_bridge::TraceaFFTKernel>>,
    
    main_pipeline: RenderPipelineState,
    instanced_pipeline: RenderPipelineState,
    instanced_gbuffer_pipeline: RenderPipelineState,
    uniform_buffer: Buffer,
    vertex_buffer: Buffer,
    
    transient_pool: Mutex<TransientPool<MetalBackend>>,

    font_texture: Option<Texture>,
    pub layer: Option<MetalLayerWrapper>,
    pub start_time: std::time::Instant,
    pub audio_data: Vec<f32>,
    pub screenshot_requested: Arc<Mutex<Option<String>>>,

    // Resolve Pipeline (Post-process)
    pub resolve_pipeline: RenderPipelineState,
    pub hdr_texture: Option<Texture>,
    pub backdrop_texture: Option<Texture>,
    pub reflection_texture: Option<Texture>,
    pub velocity_texture: Option<Texture>,
    pub dummy_velocity_texture: Texture,
    pub sampler: SamplerState,
    pub bright_pipeline: RenderPipelineState,
    pub blur_pipeline: RenderPipelineState,
    pub resolve_bloom_pipeline: RenderPipelineState,
    pub ssr_pipeline: RenderPipelineState,
    pub bloom_textures: [Option<Texture>; 3], // Bright, BlurH, BlurV
    pub ssr_history_texture: Option<Texture>,
    pub blur_uniform_buffer: Buffer,
    pub cinematic_buffer: Buffer,
    pub current_cinematic: Mutex<crate::backend::shaders::types::CinematicParams>,
    pub lut_texture: Option<Texture>,
    pub temporal_scaler: Option<metalfx::TemporalScaler>,
    pub frame_count: u64,
    pub sdf_texture: Option<Texture>,
    pub depth_texture: Option<Texture>,
    pub internal_resolution_scale: f32,
    pub cached_width: u32,
    pub cached_height: u32,
    
    // Effect specific
    pub ssr_resolve_pipeline: RenderPipelineState,
    pub motion_blur_pipeline: RenderPipelineState,

    // Command Recording State
    pub current_command_buffer: Mutex<Option<MetalIdWrapper<CommandBuffer>>>,
    pub current_drawable: Mutex<Option<MetalIdWrapper<id>>>,
    pub current_encoder: Mutex<Option<MetalIdWrapper<RenderCommandEncoder>>>,
    pub dummy_storage_buffer: Buffer,
    pub is_first_draw: Mutex<bool>,
    pub frame_resources: Arc<Mutex<Vec<MetalResource>>>,
}

#[derive(Clone, Debug)]
pub enum MetalResource {
    Buffer(metal::Buffer),
    Texture(metal::Texture),
}

impl MetalBackend {
    fn create_identity_lut(device: &Device) -> Texture {
        let size = 32;
        let desc = metal::TextureDescriptor::new();
        desc.set_texture_type(MTLTextureType::D3);
        desc.set_width(size);
        desc.set_height(size);
        desc.set_depth(size);
        desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        desc.set_usage(MTLTextureUsage::ShaderRead);
        let lut = device.new_texture(&desc);

        let mut data = Vec::with_capacity((size * size * size * 4) as usize);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    data.push(((x as f32 / (size - 1) as f32) * 255.0) as u8);
                    data.push(((y as f32 / (size - 1) as f32) * 255.0) as u8);
                    data.push(((z as f32 / (size - 1) as f32) * 255.0) as u8);
                    data.push(255);
                }
            }
        }

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: size, height: size, depth: size },
        };
        lut.replace_region_in_slice(region, 0, 0, data.as_ptr() as *const _, (size * 4) as u64, (size * size * 4) as u64);
        lut
    }

    pub fn new() -> Result<Self, String> {
        Self::new_with_config(crate::config::EngineConfig::default())
    }

    pub fn new_with_config(config: crate::config::EngineConfig) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();
        let resources = MetalResourceProvider::new(device.clone());
        
        // Pass resolution scaling to resources or pipeline provider if needed?
        // Currently pipelines are resolution agnostic, but resources might need sizing hints.
        // But backend generally recreates target textures on resize.
        // We will store the config if needed or just use it for initial setup.

        
        let shader_src = format!("{}\n{}\n{}", 
            include_str!("../shaders/metal_shader.metal"),
            include_str!("../shaders/motion_blur.metal"),
            include_str!("../shaders/ssr.metal")
        );
        let pipelines = MetalPipelineProvider::new(device.clone(), &shader_src)?;
        
        // Initialize Tracea Context with shared device
        // Using Arc explicitly to match new signature
        let tracea_context = crate::tracea_bridge::TraceaContext::new(Some(Arc::new(device.clone())))
            .unwrap_or_else(|e| {
                println!("[WARN] Failed to init Tracea: {}. Creating default context.", e);
                crate::tracea_bridge::TraceaContext::new(None).expect("Tracea init failed")
            });
        
        let main_pipeline = pipelines.create_render_pipeline(
            "Fantasmagorie Main", 
            "vs_main fs_main",
            None
        )?;

        let instanced_pipeline = pipelines.create_render_pipeline(
            "Fantasmagorie Instanced",
            "vs_instanced fs_instanced",
            None
        )?;

        let instanced_gbuffer_pipeline = pipelines.create_gbuffer_pipeline(
            "Fantasmagorie G-Buffer",
            "vs_instanced fs_instanced_gbuffer"
        )?;

        let uniform_buffer = device.new_buffer(
            std::mem::size_of::<DrawUniforms>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let vertex_buffer = device.new_buffer(
            (1024 * std::mem::size_of::<Vertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let sampler_desc = SamplerDescriptor::new();
        sampler_desc.set_min_filter(MTLSamplerMinMagFilter::Linear);
        sampler_desc.set_mag_filter(MTLSamplerMinMagFilter::Linear);
        let sampler = device.new_sampler(&sampler_desc);

        let hdr_texture_desc = metal::TextureDescriptor::new();
        hdr_texture_desc.set_width(1);
        hdr_texture_desc.set_height(1);
        hdr_texture_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        hdr_texture_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let hdr_texture = Some(device.new_texture(&hdr_texture_desc));
        
        let dummy_vel_desc = metal::TextureDescriptor::new();
        dummy_vel_desc.set_width(1);
        dummy_vel_desc.set_height(1);
        dummy_vel_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        dummy_vel_desc.set_usage(MTLTextureUsage::ShaderRead);
        let dummy_velocity_texture = device.new_texture(&dummy_vel_desc);

        // Dummy font texture (1x1 R8)
        let dummy_font_desc = metal::TextureDescriptor::new();
        dummy_font_desc.set_width(1);
        dummy_font_desc.set_height(1);
        dummy_font_desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        dummy_font_desc.set_usage(MTLTextureUsage::ShaderRead);
        let dummy_font_texture = device.new_texture(&dummy_font_desc);

        // Dummy backdrop texture (1x1 RGBA16Float)
        let dummy_backdrop_desc = metal::TextureDescriptor::new();
        dummy_backdrop_desc.set_width(1);
        dummy_backdrop_desc.set_height(1);
        dummy_backdrop_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        dummy_backdrop_desc.set_usage(MTLTextureUsage::ShaderRead);
        let dummy_backdrop_texture = device.new_texture(&dummy_backdrop_desc);

        let resolve_pipeline = pipelines.create_render_pipeline("Fantasmagorie Resolve", "vs_resolve fs_resolve", None)?;
        let bright_pipeline = pipelines.create_render_pipeline("Bright Pass", "vs_resolve fs_bright_pass", None)?;
        let blur_pipeline = pipelines.create_render_pipeline("Blur Pass", "vs_resolve fs_blur", None)?;
        let resolve_bloom_pipeline = pipelines.create_render_pipeline("Resolve Bloom", "vs_resolve fs_resolve_bloom", None)?;
        
        let mut bloom_textures = [None, None, None];
        for i in 0..3 {
            let desc = metal::TextureDescriptor::new();
            desc.set_width(1); desc.set_height(1);
            desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
            desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            bloom_textures[i] = Some(device.new_texture(&desc));
        }

        let blur_uniform_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        let cinematic_params = crate::backend::shaders::types::CinematicParams {
            exposure: 1.0, ca_strength: 0.005, vignette_intensity: 0.5, bloom_intensity: 0.4,
            tonemap_mode: 1, bloom_mode: 1, grain_strength: 0.05, time: 0.0,
            lut_intensity: 0.0, blur_radius: 0.0, motion_blur_strength: 0.0, debug_mode: 0,
            gi_intensity: 0.3, light_pos: [500.0, 300.0], volumetric_intensity: 0.5, light_color: [1.0, 0.9, 0.7, 1.0],
            jitter: [0.0, 0.0],
            render_size: [1.0, 1.0],
        };
        let cinematic_buffer = device.new_buffer(
            std::mem::size_of::<crate::backend::shaders::types::CinematicParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = cinematic_buffer.contents() as *mut crate::backend::shaders::types::CinematicParams;
            *ptr = cinematic_params;
        }

        let ssr_pipeline = pipelines.create_ssr_pipeline("Fantasmagorie SSR", "vs_ssr fs_ssr")?;
        let ssr_resolve_pipeline = pipelines.create_render_pipeline("SSR Resolve", "vs_resolve fs_ssr", None)?;
        let motion_blur_pipeline = pipelines.create_render_pipeline("MotionBlur", "vs_main fs_motion_blur", None)?;

        let dummy_storage_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            device: device.clone(), command_queue, resources, pipelines,
            tracea_context,
            tracea_blur_cache: Mutex::new(None),
            tracea_particle_cache: Mutex::new(None),
            tracea_fft_kernel: Mutex::new(None),
            main_pipeline, instanced_pipeline, instanced_gbuffer_pipeline,
            uniform_buffer, vertex_buffer,
            transient_pool: Mutex::new(TransientPool::new()),
            font_texture: Some(dummy_font_texture), backdrop_texture: Some(dummy_backdrop_texture), sampler,
            lut_texture: Some(Self::create_identity_lut(&device)),
            sdf_texture: None,
            layer: None, start_time: std::time::Instant::now(),
            audio_data: vec![0.0; 4], screenshot_requested: Arc::new(Mutex::new(None)),
            resolve_pipeline, hdr_texture, reflection_texture: None, velocity_texture: None,
            dummy_velocity_texture, bright_pipeline, blur_pipeline, resolve_bloom_pipeline,
            ssr_pipeline, ssr_resolve_pipeline, motion_blur_pipeline, bloom_textures,
            ssr_history_texture: None, blur_uniform_buffer, cinematic_buffer,
            current_cinematic: Mutex::new(cinematic_params),
            current_command_buffer: Mutex::new(None), current_drawable: Mutex::new(None),
            current_encoder: Mutex::new(None),
            dummy_storage_buffer,
            is_first_draw: Mutex::new(true),
            internal_resolution_scale: config.internal_resolution_scale,
            cached_width: 0,
            cached_height: 0,
            temporal_scaler: None,
            depth_texture: None,
            frame_count: 0,
            frame_resources: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn set_layer(&mut self, layer: *mut objc::runtime::Object) {
        self.layer = Some(MetalLayerWrapper(layer as id));
    }

    fn _quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> [Vertex; 6] {
        let x = pos.x; let y = pos.y; let w = size.x; let h = size.y;
        let c = [color.r, color.g, color.b, color.a];
        [
            Vertex { pos: [x, y], uv: [0.0, 0.0], color: c },
            Vertex { pos: [x, y + h], uv: [0.0, 1.0], color: c },
            Vertex { pos: [x + w, y + h], uv: [1.0, 1.0], color: c },
            Vertex { pos: [x, y], uv: [0.0, 0.0], color: c },
            Vertex { pos: [x + w, y + h], uv: [1.0, 1.0], color: c },
            Vertex { pos: [x + w, y], uv: [1.0, 0.0], color: c },
        ]
    }

    pub fn ensure_encoder(&self) -> Result<MetalIdWrapper<RenderCommandEncoder>, String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        if let Some(enc) = &*encoder_guard {
            return Ok(MetalIdWrapper(enc.0.to_owned()));
        }

        let d_wrapper = self.current_drawable.lock().unwrap();
        let drawable = d_wrapper.as_ref().ok_or("No drawable for encoder")?.0;
        let command_buffer_guard = self.current_command_buffer.lock().unwrap();
        let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;
        
        let mut is_first = self.is_first_draw.lock().unwrap();
        
        let render_pass_descriptor = RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor.color_attachments().object_at(0).unwrap();
        
        if let Some(ref hdr) = self.hdr_texture {
            color_attachment.set_texture(Some(hdr));
        } else {
            unsafe {
                let tex_ptr: id = msg_send![drawable, texture];
                let _: () = msg_send![color_attachment, setTexture: tex_ptr];
            }
        }
        
        if *is_first {
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_clear_color(MTLClearColor::new(0.01, 0.01, 0.02, 1.0));
            *is_first = false;
        } else {
            color_attachment.set_load_action(MTLLoadAction::Load);
        }
        color_attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor).to_owned();
        
        unsafe {
            let tex_ptr: id = msg_send![drawable, texture];
            let tex: &TextureRef = &*(tex_ptr as *mut TextureRef);
            let viewport = MTLViewport {
                originX: 0.0, originY: 0.0,
                width: tex.width() as f64, height: tex.height() as f64,
                znear: 0.0, zfar: 1.0,
            };
            encoder.set_viewport(viewport);
        }

        let wrapper = MetalIdWrapper(encoder);
        *encoder_guard = Some(MetalIdWrapper(wrapper.0.to_owned()));
        Ok(wrapper)
    }

    fn end_current_encoder(&self) {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        if let Some(enc) = encoder_guard.take() {
            enc.0.end_encoding();
        }
    }
}

impl GraphicsBackend for MetalBackend {
    fn name(&self) -> &str { "Metal" }
    fn set_resolution_scale(&mut self, scale: f32) {
        self.internal_resolution_scale = scale;
    }

    fn update_audio_data(&mut self, data: &[f32]) {
        self.audio_data = data.to_vec();
    }
    
    fn update_audio_pcm(&mut self, samples: &[f32]) {
        if !self.tracea_context.is_ready() { return; }
        
        let mut kernel_lock = self.tracea_fft_kernel.lock().unwrap();
        // Init if needed (assuming fixed size 1024 often used)
        if kernel_lock.is_none() {
             let size = samples.len().max(1024).next_power_of_two();
             if let Ok(k) = crate::tracea_bridge::TraceaFFTKernel::new(&self.tracea_context, size) {
                 *kernel_lock = Some(k);
             }
        }
        
        if let Some(kernel) = kernel_lock.as_ref() {
            // Check size mismatch
            if kernel.fft_size() != samples.len() {
                // Recreate? Or just error/log?
                // For demo simplicity, we assume consistent size.
                // If mismatch, skip or try recreate.
                if samples.len().is_power_of_two() {
                     if let Ok(k) = crate::tracea_bridge::TraceaFFTKernel::new(&self.tracea_context, samples.len()) {
                         *kernel_lock = Some(k);
                     }
                }
            }
            // Execute
            if let Some(valid_kernel) = kernel_lock.as_ref() {
                 if let Ok(spectrum) = valid_kernel.compute_spectrum(&self.tracea_context, samples) {
                      self.audio_data = spectrum;
                 }
            }
        }
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        if self.cached_width != width || self.cached_height != height {
            self.resize_internal(width, height);
        }

        let mut orchestrator = crate::renderer::orchestrator::Orchestrator::<MetalBackend>::new();
        let time = self.start_time.elapsed().as_secs_f32();

        // Sync cinematic buffer
        {
            let mut params = self.current_cinematic.lock().unwrap();
            params.time = time;
            let contents = self.cinematic_buffer.contents();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytemuck::bytes_of(&*params).as_ptr(),
                    contents as *mut u8,
                    std::mem::size_of::<crate::backend::shaders::types::CinematicParams>()
                );
            }
        }

        orchestrator.plan(dl, width, height);
        
        // 1. Begin Frame (acquire drawable, create command buffer)
        if let Err(e) = self.begin_execute() {
             eprintln!("Failed to begin Metal frame: {}", e);
             return;
        }

        if let Err(e) = orchestrator.execute(self, time, width, height) {
            eprintln!("Metal render error: {}", e);
        }

        // 2. Resolve (Final compositing to backbuffer)
        if let Err(e) = self.resolve() {
            eprintln!("Failed to resolve Metal frame: {}", e);
        }

        // 3. End Frame (commit command buffer, handle screenshots)
        if let Err(e) = self.end_execute() {
             eprintln!("Failed to end Metal frame: {}", e);
        }
        
        self.end_current_encoder();
    }

    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        let desc = metal::TextureDescriptor::new();
        desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_usage(MTLTextureUsage::ShaderRead);
        
        let tex = self.device.new_texture(&desc);
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: width as u64, height: height as u64, depth: 1 },
        };
        tex.replace_region(region, 0, data.as_ptr() as *const _, width as u64);
        self.font_texture = Some(tex);
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
            bloom_intensity: 0.4,
            tonemap_mode: match config.tonemap { Tonemap::None => 0, Tonemap::Aces => 1, Tonemap::Reinhard => 2 },
            bloom_mode: match config.bloom { Bloom::None => 0, Bloom::Soft => 1, Bloom::Cinematic => 2 },
            grain_strength: config.grain_strength,
            time: self.start_time.elapsed().as_secs_f32(),
            lut_intensity: config.lut_intensity,
            blur_radius: config.blur_radius,
            motion_blur_strength: config.motion_blur_strength,
            debug_mode: config.debug_mode,
            light_pos: [500.0, 300.0],
            gi_intensity: config.gi_intensity,
            volumetric_intensity: config.volumetric_intensity,
            light_color: [1.0, 0.9, 0.7, 1.0],
            jitter: [0.0, 0.0],
            render_size: [self.cached_width as f32, self.cached_height as f32],
        };

        *self.current_cinematic.lock().unwrap() = params;
        unsafe {
            let ptr = self.cinematic_buffer.contents() as *mut CinematicParams;
            *ptr = params;
        }
    }
}

impl MetalBackend {
    fn resize_internal(&mut self, width: u32, height: u32) {
        self.cached_width = width;
        self.cached_height = height;
        
        // Recreate HDR Texture (Scale applied in Orchestrator for Rendering, but here we might need swapping?)
        // Actually Orchestrator creates HDR_LOW_RES. 
        // Backend::hdr_texture is often used as the "Main" target or swapchain proxy in simple backends.
        // But here specific nodes use specific handles.
        // However, we need to ensure internal resources matching window size are correct.
        
        // Recreate HDR Texture
        let desc = metal::TextureDescriptor::new();
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        self.hdr_texture = Some(self.device.new_texture(&desc));

        // Recreate Depth Texture (Native resolution for post/UI, or needed for scaling?)
        // MetalFX needs Depth for the INPUT resolution. 
        // But Orchestrator manages depth for the render pass?
        // MetalBackend::depth_texture field I added is likely for global depth?
        // Let's initialize it to Native for now (UI overlay etc).
        // Orchestrator creates its own depth buffers usually.
        
        let depth_desc = metal::TextureDescriptor::new();
        depth_desc.set_width(width as u64);
        depth_desc.set_height(height as u64);
        depth_desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        depth_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        depth_desc.set_storage_mode(MTLStorageMode::Private);
        self.depth_texture = Some(self.device.new_texture(&depth_desc));

        // Initialize MetalFX if needed
        let scale = self.internal_resolution_scale;
        // Check if strictly upscaling (scale < 1.0) or if we want AA (scale == 1.0)
        // For this task, we focus on upscaling functionality.
        if (scale - 1.0).abs() > 0.001 || scale < 1.0 {
             use crate::backend::metal::metalfx::TemporalScaler;
             let input_w = (width as f32 * scale) as u64;
             let input_h = (height as f32 * scale) as u64;
             
             // MetalFX Formats
             let color_fmt = MTLPixelFormat::RGBA16Float;
             let depth_fmt = MTLPixelFormat::Depth32Float;
             let motion_fmt = MTLPixelFormat::RGBA16Float; 
             let output_fmt = MTLPixelFormat::RGBA16Float; // Upscale writes to HDR_HIGH_RES

             println!("Initializing MetalFX Scaler: {}x{} -> {}x{}", input_w, input_h, width, height);

             let scaler = TemporalScaler::new(
                 &self.device,
                 input_w as usize, input_h as usize,
                 width as usize, height as usize,
                 color_fmt, depth_fmt, motion_fmt, output_fmt
             );
             
             match scaler {
                 Some(s) => self.temporal_scaler = Some(s),
                 None => {
                     eprintln!("Failed to init MetalFX: scaler creation returned None");
                     self.temporal_scaler = None;
                 }
             }
        } else {
            self.temporal_scaler = None;
        }

        // Recreate Bloom Textures (1/2 or 1/4 size?)
        // Let's use 1/2 size for bloom in Metal
        let bloom_w = (width / 2).max(1) as u64;
        let bloom_h = (height / 2).max(1) as u64;
        for i in 0..3 {
            let desc = metal::TextureDescriptor::new();
            desc.set_width(bloom_w);
            desc.set_height(bloom_h);
            desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
            desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            self.bloom_textures[i] = Some(self.device.new_texture(&desc));
        }
    }
    
}

impl GpuExecutor for MetalBackend {
    type Buffer = Buffer;
    type Texture = Texture;
    type TextureView = Texture;
    type Sampler = SamplerState;
    type RenderPipeline = RenderPipelineState;
    type ComputePipeline = ComputePipelineState;
    type BindGroupLayout = MetalBindGroupLayout;
    type BindGroup = MetalBindGroup;

    fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String> {
        self.resources.create_buffer(size, usage, label)
    }
    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        self.resources.create_texture(desc)
    }
    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String> {
        Ok(texture.clone())
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
    fn destroy_buffer(&self, buffer: Self::Buffer) {
        self.resources.destroy_buffer(buffer)
    }
    fn destroy_texture(&self, texture: Self::Texture) {
        self.resources.destroy_texture(texture)
    }

    fn create_render_pipeline(&self, label: &str, wgsl: &str, layout: Option<&Self::BindGroupLayout>) -> Result<Self::RenderPipeline, String> {
        self.pipelines.create_render_pipeline(label, wgsl, layout)
    }
    fn get_custom_render_pipeline(&self, shader_source: &str) -> Result<Self::RenderPipeline, String> {
        self.pipelines.create_render_pipeline("Custom Pipeline", shader_source, None)
    }
    fn create_compute_pipeline(&self, shader_name: &str, shader_source: &str, entry_point: Option<&str>) -> Result<Self::ComputePipeline, String> {
        self.pipelines.create_compute_pipeline(shader_name, shader_source, entry_point)
    }
    fn get_compute_pipeline_layout(&self, _pipeline: &Self::ComputePipeline, _index: u32) -> Result<Self::BindGroupLayout, String> {
        Ok(pipeline_provider::MetalBindGroupLayout { entries: vec![] }) 
    }
    fn get_render_pipeline_layout(&self, _pipeline: &Self::RenderPipeline, _index: u32) -> Result<Self::BindGroupLayout, String> {
        Ok(pipeline_provider::MetalBindGroupLayout { entries: vec![] }) 
    }
    fn destroy_bind_group(&self, _bind_group: Self::BindGroup) {}

    fn begin_execute(&self) -> Result<(), String> {
        let wrapper = self.layer.ok_or("No CALayer set")?;
        let layer = wrapper.0;
        autoreleasepool(|| unsafe {
            let drawable: id = msg_send![layer, nextDrawable];
            if drawable == cocoa::base::nil { return Err("Failed to get next drawable".into()); }
            let _: () = msg_send![drawable, retain];
            *self.current_drawable.lock().unwrap() = Some(MetalIdWrapper(drawable));
            *self.is_first_draw.lock().unwrap() = true;
            let command_buffer = self.command_queue.new_command_buffer().to_owned();
            *self.current_command_buffer.lock().unwrap() = Some(MetalIdWrapper(command_buffer));
            Ok(())
        })
    }

    fn end_execute(&self) -> Result<(), String> {
        if let Some(wrapper) = self.current_encoder.lock().unwrap().take() {
            wrapper.0.end_encoding();
        }
        if let Some(cb_wrapper) = self.current_command_buffer.lock().unwrap().take() {
            let command_buffer = cb_wrapper.0;
            let mut req_guard = self.screenshot_requested.lock().unwrap();
            if let Some(path) = req_guard.take() {
                if let Some(d_wrapper) = *self.current_drawable.lock().unwrap() {
                    let drawable = d_wrapper.0;
                    unsafe {
                        let tex_ptr: id = msg_send![drawable, texture];
                        let src_tex: &TextureRef = &*(tex_ptr as *mut TextureRef);
                        self.perform_screenshot(src_tex, &path, &command_buffer)?;
                    }
                }
            }
            if let Some(d_wrapper) = self.current_drawable.lock().unwrap().take() {
                let drawable = d_wrapper.0;
                unsafe {
                    let _: () = msg_send![command_buffer, presentDrawable: drawable];
                    let _: () = msg_send![drawable, release];
                }
            }
            command_buffer.commit();
        }
        Ok(())
    }

    fn draw(&self, pipeline: &Self::RenderPipeline, bind_group: Option<&Self::BindGroup>, vertex_buffer: &Self::Buffer, vertex_count: u32, uniform_data: &[u8]) -> Result<(), String> {
        let wrapper = self.ensure_encoder()?;
        let encoder = &wrapper.0;
        encoder.set_render_pipeline_state(pipeline);
        encoder.set_vertex_buffer(1, Some(vertex_buffer), 0); // Slot 1 for vertices
        
        // Use set_vertex_bytes for small uniform data - zero overhead, no buffer management needed
        if !uniform_data.is_empty() {
            encoder.set_vertex_bytes(0, uniform_data.len() as u64, uniform_data.as_ptr() as *const _);
            encoder.set_fragment_bytes(0, uniform_data.len() as u64, uniform_data.as_ptr() as *const _);
        }

        if let Some(bg) = bind_group {
            let mut resources = self.frame_resources.lock().unwrap();
            for entry in &bg.entries {
                match &entry.resource {
                    pipeline_provider::MetalResource::Buffer(buf) => {
                        encoder.set_vertex_buffer(entry.binding as u64, Some(buf), 0);
                        encoder.set_fragment_buffer(entry.binding as u64, Some(buf), 0);
                        // Keep alive
                        resources.push(MetalResource::Buffer(buf.clone()));
                    },
                    pipeline_provider::MetalResource::Texture(tex) => {
                        encoder.set_fragment_texture(entry.binding as u64, Some(tex));
                        resources.push(MetalResource::Texture(tex.clone()));
                    },
                    pipeline_provider::MetalResource::Sampler(samp) => {
                        encoder.set_fragment_sampler_state(entry.binding as u64, Some(samp));
                    },
                }
            }
        }

        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
        Ok(())
    }

    fn draw_ssr(&mut self, hdr_view: &Self::TextureView, depth_view: &Self::TextureView, aux_view: &Self::TextureView, velocity_view: &Self::TextureView, output_texture: &Self::Texture) -> Result<(), String> {
        self.end_current_encoder();
        let command_buffer_guard = self.current_command_buffer.lock().unwrap();
        let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;
        let descriptor = RenderPassDescriptor::new();
        let color = descriptor.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(output_texture));
        color.set_load_action(MTLLoadAction::Load);
        color.set_store_action(MTLStoreAction::Store);
        let encoder = command_buffer.new_render_command_encoder(descriptor);
        encoder.set_render_pipeline_state(&self.ssr_pipeline);
        encoder.set_fragment_texture(0, Some(hdr_view));
        encoder.set_fragment_texture(1, Some(depth_view));
        encoder.set_fragment_texture(2, Some(aux_view));
        if let Some(hist) = &self.ssr_history_texture { encoder.set_fragment_texture(3, Some(hist)); }
        else { encoder.set_fragment_texture(3, None); }
        encoder.set_fragment_texture(4, Some(velocity_view));
        encoder.set_fragment_sampler_state(0, Some(&self.sampler));
        encoder.set_fragment_buffer(0, Some(&self.cinematic_buffer), 0);
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
        Ok(())
    }

    fn draw_instanced(&self, pipeline: &Self::RenderPipeline, bind_group: Option<&Self::BindGroup>, vertex_buffer: &Self::Buffer, instance_buffer: &Self::Buffer, vertex_count: u32, instance_count: u32) -> Result<(), String> {
        let wrapper = self.ensure_encoder()?;
        let encoder = &wrapper.0;
        encoder.set_render_pipeline_state(pipeline);
        encoder.set_vertex_buffer(1, Some(vertex_buffer), 0); // Slot 1 for vertices
        encoder.set_vertex_buffer(2, Some(instance_buffer), 0); // Slot 2 for instances
        encoder.set_fragment_sampler_state(0, Some(&self.sampler));
        
        // Keep buffers alive
        let mut resources = self.frame_resources.lock().unwrap();
        resources.push(MetalResource::Buffer(vertex_buffer.clone()));
        resources.push(MetalResource::Buffer(instance_buffer.clone()));

        if let Some(bg) = bind_group {
            for entry in &bg.entries {
                match &entry.resource {
                    pipeline_provider::MetalResource::Buffer(buf) => {
                        // Bind group buffers go to their requested binding (e.g. 0 for uniforms)
                        encoder.set_vertex_buffer(entry.binding as u64, Some(buf), 0);
                        encoder.set_fragment_buffer(entry.binding as u64, Some(buf), 0);
                        resources.push(MetalResource::Buffer(buf.clone()));
                    },
                    pipeline_provider::MetalResource::Texture(tex) => {
                        encoder.set_fragment_texture(entry.binding as u64, Some(tex));
                        resources.push(MetalResource::Texture(tex.clone()));
                    },
                    pipeline_provider::MetalResource::Sampler(samp) => encoder.set_fragment_sampler_state(entry.binding as u64, Some(samp)),
                }
            }
        }

        encoder.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, vertex_count as u64, instance_count as u64);
        Ok(())
    }

    fn draw_instanced_gbuffer(&mut self, pipeline: &Self::RenderPipeline, bind_group: Option<&Self::BindGroup>, vertex_buffer: &Self::Buffer, instance_buffer: &Self::Buffer, vertex_count: u32, instance_count: u32, aux_view: &Self::TextureView, velocity_view: &Self::TextureView, depth_view: &Self::TextureView) -> Result<(), String> {
        self.end_current_encoder();
        let d_wrapper = self.current_drawable.lock().unwrap();
        let drawable = d_wrapper.as_ref().ok_or("No drawable")?.0;
        let descriptor = RenderPassDescriptor::new();
        let color0 = descriptor.color_attachments().object_at(0).unwrap();
        if let Some(tex) = &self.hdr_texture { color0.set_texture(Some(tex)); }
        else { unsafe { let tex_ptr: id = msg_send![drawable, texture]; color0.set_texture(Some(&*(tex_ptr as *mut TextureRef))); } }
        color0.set_load_action(MTLLoadAction::Load); color0.set_store_action(MTLStoreAction::Store);
        let color1 = descriptor.color_attachments().object_at(1).unwrap();
        color1.set_texture(Some(aux_view)); color1.set_load_action(MTLLoadAction::Load); color1.set_store_action(MTLStoreAction::Store);
        let color2 = descriptor.color_attachments().object_at(2).unwrap();
        color2.set_texture(Some(velocity_view)); color2.set_load_action(MTLLoadAction::Load); color2.set_store_action(MTLStoreAction::Store);
        let depth = descriptor.depth_attachment().unwrap();
        depth.set_texture(Some(depth_view)); depth.set_load_action(MTLLoadAction::Load); depth.set_store_action(MTLStoreAction::Store);
        let cb_guard = self.current_command_buffer.lock().unwrap();
        let cb = &cb_guard.as_ref().ok_or("No command buffer")?.0;
        let encoder = cb.new_render_command_encoder(descriptor);
        encoder.set_render_pipeline_state(pipeline);
        let ds_desc = DepthStencilDescriptor::new();
        ds_desc.set_depth_compare_function(MTLCompareFunction::LessEqual);
        ds_desc.set_depth_write_enabled(true);
        encoder.set_depth_stencil_state(&self.device.new_depth_stencil_state(&ds_desc));
        encoder.set_vertex_buffer(1, Some(vertex_buffer), 0); // Slot 1 for vertices
        encoder.set_vertex_buffer(2, Some(instance_buffer), 0); // Slot 2 for instances
        
        // Keep buffers alive
        let mut resources = self.frame_resources.lock().unwrap();
        resources.push(MetalResource::Buffer(vertex_buffer.clone()));
        resources.push(MetalResource::Buffer(instance_buffer.clone()));

        if let Some(bg) = bind_group {
            for entry in &bg.entries {
                match &entry.resource {
                    pipeline_provider::MetalResource::Buffer(buf) => {
                        // Standard Slot 0 or entry.binding for bind group buffers
                        encoder.set_vertex_buffer(entry.binding as u64, Some(buf), 0);
                        encoder.set_fragment_buffer(entry.binding as u64, Some(buf), 0);
                        resources.push(MetalResource::Buffer(buf.clone()));
                    }
                    pipeline_provider::MetalResource::Texture(tex) => {
                        encoder.set_fragment_texture(entry.binding as u64, Some(tex));
                        resources.push(MetalResource::Texture(tex.clone()));
                    }
                    pipeline_provider::MetalResource::Sampler(sampler) => {
                        encoder.set_fragment_sampler_state(entry.binding as u64, Some(sampler));
                    }
                }
            }
        }
        encoder.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, vertex_count as u64, instance_count as u64);
        encoder.end_encoding();
        Ok(())
    }

    fn dispatch(&self, _p: &Self::ComputePipeline, _bg: Option<&Self::BindGroup>, _g: [u32; 3], _pc: &[u8]) -> Result<(), String> { Ok(()) }

    fn draw_particles(&mut self, pipeline: &Self::RenderPipeline, bind_group: &Self::BindGroup, particle_count: u32) -> Result<(), String> {
        self.end_current_encoder();
        let d_wrapper = self.current_drawable.lock().unwrap();
        let drawable = d_wrapper.as_ref().ok_or("No drawable")?.0;
        
        let descriptor = RenderPassDescriptor::new();
        let color = descriptor.color_attachments().object_at(0).unwrap();
        
        // Target HDR if available, else drawable
        if let Some(tex) = &self.hdr_texture {
             color.set_texture(Some(tex));
        } else {
             unsafe {
                 let tex_ptr: id = msg_send![drawable, texture];
                 color.set_texture(Some(&*(tex_ptr as *mut TextureRef)));
             }
        }
        
        color.set_load_action(MTLLoadAction::Load); // Draw ON TOP
        color.set_store_action(MTLStoreAction::Store);
        
        let cb_guard = self.current_command_buffer.lock().unwrap();
        let cb = &cb_guard.as_ref().ok_or("No command buffer")?.0;
        let encoder = cb.new_render_command_encoder(descriptor);
        
        encoder.set_render_pipeline_state(pipeline);
        
        // Bind Group
        let mut resources = self.frame_resources.lock().unwrap();
        for entry in &bind_group.entries {
            match &entry.resource {
                pipeline_provider::MetalResource::Buffer(buf) => {
                    encoder.set_vertex_buffer(entry.binding as u64, Some(buf), 0);
                    encoder.set_fragment_buffer(entry.binding as u64, Some(buf), 0);
                    resources.push(MetalResource::Buffer(buf.clone()));
                }
                pipeline_provider::MetalResource::Texture(tex) => {
                    encoder.set_vertex_texture(entry.binding as u64, Some(tex)); // For dim check in VS
                    encoder.set_fragment_texture(entry.binding as u64, Some(tex));
                    resources.push(MetalResource::Texture(tex.clone()));
                }
                pipeline_provider::MetalResource::Sampler(samp) => {
                     encoder.set_vertex_sampler_state(entry.binding as u64, Some(samp));
                     encoder.set_fragment_sampler_state(entry.binding as u64, Some(samp));
                }
            }
        }
        
        encoder.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, 4, particle_count as u64); // Quad
        encoder.end_encoding();
        Ok(())
    }
    fn copy_texture(&self, src: &Self::Texture, dst: &Self::Texture) -> Result<(), String> {
        let cb_guard = self.current_command_buffer.lock().unwrap();
        let cb = &cb_guard.as_ref().ok_or("No command buffer")?.0;
        let encoder = cb.new_blit_command_encoder();
        
        // Keep alive
        let mut resources = self.frame_resources.lock().unwrap();
        resources.push(MetalResource::Texture(src.clone()));
        resources.push(MetalResource::Texture(dst.clone()));

        encoder.copy_from_texture(src, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: src.width(), height: src.height(), depth: 1 }, dst, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 });
        encoder.end_encoding();
        Ok(())
    }
    fn generate_mipmaps(&self, texture: &Self::Texture) -> Result<(), String> {
        let cb_guard = self.current_command_buffer.lock().unwrap();
        let cb = &cb_guard.as_ref().ok_or("No command buffer")?.0;
        let encoder = cb.new_blit_command_encoder();
        
        self.frame_resources.lock().unwrap().push(MetalResource::Texture(texture.clone()));
        
        encoder.generate_mipmaps(texture);
        encoder.end_encoding();
        Ok(())
    }

    fn copy_framebuffer_to_texture(&self, dst: &Self::Texture) -> Result<(), String> {
        let cb_guard = self.current_command_buffer.lock().unwrap();
        let cb = &cb_guard.as_ref().ok_or("No command buffer")?.0;
        let encoder = cb.new_blit_command_encoder();
        
        self.frame_resources.lock().unwrap().push(MetalResource::Texture(dst.clone()));
        
        if let Some(d_wrapper) = *self.current_drawable.lock().unwrap() {
            let drawable = d_wrapper.0;
            unsafe {
                let tex_ptr: id = msg_send![drawable, texture];
                let src_tex: &TextureRef = &*(tex_ptr as *mut TextureRef);
                encoder.copy_from_texture(src_tex, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: src_tex.width(), height: src_tex.height(), depth: 1 }, dst, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 });
            }
        }
        encoder.end_encoding();
        Ok(())
    }
    fn draw_motion_blur(&self, dst: &Self::TextureView, src: &Self::TextureView, vel: &Self::TextureView, strength: f32) -> Result<(), String> {
        let cb_guard = self.current_command_buffer.lock().unwrap();
        let cb = &cb_guard.as_ref().ok_or("No command buffer")?.0;
        let desc = RenderPassDescriptor::new();
        let color = desc.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(dst)); color.set_load_action(MTLLoadAction::Load); color.set_store_action(MTLStoreAction::Store);
        let enc = cb.new_render_command_encoder(desc);
        enc.set_render_pipeline_state(&self.motion_blur_pipeline);
        enc.set_fragment_texture(0, Some(src)); enc.set_fragment_texture(1, Some(vel));
        enc.set_fragment_sampler_state(0, Some(&self.sampler));
        let str_buf = self.device.new_buffer_with_data(&strength as *const f32 as *const _, 4, MTLResourceOptions::CPUCacheModeDefaultCache);
        enc.set_fragment_buffer(0, Some(&str_buf), 0);
        enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        enc.end_encoding();
        Ok(())
    }

    fn dispatch_tracea_blur(&self, input: &Self::Texture, output: &Self::Texture, sigma: f32) -> Result<bool, String> {
        if !self.tracea_context.is_ready() { return Ok(false); }
        
        let mut cache = self.tracea_blur_cache.lock().unwrap();
        if cache.is_none() || (cache.as_ref().unwrap().sigma - sigma).abs() > 0.01 {
             let radius = (sigma * 3.0) as u32;
             let kernel = crate::tracea_bridge::TraceaBlurKernel::new(&self.tracea_context, radius)
                 .map_err(|e| format!("Tracea blur init failed: {}", e))?;
             *cache = Some(kernel);
        }
        
        let kernel = cache.as_ref().unwrap();
        kernel.execute(input, output, 1).map_err(|e| e.to_string())?;
        
        Ok(true)
    }

    fn supports_tracea_particles(&self) -> bool {
        self.tracea_context.is_ready()
    }

    fn get_tracea_particle_buffer(&self) -> Option<Self::Buffer> {
        let mut cache = self.tracea_particle_cache.lock().unwrap();
        if cache.is_none() {
             let kernel = crate::tracea_bridge::TraceaParticleKernel::new(&self.tracea_context, 100_000) // Match ParticlesNode default
                 .map_err(|e| println!("Tracea particle init failed: {}", e)).ok()?;
             *cache = Some(kernel);
        }
        
        Some(cache.as_ref()?.particle_buffer().clone())
    }

    fn dispatch_tracea_particles(&self, dt: f32, _attractor: [f32; 2], sdf_texture: Option<&Self::Texture>) -> Result<bool, String> {
        if !self.tracea_context.is_ready() { return Ok(false); }
        
        // Lazy init if not already done (should be done by get_buffer usually)
        let mut cache = self.tracea_particle_cache.lock().unwrap();
        if cache.is_none() {
            let kernel = crate::tracea_bridge::TraceaParticleKernel::new(&self.tracea_context, 100_000)
                 .map_err(|e| format!("Tracea particle init failed: {}", e))?;
             *cache = Some(kernel);
        }
        
        let kernel = cache.as_ref().unwrap();
        // Attractor not passed from ParticlesNode yet?
        // SimParams in ParticlesNode has attractor hardcoded:
        // attractor_pos: [0.0, 0.0], attractor_strength: 0.0 in reset.
        // Update passes attractor?
        // TraceaParticleKernel::update takes attractor.
        // I need to use the one passed to this function.
        // If _attractor is unused in signature, I should change signature to use it.
        // Wait, dispatch_tracea_particles signature in trait had `_attractor`.
        // I should use `attractor`.
        kernel.update(dt, _attractor, sdf_texture);
        
        Ok(true)
    }

    fn update_audio_data(&mut self, data: &[f32]) {
        self.audio_data = data.to_vec();
    }

    fn update_audio_pcm(&mut self, samples: &[f32]) {
        if !self.tracea_context.is_ready() { return; }
        
        let mut kernel_lock = self.tracea_fft_kernel.lock().unwrap();
        if kernel_lock.is_none() {
            let size = samples.len().max(1024).next_power_of_two();
            if let Ok(k) = crate::tracea_bridge::TraceaFFTKernel::new(&self.tracea_context, size) {
                *kernel_lock = Some(k);
            }
        }
        if let Some(valid_kernel) = kernel_lock.as_ref() {
            if let Ok(spectrum) = valid_kernel.compute_spectrum(&self.tracea_context, samples) {
                self.audio_data = spectrum;
            }
        }
    }
    
    fn acquire_transient_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let mut pool = self.transient_pool.lock().unwrap();
        pool.acquire_texture(self, desc)
    }
    fn release_transient_texture(&self, texture: Self::Texture, desc: &TextureDescriptor) {
        let mut pool = self.transient_pool.lock().unwrap();
        pool.release_texture(texture, desc);
    }

    fn create_bind_group(&self, _l: &Self::BindGroupLayout, entries: &[crate::backend::hal::BindGroupEntry<Self>]) -> Result<Self::BindGroup, String> {
        let mut metal_entries = Vec::new();
        for entry in entries {
            let resource = match &entry.resource {
                crate::backend::hal::BindingResource::Buffer(b) => pipeline_provider::MetalResource::Buffer((*b).clone()),
                crate::backend::hal::BindingResource::Texture(t) => pipeline_provider::MetalResource::Texture((*t).clone()),
                crate::backend::hal::BindingResource::Sampler(s) => pipeline_provider::MetalResource::Sampler((*s).clone()),
            };
            metal_entries.push(pipeline_provider::MetalBindGroupEntry {
                binding: entry.binding,
                resource,
            });
        }
        Ok(MetalBindGroup { entries: metal_entries })
    }

    fn get_font_view(&self) -> &Self::TextureView { self.font_texture.as_ref().unwrap() }
    fn get_backdrop_view(&self) -> &Self::TextureView { self.backdrop_texture.as_ref().unwrap() }
    fn get_hdr_texture(&self) -> Option<Self::Texture> { self.hdr_texture.clone() }
    fn get_backdrop_texture(&self) -> Option<Self::Texture> { self.backdrop_texture.clone() }
    fn get_extra_texture(&self) -> Option<Self::Texture> { None }
    fn get_aux_texture(&self) -> Option<Self::Texture> { None }
    fn get_velocity_texture(&self) -> Option<Self::Texture> { self.velocity_texture.clone() }
    fn get_depth_texture(&self) -> Option<Self::Texture> { self.depth_texture.clone() }
    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout { 
        static DUMMY: MetalBindGroupLayout = MetalBindGroupLayout { entries: Vec::new() };
        &DUMMY
    }
    fn get_instanced_bind_group_layout(&self) -> &Self::BindGroupLayout { 
        static DUMMY: MetalBindGroupLayout = MetalBindGroupLayout { entries: Vec::new() };
        &DUMMY
    }
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline { &self.main_pipeline }
    fn get_instanced_render_pipeline(&self) -> &Self::RenderPipeline { &self.instanced_pipeline }
    fn get_instanced_gbuffer_render_pipeline(&self) -> &Self::RenderPipeline { &self.instanced_gbuffer_pipeline }
    fn set_reflection_texture(&mut self, t: &Self::TextureView) -> Result<(), String> { self.reflection_texture = Some(t.clone()); Ok(()) }
    fn set_velocity_view(&mut self, v: &Self::TextureView) -> Result<(), String> { self.velocity_texture = Some(v.clone()); Ok(()) }
    fn set_sdf_view(&mut self, v: &Self::TextureView) -> Result<(), String> { self.sdf_texture = Some(v.clone()); Ok(()) }
    fn get_default_sampler(&self) -> &Self::Sampler { &self.sampler }
    fn get_dummy_storage_buffer(&self) -> &Self::Buffer { &self.dummy_storage_buffer }
    fn resolve(&mut self) -> Result<(), String> {
        // Now delegating to draw_post_process_pass if called via old HAL flow
        let hdr = self.hdr_texture.clone();
        if let Some(tex) = hdr {
             self.draw_post_process_pass(&tex, None)
        } else {
             Ok(())
        }
    }

    fn draw_post_process_pass(&mut self, input_view: &Self::TextureView, output_view: Option<&Self::TextureView>) -> Result<(), String> {
        self.end_current_encoder();
        
        let cb_guard = self.current_command_buffer.lock().unwrap();
        let cb = &cb_guard.as_ref().ok_or("No command buffer")?.0;

        // 1. Bloom Bright Pass
        if let Some(bright_tex) = &self.bloom_textures[0] {
            let desc = RenderPassDescriptor::new();
            let ca = desc.color_attachments().object_at(0).unwrap();
            ca.set_texture(Some(bright_tex));
            ca.set_load_action(MTLLoadAction::Clear);
            ca.set_store_action(MTLStoreAction::Store);
            
            let encoder = cb.new_render_command_encoder(desc);
            encoder.set_render_pipeline_state(&self.bright_pipeline);
            encoder.set_fragment_texture(0, Some(input_view));
            encoder.set_fragment_sampler_state(0, Some(&self.sampler));
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
        }

        // 2. Bloom Blur Pass (Horizontal)
        if let Some(bright_tex) = &self.bloom_textures[0] {
            if let Some(blur_h_tex) = &self.bloom_textures[1] {
                let desc = RenderPassDescriptor::new();
                let ca = desc.color_attachments().object_at(0).unwrap();
                ca.set_texture(Some(blur_h_tex));
                ca.set_load_action(MTLLoadAction::Clear);
                ca.set_store_action(MTLStoreAction::Store);
                
                let encoder = cb.new_render_command_encoder(desc);
                encoder.set_render_pipeline_state(&self.blur_pipeline);
                encoder.set_fragment_texture(0, Some(bright_tex));
                encoder.set_fragment_sampler_state(0, Some(&self.sampler));
                
                let horizontal: i32 = 1;
                unsafe {
                    let ptr = self.blur_uniform_buffer.contents() as *mut i32;
                    *ptr = horizontal;
                }
                encoder.set_fragment_buffer(0, Some(&self.blur_uniform_buffer), 0);
                encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
                encoder.end_encoding();
            }
        }

        // 3. Bloom Blur Pass (Vertical)
        if let Some(blur_h_tex) = &self.bloom_textures[1] {
            if let Some(blur_v_tex) = &self.bloom_textures[2] {
                let desc = RenderPassDescriptor::new();
                let ca = desc.color_attachments().object_at(0).unwrap();
                ca.set_texture(Some(blur_v_tex));
                ca.set_load_action(MTLLoadAction::Clear);
                ca.set_store_action(MTLStoreAction::Store);
                
                let encoder = cb.new_render_command_encoder(desc);
                encoder.set_render_pipeline_state(&self.blur_pipeline);
                encoder.set_fragment_texture(0, Some(blur_h_tex));
                encoder.set_fragment_sampler_state(0, Some(&self.sampler));
                
                let horizontal: i32 = 0;
                unsafe {
                    let ptr = self.blur_uniform_buffer.contents() as *mut i32;
                    *ptr = horizontal;
                }
                encoder.set_fragment_buffer(0, Some(&self.blur_uniform_buffer), 0);
                encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
                encoder.end_encoding();
            }
        }

        // 4. Final Resolve / Compositing
        let descriptor = RenderPassDescriptor::new();
        let color = descriptor.color_attachments().object_at(0).unwrap();
        
        if let Some(out) = output_view {
            color.set_texture(Some(out));
        } else {
            let d_wrapper = self.current_drawable.lock().unwrap();
            let drawable = d_wrapper.as_ref().ok_or("No drawable")?.0;
            unsafe {
                let tex_ptr: id = msg_send![drawable, texture];
                color.set_texture(Some(&*(tex_ptr as *mut TextureRef)));
            }
        }
        
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        color.set_store_action(MTLStoreAction::Store);

        let encoder = cb.new_render_command_encoder(descriptor);
        encoder.set_render_pipeline_state(&self.resolve_bloom_pipeline);
        
        encoder.set_fragment_texture(0, Some(input_view));
        if let Some(bloom) = &self.bloom_textures[2] {
            encoder.set_fragment_texture(1, Some(bloom));
        }
        if let Some(lut) = &self.lut_texture {
             // In MSL, slot 2 is for LUT (texture3d)
             encoder.set_fragment_texture(2, Some(lut));
        }
        
        // Pass extra textures (velocity, reflection, aux, extra, sdf)
        if let Some(tex) = &self.velocity_texture { encoder.set_fragment_texture(3, Some(tex)); }
        else { encoder.set_fragment_texture(3, Some(&self.dummy_velocity_texture)); }
        
        if let Some(tex) = &self.reflection_texture { encoder.set_fragment_texture(4, Some(tex)); }
        else { encoder.set_fragment_texture(4, Some(&self.dummy_velocity_texture)); }
        
        encoder.set_fragment_texture(5, Some(&self.dummy_velocity_texture)); // Aux dummy
        encoder.set_fragment_texture(6, Some(&self.dummy_velocity_texture)); // Extra dummy
        
        if let Some(tex) = &self.sdf_texture { encoder.set_fragment_texture(7, Some(tex)); }
        else { encoder.set_fragment_texture(7, Some(&self.dummy_velocity_texture)); }

        encoder.set_fragment_sampler_state(0, Some(&self.sampler));
        encoder.set_fragment_buffer(2, Some(&self.cinematic_buffer), 0);
        
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
        
        Ok(())
    }
    fn present(&self) -> Result<(), String> { Ok(()) }
    fn y_flip(&self) -> bool { false }
    fn get_cinematic_buffer(&self) -> &Self::Buffer { &self.cinematic_buffer }
    fn get_lut_texture(&self) -> Option<Self::Texture> { None } // TODO: Fix TextureRef -> Texture conversion
    fn set_lut_view(&mut self, _view: &Self::TextureView) -> Result<(), String> { Ok(()) }
    fn set_hdr_view(&mut self, view: &Self::TextureView) -> Result<(), String> { self.hdr_texture = Some(view.clone()); Ok(()) }
    fn draw_lighting_pass(&mut self, output_view: &Self::TextureView) -> Result<(), String> {
        // Fallback: Copy current HDR buffer to output
        if let Some(src_tex) = &self.hdr_texture {
             self.end_current_encoder();
             unsafe {
                let cb_guard = self.current_command_buffer.lock().unwrap();
                let cb_wrapper = cb_guard.as_ref().ok_or("No active command buffer")?;
                let cb: id = cb_wrapper.0.as_ptr() as id;
                
                let blit: id = msg_send![cb, blitCommandEncoder];
                
                let origin = MTLOrigin { x: 0, y: 0, z: 0 };
                let size = MTLSize { width: src_tex.width(), height: src_tex.height(), depth: 1 };
                
                let _: () = msg_send![blit, copyFromTexture: src_tex.as_ptr() as id
                                          sourceSlice: 0
                                          sourceLevel: 0
                                          sourceOrigin: origin
                                          sourceSize: size
                                          toTexture: output_view.as_ptr() as id
                                          destinationSlice: 0
                                          destinationLevel: 0
                                          destinationOrigin: origin];
                
                let _: () = msg_send![blit, endEncoding];
             }
        }
        Ok(())
    }
    
    fn draw_fxaa_pass(&mut self, input_view: &Self::TextureView) -> Result<(), String> {
        // In Metal, resolve_bloom_pipeline handles resolve + some filtering. 
        // For actual FXAA, we'd need a separate shader. 
        // For now, draw_post_process_pass(input, None) handles the final blit to swapchain.
        self.draw_post_process_pass(input_view, None)
    }
    fn upscale(&mut self, input: &Self::TextureView, output: &Self::TextureView, params: crate::backend::hal::UpscaleParams) -> Result<(), String> {
        self.end_current_encoder();
        if let Some(scaler) = &self.temporal_scaler {
            let cb_guard = self.current_command_buffer.lock().unwrap();
            let cb_wrapper = cb_guard.as_ref().ok_or("No active command buffer")?;
            let command_buffer: &CommandBufferRef = unsafe { std::mem::transmute(&cb_wrapper.0) }; 

            let depth_tex = self.depth_texture.as_ref().ok_or("Missing depth texture")?;
            let velocity_tex = self.velocity_texture.as_ref().ok_or("Missing velocity texture")?;

            scaler.encode(
                command_buffer,
                input, 
                depth_tex, 
                velocity_tex, 
                output, 
                params.jitter_x,
                params.jitter_y,
                params.reset_history
            );
            Ok(())
        } else {
             // Fallback: Copy input to output using direct msg_send
             unsafe {
                let cb_guard = self.current_command_buffer.lock().unwrap();
                let cb_wrapper = cb_guard.as_ref().ok_or("No active command buffer")?;
                let cb: id = cb_wrapper.0.as_ptr() as id;
                
                let blit: id = msg_send![cb, blitCommandEncoder];
                
                let origin = MTLOrigin { x: 0, y: 0, z: 0 };
                let size = MTLSize { width: input.width(), height: input.height(), depth: 1 };
                
                let _: () = msg_send![blit, copyFromTexture: input.as_ptr() as id
                                          sourceSlice: 0
                                          sourceLevel: 0
                                          sourceOrigin: origin
                                          sourceSize: size
                                          toTexture: output.as_ptr() as id
                                          destinationSlice: 0
                                          destinationLevel: 0
                                          destinationOrigin: origin];
                
                let _: () = msg_send![blit, endEncoding];
             }
             Ok(())
        }
    }
}

impl MetalBackend {
    fn perform_screenshot(&self, texture: &TextureRef, path: &str, command_buffer: &CommandBufferRef) -> Result<(), String> {
        let width = texture.width();
        let height = texture.height();
        let bytes_per_pixel = 4;
        let bytes_per_row = width * bytes_per_pixel;
        let data_size = bytes_per_row * height;

        println!("DEBUG: perform_screenshot called for {} ({}x{})", path, width, height);
        let cpu_buffer = self.device.new_buffer(data_size, MTLResourceOptions::StorageModeShared);
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_texture_to_buffer(
            texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width, height, depth: 1 },
            &cpu_buffer, 0, bytes_per_row, data_size,
            metal::MTLBlitOption::None
        );
        blit_encoder.end_encoding();

        let path = path.to_string();
        let block = ConcreteBlock::new(move |cb: &CommandBufferRef| {
            println!("DEBUG: Screenshot CommandBuffer completed for {} with status {:?}", path, cb.status());
            if cb.status() == MTLCommandBufferStatus::Completed {
                let ptr = cpu_buffer.contents() as *const u8;
                let mut data = vec![0u8; data_size as usize];
                unsafe { std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data_size as usize); }
                let mut img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width as u32, height as u32, data).unwrap();
                for pixel in img.pixels_mut() { pixel.0.swap(0, 2); } // BGRA to RGBA
                match img.save(&path) {
                    Ok(_) => println!("DEBUG: Screenshot saved successfully to {}", path),
                    Err(e) => println!("DEBUG: Failed to save screenshot: {}", e),
                }
            } else {
                println!("DEBUG: Screenshot CommandBuffer failed or cancelled.");
            }
        });
        command_buffer.add_completed_handler(&block.copy());
        Ok(())
    }
}
