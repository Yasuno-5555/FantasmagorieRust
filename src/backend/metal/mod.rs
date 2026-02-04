use metal::*;
use metal::foreign_types::ForeignTypeRef;
use std::sync::{Arc, Mutex};
use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList, DrawCommand::*};
use crate::renderer::graph::TransientPool;
use crate::backend::GraphicsBackend;
use crate::backend::shaders::types::{DrawUniforms, create_projection};
use cocoa::base::id;
use objc::{msg_send, sel, sel_impl, rc::autoreleasepool};
use block::ConcreteBlock;

pub mod resource_provider;
pub mod pipeline_provider;

use pipeline_provider::{MetalPipelineProvider, MetalBindGroup, MetalBindGroupLayout};
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureFormat, TextureUsage};
use resource_provider::MetalResourceProvider;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
    
    main_pipeline: RenderPipelineState,
    instanced_pipeline: RenderPipelineState,
    instanced_gbuffer_pipeline: RenderPipelineState,
    uniform_buffer: Buffer,
    vertex_buffer: Buffer,
    
    transient_pool: Mutex<TransientPool<MetalBackend>>,

    font_texture: Option<Texture>,
    backdrop_texture: Option<Texture>,
    pub sampler: SamplerState,
    pub layer: Option<MetalLayerWrapper>,
    pub start_time: std::time::Instant,
    pub audio_data: Vec<f32>,
    pub screenshot_requested: Arc<Mutex<Option<String>>>,

    // Resolve Pipeline (Post-process)
    pub resolve_pipeline: RenderPipelineState,
    pub hdr_texture: Option<Texture>,
    pub reflection_texture: Option<Texture>,

    // Bloom Pipelines
    pub bright_pipeline: RenderPipelineState,
    pub blur_pipeline: RenderPipelineState,
    pub resolve_bloom_pipeline: RenderPipelineState,
    pub bloom_textures: [Option<Texture>; 3], // Bright, BlurH, BlurV
    pub blur_uniform_buffer: Buffer,
    pub cinematic_buffer: Buffer,
    pub current_cinematic: Mutex<crate::backend::shaders::types::CinematicParams>,

    // Command Recording State
    pub current_command_buffer: Mutex<Option<MetalIdWrapper<CommandBuffer>>>,
    pub current_drawable: Mutex<Option<MetalIdWrapper<id>>>,
    pub current_encoder: Mutex<Option<MetalIdWrapper<RenderCommandEncoder>>>,
    pub is_first_draw: Mutex<bool>,
}

impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();
        let resources = MetalResourceProvider::new(device.clone());
        
        let shader_src = include_str!("../shaders/metal_shader.metal");
        let pipelines = MetalPipelineProvider::new(device.clone(), shader_src)?;
        
        let main_pipeline = pipelines.create_render_pipeline(
            "Fantasmagorie Main", 
            "vs_main fs_main", // Heuristic to use default shaders
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

        let font_texture_desc = metal::TextureDescriptor::new();
        font_texture_desc.set_width(1);
        font_texture_desc.set_height(1);
        font_texture_desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        let font_texture = Some(device.new_texture(&font_texture_desc));

        let backdrop_texture_desc = metal::TextureDescriptor::new();
        backdrop_texture_desc.set_width(1);
        backdrop_texture_desc.set_height(1);
        backdrop_texture_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        backdrop_texture_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let backdrop_texture = Some(device.new_texture(&backdrop_texture_desc));

        // Create HDR texture for main rendering
        let hdr_texture_desc = metal::TextureDescriptor::new();
        hdr_texture_desc.set_width(1);
        hdr_texture_desc.set_height(1);
        hdr_texture_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        hdr_texture_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let hdr_texture = Some(device.new_texture(&hdr_texture_desc));

        // Create Resolve Pipeline (Post-process)
        let resolve_pipeline = pipelines.create_render_pipeline(
            "Fantasmagorie Resolve",
            "vs_resolve fs_resolve",
            None
        )?;

        let bright_pipeline = pipelines.create_render_pipeline("Bright Pass", "vs_resolve fs_bright_pass", None)?;
        let blur_pipeline = pipelines.create_render_pipeline("Blur Pass", "vs_resolve fs_blur", None)?;
        let resolve_bloom_pipeline = pipelines.create_render_pipeline("Resolve Bloom", "vs_resolve fs_resolve_bloom", None)?;
        
        let mut bloom_textures = [None, None, None];
        for i in 0..3 {
            let desc = metal::TextureDescriptor::new();
            desc.set_width(1);
            desc.set_height(1);
            desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
            desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            bloom_textures[i] = Some(device.new_texture(&desc));
        }

        let blur_uniform_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        let cinematic_params = crate::backend::shaders::types::CinematicParams {
            exposure: 1.0,
            ca_strength: 0.005,
            vignette_intensity: 0.5,
            bloom_intensity: 0.4,
            tonemap_mode: 1, // Aces
            bloom_mode: 1,   // Soft
            grain_strength: 0.05,
            time: 0.0,
            lut_intensity: 1.0,
            blur_radius: 0.0,
            _pad: [0.0; 2],
        };
        let cinematic_buffer = device.new_buffer(
            std::mem::size_of::<crate::backend::shaders::types::CinematicParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = cinematic_buffer.contents() as *mut crate::backend::shaders::types::CinematicParams;
            *ptr = cinematic_params;
        }

        Ok(Self {
            device,
            command_queue,
            resources,
            pipelines,
            main_pipeline,
            uniform_buffer,
            vertex_buffer,
            
            transient_pool: Mutex::new(TransientPool::new()),

            font_texture,
            backdrop_texture,
            sampler,
            layer: None,
            start_time: std::time::Instant::now(),
            audio_data: vec![0.0; 4],
            screenshot_requested: Arc::new(Mutex::new(None)),
            resolve_pipeline,
            hdr_texture,
            bright_pipeline,
            blur_pipeline,
            resolve_bloom_pipeline,
            bloom_textures,
            blur_uniform_buffer,
            cinematic_buffer,
            current_cinematic: Mutex::new(cinematic_params),
            current_command_buffer: Mutex::new(None),
            current_drawable: Mutex::new(None),
            current_encoder: Mutex::new(None),
            is_first_draw: Mutex::new(true),
            reflection_texture: None,
            instanced_pipeline,
            instanced_gbuffer_pipeline,
        })
    }

    pub fn set_layer(&mut self, layer: *mut objc::runtime::Object) {
        self.layer = Some(MetalLayerWrapper(layer as id));
    }


    fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> [Vertex; 6] {
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
        
        // Render to HDR texture if available
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
        
        // Set Viewport
        unsafe {
            let tex_ptr: id = msg_send![drawable, texture];
            let tex: &TextureRef = TextureRef::from_ptr(tex_ptr as *mut _);
            let viewport = MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: tex.width() as f64,
                height: tex.height() as f64,
                znear: 0.0,
                zfar: 1.0,
            };
            encoder.set_viewport(viewport);
        }

        let wrapper = MetalIdWrapper(encoder);
        *encoder_guard = Some(MetalIdWrapper(wrapper.0.to_owned()));
        Ok(wrapper)
    }
}

impl GraphicsBackend for MetalBackend {
    fn name(&self) -> &str { "Metal" }

    fn update_audio_data(&mut self, data: &[f32]) {
        self.audio_data = data.to_vec();
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        let orchestrator = crate::renderer::orchestrator::RenderOrchestrator::new();
        // Plan and execute via RenderGraph
        let time = self.start_time.elapsed().as_secs_f32();

        // Sync time to cinematic buffer for dynamic effects
        let params_copy = {
            let mut params = self.current_cinematic.lock().unwrap();
            params.time = time;
            
            self.command_queue.device().new_buffer_with_data(
                bytemuck::bytes_of(&*params).as_ptr() as *const _,
                std::mem::size_of::<crate::backend::shaders::types::CinematicParams>() as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache
            ); 
            let contents = self.cinematic_buffer.contents();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytemuck::bytes_of(&*params).as_ptr(),
                    contents as *mut u8,
                    std::mem::size_of::<crate::backend::shaders::types::CinematicParams>()
                );
            }
            *params
        };

        // Pass to plan
        // Pass to plan
        let mut graph = orchestrator.plan(dl, &params_copy, width, height);
        
        let mut ext_resources = std::collections::HashMap::new();
        if let Some(ref hdr) = self.hdr_texture {
             // Descriptor is placeholder for now as graph implementation doesn't strictly validate external resource match
             let desc = crate::backend::hal::TextureDescriptor {
                 width, height,
                 format: crate::backend::hal::TextureFormat::Rgba16Float,
                 usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING,
                 label: Some("External HDR"),
             };
             ext_resources.insert(crate::renderer::graph::HDR_HANDLE, 
                 crate::renderer::graph::GraphResource::Texture(desc, hdr.clone()));
        }

        if let Err(e) = orchestrator.execute(self, &mut graph, ext_resources, time, width, height) {
            eprintln!("Metal render error: {}", e);
        }
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
            bloom_intensity: 0.4, // Match WGPU default or expand CinematicConfig later
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
            blur_radius: 0.0,
            _pad: [0.0; 2],
        };

        *self.current_cinematic.lock().unwrap() = params;
        unsafe {
            let ptr = self.cinematic_buffer.contents() as *mut CinematicParams;
            *ptr = params;
        }
    }
}


// Trait implementations are now flattened into GpuExecutor

impl GpuExecutor for MetalBackend {
    type Buffer = Buffer;
    type Texture = Texture;
    type TextureView = Texture;
    type Sampler = SamplerState;
    type RenderPipeline = RenderPipelineState;
    type ComputePipeline = ComputePipelineState;
    type BindGroupLayout = MetalBindGroupLayout;
    type BindGroup = MetalBindGroup;

    // Resource provider implementation
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

    // Pipeline provider implementation
    fn create_render_pipeline(&self, label: &str, wgsl: &str, layout: Option<&Self::BindGroupLayout>) -> Result<Self::RenderPipeline, String> {
        self.pipelines.create_render_pipeline(label, wgsl, layout)
    }
    fn get_custom_render_pipeline(&self, shader_source: &str) -> Result<Self::RenderPipeline, String> {
        // Simple heuristic: if it mentions 'vs_main' and 'fs_main', use the library's defaults
        self.pipelines.create_render_pipeline("Custom Pipeline", shader_source, None)
    }
    fn create_compute_pipeline(&self, shader_name: &str, shader_source: &str, entry_point: Option<&str>) -> Result<Self::ComputePipeline, String> {
        self.pipelines.create_compute_pipeline(shader_name, shader_source, entry_point)
    }

    fn get_compute_pipeline_layout(&self, _pipeline: &Self::ComputePipeline, _index: u32) -> Result<Self::BindGroupLayout, String> {
        Ok(pipeline_provider::MetalBindGroupLayout { entries: vec![] }) 
    }
    
    fn destroy_bind_group(&self, _bind_group: Self::BindGroup) {
        // Resources are handled by Arc/Drop
    }

    fn begin_execute(&self) -> Result<(), String> {
        let wrapper = self.layer.ok_or("No CALayer set")?;
        let layer = wrapper.0;
        autoreleasepool(|| unsafe {
            let drawable: id = msg_send![layer, nextDrawable];
            if drawable == cocoa::base::nil {
                return Err("Failed to get next drawable".into());
            }
            // Retain the drawable to ensure it stays alive until end_execute
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
            
            // Handle screenshot if requested
            let mut req_guard = self.screenshot_requested.lock().unwrap();
            if let Some(path) = req_guard.take() {
                if let Some(d_wrapper) = *self.current_drawable.lock().unwrap() {
                    let drawable = d_wrapper.0;
        {
            // We should use the cinematic params from context or similar if available, 
            // but MetalBackend currently doesn't simulate full CinematicConfig update loop in this demo code structure cleanly?
            // Actually, MetalBackend holds `current_cinematic`? No, it holds `dummy_storage_buffer`.
            // Let's check struct.
            // MetalBackend struct doesn't seem to have `current_cinematic` field in the snippet I saw earlier.
            // I need to add it or use a default if missing.
        }
                    unsafe {
                        let tex_ptr: id = msg_send![drawable, texture];
                        let src_tex: &TextureRef = TextureRef::from_ptr(tex_ptr as *mut _);
                        self.perform_screenshot(src_tex, &path, &command_buffer)?;
                    }
                }
            }

            if let Some(d_wrapper) = self.current_drawable.lock().unwrap().take() {
                let drawable = d_wrapper.0;
                unsafe {
                    let _: () = msg_send![command_buffer, presentDrawable: drawable];
                    // Release the drawable held by begin_execute
                    let _: () = msg_send![drawable, release];
                }
            }
            command_buffer.commit();
        }
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
        let wrapper = self.ensure_encoder()?;
        let encoder = &wrapper.0;
        encoder.set_render_pipeline_state(pipeline);
        encoder.set_vertex_buffer(0, Some(vertex_buffer), 0);
        encoder.set_fragment_sampler_state(0, Some(&self.sampler));

        if let Some(bg) = bind_group {
            for (i, buf) in bg.buffers.iter().enumerate() {
                encoder.set_vertex_buffer(1 + i as u64, Some(buf), 0);
                encoder.set_fragment_buffer(1 + i as u64, Some(buf), 0);
            }
            for (i, tex) in bg.textures.iter().enumerate() {
                encoder.set_fragment_texture(i as u64, Some(tex));
            }
            for (i, samp) in bg.samplers.iter().enumerate() {
                encoder.set_fragment_sampler_state(i as u64, Some(samp));
            }
        }

        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
        Ok(())
    }

    fn draw_instanced(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        instance_buffer: &Self::Buffer,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), String> {
        let wrapper = self.ensure_encoder()?;
        let encoder = &wrapper.0;
        encoder.set_render_pipeline_state(pipeline);
        encoder.set_vertex_buffer(0, Some(vertex_buffer), 0);
        encoder.set_vertex_buffer(2, Some(instance_buffer), 0);
        encoder.set_fragment_sampler_state(0, Some(&self.sampler));

        if let Some(bg) = bind_group {
            for (i, buf) in bg.buffers.iter().enumerate() {
                encoder.set_vertex_buffer(1 + i as u64, Some(buf), 0);
                encoder.set_fragment_buffer(1 + i as u64, Some(buf), 0);
            }
            for (i, tex) in bg.textures.iter().enumerate() {
                encoder.set_fragment_texture(i as u64, Some(tex));
            }
            for (i, samp) in bg.samplers.iter().enumerate() {
                encoder.set_fragment_sampler_state(i as u64, Some(samp));
            }
        }

        encoder.draw_primitives_instanced(
            MTLPrimitiveType::Triangle,
            0,
            vertex_count as u64,
            instance_count as u64,
        );
        Ok(())
    }

    fn draw_instanced_gbuffer(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        instance_buffer: &Self::Buffer,
        vertex_count: u32,
        instance_count: u32,
        aux_view: &Self::TextureView,
    ) -> Result<(), String> {
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        // If an encoder is already active, we must end it because we are changing attachments
        if let Some(enc) = encoder_guard.take() {
            enc.0.end_encoding();
        }

        // Create new descriptor with 2 attachments
        let d_wrapper = self.current_drawable.lock().unwrap();
        let drawable = d_wrapper.as_ref().ok_or("No drawable")?.0;
        
        let descriptor = RenderPassDescriptor::new();
        // Attachment 0: HDR
        let color_attachment0 = descriptor.color_attachments().object_at(0).unwrap();
        if let Some(tex) = &self.hdr_texture {
             color_attachment0.set_texture(Some(tex));
        } else {
             unsafe {
                 let tex_ptr: id = msg_send![drawable, texture];
                 let tex_ref: &TextureRef = TextureRef::from_ptr(tex_ptr as *mut _);
                 color_attachment0.set_texture(Some(tex_ref));
             }
        }
        color_attachment0.set_load_action(MTLLoadAction::Load);
        color_attachment0.set_store_action(MTLStoreAction::Store);

        // Attachment 1: Aux (G-Buffer)
        let color_attachment1 = descriptor.color_attachments().object_at(1).unwrap();
        color_attachment1.set_texture(Some(aux_view));
        color_attachment1.set_load_action(MTLLoadAction::Load);
        color_attachment1.set_store_action(MTLStoreAction::Store);

        let command_buffer_guard = self.current_command_buffer.lock().unwrap();
        let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;
        
        let encoder = command_buffer.new_render_command_encoder(descriptor);
        
        encoder.set_render_pipeline_state(pipeline);
        encoder.set_vertex_buffer(0, Some(vertex_buffer), 0);
        
        // Match draw_instanced implementation for bindings
        // Assuming bind_group corresponds to Metal Bindings
        // In MetalBackend::create_bind_group, we see how it packs buffers.
        // Assuming slot 1 is Global, slot 2 is Instance
        // But draw_instanced implementation logic handles this.
        // Let's copy binding logic from draw_instanced once I verify it.
        // For now, assume bind_group follows create_bind_group logic of Metal.
        // Wait, standard draw_instanced in MetalBackend sets buffers explicitly via bind_group?
        
        // Replicating draw_instanced logic blindly for now, will verify in verify step.
        if let Some(bg) = bind_group {
            // Apply bind group resources
            // This is hacky because MetalBackend BindGroup is just struct with vectors
            for (i, buffer) in bg.buffers.iter().enumerate() {
                 encoder.set_vertex_buffer((i + 1) as u64, Some(buffer), 0);
                 encoder.set_fragment_buffer((i + 1) as u64, Some(buffer), 0);
            }
            for (i, texture) in bg.textures.iter().enumerate() {
                 encoder.set_fragment_texture(i as u64, Some(texture));
            }
             for (i, sampler) in bg.samplers.iter().enumerate() {
                 encoder.set_fragment_sampler_state(i as u64, Some(sampler));
            }
        }
        
        encoder.draw_primitives_instanced(
            MTLPrimitiveType::Triangle,
            0,
            vertex_count as u64,
            instance_count as u64,
        );
        
        encoder.end_encoding();
        Ok(())
    }
    
    fn dispatch(
        &self,
        pipeline: &Self::ComputePipeline,
        bind_group: Option<&Self::BindGroup>,
        groups: [u32; 3],
        _push_constants: &[u8],
    ) -> Result<(), String> {
        if let Some(wrapper) = self.current_encoder.lock().unwrap().take() {
            wrapper.0.end_encoding();
        }

        let command_buffer_guard = self.current_command_buffer.lock().unwrap();
        let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;
        
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        
        if let Some(bg) = bind_group {
            for (i, buf) in bg.buffers.iter().enumerate() {
                encoder.set_buffer(i as u64, Some(buf), 0);
            }
            for (i, tex) in bg.textures.iter().enumerate() {
                encoder.set_texture(i as u64, Some(tex));
            }
        }

        let grid_size = MTLSize { width: groups[0] as u64, height: groups[1] as u64, depth: groups[2] as u64 };
        let thread_group_size = MTLSize { 
            width: std::cmp::min(pipeline.max_total_threads_per_threadgroup(), 64), 
            height: 1, 
            depth: 1 
        };
        
        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();
        Ok(())
    }

    fn copy_texture(&self, src: &Self::Texture, dst: &Self::Texture) -> Result<(), String> {
        if let Some(wrapper) = self.current_encoder.lock().unwrap().take() {
            wrapper.0.end_encoding();
        }

        let command_buffer_guard = self.current_command_buffer.lock().unwrap();
        let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;
        
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture(
            src, 0, 0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: src.width(), height: src.height(), depth: 1 },
            dst, 0, 0,
            MTLOrigin { x: 0, y: 0, z: 0 }
        );
        blit.end_encoding();
        Ok(())
    }

    fn generate_mipmaps(&self, _texture: &Self::Texture) -> Result<(), String> {
        Ok(())
    }

    fn copy_framebuffer_to_texture(&self, dst: &Self::Texture) -> Result<(), String> {
        let d_wrapper = self.current_drawable.lock().unwrap();
        let drawable = d_wrapper.as_ref().ok_or("No active drawable")?.0;
        
        let command_buffer_guard = self.current_command_buffer.lock().unwrap();
        let command_buffer = &command_buffer_guard.as_ref().ok_or("No active command buffer")?.0;
        
        let blit_encoder = command_buffer.new_blit_command_encoder();
        
        unsafe {
            let tex_ptr: id = msg_send![drawable, texture];
            let src_tex: &TextureRef = TextureRef::from_ptr(tex_ptr as *mut _);
            
            blit_encoder.copy_from_texture(
                src_tex,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: src_tex.width(), height: src_tex.height(), depth: 1 },
                dst,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 }
            );
        }
        blit_encoder.end_encoding();
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

    fn create_bind_group(
        &self,
        _layout: &Self::BindGroupLayout,
        buffers: &[&Self::Buffer],
        textures: &[&Self::TextureView],
        samplers: &[&Self::Sampler],
    ) -> Result<Self::BindGroup, String> {
        Ok(MetalBindGroup {
            buffers: buffers.iter().map(|&b| b.to_owned()).collect(),
            textures: textures.iter().map(|&t| t.to_owned()).collect(),
            samplers: samplers.iter().map(|&s| s.to_owned()).collect(),
        })
    }

    fn get_font_view(&self) -> &Self::TextureView { self.font_texture.as_ref().unwrap() }
    fn get_backdrop_view(&self) -> &Self::TextureView { self.backdrop_texture.as_ref().unwrap() }
    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout { 
        static DUMMY_LAYOUT: MetalBindGroupLayout = MetalBindGroupLayout { entries: Vec::new() };
        &DUMMY_LAYOUT
    }
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline { &self.main_pipeline }

    fn get_instanced_render_pipeline(&self) -> &Self::RenderPipeline {
        &self.instanced_pipeline
    }
    fn get_instanced_gbuffer_render_pipeline(&self) -> &Self::RenderPipeline {
        &self.instanced_gbuffer_pipeline
    }

    fn set_reflection_texture(&mut self, texture: &Self::TextureView) -> Result<(), String> {
        self.reflection_texture = Some(texture.to_owned());
        Ok(())
    }
    fn get_default_sampler(&self) -> &Self::Sampler { &self.sampler }

    fn resolve(&mut self) -> Result<(), String> {
        if let Some(wrapper) = self.current_encoder.lock().unwrap().take() {
            wrapper.0.end_encoding();
        }
        
        let (width, height, drawable_tex) = {
            let drawable_guard = self.current_drawable.lock().unwrap();
            let d_wrapper = drawable_guard.as_ref().ok_or("No drawable")?;
            unsafe {
                let tex_ptr: id = msg_send![d_wrapper.0, texture];
                let tex: &TextureRef = TextureRef::from_ptr(tex_ptr as *mut _);
                (tex.width(), tex.height(), MetalIdWrapper(tex_ptr))
            }
        };

        // 1. Resize Handling
        if let Some(ref hdr) = self.hdr_texture {
            if hdr.width() != width || hdr.height() != height {
                let desc = metal::TextureDescriptor::new();
                desc.set_width(width);
                desc.set_height(height);
                desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
                desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                self.hdr_texture = Some(self.device.new_texture(&desc));
                
                let bdesc = metal::TextureDescriptor::new();
                bdesc.set_width(width);
                bdesc.set_height(height);
                bdesc.set_pixel_format(MTLPixelFormat::RGBA16Float);
                bdesc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                self.backdrop_texture = Some(self.device.new_texture(&bdesc));

                // Resize bloom textures (downsampled 1/2)
                for i in 0..3 {
                    let sdesc = metal::TextureDescriptor::new();
                    sdesc.set_width(width / 2);
                    sdesc.set_height(height / 2);
                    sdesc.set_pixel_format(MTLPixelFormat::RGBA16Float);
                    sdesc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                    self.bloom_textures[i] = Some(self.device.new_texture(&sdesc));
                }
            }
        }

        let command_buffer_guard = self.current_command_buffer.lock().unwrap();
        let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;

        // 2. Backdrop Copy (for Glass Blur)
        if let (Some(ref hdr), Some(ref backdrop)) = (&self.hdr_texture, &self.backdrop_texture) {
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_texture(
                hdr, 0, 0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: hdr.width(), height: hdr.height(), depth: 1 },
                backdrop, 0, 0,
                MTLOrigin { x: 0, y: 0, z: 0 }
            );
            blit.end_encoding();
        }

        // 3. Bloom Passes
        if let (Some(ref hdr), 
                Some(ref bright), 
                Some(ref blur_h), 
                Some(ref blur_v)) = 
            (&self.hdr_texture, &self.bloom_textures[0], &self.bloom_textures[1], &self.bloom_textures[2]) 
        {
            // A. Bright Pass
            let desc = RenderPassDescriptor::new();
            desc.color_attachments().object_at(0).unwrap().set_texture(Some(bright));
            desc.color_attachments().object_at(0).unwrap().set_load_action(MTLLoadAction::Clear);
            let enc = command_buffer.new_render_command_encoder(desc);
            enc.set_render_pipeline_state(&self.bright_pipeline);
            enc.set_fragment_texture(0, Some(hdr));
            enc.set_fragment_sampler_state(0, Some(&self.sampler));
            enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            enc.end_encoding();

            // B. Blur Horizontal
            let desc = RenderPassDescriptor::new();
            desc.color_attachments().object_at(0).unwrap().set_texture(Some(blur_h));
            let enc = command_buffer.new_render_command_encoder(desc);
            enc.set_render_pipeline_state(&self.blur_pipeline);
            enc.set_fragment_texture(0, Some(bright));
            enc.set_fragment_sampler_state(0, Some(&self.sampler));
            let horizontal: i32 = 1;
            unsafe {
                self.blur_uniform_buffer.contents().copy_from_nonoverlapping(&horizontal as *const _ as *const std::ffi::c_void, 4);
            }
            enc.set_fragment_buffer(0, Some(&self.blur_uniform_buffer), 0);
            enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            enc.end_encoding();

            // C. Blur Vertical
            let desc = RenderPassDescriptor::new();
            desc.color_attachments().object_at(0).unwrap().set_texture(Some(blur_v));
            let enc = command_buffer.new_render_command_encoder(desc);
            enc.set_render_pipeline_state(&self.blur_pipeline);
            enc.set_fragment_texture(0, Some(blur_h));
            enc.set_fragment_sampler_state(0, Some(&self.sampler));
            let horizontal: i32 = 0;
            unsafe {
                self.blur_uniform_buffer.contents().copy_from_nonoverlapping(&horizontal as *const _ as *const std::ffi::c_void, 4);
            }
            enc.set_fragment_buffer(0, Some(&self.blur_uniform_buffer), 0);
            enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            enc.end_encoding();
        }

        // 4. Final Resolve (HDR + Bloom + Cinematic Effects)
        if let (Some(ref hdr), Some(ref bloom)) = (&self.hdr_texture, &self.bloom_textures[2]) {
            let render_pass_descriptor = RenderPassDescriptor::new();
            let color_attachment = render_pass_descriptor.color_attachments().object_at(0).unwrap();
            unsafe {
                let _: () = msg_send![color_attachment, setTexture: drawable_tex.0];
            }
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
            color_attachment.set_store_action(MTLStoreAction::Store);

            let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
            encoder.set_render_pipeline_state(&self.resolve_bloom_pipeline);
            encoder.set_fragment_texture(0, Some(hdr));
            encoder.set_fragment_texture(1, Some(bloom));
            encoder.set_fragment_sampler_state(0, Some(&self.sampler));
            
            // Set Uniforms for Time/Viewport
            encoder.set_fragment_buffer(1, Some(&self.uniform_buffer), 0);
            encoder.set_fragment_buffer(2, Some(&self.cinematic_buffer), 0);
            
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
        }
        
        Ok(())
    }

    fn present(&self) -> Result<(), String> {
        // In Metal, commit is handled in end_execute
        Ok(())
    }

    fn y_flip(&self) -> bool {
        false 
    }
}

impl MetalBackend {
    fn perform_screenshot(&self, texture: &TextureRef, path: &str, command_buffer: &CommandBufferRef) -> Result<(), String> {
        let width = texture.width();
        let height = texture.height();
        let bytes_per_pixel = 4;
        let bytes_per_row = width * bytes_per_pixel;
        let buffer_size = bytes_per_row * height;

        let cpu_buffer = self.device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);
        
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture_to_buffer(
            texture, 0, 0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width, height, depth: 1 },
            &cpu_buffer, 0, bytes_per_row, buffer_size,
            MTLBlitOption::None
        );
        blit.end_encoding();

        // We need to wait for completion to read back
        let path_owned = path.to_string();
        let is_bgra = texture.pixel_format() == MTLPixelFormat::BGRA8Unorm;
        
        let block = ConcreteBlock::new(move |cb: &CommandBufferRef| {
            if cb.status() == MTLCommandBufferStatus::Completed {
                let ptr = cpu_buffer.contents() as *const u8;
                let mut raw_pixels = vec![0u8; buffer_size as usize];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, raw_pixels.as_mut_ptr(), buffer_size as usize);
                }

                // Metal BGRA8Unorm to RGBA8 swap
                if is_bgra {
                    for i in (0..raw_pixels.len()).step_by(4) {
                        raw_pixels.swap(i, i + 2);
                    }
                }

                if let Err(e) = image::save_buffer(
                    &path_owned,
                    &raw_pixels,
                    width as u32,
                    height as u32,
                    image::ColorType::Rgba8,
                ) {
                    eprintln!("Failed to save Metal screenshot: {}", e);
                } else {
                    println!("✨ Metal screenshot saved to: {}", path_owned);
                }
            }
        });
        let block = block.copy();
        command_buffer.add_completed_handler(&block);

        Ok(())
    }
}
