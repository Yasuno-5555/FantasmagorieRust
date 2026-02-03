use metal::*;
use metal::foreign_types::ForeignTypeRef;
use std::sync::{Arc, Mutex};
use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList, DrawCommand::*};
use crate::backend::GraphicsBackend;
use crate::backend::shaders::types::{DrawUniforms, create_projection};
use cocoa::base::id;
use objc::{msg_send, sel, sel_impl, rc::autoreleasepool};

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
    uniform_buffer: Buffer,
    vertex_buffer: Buffer,
    
    font_texture: Option<Texture>,
    backdrop_texture: Option<Texture>,
    pub sampler: SamplerState,
    pub layer: Option<MetalLayerWrapper>,
    pub start_time: std::time::Instant,
    pub audio_data: Vec<f32>,
    pub screenshot_path: Arc<Mutex<Option<String>>>,

    // Command Recording State
    pub current_command_buffer: Mutex<Option<MetalIdWrapper<CommandBuffer>>>,
    pub current_drawable: Mutex<Option<MetalIdWrapper<id>>>,
    pub current_encoder: Mutex<Option<MetalIdWrapper<RenderCommandEncoder>>>,
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

        let uniform_buffer = device.new_buffer(
            std::mem::size_of::<DrawUniforms>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let vertex_buffer = device.new_buffer(
            (1024 * std::mem::size_of::<Vertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let sampler_desc = SamplerDescriptor::new();
        sampler_desc.set_min_filter(MTLSamplerMinMagFilter::Nearest);
        sampler_desc.set_mag_filter(MTLSamplerMinMagFilter::Nearest);
        let sampler = device.new_sampler(&sampler_desc);

        let font_texture_desc = metal::TextureDescriptor::new();
        font_texture_desc.set_width(1);
        font_texture_desc.set_height(1);
        font_texture_desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        let font_texture = Some(device.new_texture(&font_texture_desc));

        let backdrop_texture_desc = metal::TextureDescriptor::new();
        backdrop_texture_desc.set_width(1);
        backdrop_texture_desc.set_height(1);
        backdrop_texture_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let backdrop_texture = Some(device.new_texture(&backdrop_texture_desc));

        Ok(Self {
            device,
            command_queue,
            resources,
            pipelines,
            main_pipeline,
            uniform_buffer,
            vertex_buffer,
            font_texture,
            backdrop_texture,
            sampler,
            layer: None,
            start_time: std::time::Instant::now(),
            audio_data: vec![0.0; 4],
            screenshot_path: Arc::new(Mutex::new(None)),
            current_command_buffer: Mutex::new(None),
            current_drawable: Mutex::new(None),
            current_encoder: Mutex::new(None),
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
}

impl GraphicsBackend for MetalBackend {
    fn name(&self) -> &str { "Metal" }

    fn update_audio_data(&mut self, data: &[f32]) {
        self.audio_data = data.to_vec();
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        let orchestrator = crate::renderer::orchestrator::RenderOrchestrator::new();
        let tasks = orchestrator.plan(dl);
        let time = self.start_time.elapsed().as_secs_f32();
        if let Err(e) = orchestrator.execute(self, &tasks, time, width, height) {
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
        *self.screenshot_path.lock().unwrap() = Some(path.to_string());
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
    fn create_compute_pipeline(&self, label: &str, wgsl: &str, layout: Option<&Self::BindGroupLayout>) -> Result<Self::ComputePipeline, String> {
        self.pipelines.create_compute_pipeline(label, wgsl, layout)
    }
    fn destroy_bind_group(&self, bind_group: Self::BindGroup) {
        self.pipelines.destroy_bind_group(bind_group)
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
        let mut encoder_guard = self.current_encoder.lock().unwrap();
        if encoder_guard.is_none() {
            let d_wrapper = self.current_drawable.lock().unwrap();
            let drawable = d_wrapper.as_ref().ok_or("No drawable for encoder")?.0;
            let command_buffer_guard = self.current_command_buffer.lock().unwrap();
            let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;
            
            let render_pass_descriptor = RenderPassDescriptor::new();
            let color_attachment = render_pass_descriptor.color_attachments().object_at(0).unwrap();
            
            unsafe {
                let tex_ptr: id = msg_send![drawable, texture];
                let _: () = msg_send![color_attachment, setTexture: tex_ptr];
            }
            
            color_attachment.set_load_action(MTLLoadAction::Load);
            color_attachment.set_store_action(MTLStoreAction::Store);

            let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor).to_owned();
            *encoder_guard = Some(MetalIdWrapper(encoder));
        }

        let encoder = &encoder_guard.as_ref().unwrap().0;
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

    fn dispatch(
        &self,
        _pipeline: &Self::ComputePipeline,
        _bind_group: Option<&Self::BindGroup>,
        _groups: [u32; 3],
        _push_constants: &[u8],
    ) -> Result<(), String> {
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
    fn get_default_sampler(&self) -> &Self::Sampler { &self.sampler }

    fn resolve(&mut self) -> Result<(), String> {
        if let Some(wrapper) = self.current_encoder.lock().unwrap().take() {
            wrapper.0.end_encoding();
        }
        
        let d_wrapper_opt = *self.current_drawable.lock().unwrap();
        if let (Some(d_wrapper), Some(ref backdrop)) = (d_wrapper_opt, &self.backdrop_texture) {
            let drawable_ptr = d_wrapper.0;
            unsafe {
                let tex_ptr: id = msg_send![drawable_ptr, texture];
                let src_tex: &TextureRef = TextureRef::from_ptr(tex_ptr as *mut _);
                
                let command_buffer_guard = self.current_command_buffer.lock().unwrap();
                let command_buffer = &command_buffer_guard.as_ref().ok_or("No command buffer")?.0;
                
                let blit = command_buffer.new_blit_command_encoder();
                blit.copy_from_texture(
                    src_tex, 0, 0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize { width: backdrop.width(), height: backdrop.height(), depth: 1 },
                    backdrop, 0, 0,
                    MTLOrigin { x: 0, y: 0, z: 0 }
                );
                blit.end_encoding();
            }
        }
        
        Ok(())
    }

    fn present(&self) -> Result<(), String> {
        Ok(())
    }

    fn y_flip(&self) -> bool {
        false 
    }
}
