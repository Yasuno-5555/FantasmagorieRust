use metal::*;
use std::sync::{Arc, Mutex};
use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList, DrawCommand::*};
use crate::backend::GraphicsBackend;
use crate::backend::shaders::types::{DrawUniforms, create_projection};
use cocoa::base::id;
use objc::{msg_send, sel, sel_impl, rc::autoreleasepool};

pub mod resource_provider;
pub mod pipeline_provider;

use resource_provider::MetalResourceProvider;
use pipeline_provider::MetalPipelineProvider;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

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
    pub layer: Option<id>,
    pub start_time: std::time::Instant,
    pub audio_data: Vec<f32>,
    pub screenshot_path: Arc<Mutex<Option<String>>>,
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
            "vs_main", 
            "fs_main", 
            MTLPixelFormat::BGRA8Unorm // Standard macOS swapchain format
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

        Ok(Self {
            device,
            command_queue,
            resources,
            pipelines,
            main_pipeline,
            uniform_buffer,
            vertex_buffer,
            font_texture: None,
            backdrop_texture: None,
            sampler,
            layer: None,
            start_time: std::time::Instant::now(),
            audio_data: vec![0.0; 4],
            screenshot_path: Arc::new(Mutex::new(None)),
        })
    }

    pub fn set_layer(&mut self, layer: *mut objc::runtime::Object) {
        self.layer = Some(layer as id);
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
        let layer = if let Some(l) = self.layer {
            l
        } else {
            return;
        };

        autoreleasepool(|| unsafe {
            let drawable: id = msg_send![layer, nextDrawable];
            if drawable == cocoa::base::nil { return; }

            let render_pass_descriptor = RenderPassDescriptor::new();
            let color_attachment = render_pass_descriptor.color_attachments().object_at(0).unwrap();
            
            let tex_ptr: id = msg_send![drawable, texture];
            let _: () = msg_send![color_attachment, setTexture: tex_ptr];
            
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_clear_color(MTLClearColor::new(0.01, 0.01, 0.02, 1.0));
            color_attachment.set_store_action(MTLStoreAction::Store);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
            encoder.set_render_pipeline_state(&self.main_pipeline);
            encoder.set_fragment_sampler_state(0, Some(&self.sampler));

            let proj = create_projection(width as f32, height as f32, true, (0.0, 1.0));
            let mut uniforms = DrawUniforms {
                projection: std::mem::transmute(proj),
                rect: [0.0; 4],
                radii: [0.0; 4],
                border_color: [0.0; 4],
                glow_color: [0.0; 4],
                offset: [0.0; 2],
                scale: 1.0,
                border_width: 0.0,
                elevation: 0.0,
                glow_strength: 0.0,
                lut_intensity: 0.0,
                mode: 0,
                is_squircle: 0,
                time: self.start_time.elapsed().as_secs_f32(),
                viewport_size: [width as f32, height as f32],
            };

            let mut active_encoder = Some(encoder);

            for cmd in dl.commands() {
                match cmd {
                    RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, .. } => {
                        let encoder = active_encoder.as_ref().unwrap();
                        uniforms.mode = 2;
                        uniforms.rect = [pos.x, pos.y, size.x, size.y];
                        uniforms.radii = *radii;
                        uniforms.border_width = *border_width;
                        uniforms.border_color = [border_color.r, border_color.g, border_color.b, border_color.a];
                        uniforms.elevation = *elevation;
                        uniforms.is_squircle = if *is_squircle { 1 } else { 0 };
                        uniforms.glow_strength = *glow_strength;
                        uniforms.glow_color = [glow_color.r, glow_color.g, glow_color.b, glow_color.a];
                        
                        let verts = Self::quad_vertices(*pos, *size, *color);
                        encoder.set_vertex_bytes(0, std::mem::size_of_val(&verts[..]) as u64, verts.as_ptr() as *const _);
                        encoder.set_vertex_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                        encoder.set_fragment_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);
                    }
                    Text { pos, size, uv, color } => {
                        if let Some(ref tex) = self.font_texture {
                            let encoder = active_encoder.as_ref().unwrap();
                            uniforms.mode = 1;
                            let verts = [
                                Vertex { pos: [pos.x, pos.y], uv: [uv[0], uv[1]], color: [color.r, color.g, color.b, color.a] },
                                Vertex { pos: [pos.x, pos.y + size.y], uv: [uv[0], uv[3]], color: [color.r, color.g, color.b, color.a] },
                                Vertex { pos: [pos.x + size.x, pos.y + size.y], uv: [uv[2], uv[3]], color: [color.r, color.g, color.b, color.a] },
                                Vertex { pos: [pos.x, pos.y], uv: [uv[0], uv[1]], color: [color.r, color.g, color.b, color.a] },
                                Vertex { pos: [pos.x + size.x, pos.y + size.y], uv: [uv[2], uv[3]], color: [color.r, color.g, color.b, color.a] },
                                Vertex { pos: [pos.x + size.x, pos.y], uv: [uv[2], uv[1]], color: [color.r, color.g, color.b, color.a] },
                            ];
                            encoder.set_vertex_bytes(0, std::mem::size_of_val(&verts) as u64, verts.as_ptr() as *const _);
                            encoder.set_vertex_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                            encoder.set_fragment_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                            encoder.set_fragment_texture(0, Some(tex));
                            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);
                        }
                    }
                    Aurora { pos, size } => {
                        let encoder = active_encoder.as_ref().unwrap();
                        uniforms.mode = 9;
                        uniforms.rect = [pos.x, pos.y, size.x, size.y];
                        let verts = Self::quad_vertices(*pos, *size, ColorF::WHITE);
                        encoder.set_vertex_bytes(0, std::mem::size_of_val(&verts[..]) as u64, verts.as_ptr() as *const _);
                        encoder.set_vertex_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                        encoder.set_fragment_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);
                    }
                    BlurRect { pos, size, radii, sigma, .. } => {
                        // 1. End current encoder
                        if let Some(enc) = active_encoder.take() {
                            enc.end_encoding();
                        }

                        // 2. Capture background to backdrop_texture
                        let tex_ptr: id = msg_send![drawable, texture];
                        let src_tex: &TextureRef = std::mem::transmute(tex_ptr);
                        
                        // Ensure backdrop texture exists and matches size
                        if self.backdrop_texture.is_none() || 
                           self.backdrop_texture.as_ref().unwrap().width() != width as u64 ||
                           self.backdrop_texture.as_ref().unwrap().height() != height as u64 {
                            let desc = TextureDescriptor::new();
                            desc.set_pixel_format(src_tex.pixel_format());
                            desc.set_width(width as u64);
                            desc.set_height(height as u64);
                            desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite | MTLTextureUsage::RenderTarget);
                            self.backdrop_texture = Some(self.device.new_texture(&desc));
                        }

                        if let Some(ref dst_tex) = self.backdrop_texture {
                            let blit = command_buffer.new_blit_command_encoder();
                            blit.copy_from_texture(
                                src_tex, 0, 0,
                                MTLOrigin { x: 0, y: 0, z: 0 },
                                MTLSize { width: width as u64, height: height as u64, depth: 1 },
                                dst_tex, 0, 0,
                                MTLOrigin { x: 0, y: 0, z: 0 }
                            );
                            blit.end_encoding();
                        }

                        // 3. Start new encoder (LoadAction::Load to keep previous drawing)
                        let next_desc = RenderPassDescriptor::new();
                        let next_ca = next_desc.color_attachments().object_at(0).unwrap();
                        let _: () = msg_send![next_ca, setTexture: tex_ptr];
                        next_ca.set_load_action(MTLLoadAction::Load);
                        next_ca.set_store_action(MTLStoreAction::Store);
                        
                        let next_encoder = command_buffer.new_render_command_encoder(next_desc);
                        next_encoder.set_render_pipeline_state(&self.main_pipeline);
                        next_encoder.set_fragment_sampler_state(0, Some(&self.sampler));
                        if let Some(ref backdrop) = self.backdrop_texture {
                            next_encoder.set_fragment_texture(1, Some(backdrop));
                        }

                        // 4. Draw the BlurRect
                        uniforms.mode = 4;
                        uniforms.rect = [pos.x, pos.y, size.x, size.y];
                        uniforms.radii = *radii;
                        uniforms.elevation = *sigma;
                        let verts = Self::quad_vertices(*pos, *size, ColorF::WHITE);
                        next_encoder.set_vertex_bytes(0, std::mem::size_of_val(&verts[..]) as u64, verts.as_ptr() as *const _);
                        next_encoder.set_vertex_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                        next_encoder.set_fragment_bytes(1, std::mem::size_of::<DrawUniforms>() as u64, &uniforms as *const DrawUniforms as *const _);
                        next_encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);

                        active_encoder = Some(next_encoder);
                    }
                    _ => {}
                }
            }

            if let Some(enc) = active_encoder {
                enc.end_encoding();
            }

            let screenshot_req = self.screenshot_path.lock().unwrap().take();
            if let Some(path) = screenshot_req {
                let blit_encoder = command_buffer.new_blit_command_encoder();
                let tex_ptr: id = msg_send![drawable, texture];
                let tex: &TextureRef = unsafe { std::mem::transmute(tex_ptr) };
                
                let bytes_per_row = (width * 4) as u64;
                let bytes_total = bytes_per_row * (height as u64);
                let staging_buffer = self.device.new_buffer(bytes_total, MTLResourceOptions::StorageModeShared);

                blit_encoder.copy_from_texture_to_buffer(
                    tex,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize { width: width as u64, height: height as u64, depth: 1 },
                    &staging_buffer,
                    0,
                    bytes_per_row,
                    bytes_total,
                    MTLBlitOption::None,
                );
                blit_encoder.end_encoding();
                
                let _: () = msg_send![command_buffer, presentDrawable: drawable];
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let ptr = staging_buffer.contents() as *mut u8;
                let len = bytes_total as usize;
                let mut data = vec![0u8; len];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), len);
                }
                
                // BGRA to RGBA and Flip Rows (Needed for Y-up CALayer)
                let mut flipped_data = vec![0u8; len];
                for y in 0..height {
                    let src_y = height - 1 - y;
                    let src_offset = (src_y as usize) * (width as usize) * 4;
                    let dst_offset = (y as usize) * (width as usize) * 4;
                    
                    for x in 0..(width as usize) {
                        let s = src_offset + x * 4;
                        let d = dst_offset + x * 4;
                        flipped_data[d] = data[s + 2];     // R
                        flipped_data[d + 1] = data[s + 1]; // G
                        flipped_data[d + 2] = data[s];     // B
                        flipped_data[d + 3] = data[s + 3]; // A
                    }
                }
                
                // Convert to RGB
                let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
                for i in (0..len).step_by(4) {
                    rgb_data.push(flipped_data[i]);
                    rgb_data.push(flipped_data[i+1]);
                    rgb_data.push(flipped_data[i+2]);
                }
                
                if let Some(img) = image::RgbImage::from_raw(width as u32, height as u32, rgb_data) {
                    let _ = img.save(&path);
                    println!("Screenshot saved (Validated Orientation): {}", path);
                }
            } else {
                let _: () = msg_send![command_buffer, presentDrawable: drawable];
                command_buffer.commit();
            }
        });
    }

    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        let desc = TextureDescriptor::new();
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
