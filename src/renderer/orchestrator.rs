pub use crate::backend::hal::FantaRenderTask;
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use crate::draw::{DrawCommand, DrawList};
use crate::backend::shaders::types::{DrawUniforms, create_projection};
use crate::core::{ColorF, Vec2};
use bytemuck; // Ensure bytemuck is available

/// Coordinates the execution of RenderTasks across a GpuExecutor
pub struct RenderOrchestrator {
    // Current state, caching, etc.
}

impl RenderOrchestrator {
    pub fn new() -> Self {
        Self {}
    }

    /// Convert a high-level DrawList into an optimized sequence of RenderTasks
    pub fn plan(&self, dl: &DrawList) -> Vec<FantaRenderTask> {
        let mut tasks = Vec::new();
        let commands = dl.commands();
        
        let mut current_batch = Vec::new();
        let mut backdrop_captured = false;

        for cmd in commands {
            match cmd {
                DrawCommand::BackdropBlur { .. } | DrawCommand::BlurRect { .. } => {
                    // Before any blur, we must ensure backdrop is captured
                    if !backdrop_captured {
                        if !current_batch.is_empty() {
                            tasks.push(FantaRenderTask::DrawGeometry { commands: current_batch.drain(..).collect() });
                        }
                        tasks.push(FantaRenderTask::CaptureBackdrop);
                        backdrop_captured = true;
                    }
                    current_batch.push(cmd.clone());
                }
                _ => {
                    current_batch.push(cmd.clone());
                }
            }
        }

        if !current_batch.is_empty() {
            tasks.push(FantaRenderTask::DrawGeometry { commands: current_batch });
        }

        tasks.push(FantaRenderTask::Resolve);
        tasks
    }

    /// Execute a sequence of tasks using a concrete GPU implementation
    pub fn execute<E: GpuExecutor>(&self, executor: &mut E, tasks: &[crate::backend::hal::FantaRenderTask], time: f32, width: u32, height: u32) -> Result<(), String> {
        let mut should_present = false;

        executor.begin_execute()?;
        

        // Helper to create quad vertices
        fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> Vec<u8> {
             let c = [color.r, color.g, color.b, color.a];
             let mut data = Vec::with_capacity(6 * 32);
             let mk_v = |x, y, u, v| {
                 let mut v_data = Vec::new();
                 v_data.extend_from_slice(bytemuck::bytes_of(&[x, y]));
                 v_data.extend_from_slice(bytemuck::bytes_of(&[u, v]));
                 v_data.extend_from_slice(bytemuck::bytes_of(&c));
                 v_data
             };
             
             // Triangle 1: V0, V3, V2 (CCW)
             data.extend(mk_v(pos.x, pos.y, 0.0, 0.0));
             data.extend(mk_v(pos.x, pos.y + size.y, 0.0, 1.0));
             data.extend(mk_v(pos.x + size.x, pos.y + size.y, 1.0, 1.0));
             // Triangle 2: V0, V2, V1 (CCW)
             data.extend(mk_v(pos.x, pos.y, 0.0, 0.0));
             data.extend(mk_v(pos.x + size.x, pos.y + size.y, 1.0, 1.0));
             data.extend(mk_v(pos.x + size.x, pos.y, 1.0, 0.0));
             data
        }

        fn quad_vertices_uv(pos: Vec2, size: Vec2, uv: [f32;4], color: ColorF) -> Vec<u8> {
             let c = [color.r, color.g, color.b, color.a];
             let mut data = Vec::with_capacity(6 * 32);
             let mk_v = |x, y, u, v| {
                 let mut v_data = Vec::new();
                 v_data.extend_from_slice(bytemuck::bytes_of(&[x, y]));
                 v_data.extend_from_slice(bytemuck::bytes_of(&[u, v]));
                 v_data.extend_from_slice(bytemuck::bytes_of(&c));
                 v_data
             };
             
             // Triangle 1: V0, V3, V2 (CCW)
             data.extend(mk_v(pos.x, pos.y, uv[0], uv[1]));
             data.extend(mk_v(pos.x, pos.y + size.y, uv[0], uv[3]));
             data.extend(mk_v(pos.x + size.x, pos.y + size.y, uv[2], uv[3]));
             // Triangle 2: V0, V2, V1 (CCW)
             data.extend(mk_v(pos.x, pos.y, uv[0], uv[1]));
             data.extend(mk_v(pos.x + size.x, pos.y + size.y, uv[2], uv[3]));
             data.extend(mk_v(pos.x + size.x, pos.y, uv[2], uv[1]));
             data
        }

        let proj_matrix = create_projection(width as f32, height as f32, true, (-1.0, 1.0));
        let proj: [f32; 16] = *bytemuck::cast_ref(&proj_matrix);

        for task in tasks {
            match task {
                FantaRenderTask::DrawGeometry { commands } => {
                    let pipeline = executor.get_default_render_pipeline();
                    let layout = executor.get_default_bind_group_layout();
                    let sampler = executor.get_default_sampler();
                    let font_view = executor.get_font_view();
                    let backdrop_view = executor.get_backdrop_view();

                    for cmd in commands {
                         let (uniforms, verts) = match cmd {
                             DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, .. } => {
                                 (DrawUniforms {
                                     projection: proj,
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
                                     time: time,
                                     viewport_size: [width as f32, height as f32],
                                 }, quad_vertices(*pos, *size, *color))
                             }
                             DrawCommand::Text { pos, size, uv, color } => {
                                 (DrawUniforms {
                                     projection: proj,
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
                                     time: time,
                                     viewport_size: [width as f32, height as f32],
                                 }, quad_vertices_uv(*pos, *size, *uv, *color))
                             }
                             DrawCommand::Aurora { pos, size } => {
                                 (DrawUniforms {
                                     projection: proj,
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
                                     time: time,
                                     viewport_size: [width as f32, height as f32],
                                 }, quad_vertices(*pos, *size, ColorF::white()))
                             }
                             _ => continue,
                         };

                         let u_buf = executor.create_buffer(bytemuck::bytes_of(&uniforms).len() as u64, BufferUsage::Uniform, "Uniforms")?;
                         executor.write_buffer(&u_buf, 0, bytemuck::bytes_of(&uniforms));

                         let v_buf = executor.create_buffer(verts.len() as u64, BufferUsage::Vertex, "Vertices")?;
                         executor.write_buffer(&v_buf, 0, &verts);

                         let bg = executor.create_bind_group(
                             layout,
                             &[&u_buf],
                             &[font_view, backdrop_view],
                             &[sampler],
                         )?;

                         executor.draw(pipeline, Some(&bg), &v_buf, 6, &[])?;

                         // Clean up temporary resources
                         executor.destroy_bind_group(bg);
                         executor.destroy_buffer(u_buf);
                         executor.destroy_buffer(v_buf);
                    }
                }
                FantaRenderTask::CaptureBackdrop => {
                    // Logic to capture and generate mipmaps
                }
                FantaRenderTask::ComputeEffect { effect_name, params } => {
                    // Logic to dispatch compute effect
                }
                FantaRenderTask::Resolve => {
                    executor.resolve()?;
                    should_present = true;
                }
            }
        }
        executor.end_execute()?;
        
        if should_present {
            executor.present()?;
        }
        Ok(())
    }
}
