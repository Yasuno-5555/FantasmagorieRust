use crate::backend::hal::{GpuExecutor, BufferUsage};
use crate::renderer::graph::{RenderNode, RenderContext};
use crate::draw::{DrawCommand};
use crate::backend::shaders::types::{DrawUniforms, GlobalUniforms, ShapeInstance, create_projection};
use crate::core::{ColorF, Vec2};
use bytemuck::{Pod, Zeroable};

pub struct GeometryNode {
    pub commands: Vec<DrawCommand>,
    pub batching_enabled: bool,
}

impl GeometryNode {
    pub fn new(commands: Vec<DrawCommand>) -> Self {
        Self { commands, batching_enabled: true }
    }
    
    pub fn with_batching(mut self, enabled: bool) -> Self {
        self.batching_enabled = enabled;
        self
    }
}

impl<E: GpuExecutor> RenderNode<E> for GeometryNode {
    fn name(&self) -> &str { "GeometryPass" }

    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let executor = &mut ctx.executor;
        let time = ctx.time;
        let width = ctx.width;
        let height = ctx.height;

        let pipeline = executor.get_default_render_pipeline();
        let instanced_pipeline = executor.get_instanced_render_pipeline();
        let layout = executor.get_default_bind_group_layout();
        let sampler = executor.get_default_sampler();
        let font_view = executor.get_font_view();
        let backdrop_view = executor.get_backdrop_view();

        let proj_matrix = create_projection(width as f32, height as f32, executor.y_flip(), (-1.0, 1.0));
        let proj: [f32; 16] = *bytemuck::cast_ref(&proj_matrix);

        let mut i = 0;
        while i < self.commands.len() {
            let cmd = &self.commands[i];
            
            match cmd {
                DrawCommand::Text { pos, size, uv, color } => {
                    // Logic moved from orchestrator
                    let mut batched_verts = quad_vertices_uv(*pos, *size, *uv, *color);
                    let mut j = if self.batching_enabled { i + 1 } else { i + 1 };
                    if self.batching_enabled {
                        while j < self.commands.len() {
                            if let DrawCommand::Text { pos, size, uv, color } = &self.commands[j] {
                                batched_verts.extend(quad_vertices_uv(*pos, *size, *uv, *color));
                                j += 1;
                            } else { break; }
                        }
                    }
                    let uniforms = DrawUniforms {
                        projection: *bytemuck::cast_ref(&proj),
                        mode: 1, time, viewport_size: [width as f32, height as f32],
                        ..Zeroable::zeroed()
                    };
                    let u_buf = executor.create_buffer(bytemuck::bytes_of(&uniforms).len() as u64, BufferUsage::Uniform, "TextU")?;
                    executor.write_buffer(&u_buf, 0, bytemuck::bytes_of(&uniforms));
                    let v_buf = executor.create_buffer(batched_verts.len() as u64, BufferUsage::Vertex, "TextV")?;
                    executor.write_buffer(&v_buf, 0, &batched_verts);
                    let bg = executor.create_bind_group(layout, &[&u_buf], &[font_view, backdrop_view], &[sampler])?;
                    executor.draw(pipeline, Some(&bg), &v_buf, ((j - i) * 6) as u32, &[])?;
                    executor.destroy_bind_group(bg); executor.destroy_buffer(u_buf); executor.destroy_buffer(v_buf);
                    i = j;
                }
                DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, shader_inject, .. } => {
                    if shader_inject.is_some() {
                        // Fallback logic for injected shaders
                        let (uniforms, verts, custom_pipeline): (DrawUniforms, Vec<u8>, Option<E::RenderPipeline>) = {
                            let mut u: DrawUniforms = Zeroable::zeroed();
                            u.projection = *bytemuck::cast_ref(&proj); u.rect = [pos.x, pos.y, size.x, size.y];
                            u.radii = *radii; u.border_color = [border_color.r, border_color.g, border_color.b, border_color.a];
                            u.glow_color = [glow_color.r, glow_color.g, glow_color.b, glow_color.a];
                            u.border_width = *border_width; u.elevation = *elevation; u.glow_strength = *glow_strength;
                            u.mode = 2; u.is_squircle = if *is_squircle { 1 } else { 0 };
                            u.time = time; u.viewport_size = [width as f32, height as f32];
                            
                            let mut cp: Option<E::RenderPipeline> = None;
                            if let Some(source) = shader_inject {
                                if let Ok(p) = executor.get_custom_render_pipeline(source) {
                                    cp = Some(p); u.mode = 100;
                                }
                            }
                            (u, quad_vertices(*pos, *size, *color), cp)
                        };

                        let p = custom_pipeline.as_ref().unwrap_or(pipeline);
                        let u_buf = executor.create_buffer(bytemuck::bytes_of(&uniforms).len() as u64, BufferUsage::Uniform, "FallbackU")?;
                        executor.write_buffer(&u_buf, 0, bytemuck::bytes_of(&uniforms));
                        let v_buf = executor.create_buffer(verts.len() as u64, BufferUsage::Vertex, "FallbackV")?;
                        executor.write_buffer(&v_buf, 0, &verts);
                        let bg = executor.create_bind_group(layout, &[&u_buf], &[font_view, backdrop_view], &[sampler])?;
                        executor.draw(p, Some(&bg), &v_buf, 6, &[])?;
                        executor.destroy_bind_group(bg); executor.destroy_buffer(u_buf); executor.destroy_buffer(v_buf);
                        i += 1;
                    } else {
                        // Instanced path
                        let mut instances = Vec::new();
                        let mut j = i;
                        if self.batching_enabled {
                            while j < self.commands.len() {
                                if let DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, shader_inject, .. } = &self.commands[j] {
                                    if shader_inject.is_some() { break; }
                                    instances.push(ShapeInstance {
                                        rect: [pos.x, pos.y, size.x, size.y],
                                        radii: *radii,
                                        border_color: [border_color.r, border_color.g, border_color.b, border_color.a],
                                        glow_color: [glow_color.r, glow_color.g, glow_color.b, glow_color.a],
                                        params1: [*border_width, *elevation, *glow_strength, 0.0],
                                        params2: [2, if *is_squircle { 1 } else { 0 }, 0, 0],
                                    });
                                    j += 1;
                                } else { break; }
                            }
                        } else {
                            instances.push(ShapeInstance {
                                rect: [pos.x, pos.y, size.x, size.y],
                                radii: *radii,
                                border_color: [border_color.r, border_color.g, border_color.b, border_color.a],
                                glow_color: [glow_color.r, glow_color.g, glow_color.b, glow_color.a],
                                params1: [*border_width, *elevation, *glow_strength, 0.0],
                                params2: [2, if *is_squircle { 1 } else { 0 }, 0, 0],
                            });
                            j += 1;
                        }
                        
                        let global = GlobalUniforms {
                            projection: *bytemuck::cast_ref(&proj),
                            time,
                            viewport_size: [width as f32, height as f32],
                            _pad: 0.0,
                        };
                        let ui_quad = unit_quad();
                        let g_buf = executor.create_buffer(bytemuck::bytes_of(&global).len() as u64, BufferUsage::Uniform, "GlobalU")?;
                        executor.write_buffer(&g_buf, 0, bytemuck::bytes_of(&global));
                        let inst_buf = executor.create_buffer((instances.len() * std::mem::size_of::<ShapeInstance>()) as u64, BufferUsage::Storage, "InstU")?;
                        executor.write_buffer(&inst_buf, 0, bytemuck::cast_slice(&instances));
                        let v_buf = executor.create_buffer(ui_quad.len() as u64, BufferUsage::Vertex, "QuadV")?;
                        executor.write_buffer(&v_buf, 0, &ui_quad);
                        let bg = executor.create_bind_group(layout, &[&g_buf, &inst_buf], &[font_view, backdrop_view], &[sampler])?;
                        executor.draw_instanced(instanced_pipeline, Some(&bg), &v_buf, &inst_buf, 6, instances.len() as u32)?;
                        executor.destroy_bind_group(bg); executor.destroy_buffer(g_buf); executor.destroy_buffer(inst_buf); executor.destroy_buffer(v_buf);
                        i = j;
                    }
                }
                _ => {
                    // Other commands fallback
                    i += 1;
                }
            }
        }

        Ok(())
    }
}

// Helpers (Temporary duplicate from orchestrator until we unify)
fn quad_vertices_uv(pos: Vec2, size: Vec2, uv: [f32; 4], color: ColorF) -> Vec<u8> {
    let mut v = Vec::new();
    let x = pos.x; let y = pos.y; let w = size.x; let h = size.y;
    let ux = uv[0]; let uy = uv[1]; let uw = uv[2]; let uh = uv[3];
    let c = [color.r, color.g, color.b, color.a];

    // T1
    v.extend_from_slice(bytemuck::bytes_of(&[x, y, ux, uy])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x+w, y, ux+uw, uy])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x, y+h, ux, uy+uh])); v.extend_from_slice(bytemuck::bytes_of(&c));
    // T2
    v.extend_from_slice(bytemuck::bytes_of(&[x+w, y, ux+uw, uy])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x+w, y+h, ux+uw, uy+uh])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x, y+h, ux, uy+uh])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v
}

fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> Vec<u8> {
    let mut v = Vec::new();
    let x = pos.x; let y = pos.y; let w = size.x; let h = size.y;
    let c = [color.r, color.g, color.b, color.a];
    v.extend_from_slice(bytemuck::bytes_of(&[x, y, 0.0, 0.0])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x+w, y, 1.0, 0.0])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x, y+h, 0.0, 1.0])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x+w, y, 1.0, 0.0])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x+w, y+h, 1.0, 1.0])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[x, y+h, 0.0, 1.0])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v
}

fn unit_quad() -> Vec<u8> {
    let mut v = Vec::new();
    let c = [1.0, 1.0, 1.0, 1.0];
    v.extend_from_slice(bytemuck::bytes_of(&[0.0f32, 0.0f32, 0.0f32, 0.0f32])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[1.0f32, 0.0f32, 1.0f32, 0.0f32])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[0.0f32, 1.0f32, 0.0f32, 1.0f32])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[1.0f32, 0.0f32, 1.0f32, 0.0f32])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[1.0f32, 1.0f32, 1.0f32, 1.0f32])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v.extend_from_slice(bytemuck::bytes_of(&[0.0f32, 1.0f32, 0.0f32, 1.0f32])); v.extend_from_slice(bytemuck::bytes_of(&c));
    v
}
