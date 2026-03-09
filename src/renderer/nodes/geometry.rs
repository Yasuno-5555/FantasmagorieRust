use crate::backend::hal::{GpuExecutor, BufferUsage};
use crate::renderer::graph::{RenderNode, RenderContext};
use crate::draw::{DrawCommand};
use crate::backend::shaders::types::{DrawUniforms, GlobalUniforms, ShapeInstance, create_projection};
use crate::core::{ColorF, Vec2};
use bytemuck::Zeroable;

pub struct GeometryNode {
    pub commands: Vec<DrawCommand>,
    pub batching_enabled: bool,
    pub use_gpu_driven: bool,
    pub aux_handle: Option<crate::renderer::graph::ResourceHandle>,
    pub velocity_handle: Option<crate::renderer::graph::ResourceHandle>,
    pub depth_handle: Option<crate::renderer::graph::ResourceHandle>,
}

impl GeometryNode {
    pub fn new(commands: Vec<DrawCommand>) -> Self {
        Self {
            commands,
            batching_enabled: true,
            use_gpu_driven: true, // Default to true for Phase 4
            aux_handle: None,
            velocity_handle: None,
            depth_handle: None,
        }
    }
    
    pub fn with_batching(mut self, enabled: bool) -> Self {
        self.batching_enabled = enabled;
        self
    }

    pub fn with_aux(mut self, handle: crate::renderer::graph::ResourceHandle) -> Self {
        self.aux_handle = Some(handle);
        self
    }

    pub fn with_velocity(mut self, handle: crate::renderer::graph::ResourceHandle) -> Self {
        self.velocity_handle = Some(handle);
        self
    }

    pub fn with_depth(mut self, handle: crate::renderer::graph::ResourceHandle) -> Self {
        self.depth_handle = Some(handle);
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

        let pipeline = executor.get_default_render_pipeline().clone();
        let instanced_pipeline = executor.get_instanced_render_pipeline().clone();
        let layout = executor.get_default_bind_group_layout().clone();
        let instanced_layout = executor.get_instanced_bind_group_layout().clone();
        let sampler = executor.get_default_sampler().clone();
        let font_view = Some(executor.get_font_view().clone());
        let backdrop_view = executor.get_backdrop_view().clone();

        let mut proj_matrix = create_projection(width as f32, height as f32, executor.y_flip(), (-1.0, 1.0));
        
        // Apply Sub-pixel Jitter (TAA)
        // Jitter is in pixels (-0.5 to 0.5). NDC width is 2.0.
        // NDC Offset X = (jitter_x * 2.0) / width
        // NDC Offset Y = (jitter_y * 2.0) / height
        // Projection Matrix Translation is at [3][0] (X) and [3][1] (Y) derived from bounds logic.
        // We simply add the offset.
        let (jx, jy) = ctx.jitter;
        proj_matrix[3][0] += (jx * 2.0) / width as f32;
        proj_matrix[3][1] += (jy * 2.0) / height as f32;

        let proj: [f32; 16] = *bytemuck::cast_ref(&proj_matrix);
        // Diagnostic print
        // println!("GeometryPass: width={}, height={}, proj={:?}", width, height, proj);

        println!("DEBUG: GeometryNode::execute called with {} commands", self.commands.len());
        let mut i = 0;
        while i < self.commands.len() {
            let start_cmd = &self.commands[i];
            
            match start_cmd {
                DrawCommand::Text { .. } => {
                    let mut batched_verts = Vec::new();
                    let mut j = i;
                    while j < self.commands.len() {
                        if let DrawCommand::Text { pos, size, uv, color } = &self.commands[j] {
                            batched_verts.extend(quad_vertices_uv(*pos, *size, *uv, *color));
                            j += 1;
                        } else { break; }
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
                    
                    use crate::backend::hal::{BindGroupEntry, BindingResource};
                    let bg = executor.create_bind_group(&layout, &[
                        BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&u_buf) },
                        BindGroupEntry { binding: 1, resource: BindingResource::Texture(font_view.as_ref().unwrap()) },
                        BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
                        BindGroupEntry { binding: 3, resource: BindingResource::Texture(&backdrop_view) },
                        BindGroupEntry { binding: 4, resource: BindingResource::Buffer(&u_buf) },
                        BindGroupEntry { binding: 5, resource: BindingResource::Buffer(executor.get_dummy_storage_buffer()) },
                    ])?;
                    
                    executor.draw(&pipeline, Some(&bg), &v_buf, ((j - i) * 6) as u32, &[])?;
                    executor.destroy_bind_group(bg); executor.destroy_buffer(u_buf); executor.destroy_buffer(v_buf);
                    i = j;
                }
                DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, shader_inject, .. } => {
                    if shader_inject.is_some() {
                        // Custom shader path (single object)
                        let uniforms = {
                            let mut u: DrawUniforms = Zeroable::zeroed();
                            u.projection = *bytemuck::cast_ref(&proj); u.rect = [pos.x, pos.y, size.x, size.y];
                            u.radii = *radii; u.border_color = [border_color.r, border_color.g, border_color.b, border_color.a];
                            u.glow_color = [glow_color.r, glow_color.g, glow_color.b, glow_color.a];
                            u.border_width = *border_width; u.elevation = *elevation; u.glow_strength = *glow_strength;
                            u.mode = 100; u.is_squircle = if *is_squircle { 1 } else { 0 };
                            u.time = time; u.viewport_size = [width as f32, height as f32];
                            u
                        };
                        let verts = quad_vertices(*pos, *size, *color);
                        let p = executor.get_custom_render_pipeline(shader_inject.as_ref().unwrap()).unwrap_or(pipeline.clone());

                        let u_buf = executor.create_buffer(bytemuck::bytes_of(&uniforms).len() as u64, BufferUsage::Uniform, "FallbackU")?;
                        executor.write_buffer(&u_buf, 0, bytemuck::bytes_of(&uniforms));
                        let v_buf = executor.create_buffer(verts.len() as u64, BufferUsage::Vertex, "FallbackV")?;
                        executor.write_buffer(&v_buf, 0, &verts);
                        
                        use crate::backend::hal::{BindGroupEntry, BindingResource};
                        let bg = executor.create_bind_group(&layout, &[
                            BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&u_buf) },
                            BindGroupEntry { binding: 1, resource: BindingResource::Texture(font_view.as_ref().unwrap()) },
                            BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
                            BindGroupEntry { binding: 3, resource: BindingResource::Texture(&backdrop_view) },
                            BindGroupEntry { binding: 4, resource: BindingResource::Buffer(&u_buf) },
                            BindGroupEntry { binding: 5, resource: BindingResource::Buffer(executor.get_dummy_storage_buffer()) },
                        ])?;
                        executor.draw(&p, Some(&bg), &v_buf, 6, &[])?;
                        executor.destroy_bind_group(bg); executor.destroy_buffer(u_buf); executor.destroy_buffer(v_buf);
                        i += 1;
                    } else {
                        // Batch contiguous rounded rects
                        let mut instances = Vec::new();
                        let mut j = i;
                        while j < self.commands.len() {
                            if let DrawCommand::RoundedRect { 
                                pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, 
                                velocity, reflectivity, roughness, normal_map, distortion_strength, emissive_intensity, parallax_factor, shader_inject, .. 
                            } = &self.commands[j] {
                                if shader_inject.is_some() { break; }
                                instances.push(ShapeInstance {
                                    rect: [pos.x, pos.y, size.x, size.y],
                                    radii: *radii,
                                    color: [color.r, color.g, color.b, color.a],
                                    border_color: [border_color.r, border_color.g, border_color.b, border_color.a],
                                    glow_color: [glow_color.r, glow_color.g, glow_color.b, glow_color.a],
                                    params1: [*border_width, *elevation, *glow_strength, 0.0],
                                    params2: [2, if *is_squircle { 1 } else { 0 }, 0, 0],
                                    material: [velocity.x, velocity.y, *reflectivity, *roughness],
                                    pbr_params: [normal_map.unwrap_or(0) as f32, *distortion_strength, *emissive_intensity, *parallax_factor],
                                    transform: [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0],
                                });
                                j += 1;
                            } else { break; }
                        }

                        use crate::backend::hal::{BindGroupEntry, BindingResource};
                        
                        let global = GlobalUniforms {
                            projection: *bytemuck::cast_ref(&proj),
                            time, _pad0: 0.0, viewport_size: [width as f32, height as f32],
                        };
                        let ui_quad = unit_quad();
                        let g_buf = executor.create_buffer(bytemuck::bytes_of(&global).len() as u64, BufferUsage::Uniform, "GlobalU")?;
                        executor.write_buffer(&g_buf, 0, bytemuck::bytes_of(&global));
                        
                        // Input Instances (CPU -> GPU)
                        let inst_buf = executor.create_buffer((instances.len() * std::mem::size_of::<ShapeInstance>()) as u64, BufferUsage::Storage, "InputInstU")?;
                        executor.write_buffer(&inst_buf, 0, bytemuck::cast_slice(&instances));
                        
                        // Output Instances (GPU -> GPU)
                        let out_inst_buf = executor.create_buffer((instances.len() * std::mem::size_of::<ShapeInstance>()) as u64, BufferUsage::Storage, "OutputInstU")?;
                        
                        // Indirect Args
                        let indirect_args = [6u32, 0u32, 0u32, 0u32];
                        let indirect_buf = executor.create_buffer(16, BufferUsage::Indirect | BufferUsage::Storage | BufferUsage::CopyDst, "IndirectArgs")?;
                        executor.write_buffer(&indirect_buf, 0, bytemuck::cast_slice(&indirect_args));

                        let v_buf = executor.create_buffer(ui_quad.len() as u64, BufferUsage::Vertex, "QuadV")?;
                        executor.write_buffer(&v_buf, 0, &ui_quad);
                        use crate::backend::hal::{BindGroupEntry, BindingResource};
                        let culling_bg = executor.create_bind_group(executor.get_culling_bind_group_layout(), &[
                            BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&g_buf) },
                            BindGroupEntry { binding: 1, resource: BindingResource::Buffer(&inst_buf) },
                            BindGroupEntry { binding: 2, resource: BindingResource::Buffer(&out_inst_buf) },
                            BindGroupEntry { binding: 3, resource: BindingResource::Buffer(&indirect_buf) },
                        ])?;

                        let workgroups_x = (instances.len() as u32 + 63) / 64;
                        executor.dispatch(executor.get_culling_pipeline(), Some(&culling_bg), [workgroups_x, 1, 1], &[])?;
                        executor.destroy_bind_group(culling_bg);
                        
                        let bg = executor.create_bind_group(&instanced_layout, &[
                            BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&g_buf) },
                            BindGroupEntry { binding: 1, resource: BindingResource::Buffer(&out_inst_buf) },
                            BindGroupEntry { binding: 2, resource: BindingResource::Texture(font_view.as_ref().unwrap()) },
                            BindGroupEntry { binding: 3, resource: BindingResource::Texture(&backdrop_view) },
                            BindGroupEntry { binding: 4, resource: BindingResource::Sampler(&sampler) },
                        ])?;

                        if self.use_gpu_driven && executor.supports_indirect_draw() {
                            // --- GPU-Driven Path (Culling + Indirect) ---
                            let visible_indices = executor.create_buffer((instances.len() * 4) as u64, BufferUsage::Storage, "VisIdx")?;
                            let visible_counter = executor.create_buffer(4, BufferUsage::Storage | BufferUsage::CopySrc, "VisCnt")?;
                            executor.write_buffer(&visible_counter, 0, bytemuck::bytes_of(&0u32));
                            
                            let hzb = executor.get_hzb_view();
                            executor.dispatch_visibility(proj_matrix, instances.len() as u32, &inst_buf, hzb, &visible_indices, &visible_counter)?;
                            
                            let indirect_buffer = executor.create_buffer(16, BufferUsage::Indirect | BufferUsage::Storage, "IndirB")?;
                            executor.dispatch_indirect_command(&visible_counter, &indirect_buffer)?;

                            let bg = executor.create_bind_group(&instanced_layout, &[
                                BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&g_buf) },
                                BindGroupEntry { binding: 1, resource: BindingResource::Buffer(&inst_buf) },
                                BindGroupEntry { binding: 2, resource: BindingResource::Texture(font_view.as_ref().unwrap()) },
                                BindGroupEntry { binding: 3, resource: BindingResource::Texture(&backdrop_view) },
                                BindGroupEntry { binding: 4, resource: BindingResource::Sampler(&sampler) },
                                BindGroupEntry { binding: 5, resource: BindingResource::Buffer(&visible_indices) },
                            ])?;

                            let aux_view_opt = if let Some(aux_h) = self.aux_handle {
                                 if let Some(crate::renderer::graph::GraphResource::Texture(_, tex)) = ctx.resources.get(&aux_h) {
                                      Some(executor.create_texture_view(tex)?)
                                 } else { None }
                            } else { None };

                            let velocity_view_opt = if let Some(vel_h) = self.velocity_handle {
                                 if let Some(crate::renderer::graph::GraphResource::Texture(_, vtex)) = ctx.resources.get(&vel_h) {
                                      Some(executor.create_texture_view(vtex)?)
                                 } else { None }
                            } else { None };

                            let depth_view_opt = if let Some(depth_h) = self.depth_handle {
                                 if let Some(crate::renderer::graph::GraphResource::Texture(_, dtex)) = ctx.resources.get(&depth_h) {
                                      Some(executor.create_texture_view(dtex)?)
                                 } else { None }
                            } else { None };

                            if let (Some(aux), Some(vel), Some(depth)) = (&aux_view_opt, &velocity_view_opt, &depth_view_opt) {
                                let gb_pipeline = executor.get_instanced_gbuffer_render_pipeline().clone();
                                executor.draw_instanced_gbuffer_indirect(&gb_pipeline, Some(&bg), &v_buf, &out_inst_buf, &indirect_buf, 0, aux, vel, depth)?;
                            } else {
                                executor.draw_instanced_indirect(&instanced_pipeline, Some(&bg), &v_buf, &out_inst_buf, &indirect_buf, 0)?;
                            }

                            executor.destroy_bind_group(bg); executor.destroy_buffer(g_buf); executor.destroy_buffer(inst_buf); executor.destroy_buffer(out_inst_buf); executor.destroy_buffer(indirect_buf); executor.destroy_buffer(v_buf);
                        i = j;
                    }
                }
                DrawCommand::Image { pos, size, texture_id, uv, color, .. } => {
                    let uniforms = DrawUniforms {
                        projection: *bytemuck::cast_ref(&proj),
                        mode: 0, time, viewport_size: [width as f32, height as f32],
                        ..Zeroable::zeroed()
                    };
                    let verts = quad_vertices_uv(*pos, *size, *uv, *color);
                    
                    // Note: We need to resolve the texture_id to a texture view.
                    // For now, we assume the backend can provide views by ID or we fallback to font_view.
                    let tex_view = font_view.as_ref().unwrap(); // Fallback for now if not found
                    
                    let u_buf = executor.create_buffer(bytemuck::bytes_of(&uniforms).len() as u64, BufferUsage::Uniform, "ImageU")?;
                    executor.write_buffer(&u_buf, 0, bytemuck::bytes_of(&uniforms));
                    let v_buf = executor.create_buffer(verts.len() as u64, BufferUsage::Vertex, "ImageV")?;
                    executor.write_buffer(&v_buf, 0, &verts);
                    
                    use crate::backend::hal::{BindGroupEntry, BindingResource};
                    let bg = executor.create_bind_group(&layout, &[
                        BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&u_buf) },
                        BindGroupEntry { binding: 1, resource: BindingResource::Texture(tex_view) },
                        BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
                        BindGroupEntry { binding: 3, resource: BindingResource::Texture(&backdrop_view) },
                        BindGroupEntry { binding: 4, resource: BindingResource::Buffer(&u_buf) },
                        BindGroupEntry { binding: 5, resource: BindingResource::Buffer(executor.get_dummy_storage_buffer()) },
                    ])?;
                    
                    executor.draw(&pipeline, Some(&bg), &v_buf, 6, &[])?;
                    executor.destroy_bind_group(bg); executor.destroy_buffer(u_buf); executor.destroy_buffer(v_buf);
                    i += 1;
                }
                DrawCommand::TileMap { pos, size: _, texture_id, tile_count, tile_size, data, color, tiles_per_row, tile_uv_size } => {
                    if executor.supports_tilemap() {
                        let tex_view = if *texture_id > 0 {
                             if let Some(crate::renderer::graph::GraphResource::Texture(_, tex)) = ctx.resources.get(&(*texture_id as u32)) {
                                  Some(executor.create_texture_view(tex)?)
                             } else { None }
                         } else { None };
                         
                         let final_view = tex_view.as_ref().unwrap_or(font_view.as_ref().unwrap());

                         let global = GlobalUniforms {
                            projection: *bytemuck::cast_ref(&proj),
                            time, _pad0: 0.0, viewport_size: [width as f32, height as f32],
                         };
                         let g_buf = executor.create_buffer(bytemuck::bytes_of(&global).len() as u64, BufferUsage::Uniform, "TileGlobal")?;
                         executor.write_buffer(&g_buf, 0, bytemuck::bytes_of(&global));

                         let params = crate::backend::shaders::types::TilemapParams {
                             position: [pos.x, pos.y],
                             map_size: *tile_count,
                             tile_size: *tile_size,
                             uv_size: *tile_uv_size,
                             columns: *tiles_per_row,
                             _pad0: 0, _pad1: 0, _pad2: 0,
                         };

                        let aux_view_opt = if let Some(aux_h) = self.aux_handle {
                             if let Some(crate::renderer::graph::GraphResource::Texture(_, tex)) = ctx.resources.get(&aux_h) {
                                  Some(executor.create_texture_view(tex)?)
                             } else { None }
                        } else { None };

                        let velocity_view_opt = if let Some(vel_h) = self.velocity_handle {
                             if let Some(crate::renderer::graph::GraphResource::Texture(_, vtex)) = ctx.resources.get(&vel_h) {
                                  Some(executor.create_texture_view(vtex)?)
                             } else { None }
                        } else { None };

                        let depth_view_opt = if let Some(depth_h) = self.depth_handle {
                             if let Some(crate::renderer::graph::GraphResource::Texture(_, dtex)) = ctx.resources.get(&depth_h) {
                                  Some(executor.create_texture_view(dtex)?)
                             } else { None }
                        } else { None };

                         executor.draw_tilemap(params, data, final_view, &g_buf, aux_view_opt.as_ref(), velocity_view_opt.as_ref(), depth_view_opt.as_ref())?;
                         executor.destroy_buffer(g_buf);
                         i += 1;
                         continue;
                    }

                    let mut batched_verts = Vec::new();
                    
                    for y in 0..tile_count[1] {
                        for x in 0..tile_count[0] {
                            let idx = (y * tile_count[0] + x) as usize;
                            if idx >= data.len() { continue; }
                            let tile_id = data[idx];
                            if tile_id == 0 { continue; }

                            let tx = (tile_id % tiles_per_row) as f32;
                            let ty = (tile_id / tiles_per_row) as f32;
                            
                            let uv = [
                                tx * tile_uv_size[0],
                                ty * tile_uv_size[1],
                                tile_uv_size[0],
                                tile_uv_size[1]
                            ];
                            
                            let world_pos = Vec2::new(
                                pos.x + x as f32 * tile_size[0],
                                pos.y + y as f32 * tile_size[1]
                            );
                            let world_size = Vec2::new(tile_size[0], tile_size[1]);
                            
                            batched_verts.extend(quad_vertices_uv(world_pos, world_size, uv, *color));
                        }
                    }

                    if !batched_verts.is_empty() {
                        let uniforms = DrawUniforms {
                            projection: *bytemuck::cast_ref(&proj),
                            mode: 0, time, viewport_size: [width as f32, height as f32],
                            ..Zeroable::zeroed()
                        };
                        let tex_view = font_view.as_ref().unwrap(); // Fallback
                        
                        let u_buf = executor.create_buffer(bytemuck::bytes_of(&uniforms).len() as u64, BufferUsage::Uniform, "TileU")?;
                        executor.write_buffer(&u_buf, 0, bytemuck::bytes_of(&uniforms));
                        let v_buf = executor.create_buffer(batched_verts.len() as u64, BufferUsage::Vertex, "TileV")?;
                        executor.write_buffer(&v_buf, 0, &batched_verts);
                        
                        use crate::backend::hal::{BindGroupEntry, BindingResource};
                        let bg = executor.create_bind_group(&layout, &[
                            BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&u_buf) },
                            BindGroupEntry { binding: 1, resource: BindingResource::Texture(tex_view) },
                            BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
                            BindGroupEntry { binding: 3, resource: BindingResource::Texture(&backdrop_view) },
                            BindGroupEntry { binding: 4, resource: BindingResource::Buffer(&u_buf) },
                            BindGroupEntry { binding: 5, resource: BindingResource::Buffer(executor.get_dummy_storage_buffer()) },
                        ])?;
                        
                        executor.draw(&pipeline, Some(&bg), &v_buf, (batched_verts.len() / (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>())) as u32, &[])?;
                        executor.destroy_bind_group(bg); executor.destroy_buffer(u_buf); executor.destroy_buffer(v_buf);
                    }
                    i += 1;
                }
                DrawCommand::Particles { texture_id: _, particles } => {
                    let mut batched_verts = Vec::new();
                    for p in particles {
                        let uv = [0.0, 0.0, 1.0, 1.0];
                        let p_size = Vec2::new(p.size, p.size);
                        let p_pos = p.position - p_size * 0.5;
                        batched_verts.extend(quad_vertices_uv(p_pos, p_size, uv, p.color));
                    }
                    
                    if !batched_verts.is_empty() {
                        let uniforms = DrawUniforms {
                            projection: *bytemuck::cast_ref(&proj),
                            mode: 0, time, viewport_size: [width as f32, height as f32],
                            ..Zeroable::zeroed()
                        };
                        let tex_view = font_view.as_ref().unwrap(); // Fallback
                        
                        let u_buf = executor.create_buffer(bytemuck::bytes_of(&uniforms).len() as u64, BufferUsage::Uniform, "PartU")?;
                        executor.write_buffer(&u_buf, 0, bytemuck::bytes_of(&uniforms));
                        let v_buf = executor.create_buffer(batched_verts.len() as u64, BufferUsage::Vertex, "PartV")?;
                        executor.write_buffer(&v_buf, 0, &batched_verts);
                        
                        use crate::backend::hal::{BindGroupEntry, BindingResource};
                        let bg = executor.create_bind_group(&layout, &[
                            BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&u_buf) },
                            BindGroupEntry { binding: 1, resource: BindingResource::Texture(tex_view) },
                            BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
                            BindGroupEntry { binding: 3, resource: BindingResource::Texture(&backdrop_view) },
                            BindGroupEntry { binding: 4, resource: BindingResource::Buffer(&u_buf) },
                            BindGroupEntry { binding: 5, resource: BindingResource::Buffer(executor.get_dummy_storage_buffer()) },
                        ])?;
                        
                        executor.draw(&pipeline, Some(&bg), &v_buf, (batched_verts.len() / (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>())) as u32, &[])?;
                        executor.destroy_bind_group(bg); executor.destroy_buffer(u_buf); executor.destroy_buffer(v_buf);
                    }
                    i += 1;
                }
                DrawCommand::SkinnedMesh { vertices, indices, texture_id, bone_matrices } => {
                     let tex_view = if *texture_id > 0 {
                         if let Some(crate::renderer::graph::GraphResource::Texture(_, tex)) = ctx.resources.get(&(*texture_id as u32)) {
                              Some(executor.create_texture_view(tex)?)
                         } else { None }
                     } else { None };
                     let final_view = tex_view.as_ref().unwrap_or(font_view.as_ref().unwrap());

                     let global = GlobalUniforms {
                        projection: *bytemuck::cast_ref(&proj),
                        time, _pad0: 0.0, viewport_size: [width as f32, height as f32],
                     };
                     
                     // Convert Mat3 to Mat4 for shader (padded)
                     // Mat3 in Rust is [f32; 9]. WGSL expects mat4x4<f32>.
                     // We need to expand 3x3 to 4x4.
                     let mut processed_matrices = Vec::with_capacity(bone_matrices.len() * 16);
                     for m in bone_matrices {
                          let arr = m.0; // [f32; 9]
                          // Row 0
                          processed_matrices.push(arr[0]); processed_matrices.push(arr[1]); processed_matrices.push(arr[2]); processed_matrices.push(0.0);
                          // Row 1
                          processed_matrices.push(arr[3]); processed_matrices.push(arr[4]); processed_matrices.push(arr[5]); processed_matrices.push(0.0);
                          // Row 2 (Affine translation usually)
                          processed_matrices.push(arr[6]); processed_matrices.push(arr[7]); processed_matrices.push(arr[8]); processed_matrices.push(0.0);
                          // Row 3
                          processed_matrices.push(0.0); processed_matrices.push(0.0); processed_matrices.push(0.0); processed_matrices.push(1.0);
                     }
                     
                     let g_buf = executor.create_buffer(bytemuck::bytes_of(&global).len() as u64, BufferUsage::Uniform, "SkinGlobal")?;
                     executor.write_buffer(&g_buf, 0, bytemuck::bytes_of(&global));
                     
                     let v_buf = executor.create_buffer((vertices.len() * std::mem::size_of::<crate::backend::wgpu::SkinnedVertex>()) as u64, BufferUsage::Vertex, "SkinV")?;
                     executor.write_buffer(&v_buf, 0, bytemuck::cast_slice(vertices));
                     
                     let i_buf = executor.create_buffer((indices.len() * 4) as u64, BufferUsage::Index, "SkinI")?;
                     executor.write_buffer(&i_buf, 0, bytemuck::cast_slice(indices));
                     
                     let b_buf = executor.create_buffer((processed_matrices.len() * 4) as u64, BufferUsage::Storage, "SkinB")?;
                     executor.write_buffer(&b_buf, 0, bytemuck::cast_slice(&processed_matrices));
                     
                     let _ = executor.draw_skinned(&v_buf, &i_buf, indices.len() as u32, &b_buf, final_view, &g_buf);
                     
                     executor.destroy_buffer(g_buf); executor.destroy_buffer(v_buf); executor.destroy_buffer(i_buf); executor.destroy_buffer(b_buf);
                     i += 1;
                }
                _ => { i += 1; }
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

pub fn unit_quad() -> Vec<u8> {
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
