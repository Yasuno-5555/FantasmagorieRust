use crate::renderer::graph::{RenderNode, RenderContext, ResourceHandle, SDF_TEMP_A_HANDLE, SDF_TEMP_B_HANDLE, SDF_HANDLE, AUX_HANDLE};
use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage, BufferUsage};
use std::sync::Arc;

pub struct JfaSdfNode {
    pub extra_handle: ResourceHandle,
}

impl JfaSdfNode {
    pub fn new(extra_handle: ResourceHandle) -> Self {
        Self { extra_handle }
    }
}

impl<E: GpuExecutor> RenderNode<E> for JfaSdfNode {
    fn name(&self) -> &str {
        "JFA SDF Generation"
    }

    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let executor = &mut *ctx.executor;
        let width = ctx.width;
        let height = ctx.height;

        // 1. Get G-buffer resources
        let aux_tex = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = ctx.resources.get(&AUX_HANDLE) { t.clone() } else { return Err("AUX not found".into()); };
        let extra_tex = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = ctx.resources.get(&self.extra_handle) { t.clone() } else { return Err("EXTRA not found".into()); };
        
        let aux_view = executor.create_texture_view(&aux_tex)?;
        let extra_view = executor.create_texture_view(&extra_tex)?;

        // 2. Get ping-pong textures
        let temp_a = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = ctx.resources.get(&SDF_TEMP_A_HANDLE) { t.clone() } else { return Err("SDF_TEMP_A not found".into()); };
        let temp_b = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = ctx.resources.get(&SDF_TEMP_B_HANDLE) { t.clone() } else { return Err("SDF_TEMP_B not found".into()); };
        let sdf_output = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = ctx.resources.get(&SDF_HANDLE) { t.clone() } else { return Err("SDF_HANDLE not found".into()); };

        let view_a = executor.create_texture_view(&temp_a)?;
        let view_b = executor.create_texture_view(&temp_b)?;
        let view_sdf = executor.create_texture_view(&sdf_output)?;

        // 3. Prepare Compute Pipelines
        let shader_source = include_str!("../../backend/shaders/jfa.wgsl");
        let seed_pipeline = executor.create_compute_pipeline("JFA Seed", shader_source, Some("compute_seed"))?;
        let flood_pipeline = executor.create_compute_pipeline("JFA Flood", shader_source, Some("compute_flood"))?;
        let resolve_pipeline = executor.create_compute_pipeline("JFA Resolve", shader_source, Some("compute_resolve"))?;
        
        let seed_layout = executor.get_compute_pipeline_layout(&seed_pipeline, 0)?;
        let flood_layout = executor.get_compute_pipeline_layout(&flood_pipeline, 0)?;
        let resolve_layout = executor.get_compute_pipeline_layout(&resolve_pipeline, 0)?;

        // 4. Uniform Buffer for steps
        let uniform_buf = executor.create_buffer(16, BufferUsage::Uniform, "JFA Uniforms")?;

        // Helper to update uniforms
        let update_uniforms = |exec: &mut E, step: u32| {
            let data = [step, width, height, 0];
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, 16) };
            exec.write_buffer(&uniform_buf, 0, bytes);
        };

        // --- SEED PASS ---
        update_uniforms(executor, 0);
        
        // Seed bind group: input=aux (unused as texture_2d), output=view_a
        use crate::backend::hal::{BindGroupEntry, BindingResource};
        let bg_seed = executor.create_bind_group(
            &seed_layout,
            &[
                BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&uniform_buf) },
                BindGroupEntry { binding: 1, resource: BindingResource::Texture(&aux_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::Texture(&view_a) },
                BindGroupEntry { binding: 3, resource: BindingResource::Texture(&aux_view) },
                BindGroupEntry { binding: 4, resource: BindingResource::Texture(&extra_view) },
            ],
        )?;
        
        executor.dispatch(&seed_pipeline, Some(&bg_seed), [(width + 7) / 8, (height + 7) / 8, 1], &[])?;

        // --- FLOOD PASSES ---
        let mut current_step = 1u32 << (31 - (u32::max(width, height).leading_zeros()));
        let mut ping = true;

        while current_step > 0 {
            update_uniforms(executor, current_step);
            
            let (src, dst) = if ping { (&view_a, &view_b) } else { (&view_b, &view_a) };
            
            let bg_flood = executor.create_bind_group(
                &flood_layout,
                &[
                    BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&uniform_buf) },
                    BindGroupEntry { binding: 1, resource: BindingResource::Texture(src) },
                    BindGroupEntry { binding: 2, resource: BindingResource::Texture(dst) },
                    BindGroupEntry { binding: 3, resource: BindingResource::Texture(&aux_view) },
                    BindGroupEntry { binding: 4, resource: BindingResource::Texture(&extra_view) },
                ],
            )?;
            
            executor.dispatch(&flood_pipeline, Some(&bg_flood), [(width + 7) / 8, (height + 7) / 8, 1], &[])?;
            
            current_step /= 2;
            ping = !ping;
        }

        // --- RESOLVE PASS ---
        let final_src = if ping { &view_a } else { &view_b };
        let bg_resolve = executor.create_bind_group(
            &resolve_layout,
            &[
                BindGroupEntry { binding: 0, resource: BindingResource::Buffer(&uniform_buf) },
                BindGroupEntry { binding: 1, resource: BindingResource::Texture(final_src) },
                BindGroupEntry { binding: 2, resource: BindingResource::Texture(&view_sdf) },
                BindGroupEntry { binding: 3, resource: BindingResource::Texture(&aux_view) },
                BindGroupEntry { binding: 4, resource: BindingResource::Texture(&extra_view) },
            ],
        )?;
        
        executor.dispatch(&resolve_pipeline, Some(&bg_resolve), [(width + 7) / 8, (height + 7) / 8, 1], &[])?;

        // Cleanup
        executor.destroy_buffer(uniform_buf);

        Ok(())
    }
}
