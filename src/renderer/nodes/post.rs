use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, BACKDROP_HANDLE, HDR_HIGH_RES_HANDLE, LUT_HANDLE};

pub struct PostProcessNode;

impl<E: GpuExecutor> RenderNode<E> for PostProcessNode {
    fn name(&self) -> &str { "PostProcessPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        
        // Input is HDR_HIGH_RES (upscaled)
        use crate::renderer::graph::{GraphResource};
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&HDR_HIGH_RES_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            // Set this as the "HDR" texture for the post shader
            // Re-using set_hdr_view if it exists? Or pass explicitly.
            // For now, let's assume `draw_post_process` takes input view.
            use crate::renderer::graph::LDR_HANDLE;
            if let Some(GraphResource::Texture(_, ldr_tex)) = ctx.resources.get(&LDR_HANDLE) {
                let output_view = ctx.executor.create_texture_view(ldr_tex)?;
                ctx.executor.draw_post_process_pass(&view, Some(&output_view))?;
            } else {
                ctx.executor.draw_post_process_pass(&view, None)?;
            }
        } else {
             return Err("Missing HDR_HIGH_RES input".to_string());
        }

        use crate::renderer::graph::{LUT_HANDLE};
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&LUT_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.set_lut_view(&view)?;
        }
        
        Ok(())
    }
}
