use crate::backend::hal::{GpuExecutor};
use crate::renderer::graph::{RenderNode, RenderContext};

pub struct BloomNode;

impl<E: GpuExecutor> RenderNode<E> for BloomNode {
    fn name(&self) -> &str { "BloomPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        use crate::renderer::graph::{DOF_HANDLE, GraphResource};
        
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&DOF_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.draw_bloom_pass(&view)?;
        } else {
             // Fallback to HDR_HIGH_RES if DOF is missing
             use crate::renderer::graph::HDR_HIGH_RES_HANDLE;
             if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&HDR_HIGH_RES_HANDLE) {
                 let view = ctx.executor.create_texture_view(tex)?;
                 ctx.executor.draw_bloom_pass(&view)?;
             }
        }
        
        Ok(())
    }
}
