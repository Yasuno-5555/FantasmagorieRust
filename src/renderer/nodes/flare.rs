use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, HDR_HIGH_RES_HANDLE, DOF_HANDLE, FLARE_HANDLE};

pub struct LensFlareNode;

impl<E: GpuExecutor> RenderNode<E> for LensFlareNode {
    fn name(&self) -> &str { "LensFlarePass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let hdr_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&DOF_HANDLE) {
            ctx.executor.create_texture_view(tex)?
        } else {
            return Err("Missing DOF_HANDLE for LensFlare".to_string());
        };

        let output_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&FLARE_HANDLE) {
            ctx.executor.create_texture_view(tex)?
        } else {
            return Err("Missing FLARE_HANDLE target".to_string());
        };

        ctx.executor.draw_flare_pass(&hdr_view, &output_view)
    }
}
