use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, HDR_LOW_RES_HANDLE, HDR_HIGH_RES_HANDLE, VELOCITY_HANDLE};

pub struct UpscaleNode;

impl<E: GpuExecutor> RenderNode<E> for UpscaleNode {
    fn name(&self) -> &str { "UpscalePass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let input = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&HDR_LOW_RES_HANDLE) {
            ctx.executor.create_texture_view(tex)?
        } else {
            return Err("Missing HDR_LOW_RES input".to_string());
        };

        let output = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&HDR_HIGH_RES_HANDLE) {
             ctx.executor.create_texture_view(tex)?
        } else {
             return Err("Missing HDR_HIGH_RES target".to_string());
        };
        
        // Depth/Velocity might be needed for MetalFX
        // For now, simple interface
        // Get Jitter from context
        let (jx, jy) = ctx.jitter;
        let params = crate::backend::hal::UpscaleParams {
            jitter_x: jx,
            jitter_y: jy,
            reset_history: ctx.camera_cut,
        };
        ctx.executor.upscale(&input, &output, params)
    }
}
