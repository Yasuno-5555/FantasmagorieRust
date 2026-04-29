use crate::backend::hal::GpuExecutor;
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, HDR_LOW_RES_HANDLE, VELOCITY_HANDLE};

pub struct TAANode {
    pub history_handle: crate::renderer::graph::ResourceHandle,
}

impl TAANode {
    pub fn new(history_handle: crate::renderer::graph::ResourceHandle) -> Self {
        Self { history_handle }
    }
}

impl<E: GpuExecutor> RenderNode<E> for TAANode {
    fn name(&self) -> &str { "TAAPass" }

    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let current_h = HDR_LOW_RES_HANDLE;
        let vel_h = VELOCITY_HANDLE;
        let history_h = self.history_handle;

        if let (Some(current), Some(history), Some(vel)) = (
            ctx.resources.get(&current_h),
            ctx.resources.get(&history_h),
            ctx.resources.get(&vel_h),
        ) {
            if let (
                GraphResource::Texture(_, current_tex),
                GraphResource::Texture(_, history_tex),
                GraphResource::Texture(_, vel_tex),
            ) = (current, history, vel) {
                let current_view = ctx.executor.create_texture_view(current_tex)?;
                let history_view = ctx.executor.create_texture_view(history_tex)?;
                let vel_view = ctx.executor.create_texture_view(vel_tex)?;
                
                // We need an output texture. For TAA, we often ping-pong.
                // But for now, let's assume we can use a transient one and then copy back.
                let desc = crate::backend::hal::TextureDescriptor {
                    label: Some("TAA Output"),
                    width: ctx.width,
                    height: ctx.height,
                    depth: 1,
                    format: crate::backend::hal::TextureFormat::Rgba16Float,
                    usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::COPY_SRC,
                };
                
                let output_tex = ctx.executor.acquire_transient_texture(&desc)?;
                let output_view = ctx.executor.create_texture_view(&output_tex)?;
                
                ctx.executor.draw_taa_pass(&current_view, &history_view, &vel_view, &output_view)?;
                
                // Copy result back to history for next frame
                ctx.executor.copy_texture(&output_tex, history_tex)?;
                
                // Copy result to HDR_HANDLE to update the main pipeline
                ctx.executor.copy_texture(&output_tex, current_tex)?;
                
                ctx.executor.release_transient_texture(output_tex, &desc);
            }
        }

        Ok(())
    }
}
