use crate::renderer::graph::{RenderNode, RenderContext, ResourceHandle, HDR_HANDLE, VELOCITY_HANDLE};
use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};

pub struct MotionBlurNode {
    pub strength: f32,
}

impl<E: GpuExecutor> RenderNode<E> for MotionBlurNode {
    fn name(&self) -> &str { "MotionBlurNode" }

    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let hdr_res = ctx.resources.get(&HDR_HANDLE).ok_or("HDR texture not found")?;
        let vel_res = ctx.resources.get(&VELOCITY_HANDLE).ok_or("Velocity texture not found")?;

        if let (crate::renderer::graph::GraphResource::Texture(hdr_desc, hdr_tex), crate::renderer::graph::GraphResource::Texture(_, vel_tex)) = (hdr_res, vel_res) {
             let hdr_view = ctx.executor.create_texture_view(hdr_tex)?;
             let vel_view = ctx.executor.create_texture_view(vel_tex)?;
             
             let desc = TextureDescriptor {
                label: Some("Motion Blur Output"),
                width: ctx.width,
                height: ctx.height,
                depth: 1,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
             };
             let temp_tex = ctx.executor.acquire_transient_texture(&desc)?;
             let temp_view = ctx.executor.create_texture_view(&temp_tex)?;

             // Apply motion blur using HAL method
             ctx.executor.draw_motion_blur(&temp_view, &hdr_view, &vel_view, self.strength)?;
             
             // Copy back to HDR
             ctx.executor.copy_texture(&temp_tex, hdr_tex)?;
             
             ctx.executor.release_transient_texture(temp_tex, &desc);
        }
        Ok(())
    }
}
