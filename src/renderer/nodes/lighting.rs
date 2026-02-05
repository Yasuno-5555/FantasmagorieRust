use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, HDR_LOW_RES_HANDLE};

pub struct LightingNode;

impl<E: GpuExecutor> RenderNode<E> for LightingNode {
    fn name(&self) -> &str { "LightingPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        // Integrate SSR Reflection if available
        use crate::renderer::graph::{REFLECTION_HANDLE, GraphResource};
        
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&REFLECTION_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.set_reflection_texture(&view)?;
        }
        
        use crate::renderer::graph::{VELOCITY_HANDLE, SDF_HANDLE};
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&VELOCITY_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.set_velocity_view(&view)?;
        }
        
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&SDF_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.set_sdf_view(&view)?;
        }

        // We need to tell the backend to use the "Lighting" pipeline and write to HDR_LOW_RES
        // For now, we reuse `resolve` but logically it will be split.
        // We need a specific execution method on GpuExecutor for lighting pass?
        // Or we set a "mode" before calling resolve.
        
        // Let's add `draw_lighting_pass` to GpuExecutor.
        // Input: G-Buffers. Output: HDR_LOW_RES.
        
        // Retrieve output texture
        let output_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&HDR_LOW_RES_HANDLE) {
             ctx.executor.create_texture_view(tex)?
        } else {
             return Err("Missing HDR_LOW_RES target".to_string());
        };

        ctx.executor.draw_lighting_pass(&output_view)
    }
}
