use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, BACKDROP_HANDLE, HDR_HIGH_RES_HANDLE, LUT_HANDLE};

pub struct PostProcessNode;

impl<E: GpuExecutor> RenderNode<E> for PostProcessNode {
    fn name(&self) -> &str { "PostProcessPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        
        // Select input handle based on debug mode
        use crate::renderer::graph::{DebugDisplayMode, AUX_HANDLE, VELOCITY_HANDLE, DEPTH_HANDLE, EMISSIVE_HANDLE, SDF_HANDLE};
        
        let input_handle = match ctx.debug_mode {
            DebugDisplayMode::None => {
                if ctx.resources.contains_key(&crate::renderer::graph::FLARE_HANDLE) {
                    crate::renderer::graph::FLARE_HANDLE
                } else if ctx.resources.contains_key(&crate::renderer::graph::HDR_HIGH_RES_HANDLE) {
                    crate::renderer::graph::HDR_HIGH_RES_HANDLE
                } else if ctx.resources.contains_key(&crate::renderer::graph::HDR_HANDLE) {
                    crate::renderer::graph::HDR_HANDLE
                } else {
                    return Err("Missing HDR input texture in graph".to_string());
                }
            },
            DebugDisplayMode::Albedo => crate::renderer::graph::HDR_HANDLE, // In deferred, HDR before lighting is albedo + emissive
            DebugDisplayMode::Normal => crate::renderer::graph::AUX_HANDLE,
            DebugDisplayMode::Velocity => crate::renderer::graph::VELOCITY_HANDLE,
            DebugDisplayMode::Depth => crate::renderer::graph::DEPTH_HANDLE,
            DebugDisplayMode::Emissive => crate::renderer::graph::HDR_HANDLE, // Emissive is currently mixed in HDR or needs its own buffer
            DebugDisplayMode::SDF => crate::renderer::graph::SDF_HANDLE,
        };

        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&input_handle) {
            let view = ctx.executor.create_texture_view(tex)?;
            // Set this as the "HDR" texture for the backend resolve
            ctx.executor.set_hdr_view(&view)?;
            
            use crate::renderer::graph::LDR_HANDLE;
            if let Some(GraphResource::Texture(_, ldr_tex)) = ctx.resources.get(&LDR_HANDLE) {
                let output_view = ctx.executor.create_texture_view(ldr_tex)?;
                ctx.executor.draw_post_process_pass(&view, Some(&output_view))?;
            } else {
                ctx.executor.draw_post_process_pass(&view, None)?;
            }
        } else {
             return Err(format!("Handle {} found but not a texture", input_handle));
        }

        use crate::renderer::graph::{LUT_HANDLE};
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&LUT_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.set_lut_view(&view)?;
        }
        
        Ok(())
    }
}
