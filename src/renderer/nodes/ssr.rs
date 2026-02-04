use crate::renderer::graph::{RenderNode, RenderContext, ResourceHandle};
use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use std::sync::Arc;

pub struct SSRNode {
    pub gbuffer_handle: ResourceHandle,
    pub depth_handle: ResourceHandle,
    pub hdr_handle: ResourceHandle, // Previous frame or current HDR
    pub output_handle: Option<ResourceHandle>,
}

impl SSRNode {
    pub fn new(gbuffer: ResourceHandle, depth: ResourceHandle, hdr: ResourceHandle) -> Self {
        Self {
            gbuffer_handle: gbuffer,
            depth_handle: depth,
            hdr_handle: hdr,
            output_handle: None,
        }
    }
}

impl<E: GpuExecutor> RenderNode<E> for SSRNode {
    fn name(&self) -> &str { "SSR Pass" }


    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        // 1. Get Resources
        let gbuffer = ctx.resources.get(&self.gbuffer_handle).ok_or("Missing GBuffer")?;
        
        // 2. Extract TextureView
        let reflection_view = match gbuffer {
            crate::renderer::graph::GraphResource::Texture(_, tex) => {
                ctx.executor.create_texture_view(tex)?
            },
            _ => return Err("GBuffer must be a texture".to_string()),
        };

        // 3. Pass to executor for global access
        ctx.executor.set_reflection_texture(&reflection_view)?;
        
        Ok(())
    }
}
