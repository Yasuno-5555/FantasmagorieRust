use crate::renderer::graph::{RenderNode, RenderContext, ResourceHandle};
use crate::backend::hal::GpuExecutor;

pub struct SSRNode {
    pub aux_handle: ResourceHandle,
    pub depth_handle: ResourceHandle,
    pub hdr_handle: ResourceHandle,
}

impl SSRNode {
    pub fn new(aux_handle: ResourceHandle, depth_handle: ResourceHandle, hdr_handle: ResourceHandle) -> Self {
        Self { aux_handle, depth_handle, hdr_handle }
    }
}

impl<E: GpuExecutor> RenderNode<E> for SSRNode {
    fn name(&self) -> &str {
        "SSR"
    }

    fn execute(
        &mut self,
        context: &mut RenderContext<'_, E>
    ) -> Result<(), String> {
        let executor = &mut *context.executor;
        // Extract textures safely
        // Extract textures safely - using explicit Clone::clone to avoid ambiguity if E::Texture is Arc
        // Actually, if E::Texture implements Clone, t.clone() should works.
        // But maybe compiler doesn't know t's type well?
        // Let's rely on pattern matching to bind t.
        // It seems the issue is type inference.
        // Let's specify type on t.
        
        let hdr_tex = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = context.resources.get(&self.hdr_handle) { <E::Texture as Clone>::clone(t) } else { return Err("HDR not found".into()); };
        let depth_tex = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = context.resources.get(&self.depth_handle) { <E::Texture as Clone>::clone(t) } else { return Err("Depth not found".into()); };
        let aux_tex = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = context.resources.get(&self.aux_handle) { <E::Texture as Clone>::clone(t) } else { return Err("Aux not found".into()); };
        let vel_tex = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = context.resources.get(&crate::renderer::graph::VELOCITY_HANDLE) { <E::Texture as Clone>::clone(t) } else { return Err("Velocity not found".into()); };
        let ref_tex = if let Some(crate::renderer::graph::GraphResource::Texture(_, t)) = context.resources.get(&crate::renderer::graph::REFLECTION_HANDLE) { <E::Texture as Clone>::clone(t) } else { return Err("Reflection output not found".into()); };

        // Create Views
        let hdr_view = executor.create_texture_view(&hdr_tex)?;
        let depth_view = executor.create_texture_view(&depth_tex)?;
        let aux_view = executor.create_texture_view(&aux_tex)?;
        let vel_view = executor.create_texture_view(&vel_tex)?;
        
        // Draw SSR (Pass output texture directly)
        executor.draw_ssr(&hdr_view, &depth_view, &aux_view, &vel_view, &ref_tex)?;
        
        Ok(())
    }
}
