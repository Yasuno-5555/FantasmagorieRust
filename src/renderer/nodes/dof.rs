use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, HDR_HIGH_RES_HANDLE, DEPTH_HANDLE, DOF_HANDLE};

pub struct DoFNode;

impl<E: GpuExecutor> RenderNode<E> for DoFNode {
    fn name(&self) -> &str { "DepthOfFieldPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let hdr_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&HDR_HIGH_RES_HANDLE) {
            ctx.executor.create_texture_view(tex)?
        } else {
            return Err("Missing HDR_HIGH_RES for DoF".to_string());
        };

        let depth_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&DEPTH_HANDLE) {
            ctx.executor.create_texture_view(tex)?
        } else {
            return Err("Missing DEPTH for DoF".to_string());
        };

        let output_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&DOF_HANDLE) {
            ctx.executor.create_texture_view(tex)?
        } else {
            return Err("Missing DOF_HANDLE target".to_string());
        };

        // We'll need a new method in GpuExecutor for DoF, or use a custom pipeline.
        // For simplicity and to maintain parity, I'll add a trait method or use the custom pipeline API.
        // Let's assume we add `draw_dof_pass` to the HAL.
        
        // Actually, since I can't easily modify all backends at once without seeing them,
        // I will first implement the node and then update the HAL trait.
        
        ctx.executor.draw_dof_pass(&hdr_view, &depth_view, &output_view)
    }
}
