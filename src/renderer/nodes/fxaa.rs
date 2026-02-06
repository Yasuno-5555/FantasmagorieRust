use crate::backend::hal::GpuExecutor;
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, ResourceHandle};

pub const FXAA_OUTPUT_HANDLE: ResourceHandle = 40;

pub struct FXAANode<E: GpuExecutor> {
    _phantom: std::marker::PhantomData<E>,
}

impl<E: GpuExecutor> FXAANode<E> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<E: GpuExecutor> RenderNode<E> for FXAANode<E> {
    fn name(&self) -> &str {
        "FXAA Pass"
    }

    fn execute(&mut self, ctx: &mut RenderContext<E>) -> Result<(), String> {
        // Input: LDR Handle (from Post Process)
        use crate::renderer::graph::{LDR_HANDLE, GraphResource};
        
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&LDR_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.draw_fxaa_pass(&view)?;
        } else {
            // Fallback: If LDR is missing (e.g. PostPass wrote to swapchain?), do nothing?
            // Or warn.
            // But if PostPass wrote to swapchain, then FXAA is skipped implicitly in graph flow?
            // No, FXAA Node is executed.
            // If resource missing, we can't draw.
            // But we should ensure consistency.
        }
        Ok(())
    }
}
