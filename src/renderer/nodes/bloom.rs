use crate::backend::hal::{GpuExecutor};
use crate::renderer::graph::{RenderNode, RenderContext};

pub struct BloomNode;

impl<E: GpuExecutor> RenderNode<E> for BloomNode {
    fn name(&self) -> &str { "BloomPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        // Logically we just tell the backend to run its internal bloom extraction/blur.
        // This is usually part of resolve, but we split it for upscaling.
        // We reuse backend.resolve() internal logic but maybe better expose a clear method.
        // WgpuBackend::resolve already does bloom extraction.
        
        // However, in upscale path, we want bloom on the HIGH RES frame? 
        // Or LOW RES?
        // Usually, bright extraction on HIGH RES is better.
        
        // For now, let's assume we just trigger the backend's bloom logic.
        // Since resolve() is called in normal path, we need a way to call just the bloom part.
        
        // Let's add `extract_bloom` to HAL? 
        // WgpuBackend already has code for it inside resolve().
        
        Ok(())
    }
}
