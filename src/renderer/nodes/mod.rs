use crate::backend::hal::GpuExecutor;
use crate::renderer::graph::RenderContext;

pub trait RenderNode<E: GpuExecutor>: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String>;
}

pub mod geometry;
pub mod postprocess;
