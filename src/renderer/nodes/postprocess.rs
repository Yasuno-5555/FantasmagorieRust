use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage};
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, BACKDROP_HANDLE};

pub struct ResolveNode;

impl<E: GpuExecutor> RenderNode<E> for ResolveNode {
    fn name(&self) -> &str { "ResolvePass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        // Integrate SSR Reflection if available
        use crate::renderer::graph::{REFLECTION_HANDLE, GraphResource};
        
        if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&REFLECTION_HANDLE) {
            let view = ctx.executor.create_texture_view(tex)?;
            ctx.executor.set_reflection_texture(&view)?;
        }
        
        ctx.executor.resolve()
    }
}

pub struct CaptureNode;

impl<E: GpuExecutor> RenderNode<E> for CaptureNode {
    fn name(&self) -> &str { "CaptureBackdropPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        // Acquire transient texture
        let desc = TextureDescriptor {
            label: Some("Backdrop"),
            width: ctx.width,
            height: ctx.height,
            format: TextureFormat::Rgba8Unorm, // Assuming standard format
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING | TextureUsage::STORAGE_BINDING,
        };
        
        // This effectively aliases memory if the pool has a compatible texture
        let tex = ctx.executor.acquire_transient_texture(&desc)?;
        
        // Copy current screen to this texture
        ctx.executor.copy_framebuffer_to_texture(&tex)?;
        
        // Store for subsequent nodes (e.g. Blur)
        ctx.resources.insert(BACKDROP_HANDLE, GraphResource::Texture(desc, tex));
        
        Ok(())
    }
}
