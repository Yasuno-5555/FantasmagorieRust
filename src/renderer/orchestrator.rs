use crate::renderer::graph::{RenderGraph, RenderContext, ResourceHandle};
use crate::renderer::nodes::geometry::GeometryNode;
use crate::renderer::nodes::postprocess::{ResolveNode, CaptureNode};
use crate::renderer::nodes::blur::BlurNode;
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use crate::draw::{DrawCommand, DrawList};
use crate::backend::shaders::types::{DrawUniforms, GlobalUniforms, ShapeInstance, create_projection, CinematicParams};
use crate::core::{ColorF, Vec2};
use bytemuck::{self, Pod, Zeroable};

/// Coordinates the execution of RenderTasks across a GpuExecutor
pub struct RenderOrchestrator {
    pub batching_enabled: bool,
}

impl RenderOrchestrator {
    pub fn new() -> Self {
        Self { batching_enabled: true }
    }

    pub fn with_batching(mut self, enabled: bool) -> Self {
        self.batching_enabled = enabled;
        self
    }

    /// Convert a high-level DrawList into an optimized RenderGraph
    pub fn plan<E: GpuExecutor + 'static>(&self, dl: &DrawList, config: &CinematicParams, width: u32, height: u32) -> RenderGraph<E> {
        let mut graph = RenderGraph::new();
        let commands = dl.commands();
        
        let mut current_batch = Vec::new();
        let mut backdrop_captured = false;

        for cmd in commands {
            match cmd {
                DrawCommand::BackdropBlur { .. } | DrawCommand::BlurRect { .. } => {
                    if !backdrop_captured {
                        if !current_batch.is_empty() {
                            graph.add_node(GeometryNode::new(current_batch.drain(..).collect()).with_batching(self.batching_enabled));
                        }
                        graph.add_node(CaptureNode);
                        graph.add_node(BlurNode { sigma: config.blur_radius });
                        backdrop_captured = true;
                    }
                    current_batch.push(cmd.clone());
                }
                _ => {
                    current_batch.push(cmd.clone());
                }
            }
        }

        if !current_batch.is_empty() {
            let mut geo_node = GeometryNode::new(current_batch).with_batching(self.batching_enabled);
            
            // SSR Setup
            if config.bloom_mode >= 2 {
                 // Allocate Aux Buffer (G-Buffer)
                 let aux_desc = TextureDescriptor {
                     width, height,
                     format: TextureFormat::Rgba16Float,
                     usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                     label: Some("GBuffer Aux"),
                 };
                 // Use constant handle defined in graph.rs
                 let aux_h = crate::renderer::graph::REFLECTION_HANDLE; // Reusing constant or defining new?
                 // Wait, REFLECTION_HANDLE is for output. I need AUX_HANDLE.
                 // I will use raw handle for now or add AUX_HANDLE to graph.rs
                 let aux_h = ResourceHandle(12345); // Temporary
                 graph.resources.insert(aux_h, crate::renderer::graph::GraphResourceDesc::Texture(aux_desc.clone()));
                 
                 geo_node.aux_handle = Some(aux_h);
                 
                 // Add SSR Node
                 use crate::renderer::nodes::ssr::SSRNode;
                 use crate::renderer::graph::{HDR_HANDLE, REFLECTION_HANDLE};
                 
                 // Allocate Reflection Output
                 let refl_desc = TextureDescriptor { label: Some("SSR Reflection"), ..aux_desc }; // Same size/format
                 graph.resources.insert(REFLECTION_HANDLE, crate::renderer::graph::GraphResourceDesc::Texture(refl_desc));
                 
                 // I need Depth Handle. Backend should provide it via External Resources if possible.
                 // For now, assume simple case (no depth check or depth is AUX.w).
                 graph.add_node(geo_node);
                 graph.add_node(SSRNode::new(aux_h, ResourceHandle(0), HDR_HANDLE));
                 
                 // Resolve Node needs to know about Reflection Handle.
                 // But ResolveNode doesn't have fields yet?
                 // I'll update ResolveNode later.
            } else {
                 graph.add_node(geo_node);
            }
        }

        graph.add_node(ResolveNode);
        graph
    }

    /// Execute a planned RenderGraph
    pub fn execute<E: GpuExecutor + 'static>(&self, executor: &mut E, graph: &mut RenderGraph<E>, external_resources: std::collections::HashMap<ResourceHandle, crate::renderer::graph::GraphResource<E>>, time: f32, width: u32, height: u32) -> Result<(), String> {
        executor.begin_execute()?;
        graph.execute(executor, external_resources, time, width, height)?;
        executor.end_execute()?;
        executor.present()?;
        Ok(())
    }
}
