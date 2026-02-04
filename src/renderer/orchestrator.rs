use crate::renderer::graph::{RenderGraph, RenderContext};
use crate::renderer::nodes::geometry::GeometryNode;
use crate::renderer::nodes::postprocess::{ResolveNode, CaptureNode};
use crate::backend::hal::{GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use crate::draw::{DrawCommand, DrawList};
use crate::backend::shaders::types::{DrawUniforms, GlobalUniforms, ShapeInstance, create_projection};
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
    pub fn plan<E: GpuExecutor + 'static>(&self, dl: &DrawList) -> RenderGraph<E> {
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
            graph.add_node(GeometryNode::new(current_batch).with_batching(self.batching_enabled));
        }

        graph.add_node(ResolveNode);
        graph
    }

    /// Execute a planned RenderGraph
    pub fn execute<E: GpuExecutor + 'static>(&self, executor: &mut E, graph: &mut RenderGraph<E>, time: f32, width: u32, height: u32) -> Result<(), String> {
        executor.begin_execute()?;
        graph.execute(executor, time, width, height)?;
        executor.end_execute()?;
        executor.present()?;
        Ok(())
    }
}
