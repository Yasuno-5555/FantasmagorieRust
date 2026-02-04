use crate::backend::hal::{GpuExecutor, TextureDescriptor, BufferUsage, TextureUsage, TextureFormat};
use std::collections::{HashMap, VecDeque};
use std::any::Any;

/// Handle to a resource within the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle(pub u32);
pub const BACKDROP_HANDLE: ResourceHandle = ResourceHandle(10);

/// Types of resources managed by the graph
pub enum ResourceType {
    Texture,
    Buffer,
}

#[derive(Debug, Clone)]
pub enum GraphResourceDesc {
    Texture(TextureDescriptor),
    Buffer { size: u64, usage: BufferUsage, label: String },
}

/// A pool for reusing GPU resources across frames
pub struct TransientPool<E: GpuExecutor> {
    textures: HashMap<(u32, u32, TextureFormat, TextureUsage), VecDeque<E::Texture>>,
    // Buffers could be added here
}

impl<E: GpuExecutor> TransientPool<E> {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
        }
    }

    pub fn acquire_texture(&mut self, executor: &E, desc: &TextureDescriptor) -> Result<E::Texture, String> {
        let key = (desc.width, desc.height, desc.format, desc.usage);
        if let Some(queue) = self.textures.get_mut(&key) {
            if let Some(tex) = queue.pop_front() {
                return Ok(tex);
            }
        }
        executor.create_texture(desc)
    }

    pub fn release_texture(&mut self, texture: E::Texture, desc: &TextureDescriptor) {
        let key = (desc.width, desc.height, desc.format, desc.usage);
        self.textures.entry(key).or_insert_with(VecDeque::new).push_back(texture);
    }
    
    pub fn clear(&mut self, executor: &E) {
        for (_, queue) in self.textures.iter_mut() {
            for tex in queue.drain(..) {
                executor.destroy_texture(tex);
            }
        }
    }
}

#[derive(Debug)]
pub enum GraphResource<E: GpuExecutor> {
    Texture(TextureDescriptor, E::Texture), // Store descriptor for release
    Buffer(u64, BufferUsage, E::Buffer),
}

/// Context passed to each node during execution
pub struct RenderContext<'a, E: GpuExecutor> {
    pub executor: &'a mut E,
    pub resources: HashMap<ResourceHandle, GraphResource<E>>,
    pub time: f32,
    pub width: u32,
    pub height: u32,
}

/// A single step in the render graph
pub trait RenderNode<E: GpuExecutor>: Any + Send + Sync {
    fn name(&self) -> &str;
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String>;
}

/// The RenderGraph manages a sequence of rendering passes and their resources
pub struct RenderGraph<E: GpuExecutor> {
    nodes: Vec<Box<dyn RenderNode<E>>>,
    pub resources: HashMap<ResourceHandle, GraphResourceDesc>,
    next_handle: u32,
}

impl<E: GpuExecutor + 'static> RenderGraph<E> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            resources: HashMap::new(),
            next_handle: 0,
        }
    }

    pub fn add_node<N: RenderNode<E> + 'static>(&mut self, node: N) {
        self.nodes.push(Box::new(node));
    }

    pub fn execute(&mut self, executor: &mut E, time: f32, width: u32, height: u32) -> Result<(), String> {
        let mut ctx = RenderContext {
            executor,
            resources: HashMap::new(),
            time,
            width,
            height,
        };

        for node in &mut self.nodes {
            node.execute(&mut ctx)?;
        }

        // Return resources to pool
        for (_, res) in ctx.resources {
            match res {
                GraphResource::Texture(desc, tex) => {
                    ctx.executor.release_transient_texture(tex, &desc);
                }
                GraphResource::Buffer(_, _, buf) => {
                    // Pool support for buffers not yet implemented
                    ctx.executor.destroy_buffer(buf);
                }
            }
        }

        Ok(())
    }
}
