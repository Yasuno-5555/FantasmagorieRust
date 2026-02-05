use std::collections::HashMap;
use crate::backend::hal::{GpuExecutor, TextureDescriptor, BufferUsage};

pub type ResourceHandle = u32;

pub const HDR_HANDLE: ResourceHandle = 1;
pub const BACKDROP_HANDLE: ResourceHandle = 2;
pub const AUX_HANDLE: ResourceHandle = 3;
pub const DEPTH_HANDLE: ResourceHandle = 4;
pub const VELOCITY_HANDLE: ResourceHandle = 5;
pub const REFLECTION_HANDLE: ResourceHandle = 6;
pub const EXTRA_HANDLE: ResourceHandle = 8;
pub const SDF_HANDLE: ResourceHandle = 7;
pub const PARTICLE_HANDLE: ResourceHandle = 15;
pub const SDF_TEMP_A_HANDLE: ResourceHandle = 100;
pub const SDF_TEMP_B_HANDLE: ResourceHandle = 101;
pub const LUT_HANDLE: ResourceHandle = 20;

// Phase 6 Handles
pub const HDR_LOW_RES_HANDLE: ResourceHandle = 30;
pub const HDR_HIGH_RES_HANDLE: ResourceHandle = 31;
pub const LDR_HANDLE: ResourceHandle = 32; // Input to FXAA

#[derive(Clone, Debug)]
pub enum GraphResourceDesc {
    Texture(TextureDescriptor),
    Buffer { size: u64, usage: BufferUsage, label: &'static str },
}

pub enum GraphResource<E: GpuExecutor> {
    Texture(TextureDescriptor, E::Texture),
    Buffer(u64, BufferUsage, E::Buffer),
}

impl<E: GpuExecutor> GraphResource<E> {
    pub fn into_desc(&self) -> GraphResourceDesc {
        match self {
            GraphResource::Texture(desc, _) => GraphResourceDesc::Texture(desc.clone()),
            GraphResource::Buffer(size, usage, label) => GraphResourceDesc::Buffer { size: *size, usage: *usage, label: "External" },
        }
    }
}

pub struct RenderContext<'a, E: GpuExecutor> {
    pub executor: &'a mut E,
    pub resources: HashMap<ResourceHandle, GraphResource<E>>,
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub jitter: (f32, f32),
}

pub trait RenderNode<E: GpuExecutor> {
    fn name(&self) -> &str;
    fn execute(&mut self, ctx: &mut RenderContext<E>) -> Result<(), String>;
}

pub struct RenderGraph<E: GpuExecutor> {
    pub resources: HashMap<ResourceHandle, GraphResourceDesc>,
    pub nodes: Vec<Box<dyn RenderNode<E>>>,
}

impl<E: GpuExecutor> RenderGraph<E> {
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            nodes: Vec::new(),
        }
    }

    pub fn add_node<N: RenderNode<E> + 'static>(&mut self, node: N) {
        self.nodes.push(Box::new(node));
    }

    pub fn execute(&mut self, executor: &mut E, external_resources: HashMap<ResourceHandle, GraphResource<E>>, time: f32, width: u32, height: u32, jitter: (f32, f32)) -> Result<(), String> {
        let mut ctx = RenderContext {
            executor,
            resources: external_resources,
            time,
            width,
            height,
            jitter,
        };

        // Allocate declared resources
        for (handle, desc) in &self.resources {
            if ctx.resources.contains_key(handle) {
                continue;
            }
            match desc {
                GraphResourceDesc::Texture(tex_desc) => {
                    let tex = ctx.executor.acquire_transient_texture(tex_desc)?;
                    ctx.resources.insert(*handle, GraphResource::Texture(tex_desc.clone(), tex));
                }
                GraphResourceDesc::Buffer { size, usage, label } => {
                     let buf = ctx.executor.create_buffer(*size, *usage, label)?;
                     ctx.resources.insert(*handle, GraphResource::Buffer(*size, *usage, buf));
                }
            }
        }

        for node in &mut self.nodes {
            node.execute(&mut ctx)?;
        }

        Ok(())
    }
}

pub struct TransientPool<E: GpuExecutor> {
    textures: HashMap<TextureDescriptor, Vec<E::Texture>>,
}

impl<E: GpuExecutor> TransientPool<E> {
    pub fn new() -> Self {
        Self { textures: HashMap::new() }
    }

    pub fn acquire_texture(&mut self, executor: &E, desc: &TextureDescriptor) -> Result<E::Texture, String> {
        if let Some(pool) = self.textures.get_mut(desc) {
            if let Some(tex) = pool.pop() {
                return Ok(tex);
            }
        }
        executor.create_texture(desc)
    }

    pub fn release_texture(&mut self, texture: E::Texture, desc: &TextureDescriptor) {
        self.textures.entry(desc.clone()).or_insert_with(Vec::new).push(texture);
    }
}
