use std::collections::HashMap;
use std::sync::Arc;

pub type TextureId = String;

pub struct RenderContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub resources: &'a HashMap<TextureId, Arc<wgpu::TextureView>>,
    pub time: f64,
}

pub type PassExecutor<T> = Box<dyn Fn(&mut RenderContext, &mut T) + Send + Sync>;

pub struct RenderPass<T> {
    pub name: String,
    pub inputs: Vec<TextureId>,
    pub outputs: Vec<TextureId>,
    pub execute: PassExecutor<T>,
}

pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}

pub struct RenderGraph<T> {
    pub passes: Vec<RenderPass<T>>,
    pub resource_info: HashMap<TextureId, TextureDesc>,
    pub internal_textures: HashMap<TextureId, (wgpu::Texture, Arc<wgpu::TextureView>)>,
}

impl<T> RenderGraph<T> {
    pub fn new() -> Self {
        Self { 
            passes: Vec::new(),
            resource_info: HashMap::new(),
            internal_textures: HashMap::new(),
        }
    }

    pub fn create_texture(&mut self, name: impl Into<String>, width: u32, height: u32, format: wgpu::TextureFormat) {
        self.resource_info.insert(name.into(), TextureDesc {
            width,
            height,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        });
    }

    pub fn add_pass(&mut self, name: impl Into<String>, inputs: Vec<TextureId>, outputs: Vec<TextureId>, execute: PassExecutor<T>) {
        self.passes.push(RenderPass {
            name: name.into(),
            inputs,
            outputs,
            execute,
        });
    }

    pub fn execute(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, external_resources: &HashMap<TextureId, Arc<wgpu::TextureView>>, time: f64, user_ctx: &mut T) {
        // 1. Ensure all virtual resources are allocated
        for (id, desc) in &self.resource_info {
            if !external_resources.contains_key(id) && !self.internal_textures.contains_key(id) {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(id),
                    size: wgpu::Extent3d { width: desc.width, height: desc.height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: desc.format,
                    usage: desc.usage,
                    view_formats: &[],
                });
                let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                self.internal_textures.insert(id.clone(), (texture, view));
            }
        }
        
        // 2. Merge internal and external views for the context
        let mut context_resources = external_resources.clone();
        for (id, (_tex, view)) in &self.internal_textures {
            context_resources.insert(id.clone(), view.clone());
        }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Graph Encoder"),
        });

        {
            let mut ctx = RenderContext {
                device,
                queue,
                encoder: &mut encoder,
                resources: &context_resources,
                time,
            };

            for pass in &self.passes {
                (pass.execute)(&mut ctx, user_ctx);
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}
