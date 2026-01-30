use wgpu::util::DeviceExt;
use std::sync::Arc;
use crate::backend::hal::{GpuResourceProvider, BufferUsage, TextureDescriptor, TextureFormat};

pub struct WgpuResourceProvider {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl WgpuResourceProvider {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }
}

impl GpuResourceProvider for WgpuResourceProvider {
    type Buffer = wgpu::Buffer;
    type Texture = wgpu::Texture;
    type TextureView = wgpu::TextureView;
    type Sampler = wgpu::Sampler;

    fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String> {
        let wgpu_usage = match usage {
            BufferUsage::Vertex => wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Index => wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Storage => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            BufferUsage::CopySrc => wgpu::BufferUsages::COPY_SRC,
            BufferUsage::CopyDst => wgpu::BufferUsages::COPY_DST,
        };

        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu_usage,
            mapped_at_creation: false,
        }))
    }

    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let format = match desc.format {
            TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        };

        let mut usage = wgpu::TextureUsages::empty();
        if desc.usage.contains(crate::backend::hal::TextureUsage::COPY_SRC) { usage |= wgpu::TextureUsages::COPY_SRC; }
        if desc.usage.contains(crate::backend::hal::TextureUsage::COPY_DST) { usage |= wgpu::TextureUsages::COPY_DST; }
        if desc.usage.contains(crate::backend::hal::TextureUsage::TEXTURE_BINDING) { usage |= wgpu::TextureUsages::TEXTURE_BINDING; }
        if desc.usage.contains(crate::backend::hal::TextureUsage::STORAGE_BINDING) { usage |= wgpu::TextureUsages::STORAGE_BINDING; }
        if desc.usage.contains(crate::backend::hal::TextureUsage::RENDER_ATTACHMENT) { usage |= wgpu::TextureUsages::RENDER_ATTACHMENT; }

        Ok(self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label,
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        }))
    }

    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String> {
        Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }

    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String> {
        Ok(self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }))
    }

    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    fn write_texture(&self, texture: &Self::Texture, data: &[u8], width: u32, height: u32) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn destroy_buffer(&self, buffer: Self::Buffer) {
        drop(buffer);
    }

    fn destroy_texture(&self, texture: Self::Texture) {
        drop(texture);
    }

    fn destroy_bind_group(&self, bind_group: Self::BindGroup) {
        drop(bind_group);
    }
}
