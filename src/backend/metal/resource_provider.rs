use metal::*;
use crate::backend::hal::{GpuResourceProvider, BufferUsage, TextureDescriptor};
use std::sync::Arc;

pub struct MetalResourceProvider {
    device: Device,
}

impl MetalResourceProvider {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

impl GpuResourceProvider for MetalResourceProvider {
    type Buffer = Buffer;
    type Texture = Texture;
    type TextureView = Texture;
    type Sampler = SamplerState;

    fn create_buffer(&self, size: u64, usage: BufferUsage, _label: &str) -> Result<Self::Buffer, String> {
        let options = match usage {
            BufferUsage::Vertex | BufferUsage::Uniform | BufferUsage::Index => MTLResourceOptions::StorageModeShared,
            BufferUsage::Storage => MTLResourceOptions::StorageModePrivate,
            BufferUsage::CopySrc | BufferUsage::CopyDst => MTLResourceOptions::StorageModeShared,
        };
        
        Ok(self.device.new_buffer(size, options))
    }

    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let mtl_desc = metal::TextureDescriptor::new();
        mtl_desc.set_width(desc.width as u64);
        mtl_desc.set_height(desc.height as u64);
        mtl_desc.set_pixel_format(match desc.format {
            crate::backend::hal::TextureFormat::R8Unorm => MTLPixelFormat::R8Unorm,
            crate::backend::hal::TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
            crate::backend::hal::TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
            crate::backend::hal::TextureFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        });
        
        let mut usage = MTLTextureUsage::ShaderRead;
        if desc.usage.contains(crate::backend::hal::TextureUsage::RENDER_ATTACHMENT) {
            usage |= MTLTextureUsage::RenderTarget;
        }
        if desc.usage.contains(crate::backend::hal::TextureUsage::STORAGE_BINDING) {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        mtl_desc.set_usage(usage);
        
        Ok(self.device.new_texture(&mtl_desc))
    }

    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String> {
        Ok(texture.clone())
    }

    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String> {
        let desc = SamplerDescriptor::new();
        desc.set_label(label);
        desc.set_min_filter(MTLSamplerMinMagFilter::Linear);
        desc.set_mag_filter(MTLSamplerMinMagFilter::Linear);
        Ok(self.device.new_sampler(&desc))
    }

    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]) {
        let ptr = buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());
        }
    }

    fn write_texture(&self, texture: &Self::Texture, data: &[u8], width: u32, height: u32) {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: width as u64, height: height as u64, depth: 1 },
        };
        let bytes_per_row = (width * 4) as u64; // Assume 4 bytes per pixel for now
        texture.replace_region(region, 0, data.as_ptr() as *const _, bytes_per_row);
    }

    fn destroy_buffer(&self, _buffer: Self::Buffer) {
        // Automatic via drop
    }

    fn destroy_texture(&self, _texture: Self::Texture) {
        // Automatic via drop
    }
}
