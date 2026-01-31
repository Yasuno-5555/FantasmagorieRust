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

    pub fn create_buffer(&self, size: u64, usage: BufferUsage, _label: &str) -> Result<Buffer, String> {
        let options = match usage {
            BufferUsage::Vertex | BufferUsage::Uniform | BufferUsage::Index => MTLResourceOptions::StorageModeShared,
            BufferUsage::Storage => MTLResourceOptions::StorageModePrivate,
            BufferUsage::CopySrc | BufferUsage::CopyDst => MTLResourceOptions::StorageModeShared,
        };
        
        Ok(self.device.new_buffer(size, options))
    }

    pub fn create_texture(&self, desc: &TextureDescriptor) -> Result<Texture, String> {
        let mtl_desc = metal::TextureDescriptor::new();
        mtl_desc.set_width(desc.width as u64);
        mtl_desc.set_height(desc.height as u64);
        mtl_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm); // Default to RGBA8 for now
        mtl_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
        
        Ok(self.device.new_texture(&mtl_desc))
    }

    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        let ptr = buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());
        }
    }

    pub fn write_texture(&self, texture: &Texture, data: &[u8], width: u32, height: u32) {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: width as u64, height: height as u64, depth: 1 },
        };
        let bytes_per_row = (width * 4) as u64;
        texture.replace_region(region, 0, data.as_ptr() as *const _, bytes_per_row);
    }
}
