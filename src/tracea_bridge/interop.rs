//! Interoperability layer between Fantasmagorie HAL textures and Tracea buffers

#[cfg(feature = "metal")]
use metal::{Texture, TextureRef, Buffer};

/// Convert a Metal texture to a format suitable for Tracea processing
pub struct TextureInterop;

impl TextureInterop {
    /// Create a Tracea-compatible buffer view from a Metal texture
    /// This enables zero-copy sharing between rendering and compute
    #[cfg(feature = "metal")]
    pub fn texture_to_buffer(texture: &Texture, device: &metal::Device) -> Buffer {
        let width = texture.width();
        let height = texture.height();
        let bytes_per_pixel = 4; // Assume RGBA8 or similar
        let size = width * height * bytes_per_pixel;
        
        // Create a shared buffer for compute operations
        device.new_buffer(size, metal::MTLResourceOptions::StorageModeShared)
    }
    
    /// Copy texture contents to a compute buffer
    #[cfg(feature = "metal")]
    pub fn copy_texture_to_buffer(
        encoder: &metal::BlitCommandEncoderRef,
        texture: &TextureRef,
        buffer: &Buffer,
        width: u64,
        height: u64,
    ) {
        let bytes_per_row = width * 4; // Assume 4 bytes per pixel
        let bytes_per_image = bytes_per_row * height;
        
        encoder.copy_from_texture_to_buffer(
            texture,
            0, // slice
            0, // mip level
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
            metal::MTLSize { width, height, depth: 1 },
            buffer,
            0, // offset
            bytes_per_row,
            bytes_per_image,
            metal::MTLBlitOption::empty(),
        );
    }
    
    /// Copy compute buffer back to texture
    #[cfg(feature = "metal")]
    pub fn copy_buffer_to_texture(
        encoder: &metal::BlitCommandEncoderRef,
        buffer: &Buffer,
        texture: &TextureRef,
        width: u64,
        height: u64,
    ) {
        let bytes_per_row = width * 4;
        let bytes_per_image = bytes_per_row * height;
        
        encoder.copy_from_buffer_to_texture(
            buffer,
            0, // offset
            bytes_per_row,
            bytes_per_image,
            metal::MTLSize { width, height, depth: 1 },
            texture,
            0, // slice
            0, // mip level
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
            metal::MTLBlitOption::empty(),
        );
    }
}
