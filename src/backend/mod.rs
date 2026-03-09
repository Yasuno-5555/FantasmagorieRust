//! Backend module - GPU rendering backends
//! 
//! GraphicsBackend is the "Muscle" - it strictly executes drawing commands.
//! All GPU rendering goes through this trait, which is under Tracea's coordination.

use crate::draw::DrawList;
use crate::renderer::packet::DrawPacket;
use crate::config::CinematicConfig;

/// Shared shader definitions and uniform types
pub mod shaders;

/// Unified Hardware Abstraction Layer traits
pub mod hal;

/// Common interface for all graphics rendering backends.
/// This is "The Muscle" - no optimization logic allowed here.
/// Tracea is the coordinator that prepares work for GraphicsBackend.
pub trait GraphicsBackend {
    /// Get the backend as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
    /// Get the backend as Any mut for downcasting
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Get the name of the backend (e.g., "OpenGL", "Vulkan")
    fn name(&self) -> &str;

    /// High-level API: Render a DrawList to the screen (for immediate mode)
    fn render(&mut self, dl: &DrawList, width: u32, height: u32);
    
    /// Low-level API: Submit prepared DrawPackets (for optimized path)
    fn submit(&mut self, _packets: &[DrawPacket]) {
        // Default no-op, override in backends that support DrawPacket
    }
    
    /// Update the font texture atlas
    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]);

    /// Present the frame to screen
    fn present(&mut self) {
        // Default no-op, override if needed
    }

    /// Capture the current frame as an image file
    fn capture_screenshot(&mut self, _path: &str) {
        // Default no-op
    }

    /// Get GPU profiling results (timestamps and period)
    fn get_profiling_results(&self) -> Option<(Vec<u64>, f32)> {
        None
    }

    /// Update audio spectrum data (Phase 5)
    fn update_audio_data(&mut self, _spectrum: &[f32]) {
        // Default no-op
    }

    /// Update audio PCM data (for GPU FFT)
    fn update_audio_pcm(&mut self, _samples: &[f32]) {
        // Default no-op
    }

    /// Set cinematic parameters
    fn set_cinematic_config(&mut self, _config: CinematicConfig) {
        // Default no-op
    }

    /// Update the LUT 3D texture
    fn update_lut_texture(&mut self, _data: &[f32], _size: u32) {
        // Default no-op
    }

    /// Set internal resolution scale for upscaling (0.5 = 50% resolution)
    fn set_resolution_scale(&mut self, _scale: f32) {
        // Default no-op
    }

    /// Draw a tilemap
    fn draw_tilemap(
        &mut self,
        _params: crate::backend::shaders::types::TilemapParams,
        _data: &[u32],
        _texture_id: u64,
        _color: crate::core::ColorF
    ) {
        // Default no-op
    }
}

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "wgpu")]
pub use self::wgpu::WgpuBackend;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "metal")]
pub use self::metal::MetalBackend;
