//! Backend module - GPU rendering backends

use crate::draw::DrawList;

/// Common interface for all rendering backends
pub trait Backend {
    /// Get the name of the backend (e.g., "OpenGL", "Vulkan")
    fn name(&self) -> &str;

    /// Render a DrawList to the screen
    fn render(&mut self, dl: &DrawList, width: u32, height: u32);
}

#[cfg(feature = "opengl")]
pub mod opengl;

#[cfg(feature = "opengl")]
pub use opengl::OpenGLBackend;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "wgpu")]
pub use self::wgpu::WgpuBackend;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub use vulkan::VulkanBackend;

#[cfg(feature = "dx12")]
pub mod dx12;

#[cfg(feature = "dx12")]
pub use dx12::Dx12Backend;
