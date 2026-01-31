//! Shared resources for multi-window contexts
//!
//! Holds Gloabl/Shared handles for OpenGL/Vulkan resources that are shared across
//! window contexts (via context sharing).

use std::sync::Arc;
use std::cell::RefCell;

#[derive(Default)]
pub struct SharedResources {
    pub device: Option<Arc<wgpu::Device>>,
    pub queue: Option<Arc<wgpu::Queue>>,
}

thread_local! {
    pub static GLOBAL_RESOURCES: RefCell<SharedResources> = RefCell::new(SharedResources::default());
}
