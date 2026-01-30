//! Shared resources for multi-window contexts
//!
//! Holds Gloabl/Shared handles for OpenGL/Vulkan resources that are shared across
//! window contexts (via context sharing).

use glow::NativeTexture;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
pub struct SharedResources {
    pub gl: Option<Rc<glow::Context>>,
    pub font_texture: Option<NativeTexture>,
}

thread_local! {
    pub static GLOBAL_RESOURCES: RefCell<SharedResources> = RefCell::new(SharedResources::default());
}
