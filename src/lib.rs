//! # Fantasmagorie Engine (Project Crystal)
//!
//! **The Dual-Persona Engine** - A 2D-first, GPU-native rendering engine.
//!
//! ## Profiles
//! - **Lite (Sanity Mode):** "It runs on a toaster." - Embedded, Android, business apps.
//! - **Cinema (Insanity Mode):** "It melts your GPU." - Games, films, high-end demos.
//!
//! ## Architecture
//! - **Layer 1 (User API):** Fluent API / Builder pattern (The Friendly Lie)
//! - **Layer 2 (Tracea):** Optimization, compute, meaning interpretation (The Brain)
//! - **Layer 3 (Renderer):** Abstraction and translation (The Boundary)
//! - **Layer 4 (Backend):** GPU command execution (The Muscle)
//!
//! ## Philosophy
//! - The Friendly Lie (Builders) → The Strict Truth (POD AST)
//! - State lives in Rust, Logic lives in user code

pub mod config;
pub mod animation;
pub mod audio;
pub mod core;
pub mod devtools;
pub mod draw;
pub mod resource;
pub mod text;
pub mod view;
pub mod widgets;
pub mod renderer;
pub use tracea;
pub mod game;

#[cfg(any(
    feature = "opengl",
    feature = "wgpu",
    feature = "vulkan",
    feature = "dx12"
))]
pub mod backend;

#[cfg(feature = "python")]
pub mod python;

/// Convenient re-exports for common usage
pub mod prelude {
    // Core types
    pub use crate::core::{ColorF, FrameArena, Rectangle, Theme, Style, Vec2, ID};
    pub use crate::draw::DrawList;
    pub use crate::view::{Align, ViewHeader, ViewType};
    pub use crate::widgets::{BoxBuilder, ButtonBuilder, TextBuilder, UIContext};
    
    // Engine configuration
    pub use crate::config::{EngineConfig, Profile, ColorSpace, Pipeline, EngineConfigBuilder};
    
    // Renderer
    pub use crate::renderer::Renderer;

    // Interaction shortcuts
    pub use crate::view::interaction::{
        is_active, is_focused, is_hot, is_clicked, is_changed,
        get_mouse_pos, get_mouse_delta, get_scroll_delta
    };
}

// Top-level re-exports
pub use crate::core::{ColorF, Rectangle, Vec2, ID};
pub use crate::draw::DrawList;
pub use crate::view::{Align, ViewHeader, ViewType};
pub use crate::widgets::UIContext;
pub use crate::core::theme::{Theme, Style};

// ============================================================================
// PyO3 Module Entry Point
// ============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Fantasmagorie Rust Edition - Python module
///
/// Usage:
/// ```python
/// import fanta_rust as fanta
///
/// ctx = fanta.Context(1280, 720)
/// ctx.begin_frame()
///
/// fanta.Column().padding(20).bg(fanta.Color(0.1, 0.1, 0.12))
/// fanta.Text("Hello, Ouroboros!").font_size(24)
/// fanta.Button("Click Me").radius(8)
/// fanta.End()
///
/// ctx.end_frame()
/// ```
#[cfg(feature = "python")]
#[pymodule]
fn fanta_rust(py: Python, m: &PyModule) -> PyResult<()> {
    python::bindings::register(py, m)?;

    #[cfg(feature = "opengl")]
    {
        m.add_function(pyo3::wrap_pyfunction!(python::window::py_run_window, m)?)?;
    }

    Ok(())
}
