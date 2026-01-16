//! Fantasmagorie V5 Crystal - High-performance stateless UI framework
//!
//! # Architecture
//! - **Stateless AST**: Views rebuilt each frame on FrameArena
//! - **2-Pass Flexbox**: Measure (bottom-up) → Arrange (top-down)
//! - **SDF Rendering**: Signed Distance Field text/shapes
//!
//! # Philosophy
//! - The Friendly Lie (Builders) → The Strict Truth (POD AST)
//! - State lives in Rust, Logic lives in user code

pub mod core;
pub mod view;
pub mod draw;
pub mod widgets;
pub mod text;
pub mod resource;
pub mod animation;
pub mod devtools;

#[cfg(any(feature = "opengl", feature = "wgpu", feature = "vulkan", feature = "dx12"))]
pub mod backend;

#[cfg(feature = "python")]
pub mod python;

/// Convenient re-exports for common usage
pub mod prelude {
    pub use crate::core::{ColorF, Vec2, Rectangle, ID, FrameArena, Theme};
    pub use crate::view::{ViewHeader, ViewType, Align};
    pub use crate::draw::DrawList;
    pub use crate::widgets::{UIContext, BoxBuilder, TextBuilder, ButtonBuilder};
}

// Also re-export at top level for convenience
pub use crate::core::{ColorF, Vec2, Rectangle, ID};
pub use crate::view::{ViewHeader, ViewType, Align};
pub use crate::draw::DrawList;

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
