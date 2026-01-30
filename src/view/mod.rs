//! View module - AST definitions

pub mod animation;
pub mod curves;
pub mod gizmo;
pub mod grid;
pub mod header;
pub mod interaction;
pub mod layout;
pub mod plot;
pub mod renderer;
pub mod ruler;
pub mod scene3d;
pub mod views;

pub use header::{Align, ViewHeader, ViewType};
pub use interaction::{begin_interaction_pass, is_active, is_focused, is_hot};
pub use layout::compute_flex_layout;
pub use renderer::render_ui;
pub use views::*;
