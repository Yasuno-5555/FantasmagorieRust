//! View module - AST definitions and layout

pub mod header;
pub mod views;
pub mod layout;
pub mod interaction;
pub mod renderer;
pub mod animation;

pub use header::{ViewHeader, ViewType, Align};
pub use views::*;
pub use layout::compute_flex_layout;
pub use interaction::{is_hot, is_active, is_focused, begin_interaction_pass};
pub use renderer::render_ui;
