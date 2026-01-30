//! Python bindings for Fantasmagorie
//!
//! Provides PyO3 bindings to enable Python hot-reload workflow (Ouroboros)

#[cfg(feature = "python")]
pub mod bindings;

#[cfg(all(feature = "python", feature = "opengl"))]
pub mod buffer;
#[cfg(all(feature = "python", feature = "opengl"))]
pub mod dlpack;
#[cfg(all(feature = "python", feature = "opengl"))]
pub mod window;
