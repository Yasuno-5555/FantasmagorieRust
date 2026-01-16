//! Developer tools module
//!
//! Provides debugging and development utilities:
//! - UI Inspector for widget hierarchy
//! - Performance Profiler for timing
//! - Plugin system for extensions

pub mod inspector;
pub mod profiler;
pub mod plugin;

pub use inspector::{Inspector, WidgetInfo, LayoutBounds, PropertyValue, TreeItem};
pub use profiler::{Profiler, ProfilerConfig, FrameTiming, FrameStats, PerformanceLevel, ScopeTimer};
pub use plugin::{Plugin, PluginManager, PluginInfo, PluginState, PluginContext, PluginCapabilities};
