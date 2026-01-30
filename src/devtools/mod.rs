//! Developer tools module
//!
//! Provides debugging and development utilities:
//! - UI Inspector for widget hierarchy
//! - Performance Profiler for timing
//! - Plugin system for extensions

pub mod inspector;
pub mod plugin;
pub mod profiler;

pub use inspector::{Inspector, LayoutBounds, PropertyValue, TreeItem, WidgetInfo};
pub use plugin::{
    Plugin, PluginCapabilities, PluginContext, PluginInfo, PluginManager, PluginState,
};
pub use profiler::{
    FrameStats, FrameTiming, PerformanceLevel, Profiler, ProfilerConfig, ScopeTimer,
};
