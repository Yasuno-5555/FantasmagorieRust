//! Core types and infrastructure

pub mod a11y;
mod arena;
mod context;
pub mod gesture;
pub mod i18n;
mod id;
pub mod marquee;
pub mod mobile;
pub mod persistence;
pub mod resource;
pub mod snapping;
pub mod theme;
mod types;
pub mod undo;
pub mod wire;

pub use a11y::{
    is_high_contrast_mode, AccessibleInfo, AccessibleRole, AccessibleStore, FocusManager,
};
pub use arena::FrameArena;
pub use context::{EngineContext, InputContext, InteractionState, PersistentState};
pub use gesture::{GestureConfig, GestureDetector, GestureType, SwipeDirection};
pub use id::ID;
pub use marquee::{MarqueeSelection, MarqueeState, Rect, Selectable};
pub use mobile::{
    DesktopPlatform, HapticType, ImeAction, ImeHint, ImePosition, MobilePlatform, SafeAreaInsets,
};
pub use theme::Theme;
pub use types::{ColorF, Rectangle, Vec2, Vec3, WindowID};
pub use undo::{BatchCommand, CallbackCommand, Command, CommandStack};
pub use wire::{Connection, ConnectionResult, Port, PortId, PortType, WireInteraction, WireState};
