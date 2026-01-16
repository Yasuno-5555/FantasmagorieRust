//! Core types and infrastructure

mod types;
mod id;
mod arena;
mod context;
pub mod i18n;
pub mod gesture;
pub mod marquee;
pub mod wire;
pub mod mobile;
pub mod theme;
pub mod a11y;
pub mod undo;

pub use types::{ColorF, Vec2, Rectangle};
pub use id::ID;
pub use arena::FrameArena;
pub use context::{EngineContext, InputContext, PersistentState, InteractionState};
pub use gesture::{GestureDetector, GestureType, SwipeDirection, GestureConfig};
pub use marquee::{MarqueeSelection, MarqueeState, Rect, Selectable};
pub use wire::{WireInteraction, WireState, PortId, PortType, Port, Connection, ConnectionResult};
pub use mobile::{MobilePlatform, DesktopPlatform, ImeHint, ImeAction, ImePosition, HapticType, SafeAreaInsets};
pub use theme::Theme;
pub use a11y::{AccessibleInfo, AccessibleRole, FocusManager, AccessibleStore, is_high_contrast_mode};
pub use undo::{Command, CommandStack, CallbackCommand, BatchCommand};
