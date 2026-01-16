//! Animation system
//!
//! Comprehensive animation support including:
//! - Keyframe-based timeline animation
//! - Spring physics animation
//! - Sequential and parallel animation groups

pub mod keyframe;
pub mod spring;
pub mod groups;

// Re-export commonly used types
pub use keyframe::{Timeline, KeyframeTrack, Keyframe, LoopMode, PlaybackState, easing};
pub use spring::{Spring, Spring2D, SpringColor, SpringConfig, presets as spring_presets};
pub use groups::{
    Animation, AnimationManager, AnimationState,
    Tween, SequentialGroup, ParallelGroup, StaggeredGroup,
    Animatable,
};
