//! Animation system
//!
//! Comprehensive animation support including:
//! - Keyframe-based timeline animation
//! - Spring physics animation
//! - Sequential and parallel animation groups

pub mod groups;
pub mod keyframe;
pub mod spring;

// Re-export commonly used types
pub use groups::{
    Animatable, Animation, AnimationManager, AnimationState, ParallelGroup, SequentialGroup,
    StaggeredGroup, Tween,
};
pub use keyframe::{easing, Keyframe, KeyframeTrack, LoopMode, PlaybackState, Timeline};
pub use spring::{presets as spring_presets, Spring, Spring2D, SpringColor, SpringConfig};
