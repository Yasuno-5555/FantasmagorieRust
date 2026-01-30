//! # Engine Configuration (The Soul)
//!
//! This module defines the Dual Persona architecture's core switch.
//! The `Profile` determines the engine's entire behavior at startup.
//!
//! ## Philosophy
//! - **Lite (Sanity):** "It runs on a toaster." - Embedded, Android, business apps.
//! - **Cinema (Insanity):** "It melts your GPU." - Games, films, high-end demos.

/// Engine profile that determines the entire rendering behavior.
///
/// This is set at startup and **cannot be changed at runtime**.
/// Switching profiles requires restarting the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Sanity Mode: Forward rendering, RGBA8, no render graph.
    ///
    /// Target: Embedded devices, Android, Raspberry Pi, business applications.
    /// Philosophy: "It runs on a toaster."
    Lite,

    /// Insanity Mode: Satsuei Pipeline, RGBA16F (HDR), full render graph.
    ///
    /// Target: Games, films, music videos, high-end demos.
    /// Philosophy: "It melts your GPU."
    Cinema,
}

impl Default for Profile {
    fn default() -> Self {
        // Default to safety - Lite mode works everywhere
        Profile::Lite
    }
}

/// Color space configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Standard 8-bit RGBA (LDR) with sRGB transfer function.
    /// Used in Lite mode.
    Srgb,

    /// 16-bit floating point RGBA (HDR) in ACEScg working space.
    /// Used in Cinema mode.
    AcesCg,
}

/// Rendering pipeline type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pipeline {
    /// Single-pass forward rendering.
    /// Call order = Draw order.
    Forward,

    /// Multi-pass Satsuei Pipeline (Cinema only).
    /// Includes Semantics, Composite, and Lens passes.
    Satsuei,
}

/// Main engine configuration.
///
/// # Example
/// ```rust
/// use fantasmagorie::config::{EngineConfig, Profile};
///
/// // For embedded devices
/// let lite = EngineConfig::lite();
///
/// // For high-end rendering
/// let cinema = EngineConfig::cinematic();
///
/// // Custom configuration
/// let custom = EngineConfig::builder()
///     .profile(Profile::Lite)
///     .resolution(1920, 1080)
///     .vsync(true)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// The engine profile (Lite or Cinema).
    pub profile: Profile,

    /// Window/render target width.
    pub width: u32,

    /// Window/render target height.
    pub height: u32,

    /// Enable VSync.
    pub vsync: bool,

    /// Color space (derived from profile, but can be overridden).
    pub color_space: ColorSpace,

    /// Rendering pipeline (derived from profile, but can be overridden).
    pub pipeline: Pipeline,

    /// Maximum number of draw batches (Lite mode optimization).
    pub max_batches: usize,

    /// Enable debug overlays and performance counters.
    pub debug_mode: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::lite()
    }
}

impl EngineConfig {
    /// Create a Lite profile configuration.
    ///
    /// Optimized for embedded devices, mobile, and resource-constrained environments.
    pub fn lite() -> Self {
        Self {
            profile: Profile::Lite,
            width: 1280,
            height: 720,
            vsync: true,
            color_space: ColorSpace::Srgb,
            pipeline: Pipeline::Forward,
            max_batches: 256,
            debug_mode: false,
        }
    }

    /// Create an embedded profile configuration.
    ///
    /// Ultra-minimal settings for extremely constrained devices.
    pub fn embedded() -> Self {
        Self {
            profile: Profile::Lite,
            width: 800,
            height: 480,
            vsync: true,
            color_space: ColorSpace::Srgb,
            pipeline: Pipeline::Forward,
            max_batches: 64, // Conservative batching
            debug_mode: false,
        }
    }

    /// Create a Cinema profile configuration.
    ///
    /// Full-featured HDR rendering with Satsuei Pipeline.
    pub fn cinematic() -> Self {
        Self {
            profile: Profile::Cinema,
            width: 1920,
            height: 1080,
            vsync: false, // Cinema often wants uncapped for performance testing
            color_space: ColorSpace::AcesCg,
            pipeline: Pipeline::Satsuei,
            max_batches: 4096,
            debug_mode: false,
        }
    }

    /// Create a configuration builder for custom setups.
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::new()
    }

    /// Check if the engine is in Lite mode.
    #[inline]
    pub fn is_lite(&self) -> bool {
        self.profile == Profile::Lite
    }

    /// Check if the engine is in Cinema mode.
    #[inline]
    pub fn is_cinema(&self) -> bool {
        self.profile == Profile::Cinema
    }

    /// Check if HDR rendering is enabled.
    #[inline]
    pub fn is_hdr(&self) -> bool {
        self.color_space == ColorSpace::AcesCg
    }

    /// Get the resolution as a tuple.
    #[inline]
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the aspect ratio.
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// Builder for `EngineConfig`.
#[derive(Debug, Clone)]
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    /// Create a new builder with default Lite settings.
    pub fn new() -> Self {
        Self {
            config: EngineConfig::lite(),
        }
    }

    /// Set the engine profile.
    ///
    /// This will also update color_space and pipeline to match the profile's defaults.
    pub fn profile(mut self, profile: Profile) -> Self {
        self.config.profile = profile;
        // Update derived settings based on profile
        match profile {
            Profile::Lite => {
                self.config.color_space = ColorSpace::Srgb;
                self.config.pipeline = Pipeline::Forward;
            }
            Profile::Cinema => {
                self.config.color_space = ColorSpace::AcesCg;
                self.config.pipeline = Pipeline::Satsuei;
            }
        }
        self
    }

    /// Set the resolution.
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// Set VSync.
    pub fn vsync(mut self, enabled: bool) -> Self {
        self.config.vsync = enabled;
        self
    }

    /// Override the color space (use with caution).
    pub fn color_space(mut self, color_space: ColorSpace) -> Self {
        self.config.color_space = color_space;
        self
    }

    /// Set the maximum batch count.
    pub fn max_batches(mut self, count: usize) -> Self {
        self.config.max_batches = count;
        self
    }

    /// Enable debug mode.
    pub fn debug(mut self, enabled: bool) -> Self {
        self.config.debug_mode = enabled;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> EngineConfig {
        self.config
    }
}

impl Default for EngineConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lite_defaults() {
        let config = EngineConfig::lite();
        assert_eq!(config.profile, Profile::Lite);
        assert_eq!(config.color_space, ColorSpace::Srgb);
        assert_eq!(config.pipeline, Pipeline::Forward);
        assert!(config.is_lite());
        assert!(!config.is_cinema());
        assert!(!config.is_hdr());
    }

    #[test]
    fn test_cinema_defaults() {
        let config = EngineConfig::cinematic();
        assert_eq!(config.profile, Profile::Cinema);
        assert_eq!(config.color_space, ColorSpace::AcesCg);
        assert_eq!(config.pipeline, Pipeline::Satsuei);
        assert!(!config.is_lite());
        assert!(config.is_cinema());
        assert!(config.is_hdr());
    }

    #[test]
    fn test_builder() {
        let config = EngineConfig::builder()
            .profile(Profile::Cinema)
            .resolution(3840, 2160)
            .vsync(true)
            .debug(true)
            .build();

        assert_eq!(config.profile, Profile::Cinema);
        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
        assert!(config.vsync);
        assert!(config.debug_mode);
    }

    #[test]
    fn test_aspect_ratio() {
        let config = EngineConfig::builder()
            .resolution(1920, 1080)
            .build();
        
        let ratio = config.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.001);
    }
}
