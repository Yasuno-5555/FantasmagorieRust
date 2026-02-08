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

/// Bloom style configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Bloom {
    /// No bloom applied
    None,
    /// Subtle, high-quality gaussian bloom
    Soft,
    /// High-intensity, wide-spread bloom
    Cinematic,
}

/// Tone mapping algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Tonemap {
    /// No tone mapping (raw HDR output, likely clamped)
    None,
    /// ACES Filmic tone mapping
    Aces,
    /// Reinhard tone mapping
    Reinhard,
}

/// Upscaling algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UpscalerType {
    /// No upscaling (internal resolution = output resolution)
    None,
    /// Simple linear interpolation
    Linear,
    /// High-quality shader-based upscaling (FSR-style)
    HighQuality,
    /// NVIDIA Deep Learning Super Sampling (requires SDK)
    Dlss,
}

/// Detailed configuration for Cinema profile effects
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CinematicConfig {
    /// Bloom style
    pub bloom: Bloom,
    /// Tone mapping style
    pub tonemap: Tonemap,
    /// Upscaling type
    pub upscaler: UpscalerType,
    /// Overall exposure adjustment
    pub exposure: f32,
    /// Chromatic aberration strength
    pub chromatic_aberration: f32,
    /// Vignette intensity
    pub vignette: f32,
    /// Film grain strength
    pub grain_strength: f32,
    /// LUT intensity
    pub lut_intensity: f32,
    /// Depth-based blur radius
    pub blur_radius: f32,
    /// Motion blur strength
    pub motion_blur_strength: f32,
    /// Debug mode (0: None, 1: Velocity, 2: Normals, etc.)
    pub debug_mode: u32,
    /// Global Illumination intensity (Cone Tracing)
    pub gi_intensity: f32,
    /// Volumetric lighting intensity (God Rays)
    pub volumetric_intensity: f32,
    /// Light position in screen space
    pub light_pos: [f32; 2],
    /// Light color (RGBA)
    pub light_color: [f32; 4],
}

impl Default for CinematicConfig {
    fn default() -> Self {
        Self {
            bloom: Bloom::Soft,
            tonemap: Tonemap::Aces,
            upscaler: UpscalerType::None,
            exposure: 1.0,
            chromatic_aberration: 0.0015,
            vignette: 0.7,
            grain_strength: 0.00,
            lut_intensity: 0.0,
            blur_radius: 0.0,
            motion_blur_strength: 0.0,
            debug_mode: 0,
            gi_intensity: 0.5,
            volumetric_intensity: 0.0,
            light_pos: [500.0, 300.0],
            light_color: [1.0, 0.9, 0.7, 1.0],
        }
    }
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

    /// Cinematic parameters (active only in Cinema mode)
    pub cinematic: CinematicConfig,

    /// Internal resolution scale (e.g. 0.5 for 50% scale upsizing)
    pub internal_resolution_scale: f32,
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
            cinematic: CinematicConfig::default(),
            internal_resolution_scale: 1.0,
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
            cinematic: CinematicConfig::default(),
            internal_resolution_scale: 1.0,
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
            cinematic: CinematicConfig::default(),
            internal_resolution_scale: 1.0,
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

    /// Set cinematic configuration.
    pub fn cinematic(mut self, cinematic: CinematicConfig) -> Self {
        self.config.cinematic = cinematic;
        self
    }

    pub fn with_bloom(mut self, bloom: Bloom) -> Self {
        self.config.cinematic.bloom = bloom;
        self
    }

    pub fn with_tonemap(mut self, tonemap: Tonemap) -> Self {
        self.config.cinematic.tonemap = tonemap;
        self
    }

    pub fn with_exposure(mut self, exposure: f32) -> Self {
        self.config.cinematic.exposure = exposure;
        self
    }

    pub fn with_ca(mut self, ca: f32) -> Self {
        self.config.cinematic.chromatic_aberration = ca;
        self
    }

    pub fn with_vignette(mut self, vignette: f32) -> Self {
        self.config.cinematic.vignette = vignette;
        self
    }

    pub fn with_grain(mut self, grain: f32) -> Self {
        self.config.cinematic.grain_strength = grain;
        self
    }

    pub fn with_gi(mut self, intensity: f32) -> Self {
        self.config.cinematic.gi_intensity = intensity;
        self
    }

    pub fn with_volumetrics(mut self, intensity: f32) -> Self {
        self.config.cinematic.volumetric_intensity = intensity;
        self
    }

    pub fn with_light_pos(mut self, x: f32, y: f32) -> Self {
        self.config.cinematic.light_pos = [x, y];
        self
    }

    pub fn with_light_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.config.cinematic.light_color = [r, g, b, a];
        self
    }

    pub fn resolution_scale(mut self, scale: f32) -> Self {
        self.config.internal_resolution_scale = scale;
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
