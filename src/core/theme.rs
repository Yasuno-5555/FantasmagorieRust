use crate::core::ColorF;

/// Layout-related styling constants
#[derive(Clone, Debug)]
pub struct Style {
    pub rounding: f32,
    pub padding: f32,
    pub margin: f32,
    pub spacing: f32,
    pub font_size: f32,
    pub border_width: f32,
    pub elevation: f32,
    pub glow_strength: f32,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            rounding: 8.0,
            padding: 12.0,
            margin: 0.0,
            spacing: 8.0,
            font_size: 14.0,
            border_width: 1.0,
            elevation: 2.0,
            glow_strength: 0.0,
        }
    }
}

/// "Vibe" based Theme System
#[derive(Clone, Debug)]
pub struct Theme {
    /// Main background color (Window/Canvas)
    pub bg: ColorF,
    /// Panel/Container background (usually semi-transparent)
    pub panel: ColorF,
    /// Primary text color
    pub text: ColorF,
    /// Secondary text / Dimmed
    pub text_dim: ColorF,
    /// Accent color (Selection, Active states)
    pub accent: ColorF,
    /// Border color
    pub border: ColorF,
    /// Atmosphere/Glow color (for bloom effects)
    pub atmosphere: ColorF,
    /// Danger/Error color
    pub danger: ColorF,
    /// Success/Safe color
    pub success: ColorF,
}

impl Theme {
    /// Preset: Cyberpunk (Navy / Cyan / Magenta)
    pub fn cyberpunk() -> Self {
        Self {
            bg: ColorF::new(0.01, 0.01, 0.03, 1.0),   // High-spec darkness
            panel: ColorF::new(0.04, 0.04, 0.08, 0.9), // Glassy Navy
            text: ColorF::new(0.9, 0.95, 1.0, 1.0),   // Electric White
            text_dim: ColorF::new(0.4, 0.5, 0.6, 1.0),
            accent: ColorF::new(0.0, 1.0, 0.9, 1.0), // Neon Cyan
            border: ColorF::new(0.0, 0.8, 0.9, 0.4), // Laser Border
            atmosphere: ColorF::new(0.8, 0.0, 1.0, 0.5), // Magenta Vapor
            danger: ColorF::new(1.0, 0.1, 0.3, 1.0),
            success: ColorF::new(0.1, 1.0, 0.6, 1.0),
        }
    }

    /// Preset: Zen (Off-White / Soft Blue / Glass)
    pub fn zen() -> Self {
        Self {
            bg: ColorF::new(0.98, 0.98, 0.99, 1.0), 
            panel: ColorF::new(1.0, 1.0, 1.0, 0.7), 
            text: ColorF::new(0.1, 0.1, 0.15, 1.0), 
            text_dim: ColorF::new(0.5, 0.5, 0.6, 1.0),
            accent: ColorF::new(0.1, 0.5, 0.9, 1.0), 
            border: ColorF::new(0.0, 0.0, 0.0, 0.08), 
            atmosphere: ColorF::new(0.8, 0.9, 1.0, 0.4), 
            danger: ColorF::new(0.9, 0.2, 0.2, 1.0),
            success: ColorF::new(0.2, 0.8, 0.4, 1.0),
        }
    }

    /// Preset: Heat (Grey / Amber / Orange)
    pub fn heat() -> Self {
        Self {
            bg: ColorF::new(0.12, 0.11, 0.1, 1.0), 
            panel: ColorF::new(0.18, 0.17, 0.15, 0.95), 
            text: ColorF::new(0.95, 0.9, 0.85, 1.0), 
            text_dim: ColorF::new(0.6, 0.55, 0.5, 1.0),
            accent: ColorF::new(1.0, 0.5, 0.0, 1.0), 
            border: ColorF::new(0.5, 0.3, 0.1, 0.5), 
            atmosphere: ColorF::new(1.0, 0.2, 0.0, 0.5), 
            danger: ColorF::new(1.0, 0.1, 0.0, 1.0),
            success: ColorF::new(0.6, 0.9, 0.2, 1.0),
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::cyberpunk()
    }
}
