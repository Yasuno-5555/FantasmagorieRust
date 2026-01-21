use crate::core::ColorF;

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
    /// High contrast, neon vibes, deep dark background.
    pub fn cyberpunk() -> Self {
        Self {
            bg: ColorF::new(0.02, 0.02, 0.05, 1.0),   // Deep Navy/Black
            panel: ColorF::new(0.05, 0.05, 0.1, 0.8), // Dark Glass
            text: ColorF::new(0.9, 0.95, 1.0, 1.0),   // Cyan-White
            text_dim: ColorF::new(0.4, 0.5, 0.6, 1.0),
            accent: ColorF::new(0.0, 1.0, 0.9, 1.0), // Cyan Neon
            border: ColorF::new(0.0, 0.8, 0.9, 0.3), // Faint Cyan
            atmosphere: ColorF::new(0.8, 0.0, 1.0, 0.5), // Magenta Glow
            danger: ColorF::new(1.0, 0.2, 0.4, 1.0),
            success: ColorF::new(0.2, 1.0, 0.5, 1.0),
        }
    }

    /// Preset: Zen (Off-White / Soft Blue / Glass)
    /// Clean, modern, "Apple-like" aesthetic.
    pub fn zen() -> Self {
        Self {
            bg: ColorF::new(0.96, 0.96, 0.97, 1.0), // Off-White
            panel: ColorF::new(1.0, 1.0, 1.0, 0.6), // Frosted Glass white
            text: ColorF::new(0.1, 0.1, 0.15, 1.0), // Dark Slate
            text_dim: ColorF::new(0.5, 0.5, 0.55, 1.0),
            accent: ColorF::new(0.2, 0.5, 0.9, 1.0), // Soft Blue
            border: ColorF::new(0.0, 0.0, 0.0, 0.05), // Very subtle grey
            atmosphere: ColorF::new(0.8, 0.9, 1.0, 0.3), // ambient cool air
            danger: ColorF::new(0.9, 0.3, 0.3, 1.0),
            success: ColorF::new(0.3, 0.8, 0.4, 1.0),
        }
    }

    /// Preset: Heat (Grey / Amber / Orange)
    /// Analog gear, vacuum tubes, warm, industrial.
    pub fn heat() -> Self {
        Self {
            bg: ColorF::new(0.15, 0.15, 0.15, 1.0), // Industrial Grey
            panel: ColorF::new(0.1, 0.1, 0.1, 0.9), // Darker Panel
            text: ColorF::new(0.9, 0.85, 0.8, 1.0), // Warm White
            text_dim: ColorF::new(0.5, 0.45, 0.4, 1.0),
            accent: ColorF::new(1.0, 0.6, 0.0, 1.0), // Amber/Orange
            border: ColorF::new(0.4, 0.25, 0.1, 0.5), // Rusty/Copper
            atmosphere: ColorF::new(1.0, 0.3, 0.0, 0.4), // Heat radiation
            danger: ColorF::new(1.0, 0.3, 0.0, 1.0), // Red-Orange
            success: ColorF::new(0.5, 0.8, 0.2, 1.0),
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::cyberpunk()
    }
}
