use self::atlas::{FontAtlas, GlyphInfo};
use crate::core::Vec2;
use std::cell::RefCell;
use std::fs;

pub mod atlas;
pub mod markdown;
pub mod math;

pub use fontdue;

pub struct FontManager {
    pub fonts: Vec<fontdue::Font>,
    pub atlas: FontAtlas,
    pub texture_dirty: bool,
}

thread_local! {
    pub static FONT_MANAGER: RefCell<FontManager> = RefCell::new(FontManager::new());
}

impl FontManager {
    pub fn new() -> Self {
        Self {
            fonts: Vec::new(),
            atlas: FontAtlas::new(1024, 1024),
            texture_dirty: false,
        }
    }

    pub fn load_font_from_bytes(&mut self, data: &[u8]) -> usize {
        let settings = fontdue::FontSettings::default();
        let font = fontdue::Font::from_bytes(data, settings).expect("Failed to load font");
        let idx = self.fonts.len();
        self.fonts.push(font);
        idx
    }

    pub fn load_system_font(&mut self) -> usize {
        // Try Windows paths
        let paths = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", // Linux fallback
        ];

        for path in paths {
            if let Ok(bytes) = fs::read(path) {
                // println!("Loaded font: {}", path);
                return self.load_font_from_bytes(&bytes);
            }
        }

        eprintln!("Warning: No system font found!");
        0
    }

    pub fn load_icon_font(&mut self) -> usize {
        let paths = [
            "assets/fonts/remixicon.ttf",
            "assets/fonts/fa-solid-900.ttf",
            "C:/Windows/Fonts/segmdl2.ttf", // Segoe MDL2 Assets (Windows 10/11)
        ];

        for path in paths {
            if let Ok(bytes) = fs::read(path) {
                // println!("Loaded icon font: {}", path);
                return self.load_font_from_bytes(&bytes);
            }
        }

        // If we failed to find an icon font, we can just return 0 (using system font as fallback)
        // or push a clone of the system font so indices don't panic
        if !self.fonts.is_empty() {
            let font = self.fonts[0].clone();
            self.fonts.push(font);
            return 1;
        }

        eprintln!("Warning: No icon font found!");
        0
    }

    pub fn init_fonts(&mut self) {
        if self.fonts.is_empty() {
            self.load_system_font();
            self.load_icon_font();
        }
    }

    /// Measure text dimensions without rasterizing
    pub fn measure_text(&self, text: &str, size: f32) -> Vec2 {
        if self.fonts.is_empty() {
            return Vec2::ZERO;
        }
        
        let mut width = 0.0f32;

        for c in text.chars() {
            // Find font that has this glyph
            let mut found = false;
            for font in &self.fonts {
                if font.lookup_glyph_index(c) != 0 || c.is_whitespace() {
                    let metrics = font.metrics(c, size);
                    width += metrics.advance_width;
                    found = true;
                    break;
                }
            }
            if !found {
                // Fallback to first font's advance for missing glyph
                let metrics = self.fonts[0].metrics(c, size);
                width += metrics.advance_width;
            }
        }
        
        let line_height = if !self.fonts.is_empty() {
            match self.fonts[0].horizontal_line_metrics(size) {
                Some(m) => m.new_line_size,
                None => size * 1.2,
            }
        } else {
            size * 1.2
        };

        Vec2::new(width, line_height)
    }

    /// Find character index at given X offset
    pub fn find_index_at_x(&self, text: &str, size: f32, target_x: f32) -> usize {
        if self.fonts.is_empty() { return 0; }
        
        let font = &self.fonts[0];
        let mut current_x = 0.0;
        
        for (i, c) in text.chars().enumerate() {
            let metrics = font.metrics(c, size);
            let advance = metrics.advance_width;
            
            // Check if target_x is within this char's width (or closer to center)
            if target_x < current_x + advance * 0.5 {
                return i;
            }
            
            current_x += advance;
        }
        
        text.chars().count() // End
    }

    /// Get vertical metrics (ascent, descent, line_gap)
    pub fn vertical_metrics(&self, size: f32) -> Option<(f32, f32, f32)> {
        if self.fonts.is_empty() { return None; }
        // Use first font as reference for vertical metrics
        self.fonts[0].horizontal_line_metrics(size).map(|m| (m.ascent, m.descent, m.line_gap))
    }

    /// Get glyph info, rasterizing if necessary
    pub fn get_glyph(&mut self, font_idx: usize, c: char, size: f32) -> Option<GlyphInfo> {
        let px_size = size as u32;
        // 1. Try preferred font
        if let Some(info) = self.atlas.get(font_idx, c, px_size) {
            return Some(*info);
        }

        if font_idx < self.fonts.len() && (self.fonts[font_idx].lookup_glyph_index(c) != 0 || c.is_whitespace()) {
            let (metrics, bitmap) = self.fonts[font_idx].rasterize(c, size);
            if let Some(info) = self.atlas.pack_glyph(font_idx, c, px_size, metrics, &bitmap) {
                self.texture_dirty = true;
                return Some(info);
            }
        }

        // 2. Fallback: try all other fonts
        for (idx, font) in self.fonts.iter().enumerate() {
            if idx == font_idx { continue; }
            
            if let Some(info) = self.atlas.get(idx, c, px_size) {
                return Some(*info);
            }

            if font.lookup_glyph_index(c) != 0 || c.is_whitespace() {
                let (metrics, bitmap) = font.rasterize(c, size);
                if let Some(info) = self.atlas.pack_glyph(idx, c, px_size, metrics, &bitmap) {
                    self.texture_dirty = true;
                    return Some(info);
                }
            }
        }

        None
    }
}
