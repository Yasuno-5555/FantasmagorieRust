use crate::core::{Rectangle, Vec2};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct GlyphInfo {
    pub uv: Rectangle, // UV coordinates in atlas
    pub size: Vec2,    // Pixel size of glyph
    pub bearing: Vec2, // Offset from pen position
    pub advance: f32,  // Advance width
}

pub struct FontAtlas {
    pub texture_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub glyphs: HashMap<(usize, char, u32), GlyphInfo>, // (font_idx, char, px_size) -> Info

    // Packing state
    current_x: u32,
    current_y: u32,
    row_height: u32,
}

impl FontAtlas {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            texture_data: vec![0; (width * height * 3) as usize], // Triple channel (RGB8)
            width,
            height,
            glyphs: HashMap::new(),
            current_x: 0,
            current_y: 0,
            row_height: 0,
        }
    }

    /// Add a glyph to the atlas
    pub fn pack_glyph(
        &mut self,
        font_idx: usize,
        c: char,
        px_size: u32,
        metrics: fontdue::Metrics,
        bitmap: &[u8],
    ) -> Option<GlyphInfo> {
        let w = metrics.width as u32;
        let h = metrics.height as u32;

        // Add 1px padding
        let padding = 1;
        let pack_w = w + padding * 2;
        let pack_h = h + padding * 2;

        // Check if we need to move to next row
        if self.current_x + pack_w > self.width {
            self.current_x = 0;
            self.current_y += self.row_height;
            self.row_height = 0;
        }

        // Check overflow
        if self.current_y + pack_h > self.height {
            eprintln!("FontAtlas overflow!");
            return None;
        }

        // Copy bitmap to texture (RGB)
        for y in 0..h {
            for x in 0..w {
                let src_idx = (y * w + x) as usize;
                let dest_x = self.current_x + padding + x;
                let dest_y = self.current_y + padding + y;
                let dest_idx = ((dest_y * self.width + dest_x) * 3) as usize;

                if src_idx < bitmap.len() && dest_idx + 2 < self.texture_data.len() {
                    let val = bitmap[src_idx];
                    self.texture_data[dest_idx] = val;     // R
                    self.texture_data[dest_idx + 1] = val; // G
                    self.texture_data[dest_idx + 2] = val; // B
                }
            }
        }

        // Create GlyphInfo
        let u0 = (self.current_x + padding) as f32 / self.width as f32;
        let v0 = (self.current_y + padding) as f32 / self.height as f32;
        let u1 = (self.current_x + padding + w) as f32 / self.width as f32;
        let v1 = (self.current_y + padding + h) as f32 / self.height as f32;

        let info = GlyphInfo {
            uv: Rectangle::new(u0, v0, u1 - u0, v1 - v0),
            size: Vec2::new(w as f32, h as f32),
            bearing: Vec2::new(metrics.xmin as f32, metrics.ymin as f32), // fontdue ymin is from baseline up? No, usually coordinate system depends.
            advance: metrics.advance_width,
        };

        // Cache
        self.glyphs.insert((font_idx, c, px_size), info);

        // Advance packing cursor
        self.current_x += pack_w;
        self.row_height = self.row_height.max(pack_h);

        Some(info)
    }

    pub fn get(&self, font_idx: usize, c: char, px_size: u32) -> Option<&GlyphInfo> {
        self.glyphs.get(&(font_idx, c, px_size))
    }
}
