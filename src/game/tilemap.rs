use crate::core::{Vec2, Rectangle, ColorF};
use crate::resource::TextureId;
use serde::{Serialize, Deserialize};

/// A TileSet defines how a texture is sliced into tiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileSet {
    pub texture: TextureId,
    pub tile_width: u32,
    pub tile_height: u32,
    pub columns: u32,
    pub spacing: u32,
    pub margin: u32,
}

impl TileSet {
    pub fn new(texture: TextureId, tile_width: u32, tile_height: u32) -> Self {
        Self {
            texture,
            tile_width,
            tile_height,
            columns: 0, // Should be calculated or provided
            spacing: 0,
            margin: 0,
        }
    }

    /// Calculate UV rectangle for a specific tile ID
    pub fn get_tile_uv(&self, tile_id: u32, texture_width: u32, texture_height: u32) -> Rectangle {
        if self.columns == 0 { return Rectangle::new(0.0, 0.0, 1.0, 1.0); }
        
        let x = tile_id % self.columns;
        let y = tile_id / self.columns;
        
        let px_x = self.margin + x * (self.tile_width + self.spacing);
        let px_y = self.margin + y * (self.tile_height + self.spacing);
        
        Rectangle::new(
            px_x as f32 / texture_width as f32,
            px_y as f32 / texture_height as f32,
            self.tile_width as f32 / texture_width as f32,
            self.tile_height as f32 / texture_height as f32,
        )
    }
}

/// A TileMap component containing grid data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileMap {
    pub tileset: TileSet,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u32>,
    pub color: ColorF,
    pub visible: bool,
}

impl TileMap {
    pub fn new(tileset: TileSet, width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            tileset,
            width,
            height,
            data: vec![0; size],
            color: ColorF::white(),
            visible: true,
        }
    }

    pub fn set_tile(&mut self, x: u32, y: u32, tile_id: u32) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.data[idx] = tile_id;
        }
    }

    pub fn get_tile(&self, x: u32, y: u32) -> u32 {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.data[idx]
        } else {
            0
        }
    }
}
