use crate::core::{Vec2, Rectangle};

/// 2D Camera for world-space rendering.
/// Per Phase 1 Philosophy: "視点であって物体ではない"
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec2,
    pub zoom: f32,
    pub rotation: f32,
    pub screen_size: Vec2,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
            rotation: 0.0,
            screen_size: Vec2::new(1280.0, 720.0),
        }
    }
}

impl Camera {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            screen_size: Vec2::new(width, height),
            ..Default::default()
        }
    }

    /// Converts world-space coordinates to screen-space.
    pub fn world_to_screen(&self, world_pos: Vec2) -> Vec2 {
        let rel_pos = world_pos - self.position;
        
        // Apply rotation
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        let rot_x = rel_pos.x * cos_r - rel_pos.y * sin_r;
        let rot_y = rel_pos.x * sin_r + rel_pos.y * cos_r;
        
        let zoomed_x = rot_x * self.zoom;
        let zoomed_y = rot_y * self.zoom;
        
        // Center on screen
        Vec2::new(
            zoomed_x + self.screen_size.x * 0.5,
            zoomed_y + self.screen_size.y * 0.5,
        )
    }

    /// Converts screen-space coordinates back to world-space.
    pub fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        let rel_x = (screen_pos.x - self.screen_size.x * 0.5) / self.zoom;
        let rel_y = (screen_pos.y - self.screen_size.y * 0.5) / self.zoom;
        
        // Reverse rotation
        let cos_r = (-self.rotation).cos();
        let sin_r = (-self.rotation).sin();
        let rot_x = rel_x * cos_r - rel_y * sin_r;
        let rot_y = rel_x * sin_r + rel_y * cos_r;
        
        Vec2::new(rot_x + self.position.x, rot_y + self.position.y)
    }

    /// Returns the world-space bounds currently visible.
    pub fn visible_bounds(&self) -> Rectangle {
        let p0 = self.screen_to_world(Vec2::ZERO);
        let p1 = self.screen_to_world(self.screen_size);
        
        Rectangle::new(
            p0.x.min(p1.x),
            p0.y.min(p1.y),
            (p1.x - p0.x).abs(),
            (p1.y - p0.y).abs()
        )
    }
}
