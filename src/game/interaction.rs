use crate::core::Vec2;
use serde::{Deserialize, Serialize};

/// Simple AABB Collider for spatial triggers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Collider {
    pub offset: Vec2,
    pub size: Vec2,
}

impl Collider {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            offset: Vec2::new(0.0, 0.0),
            size: Vec2::new(width, height),
        }
    }

    pub fn with_offset(mut self, x: f32, y: f32) -> Self {
        self.offset = Vec2::new(x, y);
        self
    }
}

/// Tracks the interaction state of an entity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InteractionState {
    /// Is currently overlapping with something?
    pub is_touched: bool,
    /// Is currently near something?
    pub is_near: bool,
    /// Proximity value (0.0 to 1.0, where 1.0 is touching)
    pub proximity: f32,
    
    /// Internal: Used to detect "entered" or "left" transitions if needed
    pub was_touched: bool,
    pub was_near: bool,
}

impl InteractionState {
    /// Resets the transient states for the next frame's calculation.
    pub fn prepare_next_frame(&mut self) {
        self.was_touched = self.is_touched;
        self.was_near = self.is_near;
        self.is_touched = false;
        self.is_near = false;
        self.proximity = 0.0;
    }

    pub fn just_touched(&self) -> bool {
        self.is_touched && !self.was_touched
    }

    pub fn just_left(&self) -> bool {
        !self.is_touched && self.was_touched
    }
}

/// Helper for AABB overlap check.
pub fn intersects(pos_a: Vec2, col_a: &Collider, pos_b: Vec2, col_b: &Collider) -> bool {
    let a_min = pos_a + col_a.offset - col_a.size * 0.5;
    let a_max = pos_a + col_a.offset + col_a.size * 0.5;
    let b_min = pos_b + col_b.offset - col_b.size * 0.5;
    let b_max = pos_b + col_b.offset + col_b.size * 0.5;

    a_min.x < b_max.x && a_max.x > b_min.x &&
    a_min.y < b_max.y && a_max.y > b_min.y
}

/// Helper for proximity check.
pub fn get_proximity(pos_a: Vec2, col_a: &Collider, pos_b: Vec2, col_b: &Collider, radius: f32) -> f32 {
    let dist = (pos_a + col_a.offset).distance(pos_b + col_b.offset);
    let min_dist = (col_a.size.x.max(col_a.size.y) + col_b.size.x.max(col_b.size.y)) * 0.5;
    
    if dist < min_dist {
        return 1.0;
    }
    
    let range = radius;
    if dist > min_dist + range {
        return 0.0;
    }
    
    1.0 - ((dist - min_dist) / range).clamp(0.0, 1.0)
}
