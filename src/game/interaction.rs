use crate::core::Vec2;
use serde::{Deserialize, Serialize};

/// Collider types for spatial triggers and physics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Collider {
    AABB { offset: Vec2, size: Vec2 },
    Circle { offset: Vec2, radius: f32 },
    Polygon { offset: Vec2, vertices: Vec<Vec2> },
}

impl Collider {
    pub fn aabb(width: f32, height: f32) -> Self {
        Self::AABB {
            offset: Vec2::new(0.0, 0.0),
            size: Vec2::new(width, height),
        }
    }

    pub fn circle(radius: f32) -> Self {
        Self::Circle {
            offset: Vec2::new(0.0, 0.0),
            radius,
        }
    }
    
    pub fn polygon(vertices: Vec<Vec2>) -> Self {
        Self::Polygon {
            offset: Vec2::new(0.0, 0.0),
            vertices,
        }
    }

    pub fn get_offset(&self) -> Vec2 {
        match self {
            Self::AABB { offset, .. } => *offset,
            Self::Circle { offset, .. } => *offset,
            Self::Polygon { offset, .. } => *offset,
        }
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

/// Helper for collision overlap check.
pub fn intersects(pos_a: Vec2, col_a: &Collider, pos_b: Vec2, col_b: &Collider) -> bool {
    super::physics::check_collision(pos_a, col_a, pos_b, col_b).is_some()
}

/// Helper for proximity check.
pub fn get_proximity(pos_a: Vec2, col_a: &Collider, pos_b: Vec2, col_b: &Collider, radius: f32) -> f32 {
    let p_a = pos_a + col_a.get_offset();
    let p_b = pos_b + col_b.get_offset();
    let dist = p_a.distance(p_b);
    
    // Simple distance-based proximity
    if dist < 1.0 { return 1.0; }
    if dist > radius { return 0.0; }
    
    1.0 - (dist / radius).clamp(0.0, 1.0)
}
