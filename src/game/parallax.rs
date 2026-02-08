use crate::core::Vec2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParallaxLayer {
    pub factor: Vec2, // 0.0 = static (UI), 1.0 = normal, 0.5 = far background
    pub base_position: Vec2,
}

impl Default for ParallaxLayer {
    fn default() -> Self {
        Self {
            factor: Vec2::new(1.0, 1.0),
            base_position: Vec2::ZERO,
        }
    }
}
