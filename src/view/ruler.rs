use crate::core::ID;
use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RulerOrientation {
    Horizontal,
    Vertical,
}

pub struct RulerData {
    pub orientation: Cell<RulerOrientation>,
    pub start: Cell<f32>,
    pub scale: Cell<f32>,     // pixels per unit
    pub tick_step: Cell<f32>, // unit value per major tick
    pub subdivisions: Cell<u32>,
}

impl RulerData {
    pub fn new() -> Self {
        Self {
            orientation: Cell::new(RulerOrientation::Horizontal),
            start: Cell::new(0.0),
            scale: Cell::new(10.0),     // 10px = 1 unit
            tick_step: Cell::new(10.0), // Mark every 10 units
            subdivisions: Cell::new(5),
        }
    }
}
