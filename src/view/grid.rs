use crate::core::ColorF;
use std::cell::Cell;

pub struct GridData {
    pub step_x: Cell<f32>,
    pub step_y: Cell<f32>,
    pub offset_x: Cell<f32>,
    pub offset_y: Cell<f32>,
    pub scale: Cell<f32>, // Scaling factor for zoom
    pub color: Cell<ColorF>,
}

impl GridData {
    pub fn new() -> Self {
        Self {
            step_x: Cell::new(50.0),
            step_y: Cell::new(50.0),
            offset_x: Cell::new(0.0),
            offset_y: Cell::new(0.0),
            scale: Cell::new(1.0),
            color: Cell::new(ColorF::new(0.2, 0.2, 0.25, 0.3)),
        }
    }
}
