use crate::core::{Vec3, ID};
use std::cell::Cell;

pub struct Scene3DData {
    pub texture_id: Cell<u64>,
    pub camera_pos: Cell<Vec3>,
    pub camera_target: Cell<Vec3>,
    pub fov: Cell<f32>,
    pub near: Cell<f32>,
    pub far: Cell<f32>,
    pub is_ortho: Cell<bool>,
    pub scene_id: ID,
    pub lut_id: Cell<u64>,
    pub lut_intensity: Cell<f32>,
}

impl Scene3DData {
    pub fn new(scene_id: ID) -> Self {
        Self {
            texture_id: Cell::new(0),
            camera_pos: Cell::new(Vec3::new(0.0, 0.0, 5.0)),
            camera_target: Cell::new(Vec3::ZERO),
            fov: Cell::new(45.0),
            near: Cell::new(0.1),
            far: Cell::new(1000.0),
            is_ortho: Cell::new(false),
            scene_id,
            lut_id: Cell::new(0),
            lut_intensity: Cell::new(1.0),
        }
    }
}
