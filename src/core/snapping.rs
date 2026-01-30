use crate::core::{Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapType {
    Grid,
    Vertex,
    Edge,
    Midpoint,
    Center,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapContext {
    pub enabled: bool,
    pub threshold: f32, // Screen pixels
    pub grid_step: Vec3, // For 3D grid
                        // For object snapping, we would need a list of target points.
                        // For now, we only implement Grid snapping.
}

impl Default for SnapContext {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 10.0,
            grid_step: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

pub fn snap_value(val: f32, step: f32, threshold: f32) -> (f32, bool) {
    if step <= 0.0001 {
        return (val, false);
    }
    let rounded = (val / step).round() * step;
    let dist = (val - rounded).abs();
    // For grid snapping, we usually ALWAYS snap if enabled, ignoring threshold?
    // Or we only snap if close?
    // Usually Grid Snap is continuous discrete steps.
    // Object Snap is proximity based.
    // Here we treat Grid Snap as "Always Snap to Grid".
    (rounded, true)
}

pub fn snap_vec3(pos: Vec3, ctx: &SnapContext) -> (Vec3, Option<SnapType>) {
    if !ctx.enabled {
        return (pos, None);
    }

    // Grid Snapping
    let x = (pos.x / ctx.grid_step.x).round() * ctx.grid_step.x;
    let y = (pos.y / ctx.grid_step.y).round() * ctx.grid_step.y;
    let z = (pos.z / ctx.grid_step.z).round() * ctx.grid_step.z;

    (Vec3::new(x, y, z), Some(SnapType::Grid))
}

pub fn snap_vec2(pos: Vec2, step: f32) -> Vec2 {
    let x = (pos.x / step).round() * step;
    let y = (pos.y / step).round() * step;
    Vec2::new(x, y)
}
