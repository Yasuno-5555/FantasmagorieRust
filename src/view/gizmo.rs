use crate::core::Vec3;
use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    Screen, // Center/Free move
}

pub struct GizmoData {
    pub mode: Cell<GizmoMode>,
    pub position: Cell<Vec3>,
    pub rotation: Cell<Vec3>, // Euler angles in degrees
    pub scale: Cell<Vec3>,

    // Interaction state
    pub active_axis: Cell<Option<GizmoAxis>>,
    pub is_dragging: Cell<bool>,
    pub drag_start_val: Cell<Vec3>, // Initial value of pos/rot/scale
    pub drag_start_mouse: Cell<(f32, f32)>,
    pub snap_context: Cell<crate::core::snapping::SnapContext>,
}

impl GizmoData {
    pub fn new() -> Self {
        Self {
            mode: Cell::new(GizmoMode::Translate),
            position: Cell::new(Vec3::ZERO),
            rotation: Cell::new(Vec3::ZERO),
            scale: Cell::new(Vec3::ONE),
            active_axis: Cell::new(None),
            is_dragging: Cell::new(false),
            drag_start_val: Cell::new(Vec3::ZERO),
            drag_start_mouse: Cell::new((0.0, 0.0)),
            snap_context: Cell::new(crate::core::snapping::SnapContext::default()),
        }
    }
}
