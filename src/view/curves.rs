use crate::core::{ColorF, Vec2};
use std::cell::{Cell, RefCell};

/// Control point for a curve
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvePoint {
    pub pos: Vec2, // Normalized [0, 1]
    pub locked: bool,
}

/// A single color curve (e.g., Red, Green, Blue, or Master)
pub struct Curve {
    pub points: Vec<CurvePoint>,
    pub color: ColorF,
}

impl Curve {
    pub fn new(color: ColorF) -> Self {
        Self {
            points: vec![
                CurvePoint {
                    pos: Vec2::new(0.0, 0.0),
                    locked: true,
                },
                CurvePoint {
                    pos: Vec2::new(1.0, 1.0),
                    locked: true,
                },
            ],
            color,
        }
    }
    // ... evaluate remains same ...

    /// Evaluate curve at x [0, 1] using Monotonic Cubic Spline
    pub fn evaluate(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        if self.points.is_empty() {
            return x;
        }
        if self.points.len() == 1 {
            return self.points[0].pos.y;
        }

        // Find segments
        let mut idx = 0;
        while idx < self.points.len() - 1 && self.points[idx + 1].pos.x < x {
            idx += 1;
        }

        let p0 = self.points[idx].pos;
        let p1 = self.points[idx + 1].pos;

        let dx = p1.x - p0.x;
        if dx < 0.0001 {
            return p0.y;
        }

        let t = (x - p0.x) / dx;

        // Monotonic Cubic Spline requires tangents
        // For simplicity here, we use a C1 Catmull-Rom like approach but clamped to be monotonic
        let m0 = if idx > 0 {
            let p_prev = self.points[idx - 1].pos;
            (p1.y - p_prev.y) / (p1.x - p_prev.x)
        } else {
            (p1.y - p0.y) / (p1.x - p0.x)
        };

        let m1 = if idx < self.points.len() - 2 {
            let p_next = self.points[idx + 2].pos;
            (p_next.y - p0.y) / (p_next.x - p0.x)
        } else {
            (p1.y - p0.y) / (p1.x - p0.x)
        };

        // Monotonic clamping
        let d = (p1.y - p0.y) / dx;
        let m0 = m0.clamp(-3.0 * d.abs(), 3.0 * d.abs());
        let m1 = m1.clamp(-3.0 * d.abs(), 3.0 * d.abs());

        // Hermite basis
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * p0.y + h10 * dx * m0 + h01 * p1.y + h11 * dx * m1
    }
}

use crate::core::ID;
use std::collections::HashMap;

pub struct CurveEditorData {
    pub curves: Vec<Curve>,
    pub active_curve_idx: usize,
}

impl CurveEditorData {
    pub fn new() -> Self {
        Self {
            curves: vec![
                Curve::new(ColorF::WHITE),                   // Master
                Curve::new(ColorF::new(1.0, 0.2, 0.2, 1.0)), // Red
                Curve::new(ColorF::new(0.2, 1.0, 0.2, 1.0)), // Green
                Curve::new(ColorF::new(0.2, 0.2, 1.0, 1.0)), // Blue
            ],
            active_curve_idx: 0,
        }
    }
}

thread_local! {
    static CURVE_STORAGE: RefCell<HashMap<ID, CurveEditorData>> = RefCell::new(HashMap::new());
}

pub fn ensure_curve_data_exists(id: ID) {
    CURVE_STORAGE.with(|storage| {
        let mut map = storage.borrow_mut();
        if !map.contains_key(&id) {
            map.insert(id, CurveEditorData::new());
        }
    });
}

pub fn with_curve_data_mut<F, R>(id: ID, f: F) -> Option<R>
where
    F: FnOnce(&mut CurveEditorData) -> R,
{
    CURVE_STORAGE.with(|storage| {
        let mut map = storage.borrow_mut();
        map.get_mut(&id).map(f)
    })
}

pub fn with_curve_data<F, R>(id: ID, f: F) -> Option<R>
where
    F: FnOnce(&CurveEditorData) -> R,
{
    CURVE_STORAGE.with(|storage| {
        let map = storage.borrow();
        map.get(&id).map(f)
    })
}
