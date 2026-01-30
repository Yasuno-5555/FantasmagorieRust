//! Path system - SVG-like vector path builder and tessellator
//! Ported from path.hpp

use crate::core::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathVerb {
    MoveTo,
    LineTo,
    QuadTo,
    CubicTo,
    Close,
}

#[derive(Clone, Copy, Debug)]
pub struct PathSegment {
    pub verb: PathVerb,
    pub points: [Vec2; 3],
}

#[derive(Clone, Debug)]
pub struct Path {
    pub segments: Vec<PathSegment>,
}

impl Path {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn move_to(&mut self, p: Vec2) {
        self.segments.push(PathSegment {
            verb: PathVerb::MoveTo,
            points: [p, Vec2::ZERO, Vec2::ZERO],
        });
    }

    pub fn line_to(&mut self, p: Vec2) {
        self.segments.push(PathSegment {
            verb: PathVerb::LineTo,
            points: [p, Vec2::ZERO, Vec2::ZERO],
        });
    }

    pub fn quad_to(&mut self, c: Vec2, p: Vec2) {
        self.segments.push(PathSegment {
            verb: PathVerb::QuadTo,
            points: [c, p, Vec2::ZERO],
        });
    }

    pub fn cubic_to(&mut self, c1: Vec2, c2: Vec2, p: Vec2) {
        self.segments.push(PathSegment {
            verb: PathVerb::CubicTo,
            points: [c1, c2, p],
        });
    }

    pub fn close(&mut self) {
        self.segments.push(PathSegment {
            verb: PathVerb::Close,
            points: [Vec2::ZERO; 3],
        });
    }

    /// Convenience: Circle
    pub fn circle(center: Vec2, radius: f32) -> Self {
        let mut p = Self::new();
        let k = 0.55228475; // (4/3)*tan(pi/8)
        let kr = k * radius;

        p.move_to(Vec2::new(center.x + radius, center.y));
        p.cubic_to(
            Vec2::new(center.x + radius, center.y + kr),
            Vec2::new(center.x + kr, center.y + radius),
            Vec2::new(center.x, center.y + radius),
        );
        p.cubic_to(
            Vec2::new(center.x - kr, center.y + radius),
            Vec2::new(center.x - radius, center.y + kr),
            Vec2::new(center.x - radius, center.y),
        );
        p.cubic_to(
            Vec2::new(center.x - radius, center.y - kr),
            Vec2::new(center.x - kr, center.y - radius),
            Vec2::new(center.x, center.y - radius),
        );
        p.cubic_to(
            Vec2::new(center.x + kr, center.y - radius),
            Vec2::new(center.x + radius, center.y - kr),
            Vec2::new(center.x + radius, center.y),
        );
        p.close();
        p
    }

    /// Convenience: Regular polygon
    pub fn polygon(center: Vec2, radius: f32, sides: i32) -> Self {
        let mut p = Self::new();
        use std::f32::consts::PI;
        for i in 0..sides {
            let angle = (i as f32 / sides as f32) * 2.0 * PI - PI / 2.0;
            let pt = Vec2::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );
            if i == 0 {
                p.move_to(pt);
            } else {
                p.line_to(pt);
            }
        }
        p.close();
        p
    }
}

pub struct BezierTessellator {
    pub tolerance: f32,
    pub max_subdivisions: i32,
}

impl Default for BezierTessellator {
    fn default() -> Self {
        Self::new()
    }
}

impl BezierTessellator {
    pub fn new() -> Self {
        Self {
            tolerance: 0.5,
            max_subdivisions: 10,
        }
    }

    pub fn tessellate(&self, path: &Path, points: &mut Vec<Vec2>) {
        if path.segments.is_empty() {
            return;
        }

        let mut current_pos = Vec2::ZERO;
        let mut start_pos = Vec2::ZERO;

        for seg in &path.segments {
            match seg.verb {
                PathVerb::MoveTo => {
                    current_pos = seg.points[0];
                    start_pos = current_pos;
                    points.push(current_pos);
                }
                PathVerb::LineTo => {
                    current_pos = seg.points[0];
                    points.push(current_pos);
                }
                PathVerb::QuadTo => {
                    let c = seg.points[0];
                    let p = seg.points[1];
                    self.tessellate_quad_recursive(current_pos, c, p, 0, points);
                    current_pos = p;
                    points.push(current_pos);
                }
                PathVerb::CubicTo => {
                    let c1 = seg.points[0];
                    let c2 = seg.points[1];
                    let p = seg.points[2];
                    self.tessellate_cubic_recursive(current_pos, c1, c2, p, 0, points);
                    current_pos = p;
                    points.push(current_pos);
                }
                PathVerb::Close => {
                    points.push(start_pos);
                    current_pos = start_pos;
                }
            }
        }
    }

    pub fn tessellate_quad_recursive(
        &self,
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        depth: i32,
        points: &mut Vec<Vec2>,
    ) {
        if depth >= self.max_subdivisions || self.is_flat_quad(p0, p1, p2) {
            return;
        }

        let m01 = (p0 + p1) * 0.5;
        let m12 = (p1 + p2) * 0.5;
        let mid = (m01 + m12) * 0.5;

        self.tessellate_quad_recursive(p0, m01, mid, depth + 1, points);
        points.push(mid);
        self.tessellate_quad_recursive(mid, m12, p2, depth + 1, points);
    }

    pub fn tessellate_cubic_recursive(
        &self,
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
        depth: i32,
        points: &mut Vec<Vec2>,
    ) {
        if depth >= self.max_subdivisions || self.is_flat_cubic(p0, p1, p2, p3) {
            return;
        }

        let m01 = (p0 + p1) * 0.5;
        let m12 = (p1 + p2) * 0.5;
        let m23 = (p2 + p3) * 0.5;
        let m012 = (m01 + m12) * 0.5;
        let m123 = (m12 + m23) * 0.5;
        let mid = (m012 + m123) * 0.5;

        self.tessellate_cubic_recursive(p0, m01, m012, mid, depth + 1, points);
        points.push(mid);
        self.tessellate_cubic_recursive(mid, m123, m23, p3, depth + 1, points);
    }

    fn is_flat_quad(&self, p0: Vec2, p1: Vec2, p2: Vec2) -> bool {
        let d = self.point_line_distance(p1, p0, p2);
        d < self.tolerance
    }

    fn is_flat_cubic(&self, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> bool {
        let d1 = self.point_line_distance(p1, p0, p3);
        let d2 = self.point_line_distance(p2, p0, p3);
        d1.max(d2) < self.tolerance
    }

    fn point_line_distance(&self, p: Vec2, a: Vec2, b: Vec2) -> f32 {
        let ab = b - a;
        let ap = p - a;
        let len_sq = ab.x * ab.x + ab.y * ab.y;
        if len_sq < 1e-6 {
            return ap.length();
        }
        let t = (ap.x * ab.x + ap.y * ab.y) / len_sq;
        let t = t.clamp(0.0, 1.0);
        let proj = a + ab * t;
        (p - proj).length()
    }
}
