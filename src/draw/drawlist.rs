//! DrawList - Rendering command buffer
//! Ported from drawlist.hpp
//!
//! All rendering goes through DrawList commands.
//! Backend consumes these commands to produce actual GPU draws.

use crate::core::{ColorF, Vec2};

/// Draw command types
#[derive(Clone, Debug)]
pub enum DrawCommand {
    /// Rounded rectangle (with optional shadow/squircle)
    RoundedRect {
        pos: Vec2,
        size: Vec2,
        radii: [f32; 4],
        color: ColorF,
        elevation: f32,
        is_squircle: bool,
        border_width: f32,
        border_color: ColorF,
        wobble: Vec2,
        // Visual Revolution Additions
        glow_strength: f32,
        glow_color: ColorF,
    },

    /// Text glyph (SDF)
    Text {
        pos: Vec2,
        size: Vec2,
        uv: [f32; 4], // u0, v0, u1, v1
        color: ColorF,
    },

    /// Blur rectangle (glassmorphism)
    BlurRect {
        pos: Vec2,
        size: Vec2,
        radii: [f32; 4],
        sigma: f32,
    },

    /// Bezier curve
    Bezier {
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
        thickness: f32,
        color: ColorF,
    },

    /// Line
    Line {
        p0: Vec2,
        p1: Vec2,
        thickness: f32,
        color: ColorF,
    },

    /// Polyline / Path
    Polyline {
        points: Vec<Vec2>,
        color: ColorF,
        thickness: f32,
        closed: bool,
    },

    /// Push clip rectangle
    PushClip { pos: Vec2, size: Vec2 },

    /// Pop clip rectangle
    PopClip,

    /// Push transform (for scroll offset)
    PushTransform { offset: Vec2, scale: f32 },

    /// Pop transform
    PopTransform,

    /// Circle
    Circle {
        center: Vec2,
        radius: f32,
        color: ColorF,
        filled: bool,
    },

    /// Image
    Image {
        pos: Vec2,
        size: Vec2,
        texture_id: u64,
        uv: [f32; 4],
        color: ColorF,
        radii: [f32; 4],
    },

    /// Gradient Rectangle
    GradientRect {
        pos: Vec2,
        size: Vec2,
        colors: [ColorF; 4], // TL, TR, BR, BL
    },

    /// Arc (SDF)
    Arc {
        center: Vec2,
        radius: f32,
        start_angle: f32, // Radians
        end_angle: f32,   // Radians
        thickness: f32,
        color: ColorF,
    },

    /// Plot (Line graph with fill)
    Plot {
        points: Vec<Vec2>,
        color: ColorF,
        fill_color: ColorF,
        thickness: f32,
        baseline: f32, // Y-coordinate for fill bottom
    },

    /// Heatmap (optimized texture-based rendering)
    Heatmap {
        pos: Vec2,
        size: Vec2,
        data: Vec<f32>,
        width: usize,
        height: usize,
        colormap: crate::view::plot::Colormap,
        min: f32,
        max: f32,
    },

    /// 3D Viewport (embedded external scene)
    Viewport3D {
        pos: Vec2,
        size: Vec2,
        texture_id: u64,
        camera_pos: crate::core::Vec3,
        camera_target: crate::core::Vec3,
        fov: f32,
        lut_id: u64,
        lut_intensity: f32,
    },
}

/// Draw list - accumulates commands for a frame
#[derive(Default, Clone)]
pub struct DrawList {
    commands: Vec<DrawCommand>,
    clip_stack: Vec<(Vec2, Vec2)>,
    transform_stack: Vec<(Vec2, f32)>,
}

impl DrawList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all commands
    pub fn clear(&mut self) {
        self.commands.clear();
        self.clip_stack.clear();
        self.transform_stack.clear();
    }

    /// Get commands slice
    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    /// Add rounded rectangle
    pub fn add_rounded_rect(&mut self, pos: Vec2, size: Vec2, radius: f32, color: ColorF) {
        self.commands.push(DrawCommand::RoundedRect {
            pos,
            size,
            radii: [radius, radius, radius, radius],
            color,
            elevation: 0.0,
            is_squircle: false,
            border_width: 0.0,
            border_color: ColorF::transparent(),
            wobble: Vec2::ZERO,
            glow_strength: 0.0,
            glow_color: ColorF::transparent(),
        });
    }

    pub fn add_rect_ex(
        &mut self,
        pos: Vec2,
        size: Vec2,
        radii: [f32; 4], // tl, tr, br, bl
        color: ColorF,
        elevation: f32,
        is_squircle: bool,
        border_width: f32,
        border_color: ColorF,
        wobble: Vec2,
        glow_strength: f32,
        glow_color: ColorF,
    ) {
        self.commands.push(DrawCommand::RoundedRect {
            pos,
            size,
            radii,
            color,
            elevation,
            is_squircle,
            border_width,
            border_color,
            wobble,
            glow_strength,
            glow_color,
        });
    }

    /// Add rounded rectangle with full options
    pub fn add_rounded_rect_ex(
        &mut self,
        pos: Vec2,
        size: Vec2,
        radius: f32,
        color: ColorF,
        elevation: f32,
        is_squircle: bool,
        border_width: f32,
        border_color: ColorF,
        wobble: Vec2,
        glow_strength: f32,
        glow_color: ColorF,
    ) {
        self.commands.push(DrawCommand::RoundedRect {
            pos,
            size,
            radii: [radius, radius, radius, radius],
            color,
            elevation,
            is_squircle,
            border_width,
            border_color,
            wobble,
            glow_strength,
            glow_color,
        });
    }

    /// Add text glyph
    pub fn add_text(&mut self, pos: Vec2, size: Vec2, uv: [f32; 4], color: ColorF) {
        self.commands.push(DrawCommand::Text {
            pos,
            size,
            uv,
            color,
        });
    }

    /// Add blur rectangle
    pub fn add_blur_rect(&mut self, pos: Vec2, size: Vec2, radius: f32, sigma: f32) {
        self.commands.push(DrawCommand::BlurRect {
            pos,
            size,
            radii: [radius, radius, radius, radius],
            sigma,
        });
    }

    pub fn add_blur_rect_ex(&mut self, pos: Vec2, size: Vec2, radii: [f32; 4], sigma: f32) {
        self.commands.push(DrawCommand::BlurRect {
            pos,
            size,
            radii,
            sigma,
        });
    }

    /// Add bezier curve
    pub fn add_bezier(
        &mut self,
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
        thickness: f32,
        color: ColorF,
    ) {
        self.commands.push(DrawCommand::Bezier {
            p0,
            p1,
            p2,
            p3,
            thickness,
            color,
        });
    }

    /// Add line
    pub fn add_line(&mut self, p0: Vec2, p1: Vec2, thickness: f32, color: ColorF) {
        self.commands.push(DrawCommand::Line {
            p0,
            p1,
            thickness,
            color,
        });
    }

    /// Add circle
    pub fn add_circle(&mut self, center: Vec2, radius: f32, color: ColorF, filled: bool) {
        self.commands.push(DrawCommand::Circle {
            center,
            radius,
            color,
            filled,
        });
    }

    /// Add image
    pub fn add_image(
        &mut self,
        pos: Vec2,
        size: Vec2,
        texture_id: u64,
        uv: [f32; 4],
        color: ColorF,
    ) {
        self.commands.push(DrawCommand::Image {
            pos,
            size,
            texture_id,
            uv,
            color,
            radii: [0.0, 0.0, 0.0, 0.0],
        });
    }

    pub fn add_image_ex(
        &mut self,
        pos: Vec2,
        size: Vec2,
        texture_id: u64,
        uv: [f32; 4],
        color: ColorF,
        radii: [f32; 4],
    ) {
        self.commands.push(DrawCommand::Image {
            pos,
            size,
            texture_id,
            uv,
            color,
            radii,
        });
    }

    /// Add polyline
    pub fn add_polyline(&mut self, points: Vec<Vec2>, color: ColorF, thickness: f32, closed: bool) {
        self.commands.push(DrawCommand::Polyline {
            points,
            color,
            thickness,
            closed,
        });
    }

    /// Add path (tessellates into polyline)
    pub fn add_path(&mut self, path: &crate::draw::path::Path, color: ColorF, thickness: f32) {
        let mut points = Vec::new();
        let tess = crate::draw::path::BezierTessellator::new();
        tess.tessellate(path, &mut points);

        let closed = path
            .segments
            .last()
            .map(|s| s.verb == crate::draw::path::PathVerb::Close)
            .unwrap_or(false);

        self.add_polyline(points, color, thickness, closed);
    }

    /// Add gradient rectangle
    pub fn add_gradient_rect(&mut self, pos: Vec2, size: Vec2, colors: [ColorF; 4]) {
        self.commands
            .push(DrawCommand::GradientRect { pos, size, colors });
    }

    /// Add arc
    pub fn add_arc(
        &mut self,
        center: Vec2,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        thickness: f32,
        color: ColorF,
    ) {
        self.commands.push(DrawCommand::Arc {
            center,
            radius,
            start_angle,
            end_angle,
            thickness,
            color,
        });
    }

    pub fn add_plot(
        &mut self,
        points: Vec<Vec2>,
        color: ColorF,
        fill_color: ColorF,
        thickness: f32,
        baseline: f32,
    ) {
        self.commands.push(DrawCommand::Plot {
            points,
            color,
            fill_color,
            thickness,
            baseline,
        });
    }

    /// Add heatmap
    pub fn add_heatmap(
        &mut self,
        pos: Vec2,
        size: Vec2,
        data: Vec<f32>,
        width: usize,
        height: usize,
        colormap: crate::view::plot::Colormap,
        min: f32,
        max: f32,
    ) {
        self.commands.push(DrawCommand::Heatmap {
            pos,
            size,
            data,
            width,
            height,
            colormap,
            min,
            max,
        });
    }

    /// Add 3D viewport
    pub fn add_viewport_3d(
        &mut self,
        pos: Vec2,
        size: Vec2,
        texture_id: u64,
        camera_pos: crate::core::Vec3,
        camera_target: crate::core::Vec3,
        fov: f32,
        lut_id: u64,
        lut_intensity: f32,
    ) {
        self.commands.push(DrawCommand::Viewport3D {
            pos,
            size,
            texture_id,
            camera_pos,
            camera_target,
            fov,
            lut_id,
            lut_intensity,
        });
    }

    /// Push clip rectangle
    pub fn push_clip(&mut self, pos: Vec2, size: Vec2) {
        self.clip_stack.push((pos, size));
        self.commands.push(DrawCommand::PushClip { pos, size });
    }

    /// Pop clip rectangle
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
        self.commands.push(DrawCommand::PopClip);
    }

    /// Push transform
    pub fn push_transform(&mut self, offset: Vec2, scale: f32) {
        self.transform_stack.push((offset, scale));
        self.commands
            .push(DrawCommand::PushTransform { offset, scale });
    }

    /// Pop transform
    pub fn pop_transform(&mut self) {
        self.transform_stack.pop();
        self.commands.push(DrawCommand::PopTransform);
    }

    /// Get current transform offset
    pub fn current_offset(&self) -> Vec2 {
        self.transform_stack
            .last()
            .map(|(o, _)| *o)
            .unwrap_or(Vec2::ZERO)
    }

    /// Get command count
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drawlist_basic() {
        let mut dl = DrawList::new();
        dl.add_rounded_rect(
            Vec2::new(10.0, 20.0),
            Vec2::new(100.0, 50.0),
            5.0,
            ColorF::red(),
        );
        assert_eq!(dl.len(), 1);
    }

    #[test]
    fn test_clip_stack() {
        let mut dl = DrawList::new();
        dl.push_clip(Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0));
        dl.add_rounded_rect(
            Vec2::new(10.0, 10.0),
            Vec2::new(50.0, 50.0),
            0.0,
            ColorF::blue(),
        );
        dl.pop_clip();
        assert_eq!(dl.len(), 3);
    }
}
