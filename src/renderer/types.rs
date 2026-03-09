use super::super::core::{ColorF, Vec2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn pos(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub fn size(&self) -> Vec2 {
        Vec2::new(self.w, self.h)
    }
}

pub type Color = ColorF;

/// Corner radii for SDF shapes (top-left, top-right, bottom-right, bottom-left).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CornerRadii {
    pub tl: f32,
    pub tr: f32,
    pub br: f32,
    pub bl: f32,
}

impl CornerRadii {
    /// Create a uniform radius for all corners.
    pub fn uniform(r: f32) -> Self {
        Self { tl: r, tr: r, br: r, bl: r }
    }

    /// Create radii from individual corner values.
    pub fn new(tl: f32, tr: f32, br: f32, bl: f32) -> Self {
        Self { tl, tr, br, bl }
    }

    /// Convert to array [tl, tr, br, bl] for easier GPU consumption.
    pub fn to_array(&self) -> [f32; 4] {
        [self.tl, self.tr, self.br, self.bl]
    }

    pub fn as_array(&self) -> [f32; 4] {
        self.to_array()
    }
}

/// Border specification for SDF shapes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Border {
    pub width: f32,
    pub color: Color,
}

/// Glow/Outer Shadow specification for SDF shapes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Glow {
    pub strength: f32,
    pub color: Color,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorSetHandle(pub u32);

// 2D Transform (3x2 Matrix in column-major order for GPU compatibility)
// Representing:
// [ m[0] m[2] m[4] ]
// [ m[1] m[3] m[5] ]
// [  0    0    1   ]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub m: [f32; 6], 
}

impl Transform2D {
    pub fn identity() -> Self {
        Self { m: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] }
    }

    pub fn translation(x: f32, y: f32) -> Self {
        Self { m: [1.0, 0.0, 0.0, 1.0, x, y] }
    }

    pub fn rotation(angle_rad: f32) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self { m: [c, s, -s, c, 0.0, 0.0] }
    }

    pub fn scale(sx: f32, sy: f32) -> Self {
        Self { m: [sx, 0.0, 0.0, sy, 0.0, 0.0] }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let a = self.m;
        let b = other.m;
        Self {
            m: [
                a[0] * b[0] + a[2] * b[1],
                a[1] * b[0] + a[3] * b[1],
                a[0] * b[2] + a[2] * b[3],
                a[1] * b[2] + a[3] * b[3],
                a[0] * b[4] + a[2] * b[5] + a[4],
                a[1] * b[4] + a[3] * b[5] + a[5],
            ]
        }
    }

    pub fn translate(&self, x: f32, y: f32) -> Self {
        self.multiply(&Self::translation(x, y))
    }

    pub fn rotate(&self, angle_rad: f32) -> Self {
        self.multiply(&Self::rotation(angle_rad))
    }

    pub fn scale_by(&self, sx: f32, sy: f32) -> Self {
        self.multiply(&Self::scale(sx, sy))
    }

    pub fn transform_point(&self, p: Vec2) -> Vec2 {
        Vec2::new(
            self.m[0] * p.x + self.m[2] * p.y + self.m[4],
            self.m[1] * p.x + self.m[3] * p.y + self.m[5]
        )
    }

    /// Convert to 3x3 for GPU uniforms (std140 alignment)
    pub fn to_mat3x3(&self) -> [f32; 12] {
        [
            self.m[0], self.m[1], 0.0, 0.0,
            self.m[2], self.m[3], 0.0, 0.0,
            self.m[4], self.m[5], 1.0, 0.0,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UVRect {
    pub u: f32,
    pub v: f32,
    pub w: f32,
    pub h: f32,
}

impl UVRect {
    pub fn default() -> Self {
        Self { u: 0.0, v: 0.0, w: 1.0, h: 1.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DrawRange {
    pub start: u32,
    pub count: u32,
    pub vertex_offset: i32,
}
