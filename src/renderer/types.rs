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

// 2D Transform (3x2 Matrix)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub m: [f32; 6], 
}

impl Transform2D {
    pub fn identity() -> Self {
        Self { m: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] }
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
