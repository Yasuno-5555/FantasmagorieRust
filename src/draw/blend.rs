//! Blend modes for 2D rendering

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BlendMode {
    /// Standard alpha blending
    Alpha = 0,
    /// Additive blending
    Add = 1,
    /// Multiply blending
    Multiply = 2,
    /// Screen blending
    Screen = 3,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Alpha
    }
}
