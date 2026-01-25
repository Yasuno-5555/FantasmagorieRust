use super::types::*;

/// The linear description of a frame's rendering intent.
/// This is the "Meaning" layer. Tracea will consume this.
pub struct FrameDescription {
    pub commands: Vec<RenderCommand>,
}

impl FrameDescription {
    pub fn new() -> Self {
        Self {
            commands: Vec::with_capacity(1024),
        }
    }
}

/// Commands defining *what* to draw, not *how*.
pub enum RenderCommand {
    SetPipeline(PipelineHandle),
    SetTexture { slot: u32, handle: TextureHandle },
    SetTransform(Transform2D),
    SetScissor(Rect),

    /// Start of World-space rendering layer.
    BeginWorld(super::camera::Camera),
    /// End of World-space rendering layer.
    EndWorld,
    
    DrawQuad { rect: Rect, color: Color },
    DrawTexturedQuad { rect: Rect, uv: UVRect, texture: TextureHandle },
    DrawMesh(MeshHandle),

    /// High-feature SDF Shape (Mode 2)
    /// Used for rounded rects, circles, borders, and shadows.
    DrawShape {
        rect: Rect,
        color: Color,
        radii: CornerRadii,
        border: Option<Border>,
        glow: Option<Glow>,
        elevation: f32,
        is_squircle: bool,
        /// Motion Morphing weight (0.0 to 1.0)
        morph: f32,
    },
}
