use crate::core::{ColorF, Vec2};
use crate::resource::TextureId;
use crate::renderer::FrameContext;
use crate::renderer::types::{TextureHandle, UVRect, Rect};

/// Sprite component.
#[derive(Debug, Clone)]
pub struct Sprite {
    pub texture: TextureId,
    pub uv: Rect,
    pub color: ColorF,
}

impl Sprite {
    pub fn new(texture: TextureId) -> Self {
        Self {
            texture,
            uv: Rect::new(0.0, 0.0, 1.0, 1.0),
            color: ColorF::white(),
        }
    }
}

/// Immediate-mode Sprite Builder.
pub struct SpriteBuilder<'a> {
    frame: &'a mut FrameContext,
    world_pos: Vec2,
    size: Vec2,
    rotation: f32,
    animation: Option<crate::game::animation::AnimationComponent>,
}

impl<'a> SpriteBuilder<'a> {
    pub fn new(frame: &'a mut FrameContext, world_pos: Vec2, size: Vec2) -> Self {
        Self {
            frame,
            world_pos,
            size,
            rotation: 0.0,
            animation: None,
        }
    }

    pub fn rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn animate(mut self, anim: crate::game::animation::AnimationComponent) -> Self {
        self.animation = Some(anim);
        self
    }

    pub fn draw(self, sprite: &Sprite, clip: Option<&crate::game::animation::AnimationClip>) {
        let mut uv = sprite.uv;
        let mut morph = 0.0;
        let mut texture_id = sprite.texture;

        if let (Some(anim), Some(clip)) = (&self.animation, clip) {
            if anim.frame_index < clip.frames.is_empty() as usize { 
                // Safety check
            } else if anim.frame_index < clip.frames.len() {
                let frame = &clip.frames[anim.frame_index];
                texture_id = frame.texture_id;
                if let Some(uv_arr) = frame.uv_rect {
                    uv = Rect::new(uv_arr[0], uv_arr[1], uv_arr[2], uv_arr[3]);
                }
                morph = anim.morph_weight;
            }
        }

        // Draw with morphing support
        self.frame.draw(
            Rect::new(self.world_pos.x, self.world_pos.y, self.size.x, self.size.y),
            sprite.color
        )
        .morph(morph)
        .submit();
        
        // Also draw the textured quad (Renderer will need to handle blending these if both exist, 
        // but for Phase 2 we assume DrawShape handles the morphing orchestration)
        // In a real SATSUEI pipeline, these are combined.
        self.frame.draw_textured_quad(
            Rect::new(self.world_pos.x, self.world_pos.y, self.size.x, self.size.y),
            UVRect { u: uv.x, v: uv.y, w: uv.w, h: uv.h },
            TextureHandle(texture_id as u32),
        );
    }
}
