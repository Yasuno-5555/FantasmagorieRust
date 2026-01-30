use serde::{Deserialize, Serialize};

/// Semantic states for entities (The Logic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EntityState {
    #[default]
    Idle,
    Walk,
    Run,
    Attack,
    Interact,
}

/// A frame in a sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationFrame {
    /// Texture index or ID
    pub texture_id: u64,
    /// Duration of this frame in seconds
    pub duration: f32,
    /// Optional: Custom UV rect if not using a grid
    pub uv_rect: Option<[f32; 4]>,
}

/// Defines a sequence of frames for a state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationClip {
    pub frames: Vec<AnimationFrame>,
    pub loop_clip: bool,
}

/// The visual translation layer for an entity's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationComponent {
    /// Semantic state being visualized
    pub state: EntityState,
    pub timer: f32,
    pub frame_index: usize,
    
    /// Motion Morphing weight (0.0 to 1.0).
    /// Used when frames switch to provide SDF-based fluid interpolation.
    pub morph_weight: f32,
}

impl Default for AnimationComponent {
    fn default() -> Self {
        Self {
            state: EntityState::Idle,
            timer: 0.0,
            frame_index: 0,
            morph_weight: 0.0,
        }
    }
}

impl AnimationComponent {
    /// Syncs the visual state with the semantic world state.
    pub fn sync(&mut self, new_state: EntityState) {
        if self.state != new_state {
            self.state = new_state;
            self.timer = 0.0;
            self.frame_index = 0;
            self.morph_weight = 0.0;
        }
    }

    /// Updates the visual timer and triggers morphing on frame change.
    pub fn update(&mut self, dt: f32, clip: &AnimationClip) {
        if clip.frames.is_empty() { return; }
        
        // Decay morph weight
        self.morph_weight = (self.morph_weight - dt * 10.0).max(0.0);
        
        self.timer += dt;
        let current_frame_duration = clip.frames[self.frame_index].duration;
        
        if self.timer >= current_frame_duration {
            self.timer -= current_frame_duration;
            self.frame_index += 1;
            
            if self.frame_index >= clip.frames.len() {
                if clip.loop_clip {
                    self.frame_index = 0;
                } else {
                    self.frame_index = clip.frames.len() - 1;
                }
            }
            
            // Interaction: Trigger Motion Morphing (Visual Orchestration)
            self.morph_weight = 1.0;
        }
    }
}
