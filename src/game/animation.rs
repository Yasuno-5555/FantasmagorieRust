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

    // Blending support
    pub previous_state: Option<EntityState>,
    pub previous_frame_index: usize,
    pub blend_factor: f32, // 0.0 to 1.0 (0.0 = fully previous, 1.0 = fully current)
    pub blend_speed: f32,
}

impl Default for AnimationComponent {
    fn default() -> Self {
        Self {
            state: EntityState::Idle,
            timer: 0.0,
            frame_index: 0,
            morph_weight: 0.0,
            previous_state: None,
            previous_frame_index: 0,
            blend_factor: 1.0,
            blend_speed: 1.0,
        }
    }
}

impl AnimationComponent {
    /// Syncs the visual state with the semantic world state.
    pub fn sync(&mut self, new_state: EntityState) {
        if self.state != new_state {
            // Start blending
            self.previous_state = Some(self.state);
            self.previous_frame_index = self.frame_index;
            
            self.state = new_state;
            self.timer = 0.0;
            self.frame_index = 0;
            self.morph_weight = 0.0; // Reset morph for new state
            
            self.blend_factor = 0.0;
            self.blend_speed = 5.0; // Default speed
        }
    }

    /// Updates the visual timer and triggers morphing on frame change.
    pub fn update(&mut self, dt: f32, clip: &AnimationClip) {
        if clip.frames.is_empty() { return; }
        
        // Decay morph weight
        self.morph_weight = (self.morph_weight - dt * 10.0).max(0.0);
        
        // Update Blend Factor
        if self.blend_factor < 1.0 {
            self.blend_factor = (self.blend_factor + dt * self.blend_speed).min(1.0);
            if self.blend_factor >= 1.0 {
                self.previous_state = None; // Finished blending
            }
        }
        
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

/// A clip for skeletal animation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoneKeyframe {
    pub time: f32, // Normalized 0.0 to 1.0 (or seconds if duration is fixed)
    pub position: Option<crate::core::Vec2>,
    pub rotation: Option<f32>,
    pub scale: Option<crate::core::Vec2>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoneTrack {
    pub bone_name: String,
    pub keyframes: Vec<BoneKeyframe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletalAnimationClip {
    pub name: String,
    pub duration: f32,
    pub bone_tracks: Vec<BoneTrack>,
    pub loop_clip: bool,
}

/// Component for controlling skeletal animation with blending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletalAnimationComponent {
    pub current_clip: String,
    pub current_time: f32,
    
    // Blending support
    pub previous_clip: Option<String>,
    pub previous_time: f32,
    pub blend_factor: f32, // 0.0 to 1.0 (0.0 = previous, 1.0 = current)
    pub blend_speed: f32,
    
    pub speed: f32,
}

impl Default for SkeletalAnimationComponent {
    fn default() -> Self {
        Self {
            current_clip: String::new(),
            current_time: 0.0,
            previous_clip: None,
            previous_time: 0.0,
            blend_factor: 1.0,
            blend_speed: 1.0,
            speed: 1.0,
        }
    }
}

impl SkeletalAnimationComponent {
    pub fn play(&mut self, clip_name: &str, blend_duration: f32) {
        if self.current_clip == clip_name {
            return;
        }
        
        if blend_duration > 0.0 {
            self.previous_clip = Some(self.current_clip.clone());
            self.previous_time = self.current_time;
            self.blend_factor = 0.0;
            self.blend_speed = 1.0 / blend_duration;
        } else {
            self.previous_clip = None;
            self.blend_factor = 1.0;
        }
        
        self.current_clip = clip_name.to_string();
        self.current_time = 0.0;
    }
    
    pub fn update(&mut self, dt: f32, clips: &std::collections::HashMap<String, SkeletalAnimationClip>) {
        // Update main time
        if let Some(clip) = clips.get(&self.current_clip) {
            self.current_time += dt * self.speed;
            if clip.loop_clip {
                self.current_time %= clip.duration;
            } else {
                self.current_time = self.current_time.min(clip.duration);
            }
        }
        
        // Update previous time (for blending)
        if let Some(prev_name) = &self.previous_clip {
            if let Some(clip) = clips.get(prev_name) {
                self.previous_time += dt * self.speed;
                if clip.loop_clip {
                    self.previous_time %= clip.duration;
                } else {
                    self.previous_time = self.previous_time.min(clip.duration);
                }
            }
        }
        
        // Update Blend Factor
        if self.blend_factor < 1.0 {
            self.blend_factor = (self.blend_factor + dt * self.blend_speed).min(1.0);
            if self.blend_factor >= 1.0 {
                self.previous_clip = None;
            }
        }
    }
    
    /// Apply animation to skeleton
    pub fn apply(&self, skeleton: &mut crate::animation::skeleton::Skeleton, clips: &std::collections::HashMap<String, SkeletalAnimationClip>) {
        if let Some(current) = clips.get(&self.current_clip) {
            for track in &current.bone_tracks {
                // Find bone
                if let Some(bone) = skeleton.bones.iter_mut().find(|b| b.name == track.bone_name) {
                    let pose = sample_track(track, self.current_time, current.duration);
                    
                    if let Some(prev_name) = &self.previous_clip {
                        if let Some(prev_clip) = clips.get(prev_name) {
                            if let Some(prev_track) = prev_clip.bone_tracks.iter().find(|t| t.bone_name == track.bone_name) {
                                let prev_pose = sample_track(prev_track, self.previous_time, prev_clip.duration);
                                // Blend
                                if let Some(p) = pose.0 {
                                    if let Some(pp) = prev_pose.0 {
                                        bone.local_transform.local_position = pp.lerp(p, self.blend_factor);
                                    }
                                }
                                if let Some(r) = pose.1 {
                                    if let Some(pr) = prev_pose.1 {
                                        // Simple linear interpolation for angle
                                        // TODO: Shortest path interpolation
                                        bone.local_transform.local_rotation = pr + (r - pr) * self.blend_factor;
                                    }
                                }
                                if let Some(s) = pose.2 {
                                    if let Some(ps) = prev_pose.2 {
                                        bone.local_transform.local_scale = ps.lerp(s, self.blend_factor);
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    
                    // No blending or done
                    if let Some(p) = pose.0 { bone.local_transform.local_position = p; }
                    if let Some(r) = pose.1 { bone.local_transform.local_rotation = r; }
                    if let Some(s) = pose.2 { bone.local_transform.local_scale = s; }
                }
            }
        }
    }
}

fn sample_track(track: &BoneTrack, time: f32, duration: f32) -> (Option<crate::core::Vec2>, Option<f32>, Option<crate::core::Vec2>) {
    if track.keyframes.is_empty() { return (None, None, None); }
    
    // Binary search for keyframes
    // Assuming sorted
    // TODO: Optimize
    
    let t = time.clamp(0.0, duration);
    
    let mut prev_idx = 0;
    let mut next_idx = 0;
    
    for (i, kf) in track.keyframes.iter().enumerate() {
        if kf.time <= t { prev_idx = i; }
        if kf.time >= t { next_idx = i; break; }
        next_idx = i;
    }
    
    let prev = &track.keyframes[prev_idx];
    let next = &track.keyframes[next_idx];
    
    if prev_idx == next_idx {
        return (prev.position, prev.rotation, prev.scale);
    }
    
    let factor = (t - prev.time) / (next.time - prev.time);
    
    let pos = match (prev.position, next.position) {
        (Some(a), Some(b)) => Some(a.lerp(b, factor)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        _ => None,
    };
    
    let rot = match (prev.rotation, next.rotation) {
        (Some(a), Some(b)) => Some(a + (b - a) * factor),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        _ => None,
    };
    
    let scl = match (prev.scale, next.scale) {
        (Some(a), Some(b)) => Some(a.lerp(b, factor)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        _ => None,
    };
    
    (pos, rot, scl)
}
