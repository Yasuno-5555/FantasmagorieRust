use crate::core::Vec2;
use std::sync::atomic::{AtomicU64, Ordering};
use super::interaction::{Collider, InteractionState, intersects, get_proximity};
use super::animation::{AnimationComponent, EntityState, AnimationClip};
use super::input::ActionState;
use std::collections::HashMap;

static NEXT_ENTITY_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(pub u64);

#[derive(Debug, Clone)]
pub struct Transform {
    pub position: Vec2,
    pub rotation: f32,
    pub scale: Vec2,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::new(1.0, 1.0),
        }
    }
}

pub struct World {
    pub ids: Vec<EntityId>,
    pub transforms: Vec<Transform>,
    pub sprites: Vec<Option<crate::game::sprite::Sprite>>,
    pub colliders: Vec<Option<Collider>>,
    pub interaction_states: Vec<InteractionState>,
    pub animations: Vec<AnimationComponent>,
    pub input_states: Vec<ActionState>,
    pub entity_states: Vec<EntityState>,
}

impl World {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            transforms: Vec::new(),
            sprites: Vec::new(),
            colliders: Vec::new(),
            interaction_states: Vec::new(),
            animations: Vec::new(),
            input_states: Vec::new(),
            entity_states: Vec::new(),
        }
    }

    pub fn spawn(&mut self) -> EntityId {
        let id = EntityId(NEXT_ENTITY_ID.fetch_add(1, Ordering::Relaxed));
        self.ids.push(id);
        self.transforms.push(Transform::default());
        self.sprites.push(None);
        self.colliders.push(None);
        self.interaction_states.push(InteractionState::default());
        self.animations.push(AnimationComponent::default());
        self.input_states.push(ActionState::default());
        self.entity_states.push(EntityState::default());
        id
    }

    /// Sets a collider for the given entity index.
    pub fn set_collider(&mut self, index: usize, collider: Collider) {
        if index < self.colliders.len() {
            self.colliders[index] = Some(collider);
        }
    }

    /// Runs the interaction system to update states.
    /// This is an O(N^2) check, but suitable for "Phase 1.5".
    pub fn system_interaction(&mut self, proximity_radius: f32) {
        let count = self.ids.len();
        
        // 1. Prepare next frame (reset transient states)
        for state in &mut self.interaction_states {
            state.prepare_next_frame();
        }
        
        // 2. Pairwise interaction check
        for i in 0..count {
            for j in 0..count {
                if i == j { continue; }
                
                let col_a = match &self.colliders[i] { Some(c) => c, None => continue };
                let col_b = match &self.colliders[j] { Some(c) => c, None => continue };
                
                let pos_a = self.transforms[i].position;
                let pos_b = self.transforms[j].position;
                
                // Check Touch (AABB overlap)
                if intersects(pos_a, col_a, pos_b, col_b) {
                    self.interaction_states[i].is_touched = true;
                    self.interaction_states[i].proximity = 1.0;
                }
                
                // Check Approach (Proximity)
                let prox = get_proximity(pos_a, col_a, pos_b, col_b, proximity_radius);
                if prox > 0.0 {
                    self.interaction_states[i].is_near = true;
                    self.interaction_states[i].proximity = self.interaction_states[i].proximity.max(prox);
                }
            }
        }
    }

    /// Runs the animation system.
    /// Reconciles World State -> Animation Visuals.
    pub fn system_animation(&mut self, dt: f32, clips: &HashMap<EntityState, AnimationClip>) {
        for i in 0..self.ids.len() {
            let state = self.entity_states[i];
            let anim = &mut self.animations[i];
            
            // 1. Reconcile (Semantic -> Visual)
            anim.sync(state);
            
            // 2. Update Visuals
            if let Some(clip) = clips.get(&anim.state) {
                anim.update(dt, clip);
            }
        }
    }

    pub fn get_transform_mut(&mut self, index: usize) -> &mut Transform {
        &mut self.transforms[index]
    }
}
