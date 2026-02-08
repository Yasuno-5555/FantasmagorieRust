use crate::core::{Vec2, Mat3, ColorF, Rectangle};
use std::sync::atomic::{AtomicU64, Ordering};
pub use super::interaction::{Collider, InteractionState};
pub use super::state_machine::StateMachine;
pub use super::signals::{SignalBus, Signal, SignalData};
use super::animation::{AnimationComponent, EntityState, AnimationClip};
use super::input::ActionState;
use super::tilemap::TileMap;
use super::particles::{Particle, ParticleEmitter, ParticleSystem};
use super::audio::{AudioEmitter, AudioListener};
use super::parallax::ParallaxLayer;
use crate::audio::{AudioEngine, SoundType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

static NEXT_ENTITY_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub local_position: Vec2,
    pub local_rotation: f32,
    pub local_scale: Vec2,
    #[serde(skip)] // World matrix is recalculated on load
    pub world_matrix: Mat3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            local_position: Vec2::ZERO,
            local_rotation: 0.0,
            local_scale: Vec2::new(1.0, 1.0),
            world_matrix: Mat3::IDENTITY,
        }
    }
}

impl Transform {
    pub fn local_matrix(&self) -> Mat3 {
        Mat3::translation(self.local_position.x, self.local_position.y)
            * Mat3::rotation(self.local_rotation)
            * Mat3::scale(self.local_scale.x, self.local_scale.y)
    }

    /// Returns the world position derived from the world matrix.
    pub fn world_position(&self) -> Vec2 {
        Vec2::new(self.world_matrix.0[2], self.world_matrix.0[5])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsComponent {
    pub velocity: Vec2,
    pub mass: f32, // 0.0 for static
    pub friction: f32,
    pub restitution: f32,
}

impl Default for PhysicsComponent {
    fn default() -> Self {
        Self {
            velocity: Vec2::ZERO,
            mass: 1.0,
            friction: 0.1,
            restitution: 0.5,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct World {
    pub ids: Vec<EntityId>,
    pub id_to_index: HashMap<EntityId, usize>,
    pub transforms: Vec<Transform>,
    pub physics: Vec<PhysicsComponent>,
    #[serde(skip)] // Sprites usually involve textures/handles that need custom loading
    pub sprites: Vec<Option<crate::game::sprite::Sprite>>,
    pub colliders: Vec<Option<Collider>>,
    pub interactions: Vec<InteractionState>,
    pub animations: Vec<AnimationComponent>,
    pub input_states: Vec<ActionState>,
    pub entity_states: Vec<EntityState>,
    pub state_machines: Vec<Option<StateMachine>>,
    #[serde(skip)] // Signals are ephemeral
    pub signal_bus: SignalBus,
    
    // Hierarchy
    pub parents: Vec<Option<EntityId>>,
    pub children: Vec<Vec<EntityId>>,
    pub tilemaps: Vec<Option<TileMap>>,
    pub particle_emitters: Vec<Option<ParticleEmitter>>,
    pub audio_emitters: Vec<Option<AudioEmitter>>,
    pub audio_listeners: Vec<Option<AudioListener>>,
    pub parallax_layers: Vec<Option<ParallaxLayer>>,
    
    #[serde(skip)]
    pub particle_system: ParticleSystem,
    #[serde(skip)]
    pub audio_engine: Option<AudioEngine>,
}

impl World {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            id_to_index: HashMap::new(),
            transforms: Vec::new(),
            physics: Vec::new(),
            sprites: Vec::new(),
            colliders: Vec::new(),
            interactions: Vec::new(),
            animations: Vec::new(),
            input_states: Vec::new(),
            entity_states: Vec::new(),
            state_machines: Vec::new(),
            signal_bus: SignalBus::new(),
            parents: Vec::new(),
            children: Vec::new(),
            tilemaps: Vec::new(),
            particle_emitters: Vec::new(),
            audio_emitters: Vec::new(),
            audio_listeners: Vec::new(),
            parallax_layers: Vec::new(),
            particle_system: ParticleSystem::new(10000),
            audio_engine: Some(AudioEngine::new()),
        }
    }

    pub fn spawn(&mut self) -> EntityId {
        let id = EntityId(NEXT_ENTITY_ID.fetch_add(1, Ordering::Relaxed));
        let index = self.ids.len();
        self.ids.push(id);
        self.id_to_index.insert(id, index);
        self.transforms.push(Transform::default());
        self.physics.push(PhysicsComponent::default());
        self.sprites.push(None);
        self.colliders.push(None);
        self.interactions.push(InteractionState::default());
        self.animations.push(AnimationComponent::default());
        self.input_states.push(ActionState::default());
        self.entity_states.push(EntityState::default());
        self.state_machines.push(None);
        self.parents.push(None);
        self.children.push(Vec::new());
        self.tilemaps.push(None);
        self.particle_emitters.push(None);
        self.audio_emitters.push(None);
        self.audio_listeners.push(None);
        self.parallax_layers.push(None);
        id
    }

    /// Sets a collider for the given entity index.
    pub fn set_collider(&mut self, index: usize, collider: Collider) {
        if index < self.colliders.len() {
            self.colliders[index] = Some(collider);
        }
    }

    /// Runs the interaction system to update states.
    pub fn system_interaction(&mut self, proximity_radius: f32) {
        let count = self.ids.len();
        
        // 1. Prepare next frame (reset transient states)
        for state in &mut self.interactions {
            state.prepare_next_frame();
        }
        
        // 2. Pairwise interaction check
        for i in 0..count {
            for j in 0..count {
                if i == j { continue; }
                
                let col_a = match &self.colliders[i] { Some(c) => c, None => continue };
                let col_b = match &self.colliders[j] { Some(c) => c, None => continue };
                
                let pos_a = self.transforms[i].world_position();
                let pos_b = self.transforms[j].world_position();
                
                // Check Touch (AABB overlap)
                if super::interaction::intersects(pos_a, col_a, pos_b, col_b) {
                    self.interactions[i].is_touched = true;
                    self.interactions[i].proximity = 1.0;
                }
                
                // Check Approach (Proximity)
                let prox = super::interaction::get_proximity(pos_a, col_a, pos_b, col_b, proximity_radius);
                if prox > 0.0 {
                    self.interactions[i].is_near = true;
                    self.interactions[i].proximity = self.interactions[i].proximity.max(prox);
                }
            }
        }
    }

    /// Recursively updates transforms starting from root entities.
    pub fn system_transform_update(&mut self) {
        let mut process_queue = Vec::new();

        // Start with root entities (those with no parent)
        for i in 0..self.ids.len() {
            if self.parents[i].is_none() {
                self.transforms[i].world_matrix = self.transforms[i].local_matrix();
                for &child_id in &self.children[i] {
                    process_queue.push((child_id, self.transforms[i].world_matrix));
                }
            }
        }

        // Process children breadth-first
        let mut head = 0;
        while head < process_queue.len() {
            let (child_id, parent_world_matrix) = process_queue[head];
            head += 1;

            if let Some(&index) = self.id_to_index.get(&child_id) {
                self.transforms[index].world_matrix = parent_world_matrix * self.transforms[index].local_matrix();
                for &grandchild_id in &self.children[index] {
                    process_queue.push((grandchild_id, self.transforms[index].world_matrix));
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

    /// Sets the velocity for the given entity index.
    pub fn set_velocity(&mut self, index: usize, velocity: Vec2) {
        if index < self.physics.len() {
            self.physics[index].velocity = velocity;
        }
    }

    /// Updates positions based on velocities (Rigid Body Dynamics).
    pub fn system_physics_step(&mut self, dt: f32) {
        for i in 0..self.ids.len() {
            if self.physics[i].mass > 0.0 {
                let vel = self.physics[i].velocity;
                self.transforms[i].local_position = self.transforms[i].local_position + vel * dt;
            }
        }
    }

    /// Resolves collisions using impulse-based physics (Optimized with Quadtree).
    pub fn system_physics_collision(&mut self) {
        // 1. Build Quadtree
        // Use a large bounds for valid world area
        let bounds = Rectangle::new(-10000.0, -10000.0, 20000.0, 20000.0);
        let mut quadtree = super::spatial::Quadtree::new(0, bounds);
        
        let count = self.ids.len();
        
        // Insert all dynamic and static colliders
        for i in 0..count {
            if let Some(col) = &self.colliders[i] {
                let pos = self.transforms[i].world_position();
                let rect = match col {
                    Collider::AABB { offset, size } => Rectangle::new(pos.x + offset.x, pos.y + offset.y, size.x, size.y),
                    Collider::Circle { offset, radius } => Rectangle::new(pos.x + offset.x - radius, pos.y + offset.y - radius, radius * 2.0, radius * 2.0),
                    Collider::Polygon { offset, vertices } => {
                        let mut min = Vec2::new(f32::MAX, f32::MAX);
                        let mut max = Vec2::new(f32::MIN, f32::MIN);
                        for v in vertices {
                            let p = pos + *offset + *v;
                            min = min.min(p);
                            max = max.max(p);
                        }
                        if min.x > max.x { // Empty poly
                            Rectangle::new(pos.x, pos.y, 0.0, 0.0) 
                        } else {
                            Rectangle::new(min.x, min.y, max.x - min.x, max.y - min.y)
                        }
                    }
                };
                quadtree.insert(self.ids[i], rect);
            }
        }

        let mut candidates = Vec::with_capacity(32);
        
        // 2. Query and Resolve
        for i in 0..count {
            if self.colliders[i].is_none() { continue; }
            if self.physics[i].mass == 0.0 { continue; } // Optimization: Only dynamic bodies check for collisions? 
            // Note: If both are static, no resolution needed. If one is dynamic, we need to resolve.
            // If we skip static here, we won't check Static vs Dynamic if iteration order is wrong?
            // Actually, if we only iterate dynamic bodies `i`, we check against all potential `j` (static or dynamic) from tree.
            // This is correct and optimized.

            // Recompute rect (could cache this)
            let col_i = self.colliders[i].as_ref().unwrap();
            let pos_i = self.transforms[i].world_position();
            let rect_i = match col_i {
                    Collider::AABB { offset, size } => Rectangle::new(pos_i.x + offset.x, pos_i.y + offset.y, size.x, size.y),
                    Collider::Circle { offset, radius } => Rectangle::new(pos_i.x + offset.x - radius, pos_i.y + offset.y - radius, radius * 2.0, radius * 2.0),
                    Collider::Polygon { offset, vertices } => {
                        let mut min = Vec2::new(f32::MAX, f32::MAX);
                        let mut max = Vec2::new(f32::MIN, f32::MIN);
                        for v in vertices {
                            let p = pos_i + *offset + *v;
                            min = min.min(p);
                            max = max.max(p);
                        }
                        if min.x > max.x { Rectangle::new(pos_i.x, pos_i.y, 0.0, 0.0) } else { Rectangle::new(min.x, min.y, max.x - min.x, max.y - min.y) }
                    }
            };
            
            candidates.clear();
            quadtree.retrieve(&rect_i, &mut candidates);
            
            for &other_id in &candidates {
                if other_id == self.ids[i] { continue; }
                
                let j = match self.id_to_index.get(&other_id) {
                    Some(&idx) => idx,
                    None => continue, // Should not happen
                };
                
                // Avoid double resolution: iterate i, query returns j.
                // Resolution modifies both.
                // Standard approach: Only resolve if i < j OR if one is static.
                // Here we iterate all Dynamic `i`. `j` can be static or dynamic.
                // If `j` is dynamic and `j > i`, we process it now.
                // If `j` is dynamic and `j < i`, we already processed it when we were at `j`.
                // BUT, `resolve_collision` applies impulses to both.
                // So we should only process if `i < j` to avoid double counting impulse.
                // What if `j` is static?
                // Static has mass 0. Collision resolution handles it (mass infinite).
                // If `j` is static, `j` index logic check?
                // `i` is always dynamic loop.
                // If `j` is static, we MUST process it (because static loop won't run).
                // If `j` is dynamic, we enforce `i < j`.
                
                let is_j_dynamic = self.physics[j].mass > 0.0;
                if is_j_dynamic && i > j { continue; }

                let col_j = self.colliders[j].as_ref().unwrap(); 
                let pos_j = self.transforms[j].world_position();
                
                if let Some(manifold) = super::physics::check_collision(pos_i, col_i, pos_j, col_j) {
                    let m_i = self.physics[i].mass;
                    let r_i = self.physics[i].restitution;
                    let m_j = self.physics[j].mass;
                    let r_j = self.physics[j].restitution;
                    
                    let mut p_i = self.transforms[i].local_position;
                    let mut v_i = self.physics[i].velocity;
                    let mut p_j = self.transforms[j].local_position;
                    let mut v_j = self.physics[j].velocity;

                    super::physics::resolve_collision(
                        &mut p_i, &mut v_i, m_i, r_i,
                        &mut p_j, &mut v_j, m_j, r_j,
                        &manifold
                    );
                    
                    self.transforms[i].local_position = p_i;
                    self.physics[i].velocity = v_i;
                    self.transforms[j].local_position = p_j;
                    self.physics[j].velocity = v_j;
                }
            }
        }
    }

    /// Updates all state machines.
    pub fn system_state_update(&mut self, dt: f32) {
        for i in 0..self.ids.len() {
            if let Some(fsm) = &mut self.state_machines[i] {
                if let Some(new_state) = fsm.update(dt) {
                    self.signal_bus.broadcast(
                        Some(self.ids[i]),
                        SignalData::Custom(format!("state_changed:{}", new_state))
                    );
                }
            }
        }
    }

    /// Dispatches signals (placeholder for complex logic).
    pub fn system_signal_dispatch(&mut self) {
        while let Some(signal) = self.signal_bus.poll() {
            // Logic for handling signals would go here.
            // For now, we just log collision signals for physics bodies.
            if let SignalData::Collision { other, .. } = &signal.data {
                if let (Some(s), Some(t)) = (signal.source, Some(other)) {
                    // Could trigger state transitions here
                }
            }
        }
    }

    /// Updates particles and emitters.
    pub fn system_particles(&mut self, dt: f32) {
        // Update System (Simulation)
        self.particle_system.update(dt);

        // Update Emitters (Spawning)
        let count = self.ids.len();
        for i in 0..count {
            if let Some(emitter) = &mut self.particle_emitters[i] {
                if !emitter.active { continue; }
                
                emitter.accumulator += dt * emitter.rate;
                while emitter.accumulator >= 1.0 {
                    emitter.accumulator -= 1.0;
                    
                    // Spawn logic
                    let pos = self.transforms[i].world_position();
                    let angle = (super::particles::rand_f32() - 0.5) * emitter.cone_angle + emitter.direction.y.atan2(emitter.direction.x);
                    let speed = emitter.speed_range[0] + super::particles::rand_f32() * (emitter.speed_range[1] - emitter.speed_range[0]);
                    let vel = Vec2::new(angle.cos() * speed, angle.sin() * speed);
                    let life = emitter.lifetime_range[0] + super::particles::rand_f32() * (emitter.lifetime_range[1] - emitter.lifetime_range[0]);
                    let size = emitter.size_range[0] + super::particles::rand_f32() * (emitter.size_range[1] - emitter.size_range[0]);
                    
                    self.particle_system.spawn(Particle {
                        position: pos,
                        velocity: vel,
                        color: emitter.color_start, // Could lerp start/end based on variance
                        life,
                        max_life: life,
                        size,
                    });
                    
                    if emitter.one_shot {
                        if emitter.burst_count > 0 {
                            emitter.burst_count -= 1;
                        } else {
                            emitter.active = false;
                        }
                    }
                }
            }
        }
    }

    /// Updates audio spatialization
    pub fn system_audio(&mut self) {
        if let Some(engine) = &self.audio_engine {
            // 1. Update Listener
            let mut listener_pos = Vec2::ZERO;
            for i in 0..self.ids.len() {
                if let Some(l) = &self.audio_listeners[i] {
                    if l.active {
                        listener_pos = self.transforms[i].world_position();
                        break;
                    }
                }
            }
            engine.set_listener_pos(listener_pos);

            // 2. Update Emitters
            for i in 0..self.ids.len() {
                if let Some(emitter) = &self.audio_emitters[i] {
                    if emitter.playing {
                        let pos = self.transforms[i].world_position();
                        let sound_type = match emitter.params_id {
                            1 => SoundType::Noise,
                            2 => SoundType::Square(110.0),
                            _ => SoundType::Sine(440.0),
                        };
                        // Use EntityId as VoiceId
                        let id = self.ids[i].0;
                        engine.play_sound(
                            id, 
                            sound_type, 
                            pos, 
                            emitter.volume, 
                            emitter.pitch, 
                            emitter.looping
                        );
                    }
                }
            }
        }
    }

    pub fn system_parallax(&mut self, camera_pos: Vec2) {
        let one = Vec2::new(1.0, 1.0);
        for i in 0..self.ids.len() {
            if let Some(p) = &self.parallax_layers[i] {
                // Parallax offset: moves slower than camera
                // offset = camera_pos * (1.0 - factor)
                // If factor=1.0 (normal), offset=0.
                // If factor=0.5 (far), offset=camera*0.5.
                // New Position = Base + Offset
                
                let fx = 1.0 - p.factor.x;
                let fy = 1.0 - p.factor.y;
                let offset = Vec2::new(camera_pos.x * fx, camera_pos.y * fy);
                self.transforms[i].local_position = p.base_position + offset;
            }
        }
    }

    pub fn get_transform_mut(&mut self, index: usize) -> &mut Transform {
        &mut self.transforms[index]
    }

    /// Serializes the world to a JSON string.
    pub fn save_to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// Loads the world from a JSON string.
    pub fn load_from_json(json: &str) -> Result<Self, String> {
        let mut world: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        
        // Post-load: Restore transient state
        world.signal_bus = SignalBus::new();
        world.sprites = vec![None; world.ids.len()];
        world.particle_system = ParticleSystem::new(10000);
        world.audio_engine = Some(AudioEngine::new());
        
        // Ensure arrays satisfy length
        if world.tilemaps.len() < world.ids.len() { world.tilemaps.resize(world.ids.len(), None); }
        if world.particle_emitters.len() < world.ids.len() { world.particle_emitters.resize(world.ids.len(), None); }
        if world.audio_emitters.len() < world.ids.len() { world.audio_emitters.resize(world.ids.len(), None); }
        if world.audio_listeners.len() < world.ids.len() { world.audio_listeners.resize(world.ids.len(), None); }
        if world.parallax_layers.len() < world.ids.len() { world.parallax_layers.resize(world.ids.len(), None); }
        
        // Recalculate all world matrices
        world.system_transform_update();
        
        Ok(world)
    }
}
