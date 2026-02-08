use serde::{Deserialize, Serialize};
use super::{World, EntityId, Transform, PhysicsComponent, Collider, StateMachine};
use crate::core::Vec2;

/// A template for creating entities with predefined components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prefab {
    pub name: String,
    pub transform: Option<Transform>,
    pub physics: Option<PhysicsComponent>,
    pub collider: Option<Collider>,
    pub state_machine: Option<StateMachine>,
}

impl Prefab {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            transform: None,
            physics: None,
            collider: None,
            state_machine: None,
        }
    }

    /// Spawns a new entity in the world based on this prefab.
    pub fn spawn(&self, world: &mut World, position: Vec2) -> EntityId {
        let id = world.spawn();
        let index = *world.id_to_index.get(&id).unwrap();
        
        if let Some(mut t) = self.transform.clone() {
            t.local_position = position;
            world.transforms[index] = t;
        } else {
            world.transforms[index].local_position = position;
        }
        
        if let Some(p) = self.physics.clone() {
            world.physics[index] = p;
        }
        
        if let Some(c) = self.collider.clone() {
            world.colliders[index] = Some(c);
        }
        
        if let Some(s) = self.state_machine.clone() {
            world.state_machines[index] = Some(s);
        }
        
        // Recalculate transform just for this entity
        world.system_transform_update();
        
        id
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}
