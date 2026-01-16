use std::collections::{HashMap, VecDeque};
use super::node::{NodeStore, LayoutData};
use super::types::{Transform, Vec2, NodeID, StableID};
use crate::widgets::WidgetKind;

// FNV-1a 32-bit constants
const FNV_OFFSET_BASIS: u32 = 2166136261;
const FNV_PRIME: u32 = 16777619;

fn hash_str(s: &str) -> u32 {
    let mut hash = FNV_OFFSET_BASIS;
    for b in s.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[derive(Clone, Debug)]
pub struct AnimState {
    pub value: f32,
    pub initialized: bool,
}

pub struct UIContext {
    pub store: NodeStore,
    pub id_map: HashMap<StableID, NodeID>, // Hash -> Index
    
    // Stack
    pub node_stack: Vec<NodeID>, // Indices
    pub root_id: NodeID,
    pub current_id: NodeID,

    // Layout Cursor
    pub cursor_x: f32,
    pub cursor_y: f32,

    // Input
    pub mouse_pos: Vec2,
    pub mouse_down: bool,
    pub input_chars: Vec<char>, 
    
    // Transform
    pub current_transform: Transform,
    pub transform_stack: Vec<Transform>,

    // Persistence
    pub prev_layout: HashMap<StableID, LayoutData>,
    pub anims: HashMap<StableID, HashMap<i32, AnimState>>, // Prop ID -> State

    // Helpers
    pub last_node_size: Vec2,
}

impl UIContext {
    pub fn new() -> Self {
        Self {
            store: NodeStore::new(),
            id_map: HashMap::new(),
            node_stack: Vec::new(),
            root_id: 0,
            current_id: 0,
            cursor_x: 0.0,
            cursor_y: 0.0,
            mouse_pos: Vec2::default(),
            mouse_down: false,
            input_chars: Vec::new(),
            current_transform: Transform::identity(),
            transform_stack: Vec::new(),
            prev_layout: HashMap::new(),
            anims: HashMap::new(),
            last_node_size: Vec2::default(),
        }
    }

    pub fn begin_frame(&mut self) {
        // Persist layout from previous frame
        self.prev_layout.clear();
        for (stable_id, &idx) in &self.id_map {
            if idx < self.store.layout.len() {
                self.prev_layout.insert(*stable_id, self.store.layout[idx].clone());
            }
        }

        self.store.clear();
        self.id_map.clear();
        self.node_stack.clear();
        self.current_id = 0; 
        self.cursor_x = 0.0;
        self.cursor_y = 0.0;
        self.input_chars.clear();
    }

    pub fn end_frame(&mut self) {
        // Layout solve happens in application
    }

    pub fn get_id(&self, str_id: &str) -> StableID {
        let mut base = hash_str(str_id) as u64;
        
        // Combine with parent Stack ID
        if let Some(&parent_idx) = self.node_stack.last() {
            // Need to verify parent_idx is valid in store.ids
            if parent_idx < self.store.ids.len() {
                let parent_stable = self.store.ids[parent_idx];
                // simple mixing
                base = base ^ (parent_stable.wrapping_mul(16777619));
            }
        }
        base
    }

    pub fn begin_node(&mut self, id_str: &str, kind: WidgetKind) -> NodeID {
        let stable_id = self.get_id(id_str);
        
        // Check if collision? (assuming none for now)
        let nid = self.store.add_node(kind, stable_id);
        
        // Parent relationship
        if let Some(&parent) = self.node_stack.last() {
            self.store.set_parent(nid, parent);
        } else {
            self.root_id = nid;
        }
        
        self.node_stack.push(nid);
        self.current_id = nid;
        self.id_map.insert(stable_id, nid);
        
        nid
    }

    pub fn end_node(&mut self) {
        if let Some(id) = self.node_stack.pop() {
            if id < self.store.layout.len() {
                self.last_node_size = Vec2::new(self.store.layout[id].w, self.store.layout[id].h);
            }
            self.current_id = *self.node_stack.last().unwrap_or(&0);
        }
    }
}
