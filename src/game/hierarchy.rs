use super::world::{World, EntityId};

pub fn attach(world: &mut World, child: EntityId, parent: EntityId) {
    if child == parent { return; }
    
    let child_idx = match world.id_to_index.get(&child) { Some(&i) => i, None => return };
    let parent_idx = match world.id_to_index.get(&parent) { Some(&i) => i, None => return };
    
    // 1. Detach from current parent
    detach(world, child);
    
    // 2. Set new parent
    world.parents[child_idx] = Some(parent);
    world.children[parent_idx].push(child);
}

pub fn detach(world: &mut World, child: EntityId) {
    let child_idx = match world.id_to_index.get(&child) { Some(&i) => i, None => return };
    
    if let Some(old_parent) = world.parents[child_idx].take() {
        let old_parent_idx = match world.id_to_index.get(&old_parent) { Some(&i) => i, None => return };
        world.children[old_parent_idx].retain(|&id| id != child);
    }
}
