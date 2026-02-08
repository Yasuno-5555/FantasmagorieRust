use super::world::{World, EntityId};

pub fn attach(world: &mut World, child: EntityId, parent: EntityId) {
    if child == parent { return; }
    
    let child_idx = *world.id_to_index.get(&child).expect("Child entity not found");
    let parent_idx = *world.id_to_index.get(&parent).expect("Parent entity not found");
    
    // 1. Detach from current parent
    detach(world, child);
    
    // 2. Set new parent
    world.parents[child_idx] = Some(parent);
    world.children[parent_idx].push(child);
}

pub fn detach(world: &mut World, child: EntityId) {
    let child_idx = *world.id_to_index.get(&child).expect("Child entity not found");
    
    if let Some(old_parent) = world.parents[child_idx].take() {
        let old_parent_idx = *world.id_to_index.get(&old_parent).expect("Parent entity not found");
        world.children[old_parent_idx].retain(|&id| id != child);
    }
}
