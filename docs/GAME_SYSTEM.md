# Game System: Life & Motion

The game system in Fantasmagorie focuses on high-performance entity management and advanced animation techniques.

## World & Entities

The `World` struct is a lightweight ECS-like container.

```rust
let mut world = World::new();

// Spawn an entity
let player_id = world.spawn();

// Access components
world.transforms[player_id].position = Vec2::new(100.0, 100.0);
world.entity_states[player_id] = EntityState::Idle;
```

## Animation: Motion Morphing

Fantasmagorie uses a unique **Motion Morphing** system. Instead of simple sprite flipping, it can interpolate between animations using SDF (Signed Distance Fields) or custom blending weights.

### Animation Clips
An `AnimationClip` defines a sequence of frames.

```rust
let idle_clip = AnimationClip {
    frames: vec![
        AnimationFrame { texture_id: 1, duration: 0.5, uv_rect: None },
    ],
    loop_clip: true,
};
```

### Sprite Builder
The `SpriteBuilder` allows you to apply these animations to entities in the world.

```rust
SpriteBuilder::new(&mut frame, pos, size)
    .animate(anim.clone())
    .draw(&sprite, Some(&idle_clip));
```

## Input Mapping

The engine abstracts raw input (keys, mouse) into **Actions**.

```rust
let mut action_map = ActionMap::new_default();
action_map.bind(KeyCode::KeyW, Action::MoveUp);

if action_state.is_active(Action::MoveUp) {
    // Move character
}
```

This allows for easy rebinding and controller support without changing game logic.
