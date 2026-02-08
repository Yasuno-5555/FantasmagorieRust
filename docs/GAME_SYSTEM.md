# Game System Architecture

Fantasmagorie's game logic is built on a custom ECS (Entity Component System) and event-driven architecture.

## 1. The World (ECS)

The `World` struct (`src/game/world.rs`) is the heart of the simulation.

### Spawning Entities
Entities are created using a builder pattern:

```rust
let player_id = world.spawn()
    .with(Transform::from_position(Vec2::new(0.0, 0.0)))
    .with(Sprite::new("player.png"))
    .with(PhysicsComponent::new(10.0)) // Mass = 10kg
    .with(Collider::Circle { offset: Vec2::ZERO, radius: 16.0 })
    .build();
```

### System Iteration
Systems iterate over components to update state. For example, the Physics System:

```rust
pub fn update_physics(world: &mut World, dt: f32) {
    // 1. Integrate Velocity
    for i in 0..world.entities.len() {
        if let Some(phys) = &mut world.physics_components[i] {
            if let Some(trans) = &mut world.transforms[i] {
                trans.position += phys.velocity * dt;
            }
        }
    }
    // 2. Resolve Collisions...
}
```

## 2. Scene Graph
Entities support parent-child relationships. When a parent moves, all children move relative to it.

```rust
// Attach weapon to player
fanta::game::hierarchy::attach(&mut world, weapon_id, player_id);
```

The `Transform` component automatically calculates `global_matrix` by combining its `local_matrix` with the parent's global matrix.

## 3. State Machines
Complex entity logic (like AI) is managed via the `StateMachine` component (`src/game/state_machine.rs`).

```rust
// Define states
#[derive(Clone, Debug, PartialEq)]
enum EnemyState {
    Idle,
    Chase,
    Attack
}

// Update logic
fn update_ai(world: &mut World) {
    for (id, state_machine) in world.query_mut::<StateMachine>() {
        match state_machine.current_state {
            EnemyState::Idle => {
                if player_in_range() {
                    state_machine.transition_to(EnemyState::Chase);
                }
            }
            // ...
        }
    }
}
```

## 4. Signal Bus
Entities communicate without tight coupling using the `SignalBus`.

```rust
// Emit signal
signal_bus.emit(Signal {
    event: "entity_damaged",
    source: Some(entity_id),
    data: SignalData::Float(damage_amount)
});

// Handle signal
if let Some(signal) = signal_bus.pop() {
    match signal.event {
        "entity_damaged" => play_sound("hurt.wav"),
        _ => {}
    }
}
```
