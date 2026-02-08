# Fantasmagorie Engine (Project Crystal)

**The Production-Ready 2D Game Engine** - Built in Rust, powered by WGPU.

Fantasmagorie has evolved from a simple rendering framework into a full-scale 2D game engine. It bridges the gap between high-level game logic and low-level GPU execution, offering a "Dual-Persona" architecture:
-   **Lite Mode:** Efficient, minimal overhead for simple apps.
-   **Cinema Mode:** High-end visual fidelity with deferred shading and post-processing.

## 🌟 Key Features

### 🎨 Visual Revolution (WGPU)
-   **Deferred Shading:** Decoupled G-Buffer rendering (Color, Normal, Emissive).
-   **Cinematic Post-Processing:**
    -   **Bloom:** High-quality neon glow.
    -   **Tone Mapping:** ACES Filmic & Reinhard.
    -   **Vignette & Film Grain:** For that polished look.
-   **Skeletal Animation:** Bone-based skinning with smooth blending and IK support.
-   **Particle System:** High-performance CPU/GPU hybrid effects.
-   **Tilemap Renderer:** Batched rendering for massive grid-based worlds.
-   **Parallax Scrolling:** Depth-based background layers.

### ⚛️ Physics & Simulation
-   **Impulse-Based Physics:** Realistic 2D collision resolution.
-   **SAT Collision Detection:** Accurate intersections for Circles, AABBs, and Polygons.
-   **Spatial Partitioning:** Quadtree optimization for broad-phase queries.
-   **Inverse Kinematics (IK):**
    -   `TwoBoneIK`: Analytic solver for limbs.
    -   `CCDIK`: Iterative solver for chains/tentacles.

### 🔊 Audio System
-   **2D Positional Audio:** Stereo panning and distance attenuation.
-   **Fire-and-Forget SFX:** Efficient voice recycling system.

### 🧠 Gameplay Framework
-   **ECS Architecture:** Entity-Component-System for data-oriented design.
-   **Scene Management:** Stack-based transitions (Menu -> Game -> Pause).
-   **Prefab System:** JSON-based entity templates.
-   **Signal Bus:** Decoupled event communication.
-   **Input System:** Action mapping (e.g., "Jump" = Space) and state tracking.

### 🛠️ UI & Tools
-   **Immediate Mode UI:** Flex-based layout with a rich widget set (Buttons, Knobs, Nodes).
-   **Visual Debugging:** Built-in debug draw for skeletons and colliders.

## 🚀 Quick Start

```rust
use fanta_rust::prelude::*;

fn main() {
    // 1. Initialize Engine
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = std::sync::Arc::new(event_loop.create_window(winit::window::WindowAttributes::default()).unwrap());
    
    // 2. Setup Backend & Renderer
    let backend = Box::new(fanta_rust::backend::wgpu::WgpuBackend::new_async(window.clone(), 1280, 720, 1.0).unwrap());
    let mut renderer = fanta_rust::renderer::Renderer::new_lite(backend);

    // 3. Create World & Add Entities
    let mut world = fanta_rust::game::World::new();
    let entity = world.spawn()
        .with(Transform::from_position(Vec2::new(100.0, 100.0)))
        .with(Sprite::new("player.png"))
        .build();

    // 4. Run Game Loop
    event_loop.run(move |event, _, control_flow| {
        // ... (Handle input, update world, render)
    }).unwrap();
}
```

## 🎮 Demos

Explore the `examples/` directory to see the engine in action:

| Demo | Description |
| :--- | :--- |
| **`breakout_demo.rs`** | **Must See!** "The World's Most Luxurious Breakout". Showcases Neon Bloom, Particles, Screenshake, and Audio. |
| `evolution_showcase.rs` | Integrated demo of Physics, AI, and Hierarchy. |
| `skeletal_blend_demo.rs` | Skeletal animation blending and playback. |
| `ik_demo.rs` | Interactive Inverse Kinematics (Two-Bone & CCD). |
| `tilemap_demo.rs` | Infinite scrolling tilemap renderer. |
| `physics_demo.rs` | SAT collision and rigid body dynamics. |
| `visual_demo.rs` | Deferred shading and post-processing showcase. |

## 📖 Documentation

-   [Architecture Overview](docs/ARCHITECTURE.md)
-   [Walkthrough](docs/walkthrough.md) (See `walkthrough.md` in project root for latest progress)

## 🛠️ Requirements

-   Rust 1.75+
-   WGPU-compatible GPU (Vulkan, Metal, DX12, WebGPU).
