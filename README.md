# Fantasmagorie Engine (Project Crystal)

**The Dual-Persona Engine** - A 2D-first, GPU-native rendering engine built in Rust.

Fantasmagorie is designed to bridge the gap between "Logic" (the abstract game world) and "Visuals" (the concrete GPU execution). It uses a layered architecture to provide a friendly, fluent API for developers while maintaining high performance and modern GPU features like SDF-based rendering and Jump Flooding Algorithms (JFA).

## 🌟 Key Features

-   **Dual-Persona:** Lite mode for efficiency (embedded, mobile) and Cinema mode for high-end visuals.
-   **Friendly API:** Fluent builders for UI and Sprites ("The Friendly Lie").
-   **GPU Native:** Built with a focus on modern graphics APIs (Vulkan, DX12, Metal).
-   **SDF Morphing:** Advanced animation system using Signed Distance Fields for smooth transitions.
-   **Immediate Mode UI:** Lightweight, responsive UI system with a Flex-like layout.

## 🏗️ Architecture

The engine is structured into four distinct layers:

1.  **Layer 1 (User API):** Fluent API / Builder pattern (The Friendly Lie)
2.  **Layer 2 (Tracea):** Optimization, compute, meaning interpretation (The Brain)
3.  **Layer 3 (Renderer):** Abstraction and translation (The Boundary)
4.  **Layer 4 (Backend):** GPU command execution (The Muscle)

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for more details.

## 🚀 Quick Start

```rust
use fanta_rust::prelude::*;

fn main() {
    let mut renderer = Renderer::new_lite(create_backend()); // Choose your backend
    
    loop {
        let mut frame = renderer.begin_frame();
        
        // Build UI
        UIContext::new(&mut frame).column().padding(20).bg(ColorF::DARK_GRAY).build(|| {
            TextBuilder::new("Hello, Fantasmagorie!").font_size(24.0).build();
            if ButtonBuilder::new("Click Me").build().clicked() {
                println!("Button clicked!");
            }
        });

        renderer.end_frame(frame, 1280, 720);
    }
}
```

## 📖 Documentation

-   [Architecture Overview](docs/ARCHITECTURE.md)
-   [UI System](docs/UI_SYSTEM.md)
-   [Game & Animation System](docs/GAME_SYSTEM.md)
-   [Semantic Contracts (Internal)](docs/semantic_contracts.md)

## 🛠️ Requirements

-   Rust 1.75+
-   Vulkan (recommended), OpenGL, or DX12 compatible hardware.
