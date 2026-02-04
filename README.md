# Fantasmagorie Engine (Project Crystal)

**The Dual-Persona Engine** - A 2D-first, GPU-native rendering engine built in Rust.

Fantasmagorie is designed to bridge the gap between "Logic" (the abstract game world) and "Visuals" (the concrete GPU execution). It uses a layered architecture to provide a friendly, fluent API for developers while maintaining high performance and modern GPU features like SDF-based rendering and Jump Flooding Algorithms (JFA).

## 🌟 Key Features

-   **Dual-Persona:** Lite mode for efficiency (embedded, mobile) and Cinema mode for high-end visuals.
-   **Friendly API:** Fluent builders for UI and Sprites ("The Friendly Lie").
-   **WGPU Unified Backend:** Modern, stable, and cross-platform rendering (Web, Windows, Linux, macOS).
-   **SDF Visual Revolution:** High-performance UI rendering using Signed Distance Fields with Glassmorphism and Aurora effects.
-   **Immediate Mode UI:** Lightweight, responsive UI system with a Flex-like layout.

## 🏗️ Architecture

The engine uses the **V5 Crystal** architecture, consolidating implementation into two primary domains:

1.  **The Brain (Tracea):** Logic optimization and meaning interpretation.
2.  **The Muscle (Unified Backend):** GPU execution via a centralized `GpuExecutor`.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for more details.

## 🚀 Quick Start

```rust
use fanta_rust::prelude::*;
use fanta_rust::backend::WgpuBackend;
use fanta_rust::config::{EngineConfig, CinematicConfig, Bloom};

fn main() -> Result<(), String> {
    // 1. Initialize the Muscle (Backend)
    let backend = WgpuBackend::new_async(window_handle, 1280, 720)?;
    
    // 2. Configure the Brain (Cinema Profile)
    let config = EngineConfig::cinematic()
        .with_bloom(Bloom::Cinematic)
        .with_exposure(1.2);
    
    // 3. Initialize the Carrier (Renderer)
    let mut renderer = Renderer::new(Box::new(backend), config);
    
    // ... Frame loop logic ...
    
    // Update visuals at runtime (The Operational API)
    renderer.update_cinematic(CinematicConfig::default());
    
    Ok(())
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
