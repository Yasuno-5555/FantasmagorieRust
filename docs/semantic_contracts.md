# Semantic Contracts & Invariants

This document defines the interface guarantees and invariants required by the Fantasmagorie backends. All GPU passes and resource allocations must adhere to these contracts to ensure correctness and portability across WGPU and Metal.

## 1. Resource Management (GpuExecutor)

### Buffers
*   **Uniform Buffers**:
    *   Must be aligned to 256 bytes for dynamic offsets.
    *   Updated via `GpuExecutor::write_buffer`.
*   **Storage Buffers**:
    *   Created with `BufferUsage::Storage`.
    *   Used for large-scale compute data (TTG nodes, Manifold states).

### Textures
*   **Usage Flags**:
    *   Read: `TextureUsage::TEXTURE_BINDING`.
    *   Write: `TextureUsage::STORAGE_BINDING`.
    *   Render Output: `TextureUsage::RENDER_ATTACHMENT`.
*   **Implicit Synchronization**:
    *   WGPU handles layout transitions and barriers internally.
    *   GpuExecutor implementations must ensure `end_execute` is called before presenting to flush commands.

## 2. Compute Dispatch (The Muscle)

### JFA Pipeline
*   **Ping-Pong Invariant**:
    *   Input and Output textures must be swapped between steps.
    *   Step size decreases by powers of 2 (N/2, N/4 ... 1).
*   **Workgroup Size**:
    *   Standardized to [16, 16, 1] for 2D UI/SDF tasks.

## 3. Shader Interfaces (WGSL)

*   **Bind Groups**:
    *   **Group 0**: Global Context (Uniforms, Samplers).
    *   **Group 1**: Task-Specific Resources (Atlas, Backdrop).
*   **Standard Uniforms**:
    *   See `src/backend/shaders/types.rs` for the `GlobalUniforms` layout.

## 4. Coordinate Systems

*   **NDC (Normalized Device Coordinates)**:
    *   **WGPU/Metal**: Y-up, Z in [0, 1]. Top is +1.0 in UI space, but projection converts it so 0,0 is Top-Left.
    *   **Vulkan (Compatibility Layer)**: Handled by `y_flip()` in `GpuExecutor`.
*   **UVs**:
    *   [0, 0] is Top-Left, [1, 1] is Bottom-Right across all backends.
    *   SDF calculation happens in *Pixel Space* then normalized to UV units.
