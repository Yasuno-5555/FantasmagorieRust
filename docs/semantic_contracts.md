# Semantic Contracts & Invariants

This document defines the interface guarantees and invariants required by the Fantasmagorie backends. All GPU passes and resource allocations must adhere to these contracts to ensure correctness and portability across WGPU and Metal.

## 1. Resource Management

### Buffers
*   **Uniform Buffers**:
    *   Must be aligned to 256 bytes for dynamic offsets.
    *   Updated via `GpuExecutor::write_buffer`.
*   **Storage Buffers**:
    *   Created with `BufferUsage::Storage`.
    *   Used for large-scale compute data (Particle Systems, Skinning Matrices).

### Textures
*   **Usage Flags**:
    *   Read: `TextureUsage::TEXTURE_BINDING`.
    *   Write: `TextureUsage::STORAGE_BINDING`.
    *   Render Output: `TextureUsage::RENDER_ATTACHMENT`.
*   **Implicit Synchronization**:
    *   WGPU handles layout transitions and barriers internally.
    *   GpuExecutor implementations must ensure `end_execute` is called before presenting to flush commands.

## 2. Shader Interfaces (WGSL)

*   **Bind Groups**:
    *   **Group 0**: Global Context (Camera Uniforms, Samplers).
    *   **Group 1**: Material/Object Resources (Textures, Bone Matrices).
*   **Standard Uniforms**:
    *   See `src/backend/shaders/types.rs` for the `GlobalUniforms` layout.

## 3. Coordinate Systems

*   **NDC (Normalized Device Coordinates)**:
    *   **WGPU/Metal**: Y-up, Z in [0, 1]. Top is +1.0 in UI space, but projection converts it so 0,0 is Top-Left.
    *   **Vulkan (Compatibility Layer)**: Handled by `y_flip()` in `GpuExecutor` or projection matrix adjustment.
*   **UVs**:
    *   [0, 0] is Top-Left, [1, 1] is Bottom-Right across all backends.
    *   SDF calculation happens in *Pixel Space* then normalized to UV units.

## 4. Animation Invariants

*   **Bone Weights**: Use `vec4<f32>` and `vec4<u32>`. Weights must sum to 1.0. 
*   **Max Bones**: 64 bones per skeleton (limited by uniform buffer size in some backends).
