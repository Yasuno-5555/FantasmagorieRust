# Tracea Semantic Contracts & Invariants

This document defines the interface guarantees and invariants required by the Tracea backend. All GPU passes and resource allocations must adhere to these contracts to ensure correctness and portability across backends (Vulkan, DX12, Metal).

## 1. Resource Management

### Buffers
*   **Uniform Buffers**:
    *   Must be aligned to `minUniformBufferOffsetAlignment` (typically 256 bytes) when creating dynamic uniform descriptors.
    *   Host-visible uniform buffers are implicitly coherent or flushed explicitly by the backend.
*   **Storage Buffers**:
    *   Must be created with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
    *   RW Structured Buffers in HLSL/WGSL map to plain Storage Buffers in Vulkan.

### Textures (Images)
*   **Usage Flags**:
    *   Any texture read by a Compute Shader must have `SAMPLED` usage.
    *   Any texture written by a Compute Shader must have `STORAGE` usage.
    *   Textures used in Render Passes must have `COLOR_ATTACHMENT` or `DEPTH_STENCIL_ATTACHMENT` usage.
*   **Layout Transitions**:
    *   Compute Read: `VK_IMAGE_LAYOUT_GENERAL` or `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.
    *   Compute Write: `VK_IMAGE_LAYOUT_GENERAL`.
    *   The JFA pipeline uses `VK_IMAGE_LAYOUT_GENERAL` for *both* read and write to simplify Ping-Pong feedback loops without complex barrier micro-management within the loop.

## 2. Compute Dispatch

### Kernel 5: Jump Flooding Algorithm (JFA)
*   **Ping-Pong Invariant**:
    *   Input texture at binding `N` must *not* be the same resource as Output texture at binding `M` for the same dispatch.
    *   Ping-Pong is managed externally by swapping Descriptor Sets or Push Constants pointing to indices.
*   **Step Size**:
    *   The JFA `step` parameter decreases by powers of 2: $N/2, N/4, \dots, 1$.
    *   Dispatch count must exactly match $\log_2(\text{max\_dimension})$.

### Kernel 4: Cinematic Resolver
*   **Wave Intrinsics**:
    *   Resolver shaders may rely on subgroup operations (`wave_active_count`, etc.).
    *   `SIMD32` is the minimum requirement for Wave Ops; fallback paths must be provided for non-subgroup hardware (though low priority for this research engine).

## 3. Shader Interfaces (WGSL)

*   **Bind Groups**:
    *   Group 0 is reserved for Global Context (Uniforms, Samplers).
    *   Group 1 is often used for Pass-Specific Resources (Input/Output Textures).
    *   Group 2+ is for Material/Instance data.
*   **Push Constants**:
    *   128 bytes guaranteed. Used for frequent updates like `Time`, `Resolution`, `JFA Step`.

## 4. Coordinate Systems
*   **NDC**:
    *   Vulkan: Y-down, [-1, 1] Z [0, 1].
    *   Texture Space: [0, 0] is Top-Left.
*   **UVs**:
    *   [0, 1] across the texture.
    *   SDF calculation happens in *Pixel Space* then normalized to UV/World units.
