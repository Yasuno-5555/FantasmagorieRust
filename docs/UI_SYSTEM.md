# UI System: Immediate Mode Fluent API

Fantasmagorie features a powerful Immediate Mode UI (IMGUI) system with a focus on aesthetics and ease of use.

## Core Concepts

The UI is built every frame. It doesn't store state between frames; instead, it uses a **Persistence Manager** for things like scroll positions and focus.

### The UIContext

All UI building starts with the `UIContext`. It manages the current frame arena and layout stack.

```rust
let mut frame = renderer.begin_frame();
let mut ctx = UIContext::new(&mut frame);
```

## Widgets

Widgets are created using builders. Every builder provides a fluent API for styling.

### Layout (Box, Row, Column)
`BoxBuilder` is the fundamental layout unit. It supports flexbox-like properties. You can nest layouts using `ctx.begin()` and `ctx.end()`.

```rust
// Create a parent column
let col = ctx.column()
    .padding(20.0)
    .spacing(10.0)
    .bg(ColorF::BLACK)
    .build();

// Begin context for children
ctx.begin(col);

    // Add children
    ctx.text("Title").font_size(24.0).build();
    
    let row = ctx.row().spacing(5.0).build();
    ctx.begin(row);
        ctx.button("OK").build();
        ctx.button("Cancel").build();
    ctx.end(); // End row

ctx.end(); // End column
```

### Text
`TextBuilder` is used for rendering text with subpixel positioning and SDF-based fonts.

```rust
ctx.text("Hello World")
    .font_size(18.0)
    .fg(ColorF::WHITE)
    .build();
```

### Button
`ButtonBuilder` provides built-in interaction handling.

```rust
if ctx.button("Click Me")
    .radius(10.0)
    .bg(ColorF::BLUE)
    .hover(ColorF::LIGHT_BLUE)
    .build()
    .clicked() 
{
    // Action...
    play_sound();
}
```

## Rendering Backend: The SDF Revolution

All UI elements in Fantasmagorie are rendered using **Signed Distance Fields (SDF)**. This allows for:
-   **Infinite Resolution:** Corners and text stay sharp regardless of zoom or scale.
-   **Low Overhead:** Complex shapes (Rounded Boxes, Squircles, Arcs) are calculated in the fragment shader.
-   **Visual Effects:** Features like `backdrop_blur` (Glassmorphism), `glow`, and `aurora` (Dynamic Procedural Gradients) are native to the SDF pipeline.

### WGPU Integration

The UI system is optimized for the **WGPU** backend, taking advantage of modern GPU features like `Storage Textures` for backdrop capture and `Linear-to-sRGB` manual gamma correction for consistent high-dynamic-range (HDR) colors.

```rust
let glass = ctx.column()
   .backdrop_blur(10.0) // Glassmorphism
   .border(1.0, ColorF::GRAY)
   .aurora()            // Dynamic procedural noise
   .build();

ctx.begin(glass);
    // ... content ...
ctx.end();
```
