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

### Box & Layout
`BoxBuilder` (often used via `ctx.column()` or `ctx.row()`) is the fundamental layout unit. It supports flexbox-like properties.

```rust
ctx.column()
   .padding(20.0)
   .spacing(10.0)
   .bg(ColorF::BLACK)
   .build(|| {
       // Children go here
   });
```

### Text
`TextBuilder` is used for rendering text with subpixel positioning and SDF-based fonts.

```rust
TextBuilder::new("Hello World")
    .font_size(18.0)
    .fg(ColorF::WHITE)
    .build();
```

### Button
`ButtonBuilder` provides built-in interaction handling.

```rust
if ButtonBuilder::new("Click Me")
    .radius(10.0)
    .bg(ColorF::BLUE)
    .hover(ColorF::LIGHT_BLUE)
    .build()
    .clicked() 
{
    // Action...
}
```

## Styling & Effects

Fantasmagorie supports modern visual effects directly in the UI API:

-   **Radius & Squircle:** Smooth rounded corners.
-   **Backdrop Blur:** Glassmorphism effects.
-   **Glow:** Material radiance.
-   **Aurora:** Dynamic gradient effects.

Example:
```rust
ctx.column()
   .backdrop_blur(10.0)
   .border(1.0, ColorF::GRAY)
   .aurora()
   .build(|| { ... });
```
