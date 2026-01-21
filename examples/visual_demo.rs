//! Visual demo example

use fantasmagorie::prelude::*;

fn main() {
    println!("Fantasmagorie Rust - Visual Demo");
    println!("================================");

    // Create arena
    let arena = FrameArena::new();

    // Build UI
    let mut ui = UIContext::new(&arena);

    // Root container
    let root = ui
        .column()
        .size(800.0, 600.0)
        .padding(20.0)
        .bg(ColorF::new(0.1, 0.1, 0.12, 1.0))
        .build();

    ui.begin(root);

    // Header
    ui.text("Fantasmagorie V5 Crystal (Rust)")
        .font_size(24.0)
        .fg(ColorF::white());

    // Button row
    let row = ui.row().padding(10.0).build();
    ui.begin(row);

    let btn1 = ui.button("Click Me").size(120.0, 40.0).radius(8.0).build();

    let btn2 = ui
        .button("Squircle")
        .size(120.0, 40.0)
        .squircle(12.0)
        .bg(ColorF::new(0.2, 0.5, 0.8, 1.0))
        .build();

    ui.end();

    // Info box
    ui.r#box()
        .size(400.0, 100.0)
        .padding(15.0)
        .radius(10.0)
        .elevation(4.0)
        .bg(ColorF::new(0.15, 0.15, 0.18, 1.0));

    ui.end();

    // Get root and compute layout
    if let Some(root) = ui.root() {
        // Create DrawList
        let mut dl = fantasmagorie::DrawList::new();

        // Render UI (computes layout and generates draw commands)
        fantasmagorie::view::render_ui(root, 800.0, 600.0, &mut dl);

        println!("Generated {} draw commands", dl.len());

        // Print some layout info
        let rect = root.rect();
        println!(
            "Root rect: ({}, {}, {}, {})",
            rect.x, rect.y, rect.w, rect.h
        );
    }

    println!("\n[OK] Demo completed successfully!");
    println!("To run with OpenGL window, enable the 'opengl' feature and add window setup.");
}
