
use fanta_rust::backend::WgpuBackend;
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Cinema Playground");
    println!("Operations Mode: Change parameters at runtime");
    println!("Controls:");
    println!("  1: Bloom Soft      2: Bloom Cinematic   3: Bloom None");
    println!("  4: Tonemap ACES    5: Tonemap Reinhard  6: Tonemap None
  7: Increase Grain  8: Decrease Grain
  Up/Down: Exposure  Left/Right: CA Strength
  * Shader Hot-Reloading is ACTIVE. Modify src/backend/shaders/*.wgsl to see changes instantly!
");

    let event_loop = EventLoop::new()?;
    let window_attrs = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie: Cinema Playground")
        .with_inner_size(LogicalSize::new(1280, 720));
    
    let window = event_loop.create_window(window_attrs)?;
    let window = Arc::new(window);

    let size = window.inner_size();
    let backend = WgpuBackend::new_async(
        window.clone(),
        size.width,
        size.height,
    ).map_err(|e| format!("WGPU creation failed: {}", e))?;

    // --- Renderer is the USPS (Carrier) ---
    // Under V5 architecture, the Renderer is a simple boundary that carries 
    // high-level intent to the backend "Muscle".
    let mut renderer = Renderer::new(Box::new(backend), EngineConfig::cinematic());

    let mut current_width = size.width;
    let mut current_height = size.height;
    
    // Initial operational state
    use fanta_rust::config::{Bloom, Tonemap, CinematicConfig};
    let mut cinematic = CinematicConfig::default();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width > 0 && size.height > 0 {
                    current_width = size.width;
                    current_height = size.height;
                }
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event: key_event, .. }, .. } => {
                if key_event.state == ElementState::Pressed {
                    match key_event.logical_key {
                        Key::Character(s) if s == "1" => { cinematic.bloom = Bloom::Soft; println!("Bloom: Soft"); }
                        Key::Character(s) if s == "2" => { cinematic.bloom = Bloom::Cinematic; println!("Bloom: Cinematic"); }
                        Key::Character(s) if s == "3" => { cinematic.bloom = Bloom::None; println!("Bloom: None"); }
                        Key::Character(s) if s == "4" => { cinematic.tonemap = Tonemap::Aces; println!("Tonemap: ACES"); }
                        Key::Character(s) if s == "5" => { cinematic.tonemap = Tonemap::Reinhard; println!("Tonemap: Reinhard"); }
                        Key::Character(s) if s == "6" => { cinematic.tonemap = Tonemap::None; println!("Tonemap: None"); }
                        Key::Character(s) if s == "7" => { cinematic.grain_strength += 0.01; println!("Grain: {:.2}", cinematic.grain_strength); }
                        Key::Character(s) if s == "8" => { cinematic.grain_strength = (cinematic.grain_strength - 0.01).max(0.0); println!("Grain: {:.2}", cinematic.grain_strength); }
                        Key::Character(s) if s == "9" => { cinematic.lut_intensity += 0.01; println!("LUT Intensity: {:.2}", cinematic.lut_intensity); }
                        Key::Character(s) if s == "0" => { cinematic.lut_intensity = (cinematic.lut_intensity - 0.01).max(0.0); println!("LUT Intensity: {:.2}", cinematic.lut_intensity); }
                        Key::Named(NamedKey::ArrowUp) => { cinematic.exposure += 0.1; println!("Exposure: {:.2}", cinematic.exposure); }
                        Key::Named(NamedKey::ArrowDown) => { cinematic.exposure = (cinematic.exposure - 0.1).max(0.0); println!("Exposure: {:.2}", cinematic.exposure); }
                        Key::Named(NamedKey::ArrowRight) => { cinematic.chromatic_aberration += 0.0005; println!("CA: {:.4}", cinematic.chromatic_aberration); }
                        Key::Named(NamedKey::ArrowLeft) => { cinematic.chromatic_aberration = (cinematic.chromatic_aberration - 0.0005).max(0.0); println!("CA: {:.4}", cinematic.chromatic_aberration); }
                        Key::Named(NamedKey::Escape) => elwt.exit(),
                        _ => {}
                    }
                    // "Think once, then carry" - Update the backend via the carrier
                    renderer.update_cinematic(cinematic);
                }
            }
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // Simple layout for playground
                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .aurora()
                    .align(Align::Center)
                    .build();
                ui.begin(root);

                let panel = ui.column()
                    .size(450.0, 400.0)
                    .bg(ColorF::new(0.0, 0.1, 0.2, 0.5))
                    .backdrop_blur(40.0)
                    .radius(24.0)
                    .padding(40.0)
                    .build();
                ui.begin(panel);

                ui.text("CINEMA OPERATOR")
                    .font_size(28.0)
                    .fg(ColorF::new(0.0, 0.9, 1.0, 1.0))
                    .build();
                
                ui.column().height(30.0).build(); // Spacer

                ui.text(ui.arena.alloc_str(&format!("BLOOM: {:?}", cinematic.bloom))).fg(ColorF::white()).build();
                ui.text(ui.arena.alloc_str(&format!("TONEMAP: {:?}", cinematic.tonemap))).fg(ColorF::white()).build();
                ui.text(ui.arena.alloc_str(&format!("EXPOSURE: {:.2}", cinematic.exposure))).fg(ColorF::white()).build();
                ui.text(ui.arena.alloc_str(&format!("CA STRENGTH: {:.4}", cinematic.chromatic_aberration))).fg(ColorF::white()).build();
                ui.text(ui.arena.alloc_str(&format!("GRAIN: {:.2}", cinematic.grain_strength))).fg(ColorF::white()).build();
                ui.text(ui.arena.alloc_str(&format!("LUT INTENSITY: {:.2}", cinematic.lut_intensity))).fg(ColorF::white()).build();
                
                ui.column().height(30.0).build(); // Spacer
                ui.text("Use 1-6 and Arrows to adjust")
                    .font_size(14.0)
                    .fg(ColorF::new(1.0, 1.0, 1.0, 0.5))
                    .build();

                ui.end(); // panel
                ui.end(); // root

                if let Some(root_node) = ui.root() {
                    let mut dl = DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        if fm.texture_dirty {
                            // Ideally this would be carried too, but font atlas is special.
                            // We can't access backend directly, so we might need a 
                            // method on Renderer to update font.
                            renderer.update_font_texture(fm.atlas.width as u32, fm.atlas.height as u32, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                        fanta_rust::view::render_ui(
                            root_node,
                            current_width as f32,
                            current_height as f32,
                            &mut dl,
                            &mut fm,
                        );
                        // Backend update for font can't be carried easily by mailbox yet, 
                        // so we might need a better abstraction later.
                    });

                    // Renderer is the USPS: Send the letter.
                    renderer.render_list(&dl, current_width, current_height);
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
