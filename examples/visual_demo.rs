use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - visual_demo");
    println!("Visual Revolution Demo");
    println!("Controls:");
    println!("  Mouse: Move Light Source");
    println!("  Space: Toggle Bloom (None -> Soft -> Cinematic)");
    println!("  T: Toggle Tone Mapping (None -> Aces -> Reinhard)");
    println!("  D: Toggle Debug Mode (None -> Velocity -> Normals)");
    println!("  G: Toggle Film Grain");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Visual Demo")
        .with_inner_size(LogicalSize::new(1280, 800));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1280, 800, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    // Initial Configuration
    let mut config = EngineConfig::cinematic();
    config.cinematic.light_pos = [640.0, 400.0];
    config.cinematic.light_color = [1.0, 0.8, 0.6, 2.0]; // Bright warm light
    config.cinematic.bloom = fanta_rust::config::Bloom::Soft;
    config.cinematic.tonemap = fanta_rust::config::Tonemap::Aces;
    config.cinematic.grain_strength = 0.015; // Subtle grain
    
    backend.set_cinematic_config(config.cinematic);

    let mut cursor_pos = Vec2::new(640.0, 400.0);
    let mut time = 0.0f32;
    let start_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                cursor_pos = Vec2::new(position.x as f32, position.y as f32);
                
                // Update light position
                config.cinematic.light_pos = [cursor_pos.x, cursor_pos.y];
                backend.set_cinematic_config(config.cinematic);
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event: key_event, .. }, .. } => {
                if key_event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(keycode) = key_event.physical_key {
                        match keycode {
                            KeyCode::Space => {
                                config.cinematic.bloom = match config.cinematic.bloom {
                                    fanta_rust::config::Bloom::None => fanta_rust::config::Bloom::Soft,
                                    fanta_rust::config::Bloom::Soft => fanta_rust::config::Bloom::Cinematic,
                                    fanta_rust::config::Bloom::Cinematic => fanta_rust::config::Bloom::None,
                                };
                                println!("Bloom: {:?}", config.cinematic.bloom);
                                backend.set_cinematic_config(config.cinematic);
                            }
                            KeyCode::KeyT => {
                                config.cinematic.tonemap = match config.cinematic.tonemap {
                                    fanta_rust::config::Tonemap::None => fanta_rust::config::Tonemap::Aces,
                                    fanta_rust::config::Tonemap::Aces => fanta_rust::config::Tonemap::Reinhard,
                                    fanta_rust::config::Tonemap::Reinhard => fanta_rust::config::Tonemap::None,
                                };
                                println!("Tonemap: {:?}", config.cinematic.tonemap);
                                backend.set_cinematic_config(config.cinematic);
                            }
                            KeyCode::KeyD => {
                                config.cinematic.debug_mode = (config.cinematic.debug_mode + 1) % 4;
                                println!("Debug Mode: {}", config.cinematic.debug_mode);
                                backend.set_cinematic_config(config.cinematic);
                            }
                            KeyCode::KeyG => {
                                config.cinematic.grain_strength = if config.cinematic.grain_strength > 0.0 { 0.0 } else { 0.02 };
                                println!("Grain: {}", config.cinematic.grain_strength);
                                backend.set_cinematic_config(config.cinematic);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                time = start_time.elapsed().as_secs_f32();

                // Build Frame
                // Build Frame
                // Wait, use Renderer::begin_frame usually. But we don't have Renderer here.
                // We'll use FrameContext::new directly if possible, or construct DrawList manually.
                // Let's use FrameContext wrapper from lib.rs if exposed properly.
                // In lib.rs: `pub use crate::renderer::api::FrameContext;` isn't there?
                // `pub use crate::renderer::Renderer;` is there.
                
                // Actually `use fanta_rust::prelude::*` might not have FrameContext.
                // Let's use DrawList directly for simplicity since we don't need FrameContext abstraction here.
                
                let mut dl = fanta_rust::DrawList::new();
                
                // Background
                dl.add_rect(Vec2::ZERO, Vec2::new(1280.0, 800.0), ColorF::new(0.02, 0.02, 0.05, 1.0));

                // 1. Emissive Neon Circle (Bloom Test)
                dl.add_rect_ex(
                    Vec2::new(200.0, 200.0),
                    Vec2::new(100.0, 100.0),
                    [50.0, 50.0, 50.0, 50.0], // Circle
                    ColorF::new(0.0, 1.0, 1.0, 1.0), // Cyan
                    0.0, // elevation
                    true, // squircle
                    0.0, ColorF::transparent(), // border
                    Vec2::ZERO,
                    20.0, ColorF::new(0.0, 1.0, 1.0, 0.5), // glow
                    None, None,
                    Vec2::ZERO,
                    0.5, 0.2, // Reflectivity, Roughness
                    None, 0.0, 
                    3.0, // Emissive Intensity (High!) -> Should bloom
                    0.0
                );

                // 2. Shiny Metal Plate (Lighting Test)
                // Needs Normal Map? Or just roughness 0.0.
                dl.add_rect_ex(
                    Vec2::new(500.0, 400.0),
                    Vec2::new(200.0, 200.0),
                    [20.0, 20.0, 20.0, 20.0],
                    ColorF::new(0.8, 0.8, 0.8, 1.0), // Grey
                    10.0, // elevation
                    false, 
                    2.0, ColorF::white(), // border
                    Vec2::ZERO,
                    0.0, ColorF::transparent(),
                    None, None,
                    Vec2::ZERO,
                    0.9, 0.1, // High Reflectivity, Low Roughness
                    None, 0.0, 
                    0.0, // Non-emissive
                    0.0
                );

                // 3. Rough Matte Box (Lighting Test)
                dl.add_rect_ex(
                    Vec2::new(800.0, 200.0),
                    Vec2::new(150.0, 150.0),
                    [0.0, 0.0, 0.0, 0.0],
                    ColorF::new(1.0, 0.2, 0.2, 1.0), // Red
                    2.0,
                    false,
                    0.0, ColorF::transparent(),
                    Vec2::ZERO,
                    0.0, ColorF::transparent(),
                    None, None,
                    Vec2::ZERO,
                    0.1, 0.9, // Low Reflectivity, High Roughness
                    None, 0.0, 
                    0.0, 
                    0.0
                );

                // 4. Moving Emissive Object
                let move_x = 640.0 + (time * 2.0).sin() * 300.0;
                dl.add_rect_ex(
                    Vec2::new(move_x, 650.0),
                    Vec2::new(80.0, 80.0),
                    [40.0, 40.0, 40.0, 40.0],
                    ColorF::new(1.0, 0.0, 1.0, 1.0), // Magenta
                    0.0, true,
                    0.0, ColorF::transparent(),
                    Vec2::ZERO,
                    30.0, ColorF::new(1.0, 0.0, 1.0, 0.6),
                    None, None,
                    Vec2::ZERO,
                    0.5, 0.5,
                    None, 0.0,
                    5.0, // Very Bright -> Bloom
                    0.0
                );
                
                // Draw Light Source Indicator
                dl.add_circle(cursor_pos, 10.0, ColorF::new(1.0, 1.0, 0.8, 1.0), true);

                backend.render(&dl, 1280, 800);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
