use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

static mut FRAME_COUNT: u32 = 0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - SSR Glassmorphism Demo");
    println!("Showcasing: Screen-Space Reflections with Glassmorphism");

    let event_loop = EventLoop::new()?;
    let window_attrs = winit::window::WindowAttributes::default()
        .with_title("SSR Glassmorphism Demo")
        .with_inner_size(LogicalSize::new(1000, 800));
    
    let window = event_loop.create_window(window_attrs)?;
    let window = Arc::new(window);

    let size = window.inner_size();
    let mut backend = WgpuBackend::new_async(
        window.clone(),
        size.width,
        size.height,
    )
    .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut current_width = size.width;
    let mut current_height = size.height;
    let start_time = std::time::Instant::now();

    // Initial config
    let mut config = fanta_rust::config::CinematicConfig::default();
    config.bloom = fanta_rust::config::Bloom::Soft;
    config.gi_intensity = 0.8;
    config.volumetric_intensity = 0.5;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width > 0 && size.height > 0 {
                    current_width = size.width;
                    current_height = size.height;
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let time = start_time.elapsed().as_secs_f32();
                
                let mut dl = fanta_rust::DrawList::new();

                // 1. Moving Background (Stripes)
                for i in 0..20 {
                    let off = (i as f32 * 50.0 + time * 50.0) % 1000.0;
                    dl.add_rounded_rect(
                        fanta_rust::core::Vec2::new(off - 100.0, 0.0),
                        fanta_rust::core::Vec2::new(40.0, 800.0),
                        5.0,
                        fanta_rust::core::ColorF::new(0.8, 0.2, 0.2, 1.0),
                    );
                }
                
                // 2. Checkerboard
                for y in 0..10 {
                    for x in 0..10 {
                        if (x + y) % 2 == 0 {
                            dl.add_rounded_rect(
                                fanta_rust::core::Vec2::new(x as f32 * 100.0, y as f32 * 100.0),
                                fanta_rust::core::Vec2::new(100.0, 100.0),
                                0.0,
                                fanta_rust::core::ColorF::new(0.1, 0.1, 0.15, 1.0),
                            );
                        }
                    }
                }
                
                // 3. Glass Panel (Center) - High reflectivity
                dl.add_rounded_rect_ex(
                    fanta_rust::core::Vec2::new(300.0, 200.0),
                    fanta_rust::core::Vec2::new(400.0, 300.0),
                    20.0,
                    fanta_rust::core::ColorF::new(1.0, 1.0, 1.0, 0.1), // Transparent base
                    0.0,
                    false,
                    2.0,
                    fanta_rust::core::ColorF::new(1.0, 1.0, 1.0, 0.5), // Border
                    fanta_rust::core::Vec2::ZERO,
                    0.0,
                    fanta_rust::core::ColorF::transparent(),
                    None,
                    None,
                    fanta_rust::core::Vec2::ZERO,
                    0.9,  // High reflectivity
                    0.0,  // Smooth (low roughness)
                    None, 1.0, 0.0, 0.0, // Added Distortion: 1.0
                );

                // 4. Frosted Glass (Side) - High reflectivity, higher roughness
                dl.add_rounded_rect_ex(
                    fanta_rust::core::Vec2::new(100.0, 100.0),
                    fanta_rust::core::Vec2::new(150.0, 150.0),
                    15.0,
                    fanta_rust::core::ColorF::new(0.5, 0.8, 1.0, 0.2),
                    0.0,
                    false,
                    0.0,
                    fanta_rust::core::ColorF::transparent(),
                    fanta_rust::core::Vec2::ZERO,
                    0.0,
                    fanta_rust::core::ColorF::transparent(),
                    None,
                    None,
                    fanta_rust::core::Vec2::ZERO,
                    0.8,  // Reflective
                    0.5,  // Blurry reflections (roughness)
                    None, 0.0, 0.0, 0.0,
                );
                
                // 5. Moving Orb
                let orb_x = 500.0 + (time * 2.0).sin() * 200.0;
                let orb_y = 350.0 + (time * 3.0).cos() * 100.0;
                dl.add_rounded_rect_ex(
                    fanta_rust::core::Vec2::new(orb_x, orb_y),
                    fanta_rust::core::Vec2::new(80.0, 80.0),
                    40.0, // Circle
                    fanta_rust::core::ColorF::new(0.2, 1.0, 0.5, 1.0), // Bright Green
                    0.0,
                    false,
                    0.0,
                    fanta_rust::core::ColorF::transparent(),
                    fanta_rust::core::Vec2::ZERO,
                    20.0, // Glow strength
                    fanta_rust::core::ColorF::new(0.2, 1.0, 0.5, 1.0), // Glow color
                    None, // texture
                    None, // texture_uv
                    fanta_rust::core::Vec2::ZERO, // texture_scale
                    0.0,  // Reflectivity
                    0.5,  // Roughness
                    None, // normal_map
                    0.0,  // Distortion
                    2.0,  // Emissive
                    0.0,  // Emissive boost
                );

                // Update Cinematic Config
                config.volumetric_intensity = 0.5 + (time * 0.5).sin() * 0.3; // Pulse god rays
                config.gi_intensity = 0.8;
                backend.set_cinematic_config(config);

                backend.render(&dl, current_width, current_height);
                window.request_redraw();

                // Auto-screenshot for debugging
                unsafe {
                    FRAME_COUNT += 1;
                    if FRAME_COUNT % 10 == 0 {
                        println!("DEBUG: Frame {}", FRAME_COUNT);
                    }
                    if FRAME_COUNT == 60 {
                        backend.capture_screenshot("ssr_demo_internal.png");
                        println!("DEBUG: Internal screenshot saved to ssr_demo_internal.png");
                    }
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
