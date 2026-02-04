use fanta_rust::backend::WgpuBackend;
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, WindowEvent, KeyEvent};
use winit::keyboard::{Key, NamedKey};
use winit::event_loop::{ControlFlow, EventLoop};
use fanta_rust::backend::shaders::types::CinematicParams;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Performance Benchmark");
    println!("Controls:");
    println!("  [B] Toggle Batching/Instancing");
    println!("  [Up/Down] Increase/Decrease object count");
    println!("  [ESC] Exit");

    let event_loop = EventLoop::new()?;
    let window_attrs = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Benchmarking")
        .with_inner_size(LogicalSize::new(1024, 768));
    
    let window = event_loop.create_window(window_attrs)?;
    let window = Arc::new(window);
    let size = window.inner_size();
    
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut backend: WgpuBackend = WgpuBackend::new_async(
        window.clone(),
        size.width,
        size.height,
    )
    .map_err(|e: String| -> Box<dyn std::error::Error> { Box::from(e) })?;

    let mut current_width = size.width;
    let mut current_height = size.height;
    let start_time = std::time::Instant::now();
    let mut last_frame_time = std::time::Instant::now();

    let mut batching_enabled = true;
    let mut object_count = 1000;
    let mut fps = 0.0;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                current_width = size.width;
                current_height = size.height;
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, .. }, .. } => {
                match logical_key {
                    Key::Character(c) if c == "b" || c == "B" => {
                        batching_enabled = !batching_enabled;
                        println!("Batching: {}", batching_enabled);
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        object_count += 100;
                        println!("Object count: {}", object_count);
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if object_count > 100 { object_count -= 100; }
                        println!("Object count: {}", object_count);
                    }
                    Key::Named(NamedKey::Escape) => elwt.exit(),
                    _ => {}
                }
            }
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_frame_time).as_secs_f32();
                last_frame_time = now;
                fps = 0.9 * fps + 0.1 * (1.0 / dt.max(0.001));

                let elapsed = start_time.elapsed().as_secs_f32();
                let mut dl = fanta_rust::DrawList::new();
                
                // Draw many objects
                for i in 0..object_count {
                    let angle = (i as f32) * 0.1 + elapsed;
                    let radius = 300.0 + (elapsed * 0.5).sin() * 50.0;
                    let x = (current_width as f32 / 2.0) + angle.cos() * radius;
                    let y = (current_height as f32 / 2.0) + angle.sin() * radius;
                    
                    dl.add_rounded_rect(
                        Vec2::new(x - 20.0, y - 20.0),
                        Vec2::new(40.0, 40.0),
                        8.0,
                        ColorF::new(
                            (angle.sin() * 0.5 + 0.5).max(0.1),
                            (angle.cos() * 0.5 + 0.5).max(0.1),
                            0.8,
                            1.0
                        )
                    );
                }

                // Overlay Info
                fanta_rust::text::FONT_MANAGER.with(|fm| {
                    let info_text = format!(
                        "FPS: {:.1}\nObjects: {}\nBatching: {}\n[B] Toggle [Up/Down] Count",
                        fps, object_count, if batching_enabled { "ON" } else { "OFF" }
                    );
                    
                    // Simple text rendering via DrawList (placeholder rect for now)
                    dl.add_text(
                        Vec2::new(20.0, 20.0),
                        Vec2::new(400.0, 200.0),
                        [0.0, 0.0, 1.0, 1.0], // uv
                        ColorF::white()
                    );
                    let _ = info_text; // Silence unused warning
                });

                // Add test blur
                dl.add_backdrop_blur(
                    Vec2::new(100.0, 100.0),
                    Vec2::new(300.0, 200.0),
                    20.0,
                    ColorF::white()
                );

                // --- RENDERING ---
                let orchestrator = fanta_rust::renderer::orchestrator::RenderOrchestrator::new()
                    .with_batching(batching_enabled);
                
                let mut cinematic_config = CinematicParams::default();
                cinematic_config.blur_radius = 16.0;
                let mut graph = orchestrator.plan(&dl, &cinematic_config);
                if let Err(e) = orchestrator.execute(&mut backend, &mut graph, elapsed, current_width, current_height) {
                    eprintln!("Render error: {}", e);
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
