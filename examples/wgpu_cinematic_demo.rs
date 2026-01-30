
use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::view::header::Align;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - WGPU Cinematic Rich Demo");
    println!("Showcasing: HDR, SDF Glow, JFA Blur, Aurora Mesh, and Audio Reactivity");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie: Unified WGPU Cinematic Demo")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let window = Arc::new(window);

    // WGPU Setup
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let surface = unsafe { instance.create_surface(window.clone()) }?;

    let size = window.inner_size();
    let mut backend = pollster::block_on(WgpuBackend::new_async(
        &instance,
        surface,
        size.width,
        size.height,
    ))
    .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut current_width = size.width;
    let mut current_height = size.height;

    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;
    let start_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                if size.width > 0 && size.height > 0 {
                    current_width = size.width;
                    current_height = size.height;
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                cursor_x = position.x as f32;
                cursor_y = position.y as f32;
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { state, button, .. },
                ..
            } => {
                if button == MouseButton::Left {
                    mouse_pressed = state == ElementState::Pressed;
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let time = start_time.elapsed().as_secs_f32();
                
                // --- Simulate Audio Reactivity ---
                let bass = (time * 2.0).sin() * 0.5 + 0.5;
                let mid = (time * 3.5).cos() * 0.3 + 0.3;
                let _high = (time * 7.0).sin() * 0.2 + 0.2;
                let spectrum = vec![bass, mid, _high, 0.0];
                backend.update_audio_data(&spectrum);

                fanta_rust::view::interaction::update_input(
                    cursor_x,
                    cursor_y,
                    mouse_pressed,
                    false,
                    false,
                );
                fanta_rust::view::interaction::begin_interaction_pass();

                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // --- Cinematic Visual Layout ---
                let root = ui
                    .column()
                    .size(current_width as f32, current_height as f32)
                    .aurora() 
                    .align(Align::Center) 
                    .build();

                ui.begin(root);

                // Floating "Glass" Header
                let header = ui.column()
                    .size(800.0, 80.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .backdrop_blur(30.0)
                    .squircle(40.0)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.2))
                    .align(Align::Center)
                    .margin(20.0)
                    .build();
                
                ui.begin(header);
                ui.text("UNIFIED WGPU CINEMATIC STACK")
                    .font_size(36.0)
                    .fg(ColorF::new(1.0, 1.0, 1.0, 0.9))
                    .build();
                ui.end();

                // Main Content Row
                let content_row = ui.row()
                    .size(1000.0, 400.0)
                    .align(Align::Center)
                    .build();

                ui.begin(content_row);
                        
                // Left Panel: Stats
                let left_panel = ui.column()
                    .size(300.0, 350.0)
                    .bg(ColorF::new(0.0, 0.0, 0.0, 0.4))
                    .backdrop_blur(20.0)
                    .radius(20.0)
                    .padding(20.0)
                    .margin(10.0)
                    .build();
                
                ui.begin(left_panel);
                ui.text("SYSTEM STATUS")
                    .font_size(20.0)
                    .fg(ColorF::new(0.0, 1.0, 0.8, 1.0))
                    .build();
                
                ui.text(&format!("TIME: {:.2}s", time))
                    .font_size(14.0)
                    .fg(ColorF::white())
                    .build();
                
                ui.text("BACKEND: WGPU")
                    .font_size(14.0)
                    .fg(ColorF::white())
                    .build();

                // Pulsing Audio Visualizer
                let vis_row = ui.row().size(260.0, 100.0).margin(20.0).build();
                ui.begin(vis_row);
                for i in 0..5 {
                    let h = 20.0 + (time * (2.0 + i as f32)).sin().abs() * 60.0;
                    ui.column()
                        .size(30.0, h)
                        .bg(ColorF::new(0.5, 0.2, 1.0, 0.8))
                        .glow(0.5, ColorF::new(0.5, 0.2, 1.0, 1.0))
                        .radius(5.0)
                        .margin(5.0)
                        .build();
                }
                ui.end();
                ui.end(); // left_panel

                // Center: Interactive Orb
                let center_col = ui.column()
                    .size(350.0, 350.0)
                    .align(Align::Center)
                    .build();
                
                ui.begin(center_col);
                let orb_size = 200.0 + bass * 50.0;
                ui.column()
                    .size(orb_size, orb_size)
                    .bg(ColorF::new(1.0, 0.2, 0.5, 0.3))
                    .squircle(orb_size / 2.0)
                    .glow(1.2 + mid, ColorF::new(1.0, 0.1, 0.4, 1.0))
                    .build();
                ui.end();

                // Right Panel: Feature Checklist
                let right_panel = ui.column()
                    .size(300.0, 350.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .backdrop_blur(40.0)
                    .radius(20.0)
                    .padding(20.0)
                    .margin(10.0)
                    .build();
                
                ui.begin(right_panel);
                ui.text("FEATURES")
                    .font_size(20.0)
                    .fg(ColorF::new(1.0, 0.8, 0.0, 1.0))
                    .build();
                
                let features = ["HDR Pipeline", "SDF Shadows", "Temporal resolve", "JFA Blur", "Audio Reactive"];
                for f in features {
                    let f_row = ui.row().size(260.0, 30.0).build();
                    ui.begin(f_row);
                    ui.text(&format!("✔ {}", f))
                        .font_size(16.0)
                        .fg(ColorF::white())
                        .build();
                    ui.end();
                }
                ui.end(); // right_panel

                ui.end(); // content_row

                // Bottom Footer
                ui.text("Press ESC to exit • Fantasmagorie Engine v5.0")
                    .font_size(14.0)
                    .fg(ColorF::new(0.6, 0.6, 0.6, 1.0))
                    .margin(20.0)
                    .build();

                ui.end(); // root

                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::render_ui(
                            root,
                            current_width as f32,
                            current_height as f32,
                            &mut dl,
                            &mut fm,
                        );
                        if fm.texture_dirty {
                            backend.update_font_texture(fm.atlas.width as u32, fm.atlas.height as u32, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                    });

                    // Render with Unified Backend
                    backend.render(&dl, current_width, current_height);
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
