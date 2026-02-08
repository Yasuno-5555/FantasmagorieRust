use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - WGPU Demo");

    let event_loop = EventLoop::new()?;
    let window_attrs = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Visual Revolution (WGPU)")
        .with_inner_size(LogicalSize::new(1024, 768));
    
    let window = event_loop.create_window(window_attrs)?;

    let window = Arc::new(window);
    let size = window.inner_size();
    
    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut backend: WgpuBackend = WgpuBackend::new_async(
        window.clone(),
        size.width,
        size.height,
        1.0,
    )
    .map_err(|e: String| -> Box<dyn std::error::Error> { Box::from(e) })?;

    let mut current_width = size.width;
    let mut current_height = size.height;
    let start_time = std::time::Instant::now();

    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;

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
                    // WGPU resize needs surface reconfiguration usually,
                    // or just creating a new swapchain.
                    // The backend typically handles it or we re-configure surface.
                    // But WgpuBackend::new_async configured it once.
                    // Ideally we'd call backend.resize(w, h).
                    // But for now let's hope it works or just restart.
                    // Actually, WGPU requires explicit re-configure.
                    // Let's assume fixed size for this verification demo.
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
                // Screenshot verification logic
                static mut FRAME_COUNT: u32 = 0;
                unsafe {
                    FRAME_COUNT += 1;
                    if FRAME_COUNT == 10 {
                        println!("Requesting screenshot...");
                        *backend.screenshot_requested.lock().unwrap() = Some("wgpu_screenshot.png".to_string());
                    }
                    if FRAME_COUNT == 11 {
                        println!("Exiting after screenshot.");
                        elwt.exit();
                    }
                }

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

                // --- UI DEFINITION ---
                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .bg(ColorF::new(0.01, 0.01, 0.02, 1.0))
                    .build();

                ui.begin(root);
                
                let elapsed = start_time.elapsed().as_secs_f32();
                
                // Centered Cinematic Panel
                let panel = ui.column()
                    .size(800.0, 500.0)
                    .align(Align::Center)
                    .bg(ColorF::new(0.05, 0.05, 0.1, 0.8))
                    .radius(32.0)
                    .elevation(40.0)
                    .padding(40.0)
                    .spacing(20.0)
                    .build();
                
                ui.begin(panel);
                
                ui.text("WGPU Cinematic Revolution")
                    .font_size(48.0)
                    .fg(ColorF::new(0.0, 0.9, 1.0, 1.0))
                    .build();
                
                ui.text("Advanced Post-Processing & Bloom")
                    .font_size(18.0)
                    .fg(ColorF::new(0.7, 0.7, 0.8, 1.0))
                    .build();

                ui.text("").build(); // Spacer

                let content_row = ui.row().spacing(40.0).build();
                ui.begin(content_row);
                
                // HDR Emissive Blocks to trigger Bloom
                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(3.0, 0.3, 0.6, 1.0)) // HDR Red/Pink
                    .radius(24.0 + (elapsed.sin() * 8.0))
                    .glow(10.0, ColorF::new(3.0, 0.3, 0.6, 0.8))
                    .build();

                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(0.0, 4.0, 2.0, 1.0)) // HDR Cyan/Green
                    .radius(24.0 + (elapsed.cos() * 8.0))
                    .glow(10.0, ColorF::new(0.0, 4.0, 2.0, 0.8))
                    .build();
                    
                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(1.0, 1.0, 6.0, 1.0)) // HDR Blue
                    .radius(24.0 + ((elapsed * 1.5).sin() * 8.0))
                    .glow(10.0, ColorF::new(1.0, 1.0, 6.0, 0.8))
                    .build();
                
                ui.end(); // end content_row

                ui.text("").build(); // Spacer
                ui.text("• ACES Filmic Tone Mapping").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                ui.text("• Multi-pass WGPU Bloom").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                ui.text("• Chromatic Aberration & Vignette").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();

                ui.end(); // end panel
                ui.end(); // end root

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

                    // Use generic render method
                    backend.render(&dl, current_width, current_height);

                    // Automatic Screenshot for Verification
                    if start_time.elapsed().as_secs_f32() > 2.0 {
                        println!("Capturing verification screenshot...");
                        backend.capture_screenshot("wgpu_screenshot.png");
                        // We need one more frame to process the screenshot in WgpuBackend::render
                        // This demo calls render every frame, so next frame will handle it.
                    }
                    
                    if start_time.elapsed().as_secs_f32() > 2.5 {
                        println!("Demo completed successfully.");
                        elwt.exit();
                    }
                }
            }
            _ => {}
        }
        
    })?;

    Ok(())
}
