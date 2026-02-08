use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::view::header::Align;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::raw_window_handle::HasWindowHandle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - WGPU Demo");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Visual Revolution (WGPU)")
        .with_inner_size(LogicalSize::new(1024, 768));
    
    let window = event_loop.create_window(window)?;

    let window = Arc::new(window);
    
    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut backend = WgpuBackend::new_async(window.clone(), 1024, 768, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut current_width = 1024;
    let mut current_height = 768;
    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;
    let start_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                current_width = size.width;
                current_height = size.height;
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                cursor_x = position.x as f32;
                cursor_y = position.y as f32;
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                if button == MouseButton::Left {
                    mouse_pressed = state == ElementState::Pressed;
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // Screenshot verification logic
                static mut FRAME_COUNT: u32 = 0;

                fanta_rust::view::interaction::update_input(
                    cursor_x,
                    cursor_y,
                    mouse_pressed,
                    false,
                    false,
                );

                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // --- UI DEFINITION ---
                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .aurora() // Premium Aurora background
                    .align(Align::Center)
                    .build();

                ui.begin(root);
                
                let elapsed = start_time.elapsed().as_secs_f32();
                
                // Centered Cinematic Panel with Glassmorphism
                let panel = ui.column()
                    .size(800.0, 500.0)
                    .align(Align::Center)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05)) // White with low alpha
                    .backdrop_blur(40.0)                 // Strong glass effect
                    .radius(32.0)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.2)) // Subtle border
                    .elevation(40.0)
                    .padding(40.0)
                    .spacing(20.0)
                    .build();
                
                ui.begin(panel);
                
                ui.text("WGPU Render Revolution")
                    .font_size(48.0)
                    .fg(ColorF::new(0.0, 0.9, 1.0, 1.0))
                    .build();
                
                ui.text("Full HDR Pipeline & Hybrid SSR")
                    .font_size(18.0)
                    .fg(ColorF::new(0.7, 0.7, 0.8, 1.0))
                    .build();

                ui.text("").build(); // Spacer

                let content_row = ui.row().spacing(40.0).build();
                ui.begin(content_row);
                
                // HDR Emissive Blocks to trigger Bloom
                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(4.0, 0.4, 0.8, 1.0)) // HDR Pink
                    .radius(24.0 + (elapsed.sin() * 8.0))
                    .glow(15.0, ColorF::new(4.0, 0.4, 0.8, 0.8))
                    .build();

                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(0.0, 5.0, 2.5, 1.0)) // HDR Cyan
                    .radius(24.0 + (elapsed.cos() * 8.0))
                    .glow(15.0, ColorF::new(0.0, 5.0, 2.5, 0.8))
                    .build();
                    
                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(1.5, 1.5, 8.0, 1.0)) // HDR Electric Blue
                    .radius(24.0 + ((elapsed * 1.5).sin() * 8.0))
                    .glow(15.0, ColorF::new(1.5, 1.5, 8.0, 0.8))
                    .build();
                
                ui.end(); // content_row

                ui.text("").build(); // Spacer
                ui.text("• Cross-platform WGPU Performance").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                ui.text("• Multi-pass HDR Bloom").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                ui.text("• Interactive Glassmorphism").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();

                ui.end(); // panel
                ui.end(); // root
                
                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::render_ui(root, current_width as f32, current_height as f32, &mut dl, &mut fm);
                        if fm.texture_dirty {
                            backend.update_font_texture(fm.atlas.width as u32, fm.atlas.height as u32, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                    });

                    // Use generic render method
                    backend.render(&dl, current_width, current_height);

                    // Automatic Screenshot for Verification
                    if start_time.elapsed().as_secs_f32() > 2.0 {
                        unsafe {
                            if FRAME_COUNT == 0 {
                                // backend.capture_screenshot("wgpu_screenshot.png");
                                // FRAME_COUNT = 1;
                            }
                        }
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
