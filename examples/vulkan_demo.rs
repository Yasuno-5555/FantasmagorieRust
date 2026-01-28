use fanta_rust::backend::VulkanBackend;
use fanta_rust::prelude::*;
use fanta_rust::core::{Vec2, ColorF, FrameArena};
use fanta_rust::widgets::UIContext;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::ffi::c_void;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Vulkan Demo");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie Visual Revolution (Vulkan)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let window = Arc::new(window);

    // Get HWND and HINSTANCE for Vulkan (Windows specific path for this demo)
    #[cfg(target_os = "windows")]
    let (hwnd, hinstance) = match window.raw_window_handle() {
        RawWindowHandle::Win32(handle) => {
            (handle.hwnd as *mut c_void, handle.hinstance as *mut c_void)
        }
        _ => panic!("Not running on Windows/Win32"),
    };

    #[cfg(not(target_os = "windows"))]
    panic!("This verification demo currently targets Windows for Vulkan (due to surface creation logic in example)");

    let size = window.inner_size();

    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    // Initialize Vulkan Backend
    let mut backend = unsafe { VulkanBackend::new(hwnd, hinstance, size.width, size.height)? };

    let mut current_width = size.width;
    let mut current_height = size.height;
    let mut frame_count = 0;
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
                    // Note: Real app would recreate swapchain here
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
                    .aurora() // Background Aurora
                    .build();

                ui.begin(root);
                
                let elapsed = start_time.elapsed().as_secs_f32();
                
                // Centered Glassmorphism Panel
                let panel = ui.column()
                    .size(800.0, 500.0)
                    .align(Align::Center)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .squircle(32.0)
                    .backdrop_blur(15.0)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.2))
                    .elevation(30.0)
                    .padding(40.0)
                    .spacing(20.0)
                    .build();
                
                ui.begin(panel);
                
                ui.text("Fantasmagorie Ultimate Demo")
                    .font_size(48.0)
                    .fg(ColorF::white())
                    .build();
                
                ui.text("Backend: VULKAN")
                    .font_size(18.0)
                    .fg(ColorF::new(0.7, 0.8, 1.0, 1.0))
                    .build();

                let content_row = ui.row().spacing(20.0).build();
                ui.begin(content_row);
                
                // Animated Rotating Box (SDF)
                let orbit = elapsed * 2.0;
                let offset_x = orbit.cos() * 20.0;
                
                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(0.2, 0.2, 0.8, 1.0))
                    .radius(20.0 + (elapsed.sin() * 20.0))
                    .glow(1.0 + elapsed.sin().abs() * 2.0, ColorF::new(0.4, 0.4, 1.0, 1.0))
                    .margin(offset_x) 
                    .build();

                let button_col = ui.column().spacing(10.0).build();
                ui.begin(button_col);
                
                if ui.button("Standard Button").size(200.0, 50.0).radius(10.0).clicked() {
                    println!("Standard Click!");
                }

                if ui.button("Glowing Squircle")
                    .size(200.0, 50.0)
                    .squircle(15.0)
                    .bg(ColorF::new(0.8, 0.2, 0.2, 1.0))
                    .hover(ColorF::new(1.0, 0.3, 0.3, 1.0))
                    .glow(2.0, ColorF::new(0.8, 0.2, 0.2, 0.8))
                    .clicked() {
                    println!("Squircle Click!");
                }

                if ui.button("Transparent Border")
                    .size(200.0, 50.0)
                    .bg(ColorF::transparent())
                    .border(2.0, ColorF::white())
                    .radius(25.0)
                    .clicked() {
                    println!("Border Click!");
                }
                
                ui.end(); // end button_col
                ui.end(); // end content_row

                ui.text("Verification Targets:")
                    .font_size(20.0)
                    .fg(ColorF::new(0.6, 1.0, 0.6, 1.0))
                    .build();
                
                ui.text("- [x] Aurora Pixel Shader Parity")
                    .font_size(14.0).fg(ColorF::white()).build();
                ui.text("- [x] Backdrop Blur (LOD Mipmap) Sampling")
                    .font_size(14.0).fg(ColorF::white()).build();
                ui.text("- [x] Squircle Continuity (SDF)")
                    .font_size(14.0).fg(ColorF::white()).build();
                ui.text("- [x] Real-time Constant Updates")
                    .font_size(14.0).fg(ColorF::white()).build();

                ui.end(); // end panel
                ui.end(); // end root

                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::draw::DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        
                        fanta_rust::view::renderer::render_ui(
                            root,
                            current_width as f32,
                            current_height as f32,
                            &mut dl,
                            &mut fm,
                        );

                        // Sync font texture if dirty (AFTER render_ui so we have new glyphs)
                        if fm.texture_dirty {
                            let w = fm.atlas.width;
                            let h = fm.atlas.height;
                            let data = &fm.atlas.texture_data;
                            backend.update_font_texture(w, h, data);
                            fm.texture_dirty = false;
                        }
                    });
                    
                    use fanta_rust::backend::GraphicsBackend;
                    backend.render(&dl, current_width, current_height);
                    
                    frame_count += 1;
                    if frame_count == 100 {
                        backend.capture_screenshot("vulkan_screenshot.png");
                        println!("✨ Screenshot captured for visual verification!");
                    }
                    if frame_count > 100 {
                        println!("Demo completed, exiting...");
                        elwt.exit();
                    }
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
