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

                // UI Definition
                let root = ui
                    .column()
                    .size(current_width as f32, current_height as f32)
                    .padding(40.0)
                    .build();

                ui.begin(root);

                ui.text("Fantasmagorie V5: Visual Revolution (Vulkan)")
                    .font_size(32.0)
                    .fg(ColorF::white())
                    .layout_margin(10.0);

                let panel = ui
                    .column()
                    .size(500.0, 400.0)
                    .aurora() // Use the new Aurora mesh gradient
                    .bg(ColorF::new(0.0, 0.0, 0.2, 0.5))
                    .squircle(24.0)
                    .backdrop_blur(20.0) // Test glassmorphism (handled in renderer -> BlurRect)
                    .border(2.0, ColorF::new(0.5, 0.8, 1.0, 0.5))
                    .elevation(20.0)
                    .padding(20.0)
                    .build();

                ui.begin(panel);

                ui.text("Vulkan Backend Verification")
                    .font_size(24.0)
                    .fg(ColorF::white());

                ui.text("GLSL -> SPIR-V (Runtime) via Naga.")
                    .font_size(16.0)
                    .fg(ColorF::new(0.8, 0.8, 0.8, 1.0))
                    .layout_margin(10.0);

                let btn1 = ui
                    .button("Vulkan Glow")
                    .size(200.0, 50.0)
                    .bg(ColorF::new(0.4, 1.0, 0.4, 1.0))
                    .radius(12.0)
                    .glow(1.5, ColorF::new(0.4, 1.0, 0.4, 1.0))
                    .build();

                ui.end();
                ui.end();

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
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
