
use fanta_rust::backend::{GraphicsBackend, MetalBackend};
use fanta_rust::prelude::*;
use fanta_rust::view::header::Align;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use raw_window_handle::RawWindowHandle;
use cocoa::base::id;
use objc::{msg_send, sel, sel_impl};
use metal::foreign_types::ForeignType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Metal Cinematic Demo");
    println!("Showcasing: Aurora, Glassmorphism, and Native Metal Performance");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie: Metal Cinematic Revolution")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let window = Arc::new(window);
    
    // Metal Backend Setup
    let mut backend = MetalBackend::new().unwrap();
    
    // CAMetalLayer setup
    unsafe {
        use raw_window_handle::HasRawWindowHandle;
        if let RawWindowHandle::AppKit(handle) = window.raw_window_handle() {
            let view = handle.ns_view as id;
            let layer: id = msg_send![objc::class!(CAMetalLayer), new];
            let device_ptr = backend.device.as_ptr();
            let _: () = msg_send![layer, setDevice: device_ptr];
            let _: () = msg_send![layer, setFramebufferOnly: cocoa::base::NO];
            let _: () = msg_send![view, setLayer: layer];
            let _: () = msg_send![view, setWantsLayer: cocoa::base::YES];
            
            backend.set_layer(layer as *mut _);
        }
    }

    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut current_width = 1280;
    let mut current_height = 720;
    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;
    let mut frame_count = 0;
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
                let time = start_time.elapsed().as_secs_f32();
                
                // Simulate Audio Reactivity
                let bass = (time * 2.0).sin() * 0.5 + 0.5;
                let mid = (time * 3.5).cos() * 0.3 + 0.3;
                let spectrum = vec![bass, mid, 0.5, 0.0];
                backend.update_audio_data(&spectrum);

                fanta_rust::view::interaction::update_input(
                    cursor_x,
                    cursor_y,
                    mouse_pressed,
                    false,
                    false,
                );

                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // --- Cinematic UI Tree ---
                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .aurora() 
                    .align(Align::Center)
                    .build();

                ui.begin(root);

                // Glass Header
                let header = ui.column()
                    .size(800.0, 100.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .backdrop_blur(30.0)
                    .squircle(30.0)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.2))
                    .align(Align::Center)
                    .margin(40.0)
                    .padding(20.0)
                    .build();
                
                ui.begin(header);
                ui.text("METAL CINEMATIC REVOLUTION").font_size(42.0).fg(ColorF::white()).build();
                ui.text("High-Performance SDS Rendering on macOS").font_size(18.0).fg(ColorF::new(0.8, 0.8, 0.9, 1.0)).build();
                ui.end();

                // Main Content
                let content = ui.row().size(1100.0, 450.0).align(Align::Center).build();
                ui.begin(content);

                // Left Panel
                let left = ui.column()
                    .size(320.0, 400.0)
                    .bg(ColorF::new(0.0, 0.05, 0.1, 0.4))
                    .backdrop_blur(25.0)
                    .radius(20.0)
                    .padding(30.0)
                    .margin(15.0)
                    .build();
                ui.begin(left);
                ui.text("STATISTICS").font_size(18.0).fg(ColorF::new(0.0, 1.0, 0.8, 1.0)).build();
                ui.text(arena.alloc_str(&format!("TIME: {:.1}s", time))).font_size(14.0).fg(ColorF::white()).build();
                ui.text(arena.alloc_str(&format!("FPS: 60 (VSYNC)"))).font_size(14.0).fg(ColorF::white()).build();
                ui.text(arena.alloc_str(&format!("Backend: Metal 3.0"))).font_size(14.0).fg(ColorF::white()).build();
                
                // Audio Bars
                let bars = ui.row().size(260.0, 60.0).margin(20.0).build();
                ui.begin(bars);
                for i in 0..6 {
                    let h = 10.0 + (time * (1.5 + i as f32 * 0.5)).sin().abs() * 40.0;
                    ui.column().size(25.0, h).bg(ColorF::new(0.2, 0.6, 1.0, 0.8)).radius(4.0).margin(4.0).build();
                }
                ui.end();
                ui.end();

                // Center Orb
                let center_orb = ui.column().size(400.0, 400.0).align(Align::Center).build();
                ui.begin(center_orb);
                let orb_size = 220.0 + bass * 40.0;
                ui.column()
                    .size(orb_size, orb_size)
                    .bg(ColorF::new(1.0, 0.1, 0.4, 0.3))
                    .squircle(orb_size / 2.0)
                    .glow(1.5 + mid * 2.0, ColorF::new(1.0, 0.1, 0.4, 0.6))
                    .elevation(20.0)
                    .build();
                ui.end();

                // Right Panel
                let right = ui.column()
                    .size(320.0, 400.0)
                    .bg(ColorF::new(0.1, 0.0, 0.1, 0.4))
                    .backdrop_blur(25.0)
                    .radius(20.0)
                    .padding(30.0)
                    .margin(15.0)
                    .build();
                ui.begin(right);
                ui.text("ACTIVE MODULES").font_size(18.0).fg(ColorF::new(1.0, 0.8, 0.0, 1.0)).build();
                let modules = ["MSL Shaders", "SDF Engine", "Aurora Mesh", "JFA Kernels", "Linear Workflow"];
                for m in modules {
                   ui.text(arena.alloc_str(&format!("∙ {}", m))).font_size(14.0).fg(ColorF::white()).build();
                }
                ui.end();

                ui.end(); // content

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

                    backend.render(&dl, current_width, current_height);
                }

                // Interactive mode enabled - app stays open
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event: keyboard_event, .. }, .. } => {
                if keyboard_event.state == ElementState::Pressed {
                    if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyS) = keyboard_event.physical_key {
                        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                        let path = format!("/Users/yasuno/.gemini/antigravity/brain/dd0d6b5a-0a0a-47b3-8e4d-b9ea47634588/screenshot_{}.png", timestamp);
                        backend.capture_screenshot(&path);
                        println!("Manual screenshot requested: {}", path);
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
