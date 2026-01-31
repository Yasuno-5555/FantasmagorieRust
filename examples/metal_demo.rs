use fanta_rust::backend::{MetalBackend, GraphicsBackend};
use metal::foreign_types::ForeignType;
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use raw_window_handle::RawWindowHandle;

use cocoa::base::id;
use objc::{msg_send, sel, sel_impl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Metal Demo");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie Visual Revolution (Metal)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let window = Arc::new(window);
    
    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut backend = MetalBackend::new().map_err(|e| format!("Metal creation failed: {}", e))?;

    // Setup CAMetalLayer
    unsafe {
        use raw_window_handle::HasRawWindowHandle;
        if let RawWindowHandle::AppKit(handle) = window.raw_window_handle() {
            let view = handle.ns_view as id;
            let layer: id = msg_send![objc::class!(CAMetalLayer), new];
            let device_ptr = backend.device.as_ptr();
            let _: () = msg_send![layer, setDevice: device_ptr];
            let _: () = msg_send![view, setLayer: layer];
            let _: () = msg_send![view, setWantsLayer: cocoa::base::YES];
            
            backend.set_layer(layer as *mut _);
        }
    }

    let mut current_width = 1024;
    let mut current_height = 768;
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
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .bg(ColorF::new(0.05, 0.05, 0.07, 1.0))
                    .build();

                ui.begin(root);
                
                let panel = ui.column()
                    .size(600.0, 400.0)
                    .align(Align::Center)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .radius(24.0)
                    .elevation(20.0)
                    .padding(40.0)
                    .build();
                
                ui.begin(panel);
                ui.text("Fantasmagorie Metal Demo").font_size(32.0).fg(ColorF::white()).build();
                ui.text("Native Metal rendering on macOS").font_size(16.0).fg(ColorF::new(0.7, 0.7, 0.8, 1.0)).build();
                
                let elapsed = start_time.elapsed().as_secs_f32();
                ui.r#box()
                    .size(100.0, 100.0)
                    .bg(ColorF::new(0.2, 0.6, 1.0, 1.0))
                    .radius(20.0 + (elapsed.sin() * 10.0))
                    .glow(2.0, ColorF::new(0.2, 0.6, 1.0, 0.5))
                    .build();
                
                ui.end();
                ui.end();

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
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
