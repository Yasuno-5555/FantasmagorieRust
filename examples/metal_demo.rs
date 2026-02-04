use fanta_rust::backend::{MetalBackend, GraphicsBackend};
use metal::foreign_types::ForeignType;
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use cocoa::base::id;
use objc::{msg_send, sel, sel_impl};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Metal Demo");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Visual Revolution (Metal)")
        .with_inner_size(LogicalSize::new(1024, 768));
    
    let window = event_loop.create_window(window)?;

    let window = Arc::new(window);
    
    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut backend = MetalBackend::new().map_err(|e| format!("Metal creation failed: {}", e))?;

    // Setup CAMetalLayer
    unsafe {
        if let Ok(handle) = window.window_handle() {
            if let RawWindowHandle::AppKit(handle) = handle.as_raw() {
                let view = handle.ns_view.as_ptr() as id;
                let layer: id = msg_send![objc::class!(CAMetalLayer), new];
                let device_ptr = backend.device.as_ptr();
                let _: () = msg_send![layer, setDevice: device_ptr];
                let _: () = msg_send![layer, setPixelFormat: 80]; // MTLPixelFormatBGRA8Unorm
                let _: () = msg_send![view, setLayer: layer];
                let _: () = msg_send![view, setWantsLayer: cocoa::base::YES];
                
                let size = window.inner_size();
                let scale_factor = window.scale_factor();
                let _: () = msg_send![layer, setDrawableSize: cocoa::foundation::NSSize::new(
                    size.width as f64,
                    size.height as f64
                )];
                let _: () = msg_send![layer, setContentsScale: scale_factor];
                
                backend.set_layer(layer as *mut _);
            }
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
                
                // Update Metal drawable size
                unsafe {
                    if let Ok(handle) = window.window_handle() {
                        if let RawWindowHandle::AppKit(handle) = handle.as_raw() {
                            let view = handle.ns_view.as_ptr() as id;
                            let layer: id = msg_send![view, layer];
                            let scale_factor = window.scale_factor();
                            let _: () = msg_send![layer, setDrawableSize: cocoa::foundation::NSSize::new(
                                size.width as f64,
                                size.height as f64
                            )];
                            let _: () = msg_send![layer, setContentsScale: scale_factor];
                        }
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .bg(ColorF::new(0.01, 0.01, 0.02, 1.0))
                    .build();

                ui.begin(root);
                
                // Mode 12 is handled by setting elevation to a magic value or just using a specific bg
                // For now, let's just use a normal background and I'll fix the shader mode 12 trigger
                // if I can find a way to pass it. Actually mode 12 is Cyberpunk Grid.
                // In this framework, mode is usually derived from the widget type.
                
                let panel = ui.column()
                    .size(600.0, 450.0)
                    .align(Align::Center)
                    .bg(ColorF::new(0.05, 0.05, 0.1, 0.8))
                    .radius(32.0)
                    .elevation(40.0)
                    .padding(40.0)
                    .build();
                
                ui.begin(panel);
                ui.text("Visual Revolution").font_size(48.0).fg(ColorF::new(0.0, 0.9, 1.0, 1.0)).build();
                ui.text("Advanced Cinematic Post-Processing").font_size(18.0).fg(ColorF::new(0.7, 0.7, 0.8, 1.0)).build();
                
                ui.text("").build(); // Spacer

                let elapsed = start_time.elapsed().as_secs_f32();
                
                // Single row
                let row = ui.row().spacing(40.0).build();
                ui.begin(row);
                
                // High intensity blocks to trigger Bloom
                ui.r#box()
                    .size(120.0, 120.0)
                    .bg(ColorF::new(2.0, 0.2, 0.5, 1.0)) // HDR color (R > 1.0)
                    .radius(24.0 + (elapsed.sin() * 8.0))
                    .glow(10.0, ColorF::new(2.0, 0.2, 0.5, 0.8))
                    .build();

                ui.r#box()
                    .size(120.0, 120.0)
                    .bg(ColorF::new(0.0, 3.0, 1.5, 1.0)) // HDR color (G > 1.0)
                    .radius(24.0 + (elapsed.cos() * 8.0))
                    .glow(10.0, ColorF::new(0.0, 3.0, 1.5, 0.8))
                    .build();
                    
                ui.r#box()
                    .size(120.0, 120.0)
                    .bg(ColorF::new(1.0, 1.0, 5.0, 1.0)) // HDR color (B > 1.0)
                    .radius(24.0 + ((elapsed * 1.5).sin() * 8.0))
                    .glow(10.0, ColorF::new(1.0, 1.0, 5.0, 0.8))
                    .build();
                
                ui.end();

                ui.text("").build(); // Spacer
                ui.text("• ACES Filmic Tone Mapping").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                ui.text("• Multi-pass Gaussian Bloom").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                ui.text("• Chromatic Aberration & Vignette").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                
                ui.end();
                ui.end();
                
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
