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

static mut FRAME_COUNT: u32 = 0;

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
                let _: () = msg_send![layer, setFramebufferOnly: cocoa::base::NO];
                let _: () = msg_send![layer, setFramebufferOnly: cocoa::base::NO]; // Added line
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
                
                ui.text("Metal Cinematic Revolution")
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
                ui.text("• Native Metal 3.0 Performance").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
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
