use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::view::header::Align;
use fanta_rust::config::{CinematicConfig, Bloom, Tonemap};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

#[cfg(feature = "metal")]
use cocoa::base::id;
#[cfg(feature = "metal")]
use objc::{msg_send, sel, sel_impl};
#[cfg(feature = "metal")]
use metal::foreign_types::ForeignType;
#[cfg(feature = "metal")]
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
#[cfg(feature = "metal")]
use core_graphics_types::geometry::CGSize;

use fanta_rust::core::{Vec2, ColorF};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    println!("DEBUG: Args: {:?}", args);
    let use_metal = args.contains(&"--backend".to_string()) && args.contains(&"metal".to_string());
    
    if use_metal {
        println!("Fantasmagorie Rust - Visual Showcase (Metal)");
    } else {
        println!("Fantasmagorie Rust - Visual Showcase (WGPU)");
    }

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Cinematic Showcase")
        .with_inner_size(LogicalSize::new(1280, 720));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut backend: Box<dyn GraphicsBackend> = if use_metal {
        #[cfg(feature = "metal")]
        {
            use fanta_rust::backend::MetalBackend;
            let mut mb = MetalBackend::new()
                .map_err(|e| format!("Metal creation failed: {}", e))?;
            
            // Setup CAMetalLayer
            unsafe {
                if let Ok(handle) = window.window_handle() {
                    if let RawWindowHandle::AppKit(handle) = handle.as_raw() {
                        let view = handle.ns_view.as_ptr() as id;
                        let layer = metal::MetalLayer::new();
                        layer.set_device(&mb.device);
                        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
                        layer.set_framebuffer_only(false);
                        
                        let size = window.inner_size();
                        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
                        // layer.set_contents_scale(window.scale_factor()); // MetalLayer might not support this directly? check docs. 
                        // It usually inherits from view backing scale?
                        // But we set drawable size explicitly. Using physical pixels.
                        
                        let view = handle.ns_view.as_ptr() as id;
                        let _: () = msg_send![view, setLayer: layer.as_ptr()];
                        let _: () = msg_send![view, setWantsLayer: cocoa::base::YES];
                        
                        mb.set_layer(layer.as_ptr() as *mut _); 
                        // Note: MetalLayer needs to be retained? MetalLayer::new() follows creation rule (retain count +1).
                        // msg_send sets it. View retains it?
                        // MetalLayer struct drops it?
                        // Use calling `mem::forget` or extracting ptr?
                        // `layer.as_ptr()` returns `*mut Object`.
                        // `layer` is `MetalLayer` struct.
                        // Impl `Drop` calls release.
                        // We must ensure it stays alive.
                        // View usually retains layer.
                        // So letting `layer` drop (decrement) after setLayer (increment) is correct balancing?
                        // `new` returns +1.
                        // `setLayer` retains? (Assigning property). Yes. +2.
                        // `layer` drop. -1. Result +1.
                        // Correct.
                        
                    }
                }
            }
            Box::new(mb)
        }
        #[cfg(not(feature = "metal"))]
        {
            return Err("Metal feature not enabled".into());
        }
    } else {
        Box::new(WgpuBackend::new_async(window.clone(), 1280, 720, 1.0)
            .map_err(|e| format!("WGPU creation failed: {}", e))?)
    };

    let mut cinematic = CinematicConfig::default();
    cinematic.volumetric_intensity = 0.8;
    cinematic.chromatic_aberration = 0.005;
    cinematic.bloom = Bloom::Cinematic;
    cinematic.tonemap = Tonemap::Aces;
    cinematic.blur_radius = 5.0; // Depth of Field
    
    backend.set_cinematic_config(cinematic);

    // Request screenshot immediately if arg present
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|r| r == "--screenshot") {
        if let Some(path) = args.get(pos + 1) {
            backend.capture_screenshot(path);
        }
    }

    let mut current_width = 1280;
    let mut current_height = 720;
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
                let elapsed = start_time.elapsed().as_secs_f32();

                // Dynamic light follow mouse-ish or auto-orbit
                cinematic.light_pos = [
                    (current_width as f32 / 2.0) + (elapsed * 0.5).cos() * 400.0,
                    (current_height as f32 / 2.0) + (elapsed * 0.3).sin() * 200.0,
                ];
                backend.set_cinematic_config(cinematic);

                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .bg(ColorF::new(0.01, 0.01, 0.015, 1.0))
                    .align(Align::Center)
                    .build();

                ui.begin(root);
                
                // Background Grid to show God Rays & DoF
                let grid = ui.column()
                    .size(1000.0, 600.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.02))
                    .radius(32.0)
                    .backdrop_blur(20.0)
                    .align(Align::Center)
                    .spacing(40.0)
                    .build();
                ui.begin(grid);

                ui.text("Visual Revolution")
                    .font_size(72.0)
                    .fg(ColorF::new(1.0, 1.0, 1.0, 0.95))
                    .build();

                ui.text("Volumetric Rays • Bokeh Depth of Field • Anamorphotic Flares")
                    .font_size(20.0)
                    .fg(ColorF::new(0.4, 0.7, 1.0, 0.8))
                    .build();

                let row = ui.row().spacing(60.0).build();
                ui.begin(row);
                
                // High-intensity emissive geometry to trigger cinematic effects
                for i in 0..4 {
                    let color = match i {
                        0 => ColorF::new(20.0, 2.0, 0.5, 1.0),   // Solar Flare Red
                        1 => ColorF::new(0.5, 15.0, 2.0, 1.0),  // Emerald Glow
                        2 => ColorF::new(1.0, 5.0, 25.0, 1.0),  // Deep Blue Plasma
                        _ => ColorF::new(15.0, 15.0, 2.0, 1.0), // Golden Supernova
                    };
                    
                    let mut glow_color = color;
                    glow_color.r *= 0.3;
                    glow_color.g *= 0.3;
                    glow_color.b *= 0.3;
                    
                    ui.r#box()
                        .size(140.0, 140.0)
                        .bg(color)
                        .radius(70.0)
                        .glow(40.0, glow_color)
                        .build();
                }

                ui.end(); // row
                
                ui.text("Cinematic HDR Pipeline Enabled")
                    .font_size(14.0)
                    .fg(ColorF::new(0.5, 0.5, 0.6, 0.5))
                    .build();

                ui.end(); // grid
                ui.end(); // root
                
                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    
                    // Backdrop
                    let size = Vec2::new(current_width as f32, current_height as f32);
                    dl.add_rect(
                        Vec2::ZERO,
                        size,
                        ColorF::new(1.0, 0.0, 0.0, 1.0) // DEBUG: Red
                    );

                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::render_ui(root, current_width as f32, current_height as f32, &mut dl, &mut fm);
                        if fm.texture_dirty {
                            backend.update_font_texture(fm.atlas.width as u32, fm.atlas.height as u32, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                    });

                    // NEW: Direct Draw Test (bypass everything)
                    // Let's draw a full-screen magenta rect if we're debugging
                    #[cfg(feature = "metal")]
                    if use_metal {
                        if let Some(mb) = backend.as_any().downcast_ref::<fanta_rust::backend::MetalBackend>() {
                             println!("DEBUG: Direct Draw Test: hdr_texture size={:?}", mb.hdr_texture.as_ref().map(|t: &metal::Texture| (t.width(), t.height())));
                        }
                    }

                    println!("DEBUG: Inside RedrawRequested. Args len: {}", args.len());
                    if let Some(pos) = args.iter().position(|r| r == "--screenshot") {
                        println!("DEBUG: Found screenshot arg at pos {}", pos);
                        if let Some(path) = args.get(pos + 1) {
                            println!("DEBUG: Requesting screenshot to {}", path);
                            backend.capture_screenshot(path);
                        } else {
                             println!("DEBUG: Screenshot path missing");
                        }
                    } else {
                         println!("DEBUG: No screenshot arg found in {:?}", args);
                    }

                    // Render with backend
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
