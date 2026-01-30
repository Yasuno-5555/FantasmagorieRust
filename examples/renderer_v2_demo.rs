use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::renderer::types::{Rect, Color};
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

/// Stage 1 Verification Demo: Direct Renderer API
/// This demo verifies SDF shapes (rounded rects, borders, elevations)
/// without using the High-level Widget system.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Stage 1 - Renderer V2 Demo");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie Stage 1: SDF Evolution")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let window = Arc::new(window);
    let size = window.inner_size();

    // Setup WGPU Backend
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = unsafe { instance.create_surface(window.clone()) }?;
    
    let mut backend = pollster::block_on(WgpuBackend::new_async(
        &instance,
        surface,
        size.width,
        size.height,
    )).map_err(|e| format!("WGPU failure: {}", e))?;

    // Create Renderer with Lite profile (Stage 0/1 target)
    let mut renderer = Renderer::new(Box::new(backend), EngineConfig::lite());

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let size = window.inner_size();
                let mut frame = renderer.begin_frame();

                // 1. Basic Rounded Rect with Border and Elevation
                frame.draw(Rect::new(100.0, 100.0, 400.0, 200.0), Color::hex(0x2a2a2a))
                    .rounded(16.0)
                    .border(2.0, Color::white())
                    .elevation(12.0)
                    .submit();

                // 2. Individual Corner Radii
                frame.draw(Rect::new(600.0, 100.0, 300.0, 200.0), Color::hex(0x004488))
                    .radii(32.0, 0.0, 32.0, 0.0) // Top-left and bottom-right only
                    .submit();

                // 3. Glowing Pulse
                let color = Color::hex(0xff3366);
                frame.draw(Rect::new(100.0, 400.0, 200.0, 100.0), color)
                    .rounded(50.0) // Circle
                    .glow(1.0, color)
                    .submit();

                // 4. Squircle vs Rounded Rect
                frame.draw(Rect::new(400.0, 400.0, 200.0, 200.0), Color::hex(0x33aa33))
                    .rounded(40.0)
                    .submit();
                
                frame.draw(Rect::new(650.0, 400.0, 200.0, 200.0), Color::hex(0x33aa33))
                    .rounded(40.0)
                    .squircle()
                    .submit();

                renderer.end_frame(frame, size.width, size.height);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
