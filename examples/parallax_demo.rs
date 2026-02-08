use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, parallax::ParallaxLayer};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Parallax Demo");
    println!("Move mouse to scroll camera.");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Parallax Demo")
        .with_inner_size(LogicalSize::new(1280, 800));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1280, 800, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut world = World::new();

    // Layer 1 (Far Background - Slow)
    let e1 = world.spawn();
    let idx1 = *world.id_to_index.get(&e1).unwrap();
    world.parallax_layers[idx1] = Some(ParallaxLayer { 
        factor: Vec2::new(0.1, 0.1), 
        base_position: Vec2::new(100.0, 100.0) // Initial offset
    });

    // Layer 2 (Mid - Medium)
    let e2 = world.spawn();
    let idx2 = *world.id_to_index.get(&e2).unwrap();
    world.parallax_layers[idx2] = Some(ParallaxLayer { 
        factor: Vec2::new(0.5, 0.5), 
        base_position: Vec2::new(300.0, 300.0) 
    });

    // Layer 3 (Foreground - Fast/Normal)
    let e3 = world.spawn();
    let idx3 = *world.id_to_index.get(&e3).unwrap();
    world.parallax_layers[idx3] = Some(ParallaxLayer { 
        factor: Vec2::new(1.0, 1.0), 
        base_position: Vec2::new(500.0, 500.0) 
    });

    let mut cursor_pos = Vec2::ZERO;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                // Map mouse to camera position
                cursor_pos = Vec2::new(position.x as f32, position.y as f32);
            }
             Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // Simulate camera moving
                let camera_pos = cursor_pos; // Camera follows mouse

                // Update Parallax
                world.system_parallax(camera_pos);
                world.system_transform_update();

                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1280.0, 800.0), ColorF::new(0.05, 0.05, 0.05, 1.0));
                
                // Render Layers relative to "Screen" (which is stationary)
                // Wait, if camera moves, we usually translate the whole scene by -camera_pos.
                // Here we are manually moving layer transforms relative to 0,0 based on camera logic.
                // So if we draw at `transform.position`, we are drawing in "Screen Space" if we don't apply camera transform again.
                // But normally: screen_pos = world_pos - camera_pos.
                // Parallax system sets logic such that:
                // transform.local_position = base + camera*(1-factor).
                // If we draw this world_pos minus camera_pos:
                // draw_pos = (base + camera - factor*camera) - camera = base - factor*camera.
                // This gives parallax effect relative to base.
                // So we need to apply camera transform during rendering?
                // Or does Parallax system put objects in SCREEN space?
                // `offset = camera_pos * (one - p.factor)`
                // `pos = base + offset`.
                // If I draw at `pos` directly (without -camera), then:
                // pos moves WITH camera (if factor=0). Offset=camera. Pos = Base + Camera.
                // Valid for UI sticking to camera.
                // If factor=1. (normal). Offset=0. Pos = Base. Static in world.
                // If I render static world pos without camera transform, it stays on screen.
                // This implies "Camera is at 0,0".
                // BUT we said `camera_pos` is cursor.
                // So we likely want to simulate "Looking at cursor".
                // If we assume a standard game loop where we subtract camera_pos on CPU or View Matrix:
                // Let's assume we subtract camera_pos here manually to prove logic.

                // Draw Entities
                let layers = [
                    (idx1, ColorF::new(0.3, 0.3, 0.5, 1.0), 50.0), // Far (Red tint?) Blue
                    (idx2, ColorF::new(0.5, 0.8, 0.5, 1.0), 80.0), // Mid Green
                    (idx3, ColorF::new(0.8, 0.5, 0.5, 1.0), 100.0), // Near Red
                ];

                for (idx, color, size) in layers {
                    let mut pos = world.transforms[idx].world_position();
                    // Apply camera transform (World -> Screen)
                    pos = pos - camera_pos; 
                    // Centering trick: + ScreenCenter
                    pos = pos + Vec2::new(640.0, 400.0);

                    dl.add_rect(pos, Vec2::new(size, size), color);
                }

                // Draw Camera Crosshair
                dl.add_circle(Vec2::new(640.0, 400.0), 5.0, ColorF::white(), false); 
                // Camera is theoretically at `cursor_pos` in world, looking at it. Since we shift everything by -cursor_pos+center.

                backend.render(&dl, 1280, 800);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
