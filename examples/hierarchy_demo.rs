use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, EntityId, Transform, attach};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Hierarchy Demo");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Hierarchy Demo")
        .with_inner_size(LogicalSize::new(1024, 768));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1024, 768, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut world = World::new();
    
    // Create Sun
    let sun = world.spawn();
    world.transforms[*world.id_to_index.get(&sun).unwrap()].local_position = Vec2::new(512.0, 384.0);
    
    // Create Earth (child of Sun)
    let earth = world.spawn();
    world.transforms[*world.id_to_index.get(&earth).unwrap()].local_position = Vec2::new(200.0, 0.0);
    world.transforms[*world.id_to_index.get(&earth).unwrap()].local_scale = Vec2::new(0.5, 0.5);
    attach(&mut world, earth, sun);
    
    // Create Moon (child of Earth)
    let moon = world.spawn();
    world.transforms[*world.id_to_index.get(&moon).unwrap()].local_position = Vec2::new(60.0, 0.0);
    world.transforms[*world.id_to_index.get(&moon).unwrap()].local_scale = Vec2::new(0.4, 0.4);
    attach(&mut world, moon, earth);

    let start_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let elapsed = start_time.elapsed().as_secs_f32();
                
                // Rotate Sun (affects Earth and Moon)
                let sun_idx = *world.id_to_index.get(&sun).unwrap();
                world.transforms[sun_idx].local_rotation = elapsed * 0.5;
                
                // Rotate Earth (affects Moon)
                let earth_idx = *world.id_to_index.get(&earth).unwrap();
                world.transforms[earth_idx].local_rotation = elapsed * 2.0;
                
                // Update world matrices
                world.system_transform_update();

                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);
                
                let root = ui.column()
                    .size(1024.0, 768.0)
                    .bg(ColorF::new(0.05, 0.05, 0.1, 1.0))
                    .build();
                ui.begin(root);
                
                ui.text("Hierarchy Demo: Planet System").font_size(24.0).build();
                ui.text("Sun -> Earth -> Moon hierarchy via system_transform_update").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.7, 1.0)).build();
                
                ui.end();

                let mut dl = fanta_rust::DrawList::new();
                
                // Render UI
                if let Some(ui_root) = ui.root() {
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::render_ui(ui_root, 1024.0, 768.0, &mut dl, &mut fm);
                    });
                }
                
                // Simple Debug Circles for Entities
                for i in 0..world.ids.len() {
                    let pos = world.transforms[i].world_position();
                    let color = if i == sun_idx {
                        ColorF::new(1.0, 0.8, 0.0, 1.0)
                    } else if i == earth_idx {
                        ColorF::new(0.2, 0.5, 1.0, 1.0)
                    } else {
                        ColorF::new(0.7, 0.7, 0.7, 1.0)
                    };
                    
                    let radius = 20.0 * world.transforms[i].world_matrix.0[0]; // Scale with matrix
                    
                    dl.add_circle(pos, radius, color, true);
                }

                backend.render(&dl, 1024, 768);
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
