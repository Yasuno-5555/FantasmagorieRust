use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, Prefab, Collider, PhysicsComponent};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Prefab Demo");
    println!("Click to Spawn an 'Enemy' Prefab!");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Prefab Demo")
        .with_inner_size(LogicalSize::new(1024, 768));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1024, 768, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut world = World::new();

    // 1. Define an "Enemy" Prefab
    let mut enemy_prefab = Prefab::new("Enemy");
    enemy_prefab.physics = Some(PhysicsComponent {
        velocity: Vec2::ZERO,
        mass: 1.0,
        friction: 0.1,
        restitution: 0.9,
    });
    enemy_prefab.collider = Some(Collider::circle(20.0));
    
    // Serialize it to simulate loading from disk
    let prefab_json = enemy_prefab.to_json()?;
    println!("Enemy Prefab JSON:\n{}", prefab_json);
    
    // 2. Load it back
    let loaded_prefab = Prefab::from_json(&prefab_json)?;

    let mut last_time = std::time::Instant::now();
    let mut cursor_pos = Vec2::ZERO;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                cursor_pos = Vec2::new(position.x as f32, position.y as f32);
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. }, .. } => {
                // Spawn the prefab at mouse position
                loaded_prefab.spawn(&mut world, cursor_pos);
                println!("Spawned Enemy at {:?}", cursor_pos);
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.033);
                last_time = now;
                
                // Physics Simulation
                for p in &mut world.physics {
                    if p.mass > 0.0 {
                        p.velocity.y += 800.0 * dt; // Gravity
                    }
                }
                
                world.system_physics_step(dt);
                world.system_transform_update();
                world.system_physics_collision();
                world.system_transform_update();

                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1024.0, 768.0), ColorF::new(0.05, 0.1, 0.05, 1.0));
                
                // Draw entities
                for i in 0..world.ids.len() {
                    let pos = world.transforms[i].world_position();
                    let color = ColorF::new(1.0, 0.2, 0.2, 1.0);
                    
                    if let Some(col) = &world.colliders[i] {
                        match col {
                            Collider::Circle { radius, .. } => {
                                dl.add_circle(pos, *radius, color, true);
                            }
                            _ => {}
                        }
                    }
                }

                // UI Overlay
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);
                let root = ui.column().size(1024.0, 100.0).padding(20.0).build();
                ui.begin(root);
                ui.text("Prefab Demo").font_size(24.0).build();
                ui.text("Left Click to Spawn 'Enemy' Prefabs").font_size(16.0).fg(ColorF::new(0.7, 0.7, 0.7, 1.0)).build();
                ui.end();
                
                if let Some(ui_root) = ui.root() {
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::render_ui(ui_root, 1024.0, 768.0, &mut dl, &mut fm);
                        
                        if fm.texture_dirty {
                            backend.update_font_texture(fm.atlas.width as u32, fm.atlas.height as u32, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                    });
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
