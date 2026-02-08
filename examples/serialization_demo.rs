use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, Collider};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, KeyEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Serialization Demo");
    println!("Press 'S' to Save, 'L' to Load, 'C' to Clear World");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Serialization Demo")
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
    let mut saved_json: Option<String> = None;

    // Initial world setup
    spawn_entities(&mut world);

    let mut last_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { 
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, .. 
            }, .. } => {
                match logical_key {
                    Key::Character(s) if s == "s" || s == "S" => {
                        match world.save_to_json() {
                            Ok(json) => {
                                saved_json = Some(json);
                                println!("World Saved! ({} bytes)", saved_json.as_ref().unwrap().len());
                            }
                            Err(e) => println!("Save Error: {}", e),
                        }
                    }
                    Key::Character(l) if l == "l" || l == "L" => {
                        if let Some(json) = &saved_json {
                            match World::load_from_json(json) {
                                Ok(new_world) => {
                                    world = new_world;
                                    println!("World Loaded!");
                                }
                                Err(e) => println!("Load Error: {}", e),
                            }
                        } else {
                            println!("No saved data found!");
                        }
                    }
                    Key::Character(c) if c == "c" || c == "C" => {
                        world = World::new();
                        println!("World Cleared!");
                    }
                    _ => {}
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.033);
                last_time = now;
                
                // Slowly rotate everything
                for t in &mut world.transforms {
                    t.local_rotation += dt * 0.5;
                }
                
                world.system_transform_update();

                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1024.0, 768.0), ColorF::new(0.1, 0.05, 0.1, 1.0));
                
                for i in 0..world.ids.len() {
                    let pos = world.transforms[i].world_position();
                    let rot = world.transforms[i].local_rotation;
                    let color = if i % 2 == 0 { ColorF::new(0.0, 1.0, 0.5, 1.0) } else { ColorF::new(1.0, 0.5, 0.0, 1.0) };
                    
                    if let Some(col) = &world.colliders[i] {
                        match col {
                            Collider::Circle { radius, .. } => {
                                dl.add_circle(pos, *radius, color, true);
                            }
                            Collider::AABB { size, .. } => {
                                dl.add_rect(pos - *size * 0.5, *size, color);
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
                ui.text("Serialization Demo").font_size(24.0).build();
                ui.text("Press 'S' to Save, 'L' to Load, 'C' to Clear").font_size(16.0).fg(ColorF::new(0.7, 0.7, 0.7, 1.0)).build();
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

fn spawn_entities(world: &mut World) {
    for i in 0..10 {
        let id = world.spawn();
        let idx = *world.id_to_index.get(&id).unwrap();
        world.transforms[idx].local_position = Vec2::new(100.0 + i as f32 * 80.0, 300.0 + (i as f32 * 0.5).sin() * 100.0);
        if i % 2 == 0 {
            world.colliders[idx] = Some(Collider::circle(30.0));
        } else {
            world.colliders[idx] = Some(Collider::aabb(40.0, 40.0));
        }
    }
}
