use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, EntityId, Transform, PhysicsComponent, Collider};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Physics Demo");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Physics Demo")
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
    
    // Create static floor
    let _floor = world.spawn();
    world.transforms[0].local_position = Vec2::new(512.0, 700.0);
    world.physics[0].mass = 0.0; // Static
    world.colliders[0] = Some(Collider::aabb(800.0, 40.0));
    
    // Create some falling boxes
    for i in 0..5 {
        let box_entity = world.spawn();
        let idx = *world.id_to_index.get(&box_entity).unwrap();
        world.transforms[idx].local_position = Vec2::new(300.0 + i as f32 * 100.0, 100.0 + i as f32 * 50.0);
        world.physics[idx].velocity = Vec2::new(0.0, 0.0);
        world.physics[idx].restitution = 0.8;
        world.colliders[idx] = Some(Collider::aabb(40.0, 40.0));
    }
    
    // Create some falling circles
    for i in 0..5 {
        let circ_entity = world.spawn();
        let idx = *world.id_to_index.get(&circ_entity).unwrap();
        world.transforms[idx].local_position = Vec2::new(350.0 + i as f32 * 100.0, -50.0 * i as f32);
        world.physics[idx].velocity = Vec2::new(0.0, 0.0);
        world.physics[idx].restitution = 0.7;
        world.colliders[idx] = Some(Collider::circle(20.0));
    }

    // Create a large rotating polygon in the center
    let poly_entity = world.spawn();
    let poly_idx = *world.id_to_index.get(&poly_entity).unwrap();
    world.transforms[poly_idx].local_position = Vec2::new(512.0, 400.0);
    world.physics[poly_idx].mass = 0.0; // Static but rotating
    world.colliders[poly_idx] = Some(Collider::polygon(vec![
        Vec2::new(-100.0, -50.0),
        Vec2::new(100.0, -50.0),
        Vec2::new(150.0, 50.0),
        Vec2::new(-150.0, 50.0),
    ]));

    let mut last_time = std::time::Instant::now();
    let mut cursor_pos = Vec2::ZERO;
    let mut mouse_pressed = false;
    let mut dragged_entity: Option<usize> = None;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                cursor_pos = Vec2::new(position.x as f32, position.y as f32);
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                if button == winit::event::MouseButton::Left {
                    mouse_pressed = state == winit::event::ElementState::Pressed;
                    if mouse_pressed {
                        // Pick entity
                        for i in 0..world.ids.len() {
                            if world.physics[i].mass > 0.0 {
                                let pos = world.transforms[i].world_position();
                                if pos.distance(cursor_pos) < 50.0 {
                                    dragged_entity = Some(i);
                                    break;
                                }
                            }
                        }
                    } else {
                        dragged_entity = None;
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.033);
                last_time = now;
                
                // Mouse dragging logic
                if let Some(idx) = dragged_entity {
                    let target_pos = cursor_pos;
                    let current_pos = world.transforms[idx].world_position();
                    world.physics[idx].velocity = (target_pos - current_pos) * 10.0;
                }

                // Add some gravity
                for p in &mut world.physics {
                    if p.mass > 0.0 {
                        p.velocity.y += 800.0 * dt;
                    }
                }
                
                // Slowly rotate the polygon
                world.transforms[poly_idx].local_rotation += dt * 0.5;

                world.system_physics_step(dt);
                world.system_transform_update();
                world.system_physics_collision();
                world.system_transform_update();

                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1024.0, 768.0), ColorF::new(0.1, 0.1, 0.15, 1.0));
                
                for i in 0..world.ids.len() {
                    let pos = world.transforms[i].world_position();
                    let rot = world.transforms[i].local_rotation;
                    let color = if world.physics[i].mass == 0.0 {
                        ColorF::new(0.5, 0.5, 0.6, 1.0)
                    } else if dragged_entity == Some(i) {
                        ColorF::new(1.0, 1.0, 0.0, 1.0)
                    } else {
                        if i % 2 == 0 { ColorF::new(0.0, 0.8, 1.0, 1.0) } 
                        else { ColorF::new(1.0, 0.4, 0.8, 1.0) }
                    };
                    
                    if let Some(col) = &world.colliders[i] {
                        match col {
                            Collider::AABB { size, .. } => {
                                dl.add_rect(pos - *size * 0.5, *size, color);
                            }
                            Collider::Circle { radius, .. } => {
                                dl.add_circle(pos, *radius, color, true);
                            }
                            Collider::Polygon { vertices, .. } => {
                                let mut abs_verts = Vec::new();
                                let mat = Mat3::translation(pos.x, pos.y) * Mat3::rotation(rot);
                                for &v in vertices {
                                    abs_verts.push(mat * v);
                                }
                                dl.add_polyline(abs_verts, color, 2.0, true);
                            }
                        }
                    }
                }

                // UI Overlay
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);
                let root = ui.column().size(1024.0, 50.0).padding(10.0).build();
                ui.begin(root);
                ui.text("Physics Phase 2: Impulse-based Collisions").font_size(24.0).build();
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
