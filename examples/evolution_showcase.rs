use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, Prefab, Collider, PhysicsComponent, StateMachine};
use fanta_rust::game::state_machine::{State, Transition, Condition};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, MouseButton, KeyEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Final Evolution Showcase");
    println!("Commands: [S] Save | [L] Load | [Space] Spawn Asteroid | [Click] Move Station");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Evolution Showcase")
        .with_inner_size(LogicalSize::new(1280, 800));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1280, 800, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut world = World::new();
    let mut saved_json: Option<String> = None;

    // 1. Setup Space Station (Hierarchy)
    let station_root = world.spawn();
    let root_idx = *world.id_to_index.get(&station_root).unwrap();
    world.transforms[root_idx].local_position = Vec2::new(640.0, 400.0);
    
    // Modules
    for i in 0..4 {
        let angle = i as f32 * std::f32::consts::PI * 0.5;
        let module = world.spawn();
        let mod_idx = *world.id_to_index.get(&module).unwrap();
        world.transforms[mod_idx].local_position = Vec2::new(angle.cos() * 150.0, angle.sin() * 150.0);
        fanta_rust::game::attach(&mut world, module, station_root);
        
        // Rotating sub-parts
        let panel = world.spawn();
        let panel_idx = *world.id_to_index.get(&panel).unwrap();
        world.transforms[panel_idx].local_position = Vec2::new(angle.cos() * 40.0, angle.sin() * 40.0);
        world.colliders[panel_idx] = Some(Collider::aabb(60.0, 20.0));
        fanta_rust::game::attach(&mut world, panel, module);
    }

    // 2. Setup Asteroid Prefab
    let mut asteroid_prefab = Prefab::new("Asteroid");
    asteroid_prefab.physics = Some(PhysicsComponent {
        velocity: Vec2::ZERO,
        mass: 5.0,
        friction: 0.1,
        restitution: 0.8,
    });
    asteroid_prefab.collider = Some(Collider::circle(30.0));

    // 3. Setup Guard Drone with FSM
    let drone_prefab = create_drone_prefab();

    let mut last_time = std::time::Instant::now();
    let mut cursor_pos = Vec2::ZERO;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                cursor_pos = Vec2::new(position.x as f32, position.y as f32);
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. }, .. } => {
                // Move station to click
                let root_idx = *world.id_to_index.get(&station_root).unwrap();
                world.transforms[root_idx].local_position = cursor_pos;
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { 
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, .. 
            }, .. } => {
                match logical_key {
                    Key::Character(s) if s == "s" || s == "S" => {
                        if let Ok(json) = world.save_to_json() {
                            saved_json = Some(json);
                            println!("Simulation State Saved.");
                        }
                    }
                    Key::Character(l) if l == "l" || l == "L" => {
                        if let Some(json) = &saved_json {
                            if let Ok(new_world) = World::load_from_json(json) {
                                world = new_world;
                                println!("Simulation State Restored.");
                            }
                        }
                    }
                    Key::Named(NamedKey::Space) => {
                        // Spawn Asteroid at random velocity
                        let id = asteroid_prefab.spawn(&mut world, cursor_pos);
                        let idx = *world.id_to_index.get(&id).unwrap();
                        let vel = Vec2::new((rand_f32() - 0.5) * 400.0, (rand_f32() - 0.5) * 400.0);
                        world.physics[idx].velocity = vel;
                    }
                    Key::Character(d) if d == "d" || d == "D" => {
                        drone_prefab.spawn(&mut world, cursor_pos);
                    }
                    _ => {}
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.033);
                last_time = now;
                
                // Rotation Logic for Station
                if let Some(&idx) = world.id_to_index.get(&station_root) {
                    world.transforms[idx].local_rotation += dt * 0.2;
                }

                // Systems Update
                world.system_physics_step(dt);
                world.system_state_update(dt);
                world.system_physics_collision();
                world.system_transform_update();
                world.system_signal_dispatch();

                // Drawing
                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1280.0, 800.0), ColorF::new(0.02, 0.02, 0.05, 1.0));
                
                // Cyberpunk Grid Backdrop (Visual legacy)
                draw_grid(&mut dl, 1280.0, 800.0);

                for i in 0..world.ids.len() {
                    let pos = world.transforms[i].world_position();
                    let rot = world.transforms[i].local_rotation;
                    
                    if let Some(col) = &world.colliders[i] {
                        let color = get_entity_color(i);
                        match col {
                            Collider::Circle { radius, .. } => dl.add_circle(pos, *radius, color, true),
                            Collider::AABB { size, .. } => {
                                // Simple AABB drawing, logic for rotation not handled in DrawList yet 
                                // (DrawList::add_rect is axis-aligned)
                                dl.add_rect(pos - *size * 0.5, *size, color);
                            }
                            _ => {}
                        }
                    }
                }

                // UI
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);
                render_evolution_ui(&world, &mut ui);
                
                if let Some(ui_root) = ui.root() {
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::render_ui(ui_root, 1280.0, 800.0, &mut dl, &mut fm);
                        
                        if fm.texture_dirty {
                            backend.update_font_texture(fm.atlas.width as u32, fm.atlas.height as u32, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                    });
                }

                backend.render(&dl, 1280, 800);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}

fn create_drone_prefab() -> Prefab {
    let mut p = Prefab::new("Drone");
    p.physics = Some(PhysicsComponent {
        velocity: Vec2::ZERO,
        mass: 1.0,
        friction: 0.5,
        restitution: 0.5,
    });
    p.collider = Some(Collider::circle(15.0));
    
    let mut fsm = StateMachine::new("Patrol");
    fsm.add_state(State {
        name: "Patrol".to_string(),
        transitions: vec![Transition { target: "Idle".to_string(), condition: Condition::Timer(3.0) }],
    });
    fsm.add_state(State {
        name: "Idle".to_string(),
        transitions: vec![Transition { target: "Patrol".to_string(), condition: Condition::Timer(1.5) }],
    });
    p.state_machine = Some(fsm);
    p
}

fn draw_grid(dl: &mut DrawList, w: f32, h: f32) {
    let step = 50.0;
    let grid_color = ColorF::new(0.0, 0.5, 1.0, 0.1);
    for x in 0..=(w / step) as i32 {
        let x_pos = x as f32 * step;
        dl.add_polyline(vec![Vec2::new(x_pos, 0.0), Vec2::new(x_pos, h)], grid_color, 1.0, false);
    }
    for y in 0..=(h / step) as i32 {
        let y_pos = y as f32 * step;
        dl.add_polyline(vec![Vec2::new(0.0, y_pos), Vec2::new(w, y_pos)], grid_color, 1.0, false);
    }
}

fn get_entity_color(idx: usize) -> ColorF {
    match idx % 3 {
        0 => ColorF::new(0.0, 0.8, 1.0, 1.0),
        1 => ColorF::new(1.0, 0.2, 0.5, 1.0),
        _ => ColorF::new(0.8, 1.0, 0.0, 1.0),
    }
}

fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    (nanos % 1000) as f32 / 1000.0
}

fn render_evolution_ui(world: &World, ui: &mut UIContext) {
    let root = ui.column().size(400.0, 300.0).padding(30.0).build();
    ui.begin(root);
    ui.text("Engine Evolution Showcase").font_size(28.0).fg(ColorF::new(0.0, 1.0, 0.9, 1.0)).build();
    let entity_count_str = format!("Entities: {}", world.ids.len());
    ui.text(ui.arena.alloc_str(&entity_count_str)).font_size(18.0).build();
    ui.text("[Space] Spawn Asteroid").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.6, 1.0)).build();
    ui.text("[D] Spawn Drone").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.6, 1.0)).build();
    ui.text("[S/L] Save/Load State").font_size(14.0).fg(ColorF::new(0.6, 0.6, 0.6, 1.0)).build();
    ui.end();
}
