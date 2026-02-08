use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, StateMachine, SignalData, Signal};
use fanta_rust::game::state_machine::{State, Transition, Condition};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Gameplay Systems Demo");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Gameplay Demo")
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
    
    // 1. Create a "Drone" entity with a State Machine
    let drone = world.spawn();
    let drone_idx = *world.id_to_index.get(&drone).unwrap();
    world.transforms[drone_idx].local_position = Vec2::new(512.0, 384.0);
    
    let mut fsm = StateMachine::new("PatrolLeft");
    
    // Define PatrolLeft state
    fsm.add_state(State {
        name: "PatrolLeft".to_string(),
        transitions: vec![Transition {
            target: "Idle".to_string(),
            condition: Condition::Timer(2.0),
        }],
    });
    
    // Define Idle state
    fsm.add_state(State {
        name: "Idle".to_string(),
        transitions: vec![Transition {
            target: "PatrolRight".to_string(),
            condition: Condition::Timer(1.0),
        }],
    });
    
    // Define PatrolRight state
    fsm.add_state(State {
        name: "PatrolRight".to_string(),
        transitions: vec![Transition {
            target: "IdleAgain".to_string(),
            condition: Condition::Timer(2.0),
        }],
    });
    
    // Define IdleAgain state
    fsm.add_state(State {
        name: "IdleAgain".to_string(),
        transitions: vec![Transition {
            target: "PatrolLeft".to_string(),
            condition: Condition::Timer(1.0),
        }],
    });
    
    world.state_machines[drone_idx] = Some(fsm);

    // 2. Create a "Signal Receiver" entity (e.g., a flashing light)
    let light = world.spawn();
    let light_idx = *world.id_to_index.get(&light).unwrap();
    world.transforms[light_idx].local_position = Vec2::new(512.0, 200.0);
    let mut light_color = ColorF::new(0.2, 0.2, 0.2, 1.0);

    let mut last_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.033);
                last_time = now;
                
                // Update Logic
                world.system_state_update(dt);
                
                // Process signals (Signal dispatch logic)
                while let Some(signal) = world.signal_bus.poll() {
                    match signal.data {
                        SignalData::Custom(ref s) if s.starts_with("state_changed:") => {
                            let state = &s["state_changed:".len()..];
                            match state {
                                "Idle" | "IdleAgain" => light_color = ColorF::new(1.0, 1.0, 0.0, 1.0), // Yellow
                                "PatrolLeft" => light_color = ColorF::new(0.0, 1.0, 1.0, 1.0),      // Cyan
                                "PatrolRight" => light_color = ColorF::new(1.0, 0.0, 1.0, 1.0),     // Magenta
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
                
                // Move drone based on state
                if let Some(fsm) = &world.state_machines[drone_idx] {
                    match fsm.current_state.as_str() {
                        "PatrolLeft" => world.transforms[drone_idx].local_position.x -= 100.0 * dt,
                        "PatrolRight" => world.transforms[drone_idx].local_position.x += 100.0 * dt,
                        _ => {}
                    }
                }

                world.system_transform_update();

                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1024.0, 768.0), ColorF::new(0.05, 0.05, 0.1, 1.0));
                
                // Draw Signal Receiver (Light)
                dl.add_circle(world.transforms[light_idx].world_position(), 30.0, light_color, true);
                dl.add_circle(world.transforms[light_idx].world_position(), 40.0, ColorF::new(light_color.r, light_color.g, light_color.b, 0.3), false);

                // Draw Drone
                let drone_pos = world.transforms[drone_idx].world_position();
                dl.add_rect(drone_pos - Vec2::new(20.0, 20.0), Vec2::new(40.0, 40.0), ColorF::new(0.8, 0.8, 0.8, 1.0));
                
                // Text Overlay
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);
                let root = ui.column().size(1024.0, 200.0).padding(40.0).build();
                ui.begin(root);
                
                ui.text("Phase 3: Gameplay Systems").font_size(32.0).fg(ColorF::new(0.0, 1.0, 0.8, 1.0)).build();
                
                if let Some(fsm) = &world.state_machines[drone_idx] {
                    let drone_state_str = format!("Drone State: {}", fsm.current_state);
                    ui.text(ui.arena.alloc_str(&drone_state_str))
                        .font_size(24.0)
                        .fg(ColorF::new(1.0, 1.0, 1.0, 1.0))
                        .build();
                    let timer_str = format!("Timer: {:.2}s", fsm.timer);
                    ui.text(ui.arena.alloc_str(&timer_str))
                        .font_size(18.0)
                        .fg(ColorF::new(0.6, 0.6, 0.6, 1.0))
                        .build();
                }
                
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
