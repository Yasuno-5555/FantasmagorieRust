use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, audio::{AudioEmitter, AudioListener}};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Audio Demo");
    println!("Sound source orbits the center.");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Audio Demo")
        .with_inner_size(LogicalSize::new(1280, 800));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1280, 800, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut world = World::new();

    // 1. Spawn Listener (Center)
    let listener_entity = world.spawn();
    let l_idx = *world.id_to_index.get(&listener_entity).unwrap();
    world.transforms[l_idx].local_position = Vec2::new(640.0, 400.0);
    world.audio_listeners[l_idx] = Some(AudioListener { active: true });

    // 2. Spawn Emitter (Orbiting)
    let emitter_entity = world.spawn();
    let e_idx = *world.id_to_index.get(&emitter_entity).unwrap();
    world.transforms[e_idx].local_position = Vec2::new(640.0 + 300.0, 400.0);
    world.audio_emitters[e_idx] = Some(AudioEmitter {
        params_id: 2, // Pulse/Square
        volume: 0.5,
        pitch: 1.0,
        looping: true,
        playing: true,
        ..Default::default()
    });

    let mut last_time = std::time::Instant::now();
    let mut angle = 0.0f32;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
             Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.033);
                last_time = now;

                // Move emitter
                angle += dt * 0.5; // Radians per sec
                let orbit_radius = 400.0;
                let center = Vec2::new(640.0, 400.0);
                world.transforms[e_idx].local_position = center + Vec2::new(angle.cos() * orbit_radius, angle.sin() * orbit_radius);

                // Update Systems
                world.system_audio();
                world.system_transform_update();

                // Draw
                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1280.0, 800.0), ColorF::new(0.05, 0.05, 0.05, 1.0));
                
                // Draw Listener (Blue Ear)
                dl.add_circle(world.transforms[l_idx].local_position, 15.0, ColorF::new(0.2, 0.5, 1.0, 1.0), true);

                // Draw Emitter (Red Source)
                dl.add_circle(world.transforms[e_idx].local_position, 10.0, ColorF::new(1.0, 0.2, 0.2, 1.0), true);
                
                // Draw connecting line
                dl.add_polyline(vec![world.transforms[l_idx].local_position, world.transforms[e_idx].local_position], ColorF::new(1.0, 1.0, 1.0, 0.2), 2.0, false);

                backend.render(&dl, 1280, 800);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
