use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, particles::{ParticleEmitter}};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Particle Demo");
    println!("Click to move emitter.");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Particle Demo")
        .with_inner_size(LogicalSize::new(1280, 800));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1280, 800, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut world = World::new();

    // Spawn Emitter
    let emitter_entity = world.spawn();
    let idx = *world.id_to_index.get(&emitter_entity).unwrap();
    world.transforms[idx].local_position = Vec2::new(640.0, 400.0);
    
    world.particle_emitters[idx] = Some(ParticleEmitter {
        rate: 500.0, // High rate
        lifetime_range: [1.0, 2.0],
        speed_range: [50.0, 150.0],
        color_start: ColorF::new(1.0, 0.5, 0.0, 1.0), // Orange
        color_end: ColorF::new(1.0, 0.0, 0.0, 0.0), // Red transparent
        size_range: [5.0, 10.0],
        cone_angle: std::f32::consts::PI * 2.0,
        ..Default::default()
    });

    let mut last_time = std::time::Instant::now();
    let mut cursor_pos = Vec2::new(640.0, 400.0);

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                cursor_pos = Vec2::new(position.x as f32, position.y as f32);
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. }, .. } => {
                world.transforms[idx].local_position = cursor_pos;
            }
             Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.033);
                last_time = now;

                // Move emitter to cursor if needed (or just click)
                // world.transforms[idx].local_position = cursor_pos; // Follow cursor continuously

                // Update
                world.system_particles(dt);

                // Draw
                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1280.0, 800.0), ColorF::new(0.05, 0.05, 0.05, 1.0));
                
                // Draw Emitter location
                dl.add_circle(world.transforms[idx].local_position, 5.0, ColorF::white(), true);

                // Draw Particles
                dl.add_particles(0, world.particle_system.particles.clone());
                
                // Simple debug print
                println!("Particles: {}", world.particle_system.particles.len());

                backend.render(&dl, 1280, 800);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
