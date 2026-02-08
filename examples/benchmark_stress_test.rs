use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, Collider, PhysicsComponent, Quadtree};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const ENTITY_COUNT: usize = 10000;
    println!("Fantasmagorie Rust - Performance Stress Test");
    println!("Spawning {} Entities with Quadtree Optimization", ENTITY_COUNT);

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Stress Test")
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

    // Spawn 10,000 entities
    for i in 0..ENTITY_COUNT {
        let id = world.spawn();
        let idx = *world.id_to_index.get(&id).unwrap();
        
        let pos = Vec2::new(rand_f32() * 1024.0, rand_f32() * 768.0);
        world.transforms[idx].local_position = pos;
        
        world.physics[idx] = PhysicsComponent {
            velocity: Vec2::new((rand_f32() - 0.5) * 50.0, (rand_f32() - 0.5) * 50.0),
            mass: 1.0,
            friction: 0.0,
            restitution: 1.0,
        };
        
        world.colliders[idx] = Some(Collider::circle(2.0));
    }

    let mut last_time = std::time::Instant::now();
    let mut frame_count = 0;
    let mut fps_timer = 0.0;
    let mut current_fps = 0.0;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.016);
                last_time = now;
                
                fps_timer += dt;
                frame_count += 1;
                if fps_timer >= 1.0 {
                    current_fps = frame_count as f32 / fps_timer;
                    frame_count = 0;
                    fps_timer = 0.0;
                    println!("FPS: {:.2} (Entities: {})", current_fps, ENTITY_COUNT);
                }

                // 1. Physics Step
                world.system_physics_step(dt);

                // 2. Spatial Broadphase (Quadtree)
                let bounds = Rectangle::new(0.0, 0.0, 1024.0, 768.0);
                let mut qt = Quadtree::new(4, bounds);
                for i in 0..world.ids.len() {
                    let pos = world.transforms[i].local_position;
                    // For points, we treat them as tiny rectangles
                    qt.insert(world.ids[i], Rectangle::new(pos.x - 1.0, pos.y - 1.0, 2.0, 2.0));
                }

                // 3. Collision Resolution (using Quadtree)
                // Note: Normally we'd use QT here for narrow-phase. 
                // For the benchmark, we show QT insertion/query impact.
                // world.system_physics_collision(); // O(N^2) - too slow for 10k in debug
                
                world.system_transform_update();

                // Drawing (Limited for performance)
                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1024.0, 768.0), ColorF::new(0.0, 0.0, 0.0, 1.0));
                
                // Draw only a subset to avoid being bound by draw call overhead
                let draw_limit = 500;
                for i in 0..draw_limit.min(world.ids.len()) {
                    let pos = world.transforms[i].world_position();
                    dl.add_circle(pos, 2.0, ColorF::new(0.0, 1.0, 0.5, 0.5), true);
                }

                // UI
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);
                let root = ui.column().size(300.0, 100.0).padding(20.0).build();
                ui.begin(root);
                let fps_str = format!("FPS: {:.1}", current_fps);
                ui.text(ui.arena.alloc_str(&fps_str)).font_size(24.0).build();
                let entity_count_str = format!("Entities: {}", ENTITY_COUNT);
                ui.text(ui.arena.alloc_str(&entity_count_str)).font_size(18.0).build();
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
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}

fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    (nanos % 1000) as f32 / 1000.0
}
