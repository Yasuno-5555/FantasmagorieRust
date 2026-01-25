use fanta_rust::prelude::*;
use fanta_rust::game::*;
use fanta_rust::backend::GraphicsBackend;
use fanta_rust::draw::DrawList;
use fanta_rust::renderer::types::Rect;

struct NullBackend;
impl GraphicsBackend for NullBackend {
    fn name(&self) -> &str { "Null" }
    fn render(&mut self, _dl: &DrawList, _width: u32, _height: u32) {}
    fn update_font_texture(&mut self, _w: u32, _h: u32, _d: &[u8]) {}
}

fn main() {
    println!("Fantasmagorie Game Engine - Interaction Kernel Demo");
    println!("====================================================");

    // 1. Setup Game World
    let mut world = World::new();
    
    // Spawn Player
    let _player_id = world.spawn();
    world.transforms[0].position = Vec2::new(0.0, 0.0);
    world.set_collider(0, Collider::new(50.0, 50.0));
    
    // Spawn Reactors (Entities that react to the player)
    for i in 1..6 {
        world.spawn();
        let angle = i as f32 * 1.2;
        world.transforms[i].position = Vec2::new(angle.cos() * 300.0, angle.sin() * 300.0);
        world.set_collider(i, Collider::new(80.0, 80.0));
    }

    // 2. Setup Camera
    let camera = Camera::new(1280.0, 720.0);
    
    // 3. Setup Renderer
    let mut renderer = Renderer::new_lite(Box::new(NullBackend));

    // 4. Simulation Constants
    let dt = 1.0 / 60.0;
    let proximity_radius = 150.0;

    println!("Simulating 120 frames of interaction...");

    for frame_count in 0..120 {
        // Move player in a circle to trigger interactions
        let t = (frame_count as f32) * 0.1;
        world.transforms[0].position = Vec2::new(t.cos() * 300.0, t.sin() * 300.0);

        // Run interaction system
        world.system_interaction(proximity_radius);

        let mut frame = renderer.begin_frame();
        frame.begin_world(&camera);

        for i in 0..world.ids.len() {
            let pos = world.transforms[i].position;
            let state = &world.interaction_states[i];
            
            // --- REACTIVE RENDERING ---
            let base_color = if i == 0 {
                ColorF::new(1.0, 1.0, 1.0, 1.0) // Player is white
            } else {
                ColorF::new(0.2, 0.2, 0.3, 1.0) // Reactors are dark blue
            };

            // Calculate reaction effects
            let mut color = base_color;
            let mut glow_strength = 0.0;
            let mut elevation = 0.0;
            let mut corner_radius = 4.0;

            if i > 0 {
                // If touched: pulse color and increase thickness
                if state.is_touched {
                    color = ColorF::new(0.4, 0.8, 1.0, 1.0);
                    corner_radius = 20.0; // Morph to more rounded
                }
                
                // If near: glow based on proximity
                if state.is_near {
                    glow_strength = state.proximity * 15.0;
                    elevation = state.proximity * 10.0;
                }
            }

            // Draw using high-feature SDF shapes
            frame.draw(
                Rect::new(pos.x - 40.0, pos.y - 40.0, 80.0, 80.0),
                color
            )
            .rounded(corner_radius)
            .glow(glow_strength, ColorF::new(0.0, 0.5, 1.0, 0.5))
            .elevation(elevation)
            .submit();
        }

        frame.end_world();
        renderer.end_frame(frame, 1280, 720);

        if frame_count % 30 == 0 {
            let near_count = world.interaction_states.iter().filter(|s| s.is_near).count();
            let touch_count = world.interaction_states.iter().filter(|s| s.is_touched).count();
            println!("Frame {}: Near={}, Touched={}", frame_count, near_count, touch_count);
        }
    }

    println!("[OK] Interaction Demo completed. The world is alive.");
}
