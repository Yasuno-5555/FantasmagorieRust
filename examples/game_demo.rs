use fanta_rust::prelude::*;
use fanta_rust::game::*;
use fanta_rust::backend::GraphicsBackend;
use fanta_rust::draw::DrawList;
use std::collections::HashMap;

struct NullBackend;
impl GraphicsBackend for NullBackend {
    fn name(&self) -> &str { "Null" }
    fn render(&mut self, _dl: &DrawList, _width: u32, _height: u32) {}
    fn update_font_texture(&mut self, _w: u32, _h: u32, _d: &[u8]) {}
}

fn main() {
    println!("Fantasmagorie Game Engine - Life & Motion Demo");
    println!("==============================================");

    // 1. Setup Game World
    let mut world = World::new();
    
    // Spawn Player
    let player_id = world.spawn();
    world.transforms[0].position = Vec2::new(0.0, 0.0);
    
    // 2. Setup Animation Clips (Visual Translations)
    let mut clips = HashMap::new();
    
    // Idle Clip
    clips.insert(EntityState::Idle, AnimationClip {
        frames: vec![
            AnimationFrame { texture_id: 1, duration: 0.5, uv_rect: None },
            AnimationFrame { texture_id: 1, duration: 0.5, uv_rect: None },
        ],
        loop_clip: true,
    });
    
    // Walk Clip (Simulating frames)
    clips.insert(EntityState::Walk, AnimationClip {
        frames: vec![
            AnimationFrame { texture_id: 2, duration: 0.15, uv_rect: None },
            AnimationFrame { texture_id: 3, duration: 0.15, uv_rect: None },
            AnimationFrame { texture_id: 4, duration: 0.15, uv_rect: None },
            AnimationFrame { texture_id: 5, duration: 0.15, uv_rect: None },
        ],
        loop_clip: true,
    });

    // 3. Setup Input Mapping (Abstraction)
    let action_map = ActionMap::new_default();
    let mut action_state = ActionState::default();

    // 4. Setup Renderer
    let mut renderer = Renderer::new_lite(Box::new(NullBackend));
    let camera = Camera::new(1280.0, 720.0);

    // 5. Game Loop Mock
    let dt = 1.0 / 60.0;
    println!("Simulating 180 frames of Life & Motion...");

    for frame_count in 0..180 {
        // --- INPUT SYSTEM ---
        // Mocking user input: Holding "D" (Right) for a while, then stopping
        let moving = frame_count > 30 && frame_count < 150;
        action_state.set_active(Action::MoveRight, moving);

        // --- WORLD STATE LOGIC (The Brain) ---
        // Player moves based on Input Action Mapping
        let player_idx = 0;
        if action_state.is_active(Action::MoveRight) {
            world.transforms[player_idx].position.x += 200.0 * dt;
            world.entity_states[player_idx] = EntityState::Walk;
        } else {
            world.entity_states[player_idx] = EntityState::Idle;
        }

        // --- ANIMATION SYSTEM (The Visual Translator) ---
        world.system_animation(dt, &clips);

        // --- RENDERING ---
        let mut frame = renderer.begin_frame();
        frame.begin_world(&camera);

        for i in 0..world.ids.len() {
            let pos = world.transforms[i].position;
            let anim = &world.animations[i];
            
            // Draw Player using SpriteBuilder with Motion Morphing
            let player_sprite = Sprite::new(1); // Default texture
            
            SpriteBuilder::new(&mut frame, pos, Vec2::new(128.0, 128.0))
                .animate(anim.clone())
                .draw(&player_sprite, clips.get(&anim.state));
        }

        frame.end_world();
        renderer.end_frame(frame, 1280, 720);

        if frame_count % 30 == 0 {
            let state = world.entity_states[0];
            let morph = world.animations[0].morph_weight;
            println!("Frame {}: State={:?}, Pos={:?}, Morph={:.2}", 
                frame_count, state, world.transforms[0].position, morph);
        }
    }

    println!("[OK] Life & Motion Demo completed.");
    println!("Visuals successfully translated World Logic into SDF-morphed animation.");
}
