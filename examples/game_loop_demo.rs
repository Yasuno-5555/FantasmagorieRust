use fanta_rust::prelude::*;
use fanta_rust::game::scene::{Scene, SceneManager, SceneTransition};
use fanta_rust::game::input::{Action, ActionMap, ActionState};
use fanta_rust::backend::GraphicsBackend;
use winit::event::{Event, WindowEvent, ElementState, KeyEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowAttributes;
use std::sync::Arc;
use std::time::Instant;

const SCREEN_WIDTH: f32 = 800.0;
const SCREEN_HEIGHT: f32 = 600.0;

// ==================================================================================
// Scenes
// ==================================================================================

struct TitleScene;
impl Scene for TitleScene {
    fn update(&mut self, _dt: f32, input: &ActionState) -> SceneTransition {
        if input.is_active(Action::Start) {
            return SceneTransition::Replace(Box::new(GameScene::new()));
        }
        SceneTransition::None
    }

    fn draw(&mut self, dl: &mut DrawList) {
        dl.add_rect(Vec2::ZERO, Vec2::new(SCREEN_WIDTH, SCREEN_HEIGHT), ColorF::new(0.1, 0.1, 0.2, 1.0));
        dl.add_text_simple(Vec2::new(260.0, 200.0), "FANTASMAGORIE", 40.0, ColorF::new(0.5, 0.8, 1.0, 1.0));
        dl.add_text_simple(Vec2::new(250.0, 300.0), "Press SPACE to Start", 20.0, ColorF::white());
        dl.add_text_simple(Vec2::new(10.0, 570.0), "Scene: Title", 16.0, ColorF::new(0.5, 0.5, 0.5, 1.0));
    }
}

struct GameScene {
    ball_pos: Vec2,
    ball_vel: Vec2,
}

impl GameScene {
    fn new() -> Self {
        Self {
            ball_pos: Vec2::new(400.0, 300.0),
            ball_vel: Vec2::new(200.0, -200.0),
        }
    }
}

impl Scene for GameScene {
    fn update(&mut self, dt: f32, input: &ActionState) -> SceneTransition {
        if input.is_active(Action::Back) {
            return SceneTransition::Replace(Box::new(GameOverScene));
        }

        // Simple bounce logic
        self.ball_pos.x += self.ball_vel.x * dt;
        self.ball_pos.y += self.ball_vel.y * dt;

        if self.ball_pos.x < 10.0 || self.ball_pos.x > SCREEN_WIDTH - 10.0 {
            self.ball_vel.x *= -1.0;
        }
        if self.ball_pos.y < 10.0 || self.ball_pos.y > SCREEN_HEIGHT - 10.0 {
            self.ball_vel.y *= -1.0;
        }

        SceneTransition::None
    }

    fn draw(&mut self, dl: &mut DrawList) {
        dl.add_rect(Vec2::ZERO, Vec2::new(SCREEN_WIDTH, SCREEN_HEIGHT), ColorF::new(0.05, 0.05, 0.1, 1.0));
        dl.add_circle(self.ball_pos, 10.0, ColorF::red(), true);
        dl.add_text_simple(Vec2::new(320.0, 50.0), "GAME RUNNING", 24.0, ColorF::green());
        dl.add_text_simple(Vec2::new(10.0, 570.0), "Press ESC to End", 16.0, ColorF::new(0.5, 0.5, 0.5, 1.0));
    }
}

struct GameOverScene;
impl Scene for GameOverScene {
    fn update(&mut self, _dt: f32, input: &ActionState) -> SceneTransition {
        if input.is_active(Action::Restart) {
            return SceneTransition::Replace(Box::new(TitleScene));
        }
        SceneTransition::None
    }

    fn draw(&mut self, dl: &mut DrawList) {
        dl.add_rect(Vec2::ZERO, Vec2::new(SCREEN_WIDTH, SCREEN_HEIGHT), ColorF::new(0.2, 0.0, 0.0, 1.0));
        dl.add_text_simple(Vec2::new(300.0, 250.0), "GAME OVER", 40.0, ColorF::red());
        dl.add_text_simple(Vec2::new(280.0, 350.0), "Press R to Restart", 20.0, ColorF::white());
    }
}

// ==================================================================================
// Main
// ==================================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window_attrs = WindowAttributes::default()
        .with_title("Fantasmagorie - Scene Demo")
        .with_inner_size(winit::dpi::LogicalSize::new(SCREEN_WIDTH, SCREEN_HEIGHT));
    let window = Arc::new(event_loop.create_window(window_attrs)?);

    let mut backend = fanta_rust::backend::wgpu::WgpuBackend::new_async(window.clone(), SCREEN_WIDTH as u32, SCREEN_HEIGHT as u32, 1.0).unwrap();

    // Setup Input Map
    let mut map = ActionMap::new_default();
    // Add Start/Back/Restart mappings
    // Check winit KeyCodes or use general scan codes if possible, 
    // but winit PhysicalKey::Code(KeyCode::Space) etc is robust.
    // map.keyboard uses u32 scancodes traditionally in this engine (from legacy code)
    // But `breakout_demo` uses KeyCode matching.
    // Here we need to map KeyCode -> u32 if ActionMap uses u32.
    // ActionMap definition: pub keyboard: HashMap<u32, Action>
    // So we need to decide on a mapping convention.
    // Let's assume standard scancodes or just map KeyCode to u32 arbitrarily here.
    
    // Bindings (Arbitrary u32 for internal map, we just need to match in loop)
    map.keyboard.insert(100, Action::Start);   // Space
    map.keyboard.insert(101, Action::Back);    // Escape
    map.keyboard.insert(102, Action::Restart); // R

    let mut action_state = ActionState::default();
    let mut sm = SceneManager::new(Box::new(TitleScene));
    let mut last_time = Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. } => {
               if let PhysicalKey::Code(code) = event.physical_key {
                   let pressed = event.state == ElementState::Pressed;
                   
                   // Map KeyCode to our u32 ID
                   let id = match code {
                       KeyCode::Space => 100,
                       KeyCode::Escape => 101,
                       KeyCode::KeyR => 102,
                       _ => 0,
                   };
                   
                   if let Some(action) = map.keyboard.get(&id) {
                       action_state.set_active(*action, pressed);
                   }
               }
            }

            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.05);
                last_time = now;
                
                // Update Scene Manager
                if !sm.update(dt, &action_state) {
                    elwt.exit(); // Quit
                }
                
                // Draw
                let mut dl = fanta_rust::DrawList::new();
                sm.draw(&mut dl);
                
                backend.render(&dl, SCREEN_WIDTH as u32, SCREEN_HEIGHT as u32);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
