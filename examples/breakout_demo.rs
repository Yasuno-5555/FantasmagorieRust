use fanta_rust::prelude::*;
use fanta_rust::backend::GraphicsBackend;
use fanta_rust::game::particles::{Particle, ParticleSystem};
use winit::event::{Event, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowAttributes;
use std::sync::Arc;
use std::time::Instant;

// ==================================================================================
// Simple RNG (Xorshift)
// ==================================================================================
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: if seed == 0 { 123456789 } else { seed } }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    fn vec2_range(&mut self, min: f32, max: f32) -> Vec2 {
        Vec2::new(self.range(min, max), self.range(min, max))
    }
    
    /// Random point in unit circle
    fn in_unit_circle(&mut self) -> Vec2 {
        let angle = self.range(0.0, std::f32::consts::PI * 2.0);
        let r = self.next_f32().sqrt();
        Vec2::new(angle.cos() * r, angle.sin() * r)
    }
}

// ==================================================================================
// Constants & Types
// ==================================================================================
const SCREEN_WIDTH: f32 = 1280.0;
const SCREEN_HEIGHT: f32 = 900.0; // Slightly taller for room
const PADDLE_WIDTH: f32 = 120.0;
const PADDLE_HEIGHT: f32 = 24.0;
const BALL_RADIUS: f32 = 12.0;
const BRICK_ROWS: usize = 8;
const BRICK_COLS: usize = 14;
const BRICK_HEIGHT: f32 = 30.0;
const BRICK_SPACING: f32 = 8.0;

#[derive(Clone, Copy)]
struct Ball {
    pos: Vec2,
    vel: Vec2,
    active: bool,
}

struct Paddle {
    pos: Vec2,
    width: f32,
    target_x: f32, // For smooth movement
}

struct Brick {
    rect: Rectangle,
    color: ColorF,
    active: bool,
    health: u32,
}

// ==================================================================================
// Main Game State
// ==================================================================================
struct Game {
    ball: Ball,
    paddle: Paddle,
    bricks: Vec<Brick>,
    particles: ParticleSystem,
    rng: Rng,
    
    // Juice State
    screenshake_timer: f32,
    screenshake_intensity: f32,
    hitlag_timer: f32,
    
    // Inputs
    input_left: bool,
    input_right: bool,
    mouse_pos: Vec2,
    using_mouse: bool,
}

impl Game {
    fn new() -> Self {
        let mut rng = Rng::new(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u32);
        
        // Generate Bricks
        let mut bricks = Vec::new();
        let total_brick_width = (SCREEN_WIDTH - (BRICK_COLS + 1) as f32 * BRICK_SPACING) / BRICK_COLS as f32;
        
        let colors = [
            ColorF::new(4.0, 0.2, 0.2, 1.0), // HDR Red
            ColorF::new(4.0, 2.0, 0.0, 1.0), // HDR Orange
            ColorF::new(4.0, 4.0, 0.0, 1.0), // HDR Yellow
            ColorF::new(0.2, 4.0, 0.2, 1.0), // HDR Green
            ColorF::new(0.0, 4.0, 4.0, 1.0), // HDR Cyan
            ColorF::new(0.2, 0.2, 5.0, 1.0), // HDR Blue
            ColorF::new(3.0, 0.0, 4.0, 1.0), // HDR Purple
            ColorF::new(4.0, 0.0, 2.0, 1.0), // HDR Magenta
        ];

        for r in 0..BRICK_ROWS {
            for c in 0..BRICK_COLS {
                let x = BRICK_SPACING + c as f32 * (total_brick_width + BRICK_SPACING);
                let y = BRICK_SPACING + 80.0 + r as f32 * (BRICK_HEIGHT + BRICK_SPACING);
                bricks.push(Brick {
                    rect: Rectangle::new(x, y, total_brick_width, BRICK_HEIGHT),
                    color: colors[r % colors.len()],
                    active: true,
                    health: 1,
                });
            }
        }

        Self {
            ball: Ball {
                pos: Vec2::new(SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0),
                vel: Vec2::new(300.0, -300.0),
                active: false,
            },
            paddle: Paddle {
                pos: Vec2::new(SCREEN_WIDTH / 2.0, SCREEN_HEIGHT - 60.0),
                width: PADDLE_WIDTH,
                target_x: SCREEN_WIDTH / 2.0,
            },
            bricks,
            particles: ParticleSystem::new(20000),
            rng,
            screenshake_timer: 0.0,
            screenshake_intensity: 0.0,
            hitlag_timer: 0.0,
            input_left: false,
            input_right: false,
            mouse_pos: Vec2::ZERO,
            using_mouse: true,
        }
    }

    fn update(&mut self, dt: f32) {
        // Hitlag Logic
        if self.hitlag_timer > 0.0 {
            self.hitlag_timer -= dt;
            if self.hitlag_timer > 0.0 {
                return; // Stop time!
            }
        }
    
        // Screenshake decay
        if self.screenshake_timer > 0.0 {
            self.screenshake_timer -= dt;
            if self.screenshake_timer <= 0.0 {
                self.screenshake_intensity = 0.0;
            }
        }

        // Paddle Movement
        let speed = 800.0 * dt;
        if self.input_left {
            self.paddle.target_x -= speed;
            self.using_mouse = false;
        }
        if self.input_right {
            self.paddle.target_x += speed;
            self.using_mouse = false;
        }
        if self.using_mouse {
            self.paddle.target_x = self.mouse_pos.x;
        }

        // Smooth paddle follow
        self.paddle.pos.x += (self.paddle.target_x - self.paddle.pos.x) * 20.0 * dt;
        self.paddle.pos.x = self.paddle.pos.x.clamp(self.paddle.width/2.0, SCREEN_WIDTH - self.paddle.width/2.0);
        let paddle_rect = Rectangle::new(self.paddle.pos.x - self.paddle.width/2.0, self.paddle.pos.y - PADDLE_HEIGHT/2.0, self.paddle.width, PADDLE_HEIGHT);

        // Ball Logic
        if self.ball.active {
            // Ball Trail
            let trail_rate = (self.ball.vel.length() / 20.0).max(1.0); // More speed = more particles
            for _ in 0..trail_rate as usize {
                 self.particles.spawn(Particle {
                    position: self.ball.pos + self.rng.in_unit_circle() * 4.0,
                    velocity: self.ball.vel.normalized() * -50.0 + self.rng.in_unit_circle() * 10.0,
                    color: ColorF::new(0.5, 0.8, 1.0, 0.8), // Cyan trail
                    life: 0.3,
                    max_life: 0.3,
                    size: self.rng.range(3.0, 8.0),
                });
            }
        
            // Move Ball
            self.ball.pos = self.ball.pos + self.ball.vel * dt;

            // Wall Collisions
            if self.ball.pos.x < BALL_RADIUS {
                self.ball.pos.x = BALL_RADIUS;
                self.ball.vel.x *= -1.0;
                self.trigger_wall_hit(Vec2::new(-1.0, 0.0));
            }
            if self.ball.pos.x > SCREEN_WIDTH - BALL_RADIUS {
                self.ball.pos.x = SCREEN_WIDTH - BALL_RADIUS;
                self.ball.vel.x *= -1.0;
                self.trigger_wall_hit(Vec2::new(1.0, 0.0));
            }
            if self.ball.pos.y < BALL_RADIUS {
                self.ball.pos.y = BALL_RADIUS;
                self.ball.vel.y *= -1.0;
                self.trigger_wall_hit(Vec2::new(0.0, -1.0));
            }
            
            // Paddle Collision
            let ball_rect = Rectangle::new(self.ball.pos.x - BALL_RADIUS, self.ball.pos.y - BALL_RADIUS, BALL_RADIUS*2.0, BALL_RADIUS*2.0);
            if check_collision(&ball_rect, &paddle_rect) && self.ball.vel.y > 0.0 {
                 // Hit paddle
                 self.ball.vel.y *= -1.0;
                 // Deflect based on hit position
                 let diff = self.ball.pos.x - self.paddle.pos.x;
                 self.ball.vel.x += diff * 5.0; 
                 
                 // Normalize speed and increase slightly
                 let mut speed = self.ball.vel.length();
                 speed = (speed + 20.0).min(1500.0); // Cap speed
                 self.ball.vel = self.ball.vel.normalized() * speed;
                 
                 self.trigger_paddle_hit();
            }

            // Brick Collision
            let mut hit_data = None;
            for brick in &mut self.bricks {
                if !brick.active { continue; }
                
                if check_collision(&ball_rect, &brick.rect) {
                    brick.active = false;
                    
                    // Simple reflection (naive)
                    // Determine side
                    let overlap_x = (ball_rect.center().x - brick.rect.center().x).abs();
                    let overlap_y = (ball_rect.center().y - brick.rect.center().y).abs();
                    let w = (ball_rect.w + brick.rect.w) / 2.0;
                    let h = (ball_rect.h + brick.rect.h) / 2.0;
                    
                    if (overlap_x / w) > (overlap_y / h) {
                        self.ball.vel.x *= -1.0;
                    } else {
                        self.ball.vel.y *= -1.0;
                    }
                    
                    hit_data = Some((brick.rect.center(), brick.color));
                    break; // One brick per frame to prevent tunneling issues in simple physics
                }
            }

            if let Some((pos, color)) = hit_data {
                self.trigger_brick_hit(pos, color);
            }

            // Death
            if self.ball.pos.y > SCREEN_HEIGHT + 50.0 {
                self.ball.active = false;
                self.screenshake(10.0, 0.5); // Big shake on death
            }

        } else {
            // Stuck to paddle
            self.ball.pos = self.paddle.pos + Vec2::new(0.0, -30.0);
            if self.input_left || self.input_right || self.using_mouse { // Launch on input
                // self.ball.active = true;
                // self.ball.vel = Vec2::new(0.0, -600.0);
            }
        }
        
        // Update particles
        self.particles.update(dt);
    }
    
    fn launch_ball(&mut self) {
        if !self.ball.active {
            self.ball.active = true;
            let angle = self.rng.range(-0.5, 0.5); // Radians
            self.ball.vel = Vec2::new(angle.sin(), -angle.cos()) * 600.0;
        }
    }

    fn screenshake(&mut self, intensity: f32, duration: f32) {
        if intensity > self.screenshake_intensity {
            self.screenshake_intensity = intensity;
            self.screenshake_timer = duration;
        }
    }

    fn trigger_wall_hit(&mut self, normal: Vec2) {
        self.screenshake(2.0, 0.1);
        // Sparks
        for _ in 0..10 {
             self.particles.spawn(Particle {
                position: self.ball.pos,
                velocity: normal * 100.0 + self.rng.in_unit_circle() * 100.0,
                color: ColorF::new(0.8, 0.8, 1.0, 1.0),
                life: 0.5,
                max_life: 0.5,
                size: self.rng.range(2.0, 5.0),
            });
        }
    }
    
    fn trigger_paddle_hit(&mut self) {
         self.screenshake(5.0, 0.2);
         // Burst
         for _ in 0..20 {
             self.particles.spawn(Particle {
                position: self.ball.pos,
                velocity: Vec2::new(0.0, -1.0) * 200.0 + self.rng.in_unit_circle() * 100.0,
                color: ColorF::new(0.0, 5.0, 5.0, 1.0), // Cyan HDR
                life: 0.6,
                max_life: 0.6,
                size: self.rng.range(3.0, 6.0),
            });
        }
    }

    fn trigger_brick_hit(&mut self, pos: Vec2, color: ColorF) {
        self.screenshake(8.0, 0.15);
        self.hitlag_timer = 0.05; // 50ms freeze
        
        // Explosion
        for _ in 0..40 {
             let speed = self.rng.range(100.0, 500.0);
             let dir = self.rng.in_unit_circle();
             self.particles.spawn(Particle {
                position: pos,
                velocity: dir * speed,
                color: color,
                life: self.rng.range(0.5, 1.2),
                max_life: 1.2,
                size: self.rng.range(4.0, 10.0),
            });
        }
    }
}

fn check_collision(r1: &Rectangle, r2: &Rectangle) -> bool {
    r1.x < r2.x + r2.w &&
    r1.x + r1.w > r2.x &&
    r1.y < r2.y + r2.h &&
    r1.y + r1.h > r2.y
}

// ==================================================================================
// Main Entry
// ==================================================================================
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window_attrs = WindowAttributes::default()
        .with_title("Fantasmagorie - Luxurious Breakout")
        .with_inner_size(winit::dpi::LogicalSize::new(SCREEN_WIDTH, SCREEN_HEIGHT));
    let window = Arc::new(event_loop.create_window(window_attrs)?);

    // Initialize Engine with Cinematic Config for Bloom
    let mut config = EngineConfig::cinematic();
    config.cinematic.bloom = fanta_rust::config::Bloom::Cinematic;
    config.cinematic.blur_radius = 0.6;
    config.cinematic.tonemap = fanta_rust::config::Tonemap::Aces;
    
    let mut backend = fanta_rust::backend::wgpu::WgpuBackend::new_async(window.clone(), 1280, 900, 1.0).unwrap();
    backend.set_cinematic_config(config.cinematic.clone());

    let mut game = Game::new();
    let mut last_time = Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                game.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
                game.using_mouse = true;
            }
            
            Event::WindowEvent { event: WindowEvent::MouseInput { state: ElementState::Pressed, .. }, .. } => {
                game.launch_ball();
            }

            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. } => {
               if let PhysicalKey::Code(code) = event.physical_key {
                   let pressed = event.state == ElementState::Pressed;
                   match code {
                       KeyCode::KeyA | KeyCode::ArrowLeft => game.input_left = pressed,
                       KeyCode::KeyD | KeyCode::ArrowRight => game.input_right = pressed,
                       KeyCode::Space => if pressed { game.launch_ball(); },
                       _ => {}
                   }
               }
            }

            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32().min(0.05); // Cap at 20fps for physics stability
                last_time = now;
                
                game.update(dt);
                
                // Rendering
                let mut dl = fanta_rust::DrawList::new();
                
                // Apply Screenshake
                let shake_offset = if game.screenshake_intensity > 0.0 {
                    Vec2::new(
                        game.rng.range(-1.0, 1.0) * game.screenshake_intensity,
                        game.rng.range(-1.0, 1.0) * game.screenshake_intensity
                    )
                } else {
                    Vec2::ZERO
                };
                
                dl.push_transform(shake_offset, 1.0);
                
                // Background (Dark Grid)
                dl.add_rect(Vec2::ZERO, Vec2::new(SCREEN_WIDTH, SCREEN_HEIGHT), ColorF::new(0.01, 0.01, 0.02, 1.0));
                
                // Bricks
                for brick in &game.bricks {
                    if brick.active {
                        dl.add_rect(brick.rect.pos(), brick.rect.size(), brick.color);
                        // Add inner glow (lighter center)
                        let center_rect = Rectangle::new(brick.rect.x + 4.0, brick.rect.y + 4.0, brick.rect.w - 8.0, brick.rect.h - 8.0);
                        dl.add_rect(center_rect.pos(), center_rect.size(), brick.color.lighten(0.5));
                    }
                }
                
                // Paddle
                let paddle_rect = Rectangle::new(game.paddle.pos.x - game.paddle.width/2.0, game.paddle.pos.y - PADDLE_HEIGHT/2.0, game.paddle.width, PADDLE_HEIGHT);
                dl.add_rect(paddle_rect.pos(), paddle_rect.size(), ColorF::new(0.2, 2.0, 4.0, 1.0));
                
                // Ball
                if game.ball.active || true { // Always draw ball
                    dl.add_circle(game.ball.pos, BALL_RADIUS, ColorF::new(4.0, 4.0, 4.0, 1.0), true);
                    // Glow halo
                    dl.add_circle(game.ball.pos, BALL_RADIUS * 2.0, ColorF::new(1.0, 1.0, 2.0, 0.2), true);
                }
                
                // Particles
                dl.add_particles(0, game.particles.particles.clone());
                
                dl.pop_transform();

                backend.render(&dl, 1280, 900);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
