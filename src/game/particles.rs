use crate::core::{Vec2, ColorF};
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub color: ColorF,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticleEmitter {
    pub rate: f32, // particles per second
    pub accumulator: f32,
    pub active: bool,
    pub lifetime_range: [f32; 2],
    pub speed_range: [f32; 2],
    pub color_start: ColorF,
    pub color_end: ColorF,
    pub size_range: [f32; 2],
    pub cone_angle: f32, // Spread angle in radians
    pub direction: Vec2,
    pub one_shot: bool,
    pub burst_count: u32,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            rate: 10.0,
            accumulator: 0.0,
            active: true,
            lifetime_range: [0.5, 1.0],
            speed_range: [50.0, 100.0],
            color_start: ColorF::white(),
            color_end: ColorF::new(1.0, 1.0, 1.0, 0.0),
            size_range: [2.0, 5.0],
            cone_angle: std::f32::consts::PI * 2.0,
            direction: Vec2::new(0.0, -1.0),
            one_shot: false,
            burst_count: 0,
        }
    }
}

pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub max_particles: usize,
}

impl ParticleSystem {
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: Vec::with_capacity(max_particles),
            max_particles,
        }
    }

    pub fn update(&mut self, dt: f32) {
        let mut i = 0;
        while i < self.particles.len() {
            self.particles[i].life -= dt;
            if self.particles[i].life <= 0.0 {
                self.particles.swap_remove(i);
            } else {
                let p = &mut self.particles[i];
                p.position = p.position + p.velocity * dt;
                // Optional: Gravity
                // p.velocity.y += 98.0 * dt; 
                i += 1;
            }
        }
    }

    pub fn spawn(&mut self, p: Particle) {
        if self.particles.len() < self.max_particles {
            self.particles.push(p);
        }
    }
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new(10000)
    }
}

pub fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    // Simple hash to get pseudo-random float
    // Note: In real engine, use proper RNG.
    // This is a naive implementation that might not be random enough per frame.
    // Ideally use `rand` crate or passed seed.
    // For demo, we rely on rapid calls changing nanos.
    ((nanos % 1000000) as f32) / 1000000.0
}
