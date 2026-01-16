//! Animation logic - Easing, Springs, and Transitions
//! Ported from animation.hpp and spring_physics.hpp

use crate::core::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Easing {
    Linear,
    QuadIn, QuadOut, QuadInOut,
    CubicIn, CubicOut, CubicInOut,
    ExpoIn, ExpoOut, ExpoInOut,
    ElasticIn, ElasticOut, ElasticInOut,
    BackIn, BackOut, BackInOut,
    Spring,
}

pub struct Spring {
    pub stiffness: f32, // k
    pub damping: f32,   // c
    pub mass: f32,      // m
    pub velocity: f32,
}

impl Default for Spring {
    fn default() -> Self {
        Self {
            stiffness: 170.0,
            damping: 26.0,
            mass: 1.0,
            velocity: 0.0,
        }
    }
}

impl Spring {
    pub fn update(&mut self, current: f32, target: f32, dt: f32) -> f32 {
        let f_spring = -self.stiffness * (current - target);
        let f_damping = -self.damping * self.velocity;
        let acceleration = (f_spring + f_damping) / self.mass;
        self.velocity += acceleration * dt;
        current + self.velocity * dt
    }
}

pub fn ease(t: f32, easing: Easing) -> f32 {
    match easing {
        Easing::Linear => t,
        Easing::QuadIn => t * t,
        Easing::QuadOut => t * (2.0 - t),
        Easing::QuadInOut => if t < 0.5 { 2.0 * t * t } else { -1.0 + (4.0 - 2.0 * t) * t },
        Easing::CubicIn => t * t * t,
        Easing::CubicOut => { let t = t - 1.0; t * t * t + 1.0 },
        Easing::CubicInOut => if t < 0.5 { 4.0 * t * t * t } else { (t - 1.0) * (2.0 * t - 2.0).powi(2) + 1.0 },
        Easing::ExpoIn => if t == 0.0 { 0.0 } else { (2.0f32).powf(10.0 * (t - 1.0)) },
        Easing::ExpoOut => if t == 1.0 { 1.0 } else { 1.0 - (2.0f32).powf(-10.0 * t) },
        Easing::ExpoInOut => {
            if t == 0.0 { 0.0 }
            else if t == 1.0 { 1.0 }
            else if t < 0.5 { (2.0f32).powf(20.0 * t - 10.0) / 2.0 }
            else { (2.0 - (2.0f32).powf(-20.0 * t + 10.0)) / 2.0 }
        }
        Easing::ElasticOut => {
            let c4 = (2.0 * std::f32::consts::PI) / 3.0;
            if t == 0.0 { 0.0 }
            else if t == 1.0 { 1.0 }
            else { (2.0f32).powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0 }
        }
        Easing::BackOut => {
            let c1 = 1.70158;
            let c3 = c1 + 1.0;
            let t = t - 1.0;
            1.0 + c3 * t.powi(3) + c1 * t.powi(2)
        }
        _ => t, // Fallback
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnimationStateEx {
    pub value: f32,
    pub start_value: f32,
    pub target: f32,
    pub velocity: f32,
    pub time: f32,     // Time since start or current progress
    pub duration: f32,
    pub easing: Easing,
}

impl Default for AnimationStateEx {
    fn default() -> Self {
        Self {
            value: 0.0,
            start_value: 0.0,
            target: 0.0,
            velocity: 0.0,
            time: 0.0,
            duration: 0.3,
            easing: Easing::ExpoOut,
        }
    }
}
