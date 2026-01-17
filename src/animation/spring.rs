//! Spring physics animation
//!
//! Provides spring-based animation with:
//! - Configurable stiffness, damping, and mass
//! - Preset library for common behaviors
//! - Critical damping calculation

/// Spring configuration
#[derive(Debug, Clone, Copy)]
pub struct SpringConfig {
    /// Spring stiffness (higher = faster)
    pub stiffness: f32,
    /// Damping ratio (1.0 = critical damping)
    pub damping: f32,
    /// Mass of the object
    pub mass: f32,
}

impl SpringConfig {
    pub fn new(stiffness: f32, damping: f32, mass: f32) -> Self {
        Self {
            stiffness,
            damping,
            mass,
        }
    }

    /// Calculate critical damping coefficient
    pub fn critical_damping(&self) -> f32 {
        2.0 * (self.stiffness * self.mass).sqrt()
    }

    /// Check if this spring is critically damped
    pub fn is_critically_damped(&self) -> bool {
        (self.damping - self.critical_damping()).abs() < 0.001
    }

    /// Check if this spring is overdamped (no oscillation)
    pub fn is_overdamped(&self) -> bool {
        self.damping > self.critical_damping()
    }

    /// Check if this spring is underdamped (oscillates)
    pub fn is_underdamped(&self) -> bool {
        self.damping < self.critical_damping()
    }
}

/// Preset spring configurations
pub mod presets {
    use super::SpringConfig;

    /// Default spring - balanced feel
    pub const DEFAULT: SpringConfig = SpringConfig {
        stiffness: 170.0,
        damping: 26.0,
        mass: 1.0,
    };

    /// Gentle spring - slow and smooth
    pub const GENTLE: SpringConfig = SpringConfig {
        stiffness: 120.0,
        damping: 14.0,
        mass: 1.0,
    };

    /// Wobbly spring - bouncy with oscillation
    pub const WOBBLY: SpringConfig = SpringConfig {
        stiffness: 180.0,
        damping: 12.0,
        mass: 1.0,
    };

    /// Stiff spring - fast and snappy
    pub const STIFF: SpringConfig = SpringConfig {
        stiffness: 210.0,
        damping: 20.0,
        mass: 1.0,
    };

    /// Slow spring - very slow movement
    pub const SLOW: SpringConfig = SpringConfig {
        stiffness: 280.0,
        damping: 60.0,
        mass: 1.0,
    };

    /// Molasses spring - extremely slow
    pub const MOLASSES: SpringConfig = SpringConfig {
        stiffness: 280.0,
        damping: 120.0,
        mass: 1.0,
    };

    /// No wobble - critically damped, no overshoot
    pub const NO_WOBBLE: SpringConfig = SpringConfig {
        stiffness: 170.0,
        damping: 26.0,
        mass: 1.0,
    };

    /// iOS default spring
    pub const IOS_DEFAULT: SpringConfig = SpringConfig {
        stiffness: 400.0,
        damping: 30.0,
        mass: 1.0,
    };

    /// Material Design spring
    pub const MATERIAL: SpringConfig = SpringConfig {
        stiffness: 600.0,
        damping: 40.0,
        mass: 1.0,
    };

    /// Bouncy spring - lots of oscillation
    pub const BOUNCY: SpringConfig = SpringConfig {
        stiffness: 300.0,
        damping: 10.0,
        mass: 1.0,
    };
}

/// Spring animator state
#[derive(Debug, Clone)]
pub struct Spring {
    config: SpringConfig,
    /// Current value
    value: f32,
    /// Target value
    target: f32,
    /// Current velocity
    velocity: f32,
    /// Rest threshold (stops when delta < threshold)
    rest_threshold: f32,
    /// Velocity threshold (stops when velocity < threshold)
    velocity_threshold: f32,
}

impl Spring {
    /// Create a new spring with given config
    pub fn new(config: SpringConfig) -> Self {
        Self {
            config,
            value: 0.0,
            target: 0.0,
            velocity: 0.0,
            rest_threshold: 0.001,
            velocity_threshold: 0.001,
        }
    }

    /// Create with preset
    pub fn with_preset(preset: SpringConfig) -> Self {
        Self::new(preset)
    }

    /// Set the current value
    pub fn set_value(&mut self, value: f32) {
        self.value = value;
    }

    /// Set the target value
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Set both value and target (jump to position)
    pub fn jump_to(&mut self, value: f32) {
        self.value = value;
        self.target = value;
        self.velocity = 0.0;
    }

    /// Set initial velocity (for fling gestures)
    pub fn set_velocity(&mut self, velocity: f32) {
        self.velocity = velocity;
    }

    /// Update the spring simulation
    pub fn update(&mut self, delta_seconds: f32) -> f32 {
        if self.is_at_rest() {
            return self.value;
        }

        // Spring force: F = -k * x
        // Damping force: F = -c * v
        // F = ma, so a = F/m

        let displacement = self.value - self.target;
        let spring_force = -self.config.stiffness * displacement;
        let damping_force = -self.config.damping * self.velocity;
        let acceleration = (spring_force + damping_force) / self.config.mass;

        self.velocity += acceleration * delta_seconds;
        self.value += self.velocity * delta_seconds;

        self.value
    }

    /// Check if the spring is at rest
    pub fn is_at_rest(&self) -> bool {
        let displacement = (self.value - self.target).abs();
        displacement < self.rest_threshold && self.velocity.abs() < self.velocity_threshold
    }

    /// Get current value
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Get current velocity
    pub fn velocity(&self) -> f32 {
        self.velocity
    }

    /// Get target value
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Get the config
    pub fn config(&self) -> &SpringConfig {
        &self.config
    }

    /// Set rest thresholds
    pub fn set_thresholds(&mut self, rest: f32, velocity: f32) {
        self.rest_threshold = rest;
        self.velocity_threshold = velocity;
    }
}

/// 2D Spring for Vec2 animation
#[derive(Debug, Clone)]
pub struct Spring2D {
    x: Spring,
    y: Spring,
}

impl Spring2D {
    /// Create a new 2D spring
    pub fn new(config: SpringConfig) -> Self {
        Self {
            x: Spring::new(config),
            y: Spring::new(config),
        }
    }

    /// Set current position
    pub fn set_value(&mut self, x: f32, y: f32) {
        self.x.set_value(x);
        self.y.set_value(y);
    }

    /// Set target position
    pub fn set_target(&mut self, x: f32, y: f32) {
        self.x.set_target(x);
        self.y.set_target(y);
    }

    /// Jump to position
    pub fn jump_to(&mut self, x: f32, y: f32) {
        self.x.jump_to(x);
        self.y.jump_to(y);
    }

    /// Set velocity
    pub fn set_velocity(&mut self, vx: f32, vy: f32) {
        self.x.set_velocity(vx);
        self.y.set_velocity(vy);
    }

    /// Update and return new position
    pub fn update(&mut self, delta_seconds: f32) -> (f32, f32) {
        (self.x.update(delta_seconds), self.y.update(delta_seconds))
    }

    /// Check if at rest
    pub fn is_at_rest(&self) -> bool {
        self.x.is_at_rest() && self.y.is_at_rest()
    }

    /// Get current position
    pub fn value(&self) -> (f32, f32) {
        (self.x.value(), self.y.value())
    }

    /// Get current velocity
    pub fn velocity(&self) -> (f32, f32) {
        (self.x.velocity(), self.y.velocity())
    }
}

/// Color spring for smooth color transitions
#[derive(Debug, Clone)]
pub struct SpringColor {
    r: Spring,
    g: Spring,
    b: Spring,
    a: Spring,
}

impl SpringColor {
    /// Create a new color spring
    pub fn new(config: SpringConfig) -> Self {
        Self {
            r: Spring::new(config),
            g: Spring::new(config),
            b: Spring::new(config),
            a: Spring::new(config),
        }
    }

    /// Set current color (RGBA 0-1)
    pub fn set_value(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.r.set_value(r);
        self.g.set_value(g);
        self.b.set_value(b);
        self.a.set_value(a);
    }

    /// Set target color
    pub fn set_target(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.r.set_target(r);
        self.g.set_target(g);
        self.b.set_target(b);
        self.a.set_target(a);
    }

    /// Update and return new color
    pub fn update(&mut self, delta_seconds: f32) -> (f32, f32, f32, f32) {
        (
            self.r.update(delta_seconds).clamp(0.0, 1.0),
            self.g.update(delta_seconds).clamp(0.0, 1.0),
            self.b.update(delta_seconds).clamp(0.0, 1.0),
            self.a.update(delta_seconds).clamp(0.0, 1.0),
        )
    }

    /// Check if at rest
    pub fn is_at_rest(&self) -> bool {
        self.r.is_at_rest() && self.g.is_at_rest() && self.b.is_at_rest() && self.a.is_at_rest()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spring_reaches_target() {
        let mut spring = Spring::new(presets::STIFF);
        spring.set_value(0.0);
        spring.set_target(100.0);

        // Simulate for 1 second
        for _ in 0..60 {
            spring.update(1.0 / 60.0);
        }

        assert!((spring.value() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_spring_at_rest() {
        let mut spring = Spring::new(presets::DEFAULT);
        spring.set_value(100.0);
        spring.set_target(100.0);

        assert!(spring.is_at_rest());
    }

    #[test]
    fn test_presets() {
        assert!(presets::WOBBLY.is_underdamped());
        assert!(presets::MOLASSES.is_overdamped());
    }
}
