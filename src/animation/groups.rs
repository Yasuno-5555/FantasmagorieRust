//! Animation groups
//!
//! Provides sequential and parallel animation composition:
//! - Sequential groups (one after another)
//! - Parallel groups (all at once)
//! - Staggered groups (delayed starts)
//! - Chained callbacks

use std::collections::HashMap;

/// Animation completion callback
pub type CompletionCallback = Box<dyn FnOnce() + Send>;

/// Trait for animatable values
pub trait Animatable: Clone {
    fn lerp(&self, target: &Self, t: f32) -> Self;
}

impl Animatable for f32 {
    fn lerp(&self, target: &Self, t: f32) -> Self {
        self + (target - self) * t
    }
}

impl Animatable for (f32, f32) {
    fn lerp(&self, target: &Self, t: f32) -> Self {
        (
            self.0 + (target.0 - self.0) * t,
            self.1 + (target.1 - self.1) * t,
        )
    }
}

/// Animation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationState {
    /// Not started
    Pending,
    /// Currently running
    Running,
    /// Completed
    Completed,
    /// Cancelled
    Cancelled,
}

/// A single tween animation
pub struct Tween<T: Animatable> {
    start: T,
    end: T,
    duration_ms: f32,
    elapsed_ms: f32,
    easing: fn(f32) -> f32,
    state: AnimationState,
    delay_ms: f32,
}

impl<T: Animatable> Tween<T> {
    pub fn new(start: T, end: T, duration_ms: f32) -> Self {
        Self {
            start,
            end,
            duration_ms,
            elapsed_ms: 0.0,
            easing: |t| t, // linear
            state: AnimationState::Pending,
            delay_ms: 0.0,
        }
    }

    pub fn with_easing(mut self, easing: fn(f32) -> f32) -> Self {
        self.easing = easing;
        self
    }

    pub fn with_delay(mut self, delay_ms: f32) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    pub fn start(&mut self) {
        self.state = AnimationState::Running;
    }

    pub fn update(&mut self, delta_ms: f32) -> T {
        if self.state != AnimationState::Running {
            return self.start.clone();
        }

        self.elapsed_ms += delta_ms;

        // Handle delay
        if self.elapsed_ms < self.delay_ms {
            return self.start.clone();
        }

        let effective_elapsed = self.elapsed_ms - self.delay_ms;
        let t = (effective_elapsed / self.duration_ms).clamp(0.0, 1.0);
        let eased_t = (self.easing)(t);

        if t >= 1.0 {
            self.state = AnimationState::Completed;
            return self.end.clone();
        }

        self.start.lerp(&self.end, eased_t)
    }

    pub fn is_complete(&self) -> bool {
        self.state == AnimationState::Completed
    }

    pub fn progress(&self) -> f32 {
        let effective_elapsed = (self.elapsed_ms - self.delay_ms).max(0.0);
        (effective_elapsed / self.duration_ms).clamp(0.0, 1.0)
    }
}

/// Sequential animation group
pub struct SequentialGroup {
    animations: Vec<Box<dyn Animation>>,
    current_index: usize,
    state: AnimationState,
}

impl SequentialGroup {
    pub fn new() -> Self {
        Self {
            animations: Vec::new(),
            current_index: 0,
            state: AnimationState::Pending,
        }
    }

    pub fn add<A: Animation + 'static>(mut self, animation: A) -> Self {
        self.animations.push(Box::new(animation));
        self
    }

    pub fn start(&mut self) {
        self.state = AnimationState::Running;
        self.current_index = 0;
        if let Some(anim) = self.animations.get_mut(0) {
            anim.start();
        }
    }

    pub fn update(&mut self, delta_ms: f32) {
        if self.state != AnimationState::Running {
            return;
        }

        if let Some(current) = self.animations.get_mut(self.current_index) {
            current.update(delta_ms);

            if current.is_complete() {
                self.current_index += 1;
                if self.current_index >= self.animations.len() {
                    self.state = AnimationState::Completed;
                } else if let Some(next) = self.animations.get_mut(self.current_index) {
                    next.start();
                }
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.state == AnimationState::Completed
    }

    pub fn progress(&self) -> f32 {
        if self.animations.is_empty() {
            return 1.0;
        }

        let completed = self.current_index as f32;
        let current_progress = self
            .animations
            .get(self.current_index)
            .map(|a| a.progress())
            .unwrap_or(0.0);

        (completed + current_progress) / self.animations.len() as f32
    }
}

impl Default for SequentialGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel animation group
pub struct ParallelGroup {
    animations: Vec<Box<dyn Animation>>,
    state: AnimationState,
}

impl ParallelGroup {
    pub fn new() -> Self {
        Self {
            animations: Vec::new(),
            state: AnimationState::Pending,
        }
    }

    pub fn add<A: Animation + 'static>(mut self, animation: A) -> Self {
        self.animations.push(Box::new(animation));
        self
    }

    pub fn start(&mut self) {
        self.state = AnimationState::Running;
        for anim in &mut self.animations {
            anim.start();
        }
    }

    pub fn update(&mut self, delta_ms: f32) {
        if self.state != AnimationState::Running {
            return;
        }

        let mut all_complete = true;
        for anim in &mut self.animations {
            anim.update(delta_ms);
            if !anim.is_complete() {
                all_complete = false;
            }
        }

        if all_complete {
            self.state = AnimationState::Completed;
        }
    }

    pub fn is_complete(&self) -> bool {
        self.state == AnimationState::Completed
    }

    pub fn progress(&self) -> f32 {
        if self.animations.is_empty() {
            return 1.0;
        }

        // Return the minimum progress (all must complete)
        self.animations
            .iter()
            .map(|a| a.progress())
            .fold(1.0f32, |acc, p| acc.min(p))
    }
}

impl Default for ParallelGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Staggered group (parallel with delays)
pub struct StaggeredGroup {
    animations: Vec<(f32, Box<dyn Animation>)>, // (delay_ms, animation)
    state: AnimationState,
    elapsed_ms: f32,
}

impl StaggeredGroup {
    pub fn new() -> Self {
        Self {
            animations: Vec::new(),
            state: AnimationState::Pending,
            elapsed_ms: 0.0,
        }
    }

    /// Add animation with stagger delay
    pub fn add<A: Animation + 'static>(mut self, delay_ms: f32, animation: A) -> Self {
        self.animations.push((delay_ms, Box::new(animation)));
        self
    }

    /// Add animations with uniform stagger
    pub fn stagger<A: Animation + 'static>(mut self, stagger_ms: f32, animations: Vec<A>) -> Self {
        for (i, anim) in animations.into_iter().enumerate() {
            let delay = i as f32 * stagger_ms;
            self.animations.push((delay, Box::new(anim)));
        }
        self
    }

    pub fn start(&mut self) {
        self.state = AnimationState::Running;
        self.elapsed_ms = 0.0;
    }

    pub fn update(&mut self, delta_ms: f32) {
        if self.state != AnimationState::Running {
            return;
        }

        self.elapsed_ms += delta_ms;

        let mut all_complete = true;
        for (delay, anim) in &mut self.animations {
            if self.elapsed_ms >= *delay {
                if !anim.is_running() && !anim.is_complete() {
                    anim.start();
                }
                anim.update(delta_ms);
            }
            if !anim.is_complete() {
                all_complete = false;
            }
        }

        if all_complete && !self.animations.is_empty() {
            self.state = AnimationState::Completed;
        }
    }

    pub fn is_complete(&self) -> bool {
        self.state == AnimationState::Completed
    }

    pub fn progress(&self) -> f32 {
        if self.animations.is_empty() {
            return 1.0;
        }

        self.animations
            .iter()
            .map(|(_, a)| a.progress())
            .sum::<f32>()
            / self.animations.len() as f32
    }
}

impl Default for StaggeredGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Animation trait for composability
pub trait Animation: Send {
    fn start(&mut self);
    fn update(&mut self, delta_ms: f32);
    fn is_complete(&self) -> bool;
    fn is_running(&self) -> bool;
    fn progress(&self) -> f32;
}

impl<T: Animatable + Send> Animation for Tween<T> {
    fn start(&mut self) {
        self.start();
    }
    fn update(&mut self, delta_ms: f32) {
        let _ = self.update(delta_ms);
    }
    fn is_complete(&self) -> bool {
        self.is_complete()
    }
    fn is_running(&self) -> bool {
        self.state == AnimationState::Running
    }
    fn progress(&self) -> f32 {
        self.progress()
    }
}

impl Animation for SequentialGroup {
    fn start(&mut self) {
        self.start();
    }
    fn update(&mut self, delta_ms: f32) {
        self.update(delta_ms);
    }
    fn is_complete(&self) -> bool {
        self.is_complete()
    }
    fn is_running(&self) -> bool {
        self.state == AnimationState::Running
    }
    fn progress(&self) -> f32 {
        self.progress()
    }
}

impl Animation for ParallelGroup {
    fn start(&mut self) {
        self.start();
    }
    fn update(&mut self, delta_ms: f32) {
        self.update(delta_ms);
    }
    fn is_complete(&self) -> bool {
        self.is_complete()
    }
    fn is_running(&self) -> bool {
        self.state == AnimationState::Running
    }
    fn progress(&self) -> f32 {
        self.progress()
    }
}

impl Animation for StaggeredGroup {
    fn start(&mut self) {
        self.start();
    }
    fn update(&mut self, delta_ms: f32) {
        self.update(delta_ms);
    }
    fn is_complete(&self) -> bool {
        self.is_complete()
    }
    fn is_running(&self) -> bool {
        self.state == AnimationState::Running
    }
    fn progress(&self) -> f32 {
        self.progress()
    }
}

/// Animation manager for running multiple animations
pub struct AnimationManager {
    animations: HashMap<String, Box<dyn Animation>>,
}

impl AnimationManager {
    pub fn new() -> Self {
        Self {
            animations: HashMap::new(),
        }
    }

    /// Add a named animation
    pub fn add<A: Animation + 'static>(&mut self, name: &str, animation: A) {
        self.animations
            .insert(name.to_string(), Box::new(animation));
    }

    /// Start an animation by name
    pub fn start(&mut self, name: &str) {
        if let Some(anim) = self.animations.get_mut(name) {
            anim.start();
        }
    }

    /// Update all animations
    pub fn update(&mut self, delta_ms: f32) {
        for anim in self.animations.values_mut() {
            anim.update(delta_ms);
        }
    }

    /// Remove completed animations
    pub fn cleanup(&mut self) {
        self.animations.retain(|_, anim| !anim.is_complete());
    }

    /// Check if animation is complete
    pub fn is_complete(&self, name: &str) -> bool {
        self.animations
            .get(name)
            .map(|a| a.is_complete())
            .unwrap_or(true)
    }

    /// Get animation progress
    pub fn progress(&self, name: &str) -> f32 {
        self.animations
            .get(name)
            .map(|a| a.progress())
            .unwrap_or(1.0)
    }
}

impl Default for AnimationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tween() {
        let mut tween = Tween::new(0.0f32, 100.0, 1000.0);
        tween.start();

        let val = tween.update(500.0);
        assert!((val - 50.0).abs() < 0.01);

        let val = tween.update(500.0);
        assert!((val - 100.0).abs() < 0.01);
        assert!(tween.is_complete());
    }

    #[test]
    fn test_sequential() {
        let mut group = SequentialGroup::new()
            .add(Tween::new(0.0f32, 50.0, 500.0))
            .add(Tween::new(50.0f32, 100.0, 500.0));

        group.start();

        group.update(250.0);
        assert!(!group.is_complete());

        group.update(250.0);
        assert!(!group.is_complete()); // First complete, second starting

        group.update(500.0);
        assert!(group.is_complete());
    }
}
