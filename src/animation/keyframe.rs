//! Keyframe-based animation system
//!
//! Provides timeline animation with:
//! - Keyframe interpolation
//! - Multiple easing functions
//! - Property targeting
//! - Looping and reversing

use std::collections::HashMap;

/// Easing function type
pub type EasingFn = fn(f32) -> f32;

/// Built-in easing functions
pub mod easing {
    /// Linear interpolation
    pub fn linear(t: f32) -> f32 { t }
    
    /// Ease in (slow start)
    pub fn ease_in_quad(t: f32) -> f32 { t * t }
    pub fn ease_in_cubic(t: f32) -> f32 { t * t * t }
    pub fn ease_in_quart(t: f32) -> f32 { t * t * t * t }
    pub fn ease_in_quint(t: f32) -> f32 { t * t * t * t * t }
    pub fn ease_in_sine(t: f32) -> f32 { 1.0 - (t * std::f32::consts::FRAC_PI_2).cos() }
    pub fn ease_in_expo(t: f32) -> f32 { if t == 0.0 { 0.0 } else { (2.0_f32).powf(10.0 * t - 10.0) } }
    pub fn ease_in_circ(t: f32) -> f32 { 1.0 - (1.0 - t * t).sqrt() }
    
    /// Ease out (slow end)
    pub fn ease_out_quad(t: f32) -> f32 { 1.0 - (1.0 - t) * (1.0 - t) }
    pub fn ease_out_cubic(t: f32) -> f32 { 1.0 - (1.0 - t).powi(3) }
    pub fn ease_out_quart(t: f32) -> f32 { 1.0 - (1.0 - t).powi(4) }
    pub fn ease_out_quint(t: f32) -> f32 { 1.0 - (1.0 - t).powi(5) }
    pub fn ease_out_sine(t: f32) -> f32 { (t * std::f32::consts::FRAC_PI_2).sin() }
    pub fn ease_out_expo(t: f32) -> f32 { if t == 1.0 { 1.0 } else { 1.0 - (2.0_f32).powf(-10.0 * t) } }
    pub fn ease_out_circ(t: f32) -> f32 { (1.0 - (t - 1.0) * (t - 1.0)).sqrt() }
    
    /// Ease in-out (slow start and end)
    pub fn ease_in_out_quad(t: f32) -> f32 {
        if t < 0.5 { 2.0 * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(2) / 2.0 }
    }
    pub fn ease_in_out_cubic(t: f32) -> f32 {
        if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 }
    }
    pub fn ease_in_out_quart(t: f32) -> f32 {
        if t < 0.5 { 8.0 * t.powi(4) } else { 1.0 - (-2.0 * t + 2.0).powi(4) / 2.0 }
    }
    pub fn ease_in_out_sine(t: f32) -> f32 {
        -((std::f32::consts::PI * t).cos() - 1.0) / 2.0
    }
    
    /// Elastic easing (spring-like overshoot)
    pub fn ease_in_elastic(t: f32) -> f32 {
        if t == 0.0 || t == 1.0 { return t; }
        let c4 = (2.0 * std::f32::consts::PI) / 3.0;
        -(2.0_f32.powf(10.0 * t - 10.0)) * ((t * 10.0 - 10.75) * c4).sin()
    }
    pub fn ease_out_elastic(t: f32) -> f32 {
        if t == 0.0 || t == 1.0 { return t; }
        let c4 = (2.0 * std::f32::consts::PI) / 3.0;
        2.0_f32.powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
    }
    
    /// Back easing (slight overshoot)
    pub fn ease_in_back(t: f32) -> f32 {
        let c1 = 1.70158;
        let c3 = c1 + 1.0;
        c3 * t * t * t - c1 * t * t
    }
    pub fn ease_out_back(t: f32) -> f32 {
        let c1 = 1.70158;
        let c3 = c1 + 1.0;
        1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
    }
    
    /// Bounce easing
    pub fn ease_out_bounce(t: f32) -> f32 {
        let n1 = 7.5625;
        let d1 = 2.75;
        if t < 1.0 / d1 {
            n1 * t * t
        } else if t < 2.0 / d1 {
            let t = t - 1.5 / d1;
            n1 * t * t + 0.75
        } else if t < 2.5 / d1 {
            let t = t - 2.25 / d1;
            n1 * t * t + 0.9375
        } else {
            let t = t - 2.625 / d1;
            n1 * t * t + 0.984375
        }
    }
    pub fn ease_in_bounce(t: f32) -> f32 { 1.0 - ease_out_bounce(1.0 - t) }
}

/// A single keyframe
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Time position (0.0 to 1.0 normalized, or absolute ms)
    pub time: f32,
    /// Value at this keyframe
    pub value: f32,
    /// Easing function to use when interpolating TO this keyframe
    pub easing: EasingFn,
}

impl Keyframe {
    pub fn new(time: f32, value: f32) -> Self {
        Self { time, value, easing: easing::linear }
    }

    pub fn with_easing(mut self, easing: EasingFn) -> Self {
        self.easing = easing;
        self
    }
}

/// Animation loop mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Play once and stop
    Once,
    /// Loop from start
    Loop,
    /// Ping-pong (forward then reverse)
    PingPong,
}

/// Animation playback state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

/// Animation direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Reverse,
}

/// A keyframe track for a single property
#[derive(Debug, Clone)]
pub struct KeyframeTrack {
    name: String,
    keyframes: Vec<Keyframe>,
    /// Duration in milliseconds
    duration_ms: f32,
}

impl KeyframeTrack {
    pub fn new(name: &str, duration_ms: f32) -> Self {
        Self {
            name: name.to_string(),
            keyframes: Vec::new(),
            duration_ms,
        }
    }

    /// Add a keyframe
    pub fn add_keyframe(&mut self, keyframe: Keyframe) -> &mut Self {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        self
    }

    /// Add keyframe with builder pattern
    pub fn keyframe(mut self, time: f32, value: f32) -> Self {
        self.add_keyframe(Keyframe::new(time, value));
        self
    }

    /// Add keyframe with easing
    pub fn keyframe_eased(mut self, time: f32, value: f32, easing: EasingFn) -> Self {
        self.add_keyframe(Keyframe::new(time, value).with_easing(easing));
        self
    }

    /// Get interpolated value at time (0.0 to 1.0)
    pub fn sample(&self, t: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].value;
        }

        let t = t.clamp(0.0, 1.0);

        // Find surrounding keyframes
        let mut prev_idx = 0;
        let mut next_idx = 0;
        
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.time <= t {
                prev_idx = i;
            }
            if kf.time >= t {
                next_idx = i;
                break;
            }
            next_idx = i;
        }

        if prev_idx == next_idx {
            return self.keyframes[prev_idx].value;
        }

        let prev = &self.keyframes[prev_idx];
        let next = &self.keyframes[next_idx];

        // Calculate local t between keyframes
        let local_t = if next.time > prev.time {
            (t - prev.time) / (next.time - prev.time)
        } else {
            0.0
        };

        // Apply easing
        let eased_t = (next.easing)(local_t);

        // Interpolate
        prev.value + (next.value - prev.value) * eased_t
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn duration_ms(&self) -> f32 { self.duration_ms }
}

/// Timeline animation with multiple tracks
pub struct Timeline {
    tracks: HashMap<String, KeyframeTrack>,
    /// Total duration in milliseconds
    duration_ms: f32,
    /// Current time in milliseconds
    current_time_ms: f32,
    /// Playback state
    state: PlaybackState,
    /// Playback direction
    direction: Direction,
    /// Loop mode
    loop_mode: LoopMode,
    /// Playback speed multiplier
    speed: f32,
}

impl Timeline {
    pub fn new(duration_ms: f32) -> Self {
        Self {
            tracks: HashMap::new(),
            duration_ms,
            current_time_ms: 0.0,
            state: PlaybackState::Stopped,
            direction: Direction::Forward,
            loop_mode: LoopMode::Once,
            speed: 1.0,
        }
    }

    /// Add a track to the timeline
    pub fn add_track(&mut self, track: KeyframeTrack) {
        self.tracks.insert(track.name.clone(), track);
    }

    /// Set loop mode
    pub fn set_loop_mode(&mut self, mode: LoopMode) {
        self.loop_mode = mode;
    }

    /// Set playback speed
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Start playing
    pub fn play(&mut self) {
        self.state = PlaybackState::Playing;
    }

    /// Pause playback
    pub fn pause(&mut self) {
        self.state = PlaybackState::Paused;
    }

    /// Stop and reset
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
        self.current_time_ms = 0.0;
        self.direction = Direction::Forward;
    }

    /// Seek to time
    pub fn seek(&mut self, time_ms: f32) {
        self.current_time_ms = time_ms.clamp(0.0, self.duration_ms);
    }

    /// Update the timeline
    pub fn update(&mut self, delta_ms: f32) {
        if self.state != PlaybackState::Playing {
            return;
        }

        let delta = delta_ms * self.speed;
        
        match self.direction {
            Direction::Forward => {
                self.current_time_ms += delta;
                if self.current_time_ms >= self.duration_ms {
                    match self.loop_mode {
                        LoopMode::Once => {
                            self.current_time_ms = self.duration_ms;
                            self.state = PlaybackState::Stopped;
                        }
                        LoopMode::Loop => {
                            self.current_time_ms %= self.duration_ms;
                        }
                        LoopMode::PingPong => {
                            self.current_time_ms = self.duration_ms;
                            self.direction = Direction::Reverse;
                        }
                    }
                }
            }
            Direction::Reverse => {
                self.current_time_ms -= delta;
                if self.current_time_ms <= 0.0 {
                    match self.loop_mode {
                        LoopMode::Once => {
                            self.current_time_ms = 0.0;
                            self.state = PlaybackState::Stopped;
                        }
                        LoopMode::Loop => {
                            self.current_time_ms = self.duration_ms;
                        }
                        LoopMode::PingPong => {
                            self.current_time_ms = 0.0;
                            self.direction = Direction::Forward;
                        }
                    }
                }
            }
        }
    }

    /// Get current value for a track
    pub fn get(&self, track_name: &str) -> Option<f32> {
        self.tracks.get(track_name).map(|track| {
            let t = self.current_time_ms / self.duration_ms;
            track.sample(t)
        })
    }

    /// Get all current values
    pub fn get_all(&self) -> HashMap<String, f32> {
        let t = self.current_time_ms / self.duration_ms;
        self.tracks.iter()
            .map(|(name, track)| (name.clone(), track.sample(t)))
            .collect()
    }

    /// Get normalized progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        self.current_time_ms / self.duration_ms
    }

    /// Check if animation is complete
    pub fn is_complete(&self) -> bool {
        self.state == PlaybackState::Stopped && self.current_time_ms >= self.duration_ms
    }

    /// Check if playing
    pub fn is_playing(&self) -> bool {
        self.state == PlaybackState::Playing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_interpolation() {
        let track = KeyframeTrack::new("opacity", 1000.0)
            .keyframe(0.0, 0.0)
            .keyframe(0.5, 1.0)
            .keyframe(1.0, 0.5);

        assert_eq!(track.sample(0.0), 0.0);
        assert_eq!(track.sample(0.5), 1.0);
        assert_eq!(track.sample(1.0), 0.5);
        assert!((track.sample(0.25) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_timeline_playback() {
        let mut timeline = Timeline::new(1000.0);
        timeline.add_track(
            KeyframeTrack::new("x", 1000.0)
                .keyframe(0.0, 0.0)
                .keyframe(1.0, 100.0)
        );
        
        timeline.play();
        timeline.update(500.0);
        
        assert!((timeline.get("x").unwrap() - 50.0).abs() < 0.01);
    }
}
