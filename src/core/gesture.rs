//! Gesture detection engine
//!
//! Detects touch/mouse gestures including:
//! - Swipe (direction + velocity)
//! - Pinch (zoom in/out)
//! - Long Press
//! - Drag
//! - Double Tap

use crate::core::Vec2;

/// Gesture types that can be detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GestureType {
    /// No gesture detected
    None,
    /// Tap gesture (quick touch)
    Tap { pos: Vec2 },
    /// Double tap gesture
    DoubleTap { pos: Vec2 },
    /// Long press (held for duration)
    LongPress { pos: Vec2, duration_ms: u32 },
    /// Swipe gesture with direction and velocity
    Swipe { start: Vec2, end: Vec2, velocity: Vec2, direction: SwipeDirection },
    /// Drag gesture (ongoing movement)
    Drag { start: Vec2, current: Vec2, delta: Vec2 },
    /// Pinch gesture (two-finger zoom)
    Pinch { center: Vec2, scale: f32, rotation: f32 },
    /// Pan gesture (two-finger drag)
    Pan { delta: Vec2 },
}

/// Swipe direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwipeDirection {
    Left,
    Right,
    Up,
    Down,
}

/// Touch point state
#[derive(Debug, Clone, Copy)]
struct TouchPoint {
    id: u32,
    pos: Vec2,
    start_pos: Vec2,
    start_time_ms: u64,
}

/// Configuration for gesture detection
#[derive(Debug, Clone)]
pub struct GestureConfig {
    /// Minimum distance to trigger a swipe (pixels)
    pub swipe_threshold: f32,
    /// Minimum velocity for swipe (pixels/sec)
    pub swipe_velocity_threshold: f32,
    /// Duration for long press (ms)
    pub long_press_duration_ms: u32,
    /// Maximum distance for tap (pixels)
    pub tap_threshold: f32,
    /// Maximum time between double taps (ms)
    pub double_tap_interval_ms: u32,
    /// Minimum scale change to trigger pinch
    pub pinch_threshold: f32,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            swipe_threshold: 50.0,
            swipe_velocity_threshold: 200.0,
            long_press_duration_ms: 500,
            tap_threshold: 10.0,
            double_tap_interval_ms: 300,
            pinch_threshold: 0.1,
        }
    }
}

/// Gesture detector state machine
pub struct GestureDetector {
    config: GestureConfig,
    touches: Vec<TouchPoint>,
    last_tap_time_ms: u64,
    last_tap_pos: Vec2,
    current_time_ms: u64,
    pending_gesture: GestureType,
}

impl GestureDetector {
    /// Create a new gesture detector with default config
    pub fn new() -> Self {
        Self::with_config(GestureConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GestureConfig) -> Self {
        Self {
            config,
            touches: Vec::new(),
            last_tap_time_ms: 0,
            last_tap_pos: Vec2::ZERO,
            current_time_ms: 0,
            pending_gesture: GestureType::None,
        }
    }

    /// Update the current time (call each frame with delta_ms)
    pub fn update(&mut self, current_time_ms: u64) {
        self.current_time_ms = current_time_ms;
        
        // Check for long press
        if self.touches.len() == 1 {
            let touch = &self.touches[0];
            let duration = (self.current_time_ms - touch.start_time_ms) as u32;
            let distance = (touch.pos - touch.start_pos).length();
            
            if duration >= self.config.long_press_duration_ms && distance < self.config.tap_threshold {
                self.pending_gesture = GestureType::LongPress {
                    pos: touch.pos,
                    duration_ms: duration,
                };
            }
        }
    }

    /// Handle touch/mouse down event
    pub fn on_touch_start(&mut self, id: u32, pos: Vec2) {
        self.touches.push(TouchPoint {
            id,
            pos,
            start_pos: pos,
            start_time_ms: self.current_time_ms,
        });
        self.pending_gesture = GestureType::None;
    }

    /// Handle touch/mouse move event
    pub fn on_touch_move(&mut self, id: u32, pos: Vec2) -> GestureType {
        if let Some(touch) = self.touches.iter_mut().find(|t| t.id == id) {
            let delta = pos - touch.pos;
            let start = touch.start_pos;
            touch.pos = pos;

            // Single touch drag
            if self.touches.len() == 1 {
                return GestureType::Drag {
                    start,
                    current: pos,
                    delta,
                };
            }
            
            // Two-touch pinch/pan
            if self.touches.len() == 2 {
                return self.detect_pinch_pan();
            }
        }
        GestureType::None
    }

    /// Handle touch/mouse up event
    pub fn on_touch_end(&mut self, id: u32, pos: Vec2) -> GestureType {
        let gesture = if let Some(idx) = self.touches.iter().position(|t| t.id == id) {
            let touch = self.touches.remove(idx);
            self.detect_end_gesture(&touch, pos)
        } else {
            GestureType::None
        };
        
        // Return pending gesture if we have one
        if self.pending_gesture != GestureType::None {
            let pending = self.pending_gesture;
            self.pending_gesture = GestureType::None;
            return pending;
        }
        
        gesture
    }

    /// Detect gesture on touch end
    fn detect_end_gesture(&mut self, touch: &TouchPoint, end_pos: Vec2) -> GestureType {
        let delta = end_pos - touch.start_pos;
        let distance = delta.length();
        let duration_ms = (self.current_time_ms - touch.start_time_ms) as f32;
        
        // Calculate velocity (pixels per second)
        let velocity = if duration_ms > 0.0 {
            delta * (1000.0 / duration_ms)
        } else {
            Vec2::ZERO
        };
        let speed = velocity.length();

        // Check for swipe
        if distance >= self.config.swipe_threshold && speed >= self.config.swipe_velocity_threshold {
            let direction = self.get_swipe_direction(delta);
            return GestureType::Swipe {
                start: touch.start_pos,
                end: end_pos,
                velocity,
                direction,
            };
        }

        // Check for tap
        if distance < self.config.tap_threshold && duration_ms < self.config.long_press_duration_ms as f32 {
            // Check for double tap
            let time_since_last_tap = self.current_time_ms - self.last_tap_time_ms;
            let dist_from_last_tap = (end_pos - self.last_tap_pos).length();
            
            if time_since_last_tap < self.config.double_tap_interval_ms as u64 
               && dist_from_last_tap < self.config.tap_threshold * 2.0 {
                self.last_tap_time_ms = 0;
                return GestureType::DoubleTap { pos: end_pos };
            }
            
            self.last_tap_time_ms = self.current_time_ms;
            self.last_tap_pos = end_pos;
            return GestureType::Tap { pos: end_pos };
        }

        GestureType::None
    }

    /// Detect pinch/pan gesture from two touches
    fn detect_pinch_pan(&self) -> GestureType {
        if self.touches.len() != 2 {
            return GestureType::None;
        }

        let t0 = &self.touches[0];
        let t1 = &self.touches[1];

        // Current state
        let center = (t0.pos + t1.pos) * 0.5;
        let current_dist = (t0.pos - t1.pos).length();
        
        // Initial state
        let initial_dist = (t0.start_pos - t1.start_pos).length();
        
        // Calculate scale
        let scale = if initial_dist > 0.0 {
            current_dist / initial_dist
        } else {
            1.0
        };

        // Calculate rotation (angle between initial and current vectors)
        let initial_vec = t1.start_pos - t0.start_pos;
        let current_vec = t1.pos - t0.pos;
        let rotation = current_vec.y.atan2(current_vec.x) - initial_vec.y.atan2(initial_vec.x);

        // Check if it's more of a pinch or pan
        if (scale - 1.0).abs() >= self.config.pinch_threshold {
            GestureType::Pinch { center, scale, rotation }
        } else {
            let initial_center = (t0.start_pos + t1.start_pos) * 0.5;
            let delta = center - initial_center;
            GestureType::Pan { delta }
        }
    }

    /// Get swipe direction from delta
    fn get_swipe_direction(&self, delta: Vec2) -> SwipeDirection {
        if delta.x.abs() > delta.y.abs() {
            if delta.x > 0.0 { SwipeDirection::Right } else { SwipeDirection::Left }
        } else {
            if delta.y > 0.0 { SwipeDirection::Down } else { SwipeDirection::Up }
        }
    }

    /// Check if any touches are active
    pub fn is_touching(&self) -> bool {
        !self.touches.is_empty()
    }

    /// Get number of active touches
    pub fn touch_count(&self) -> usize {
        self.touches.len()
    }

    /// Get the pending gesture (if any)
    pub fn pending(&self) -> GestureType {
        self.pending_gesture
    }

    /// Clear all state
    pub fn reset(&mut self) {
        self.touches.clear();
        self.pending_gesture = GestureType::None;
    }
}

impl Default for GestureDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tap_detection() {
        let mut detector = GestureDetector::new();
        
        detector.on_touch_start(0, Vec2::new(100.0, 100.0));
        detector.update(50);
        let gesture = detector.on_touch_end(0, Vec2::new(102.0, 101.0));
        
        assert!(matches!(gesture, GestureType::Tap { .. }));
    }

    #[test]
    fn test_swipe_detection() {
        let mut detector = GestureDetector::new();
        
        detector.on_touch_start(0, Vec2::new(100.0, 100.0));
        detector.update(100);
        let gesture = detector.on_touch_end(0, Vec2::new(200.0, 100.0));
        
        assert!(matches!(gesture, GestureType::Swipe { direction: SwipeDirection::Right, .. }));
    }
}
