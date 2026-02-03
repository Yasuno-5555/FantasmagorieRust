use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Time {
    pub seconds: f64,
    pub frame: u64,
}

impl Default for Time {
    fn default() -> Self {
        Self { seconds: 0.0, frame: 0 }
    }
}

pub trait TimeEvaluator {
    fn evaluate(&mut self, time: Time);
}

pub struct MasterClock {
    start_time: Instant,
    current_time: Time,
    playback_rate: f64, // 1.0 = normal, 0.0 = paused, -1.0 = reverse
    is_paused: bool,
    
    // For manual seeking/scrubbing
    offset_seconds: f64,
}

impl MasterClock {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            current_time: Time::default(),
            playback_rate: 1.0,
            is_paused: false,
            offset_seconds: 0.0,
        }
    }

    pub fn tick(&mut self, dt: f64) {
        if !self.is_paused {
            // In a real cinematic engine, we might lock to audio clock or
            // accumulate dt * playback_rate.
            // basic accumulation for now, but respecting the "Constitution" that time creates the state.
             
            // However, to support "Seeking", we should calculate time based on 
            // reference_start + elapsed * rate, OR accumulate.
            // For stability without drift, accumulated time is often safer for games, 
            // but for "absolute synchronization", we might want absolute calculations.
            
            // Let's stick to simple accumulation for the "tick" but expose "seek".
            self.current_time.seconds += dt * self.playback_rate;
        }
        
        self.current_time.frame += 1;
    }

    pub fn time(&self) -> Time {
        self.current_time
    }
    
    pub fn set_time(&mut self, seconds: f64) {
        self.current_time.seconds = seconds;
        // Frame count monotonically increases even if we seek backwards? 
        // Or does it reset? For rendering logic, monotonic increase is safer for cache invalidation.
        // But for "Timeline", we really care about the 'seconds'.
    }

    pub fn pause(&mut self) {
        self.is_paused = true;
    }

    pub fn play(&mut self) {
        self.is_paused = false;
        // Resuming.
    }
    
    pub fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
    }
    
    pub fn seek(&mut self, time: f64) {
        self.current_time.seconds = time;
    }

    pub fn is_paused(&self) -> bool {
        self.is_paused
    }
}
