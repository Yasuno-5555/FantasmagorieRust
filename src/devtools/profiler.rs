//! Performance profiler for UI rendering
//!
//! Tracks and displays:
//! - Frame timing
//! - Layout performance
//! - Render statistics
//! - Widget counts

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Single frame timing data
#[derive(Debug, Clone, Copy)]
pub struct FrameTiming {
    /// Total frame time
    pub total_ms: f32,
    /// Layout calculation time
    pub layout_ms: f32,
    /// Render/draw time
    pub render_ms: f32,
    /// Input processing time
    pub input_ms: f32,
    /// Widget count
    pub widget_count: u32,
    /// Draw call count
    pub draw_calls: u32,
    /// Vertex count
    pub vertices: u32,
    /// Texture switches
    pub texture_switches: u32,
}

impl Default for FrameTiming {
    fn default() -> Self {
        Self {
            total_ms: 0.0,
            layout_ms: 0.0,
            render_ms: 0.0,
            input_ms: 0.0,
            widget_count: 0,
            draw_calls: 0,
            vertices: 0,
            texture_switches: 0,
        }
    }
}

/// Scope timer for measuring code sections
pub struct ScopeTimer {
    start: Instant,
    name: &'static str,
}

impl ScopeTimer {
    pub fn new(name: &'static str) -> Self {
        Self {
            start: Instant::now(),
            name,
        }
    }

    pub fn elapsed_ms(&self) -> f32 {
        self.start.elapsed().as_secs_f32() * 1000.0
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl Drop for ScopeTimer {
    fn drop(&mut self) {
        // Could log the timing here
    }
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Number of frames to keep in history
    pub history_size: usize,
    /// Show FPS counter
    pub show_fps: bool,
    /// Show frame graph
    pub show_graph: bool,
    /// Show detailed breakdown
    pub show_details: bool,
    /// Warning threshold (ms)
    pub warning_threshold_ms: f32,
    /// Critical threshold (ms)
    pub critical_threshold_ms: f32,
    /// Target frame time (ms) - 60fps = 16.67ms
    pub target_frame_ms: f32,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            history_size: 120,
            show_fps: true,
            show_graph: true,
            show_details: false,
            warning_threshold_ms: 16.67,
            critical_threshold_ms: 33.33,
            target_frame_ms: 16.67,
        }
    }
}

/// Performance profiler
pub struct Profiler {
    config: ProfilerConfig,
    /// Frame timing history
    history: VecDeque<FrameTiming>,
    /// Current frame being built
    current: FrameTiming,
    /// Frame start time
    frame_start: Option<Instant>,
    /// Section timers
    section_start: Option<Instant>,
    /// Is profiler visible
    visible: bool,
    /// Pause data collection
    paused: bool,
}

impl Profiler {
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    pub fn with_config(config: ProfilerConfig) -> Self {
        let history_size = config.history_size;
        Self {
            config,
            history: VecDeque::with_capacity(history_size),
            current: FrameTiming::default(),
            frame_start: None,
            section_start: None,
            visible: false,
            paused: false,
        }
    }

    /// Start a new frame
    pub fn begin_frame(&mut self) {
        if self.paused {
            return;
        }
        self.frame_start = Some(Instant::now());
        self.current = FrameTiming::default();
    }

    /// End the current frame
    pub fn end_frame(&mut self) {
        if self.paused {
            return;
        }
        if let Some(start) = self.frame_start.take() {
            self.current.total_ms = start.elapsed().as_secs_f32() * 1000.0;

            // Add to history
            if self.history.len() >= self.config.history_size {
                self.history.pop_front();
            }
            self.history.push_back(self.current);
        }
    }

    /// Begin timing a section
    pub fn begin_section(&mut self) {
        if !self.paused {
            self.section_start = Some(Instant::now());
        }
    }

    /// End layout timing
    pub fn end_layout(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.current.layout_ms = start.elapsed().as_secs_f32() * 1000.0;
        }
    }

    /// End render timing
    pub fn end_render(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.current.render_ms = start.elapsed().as_secs_f32() * 1000.0;
        }
    }

    /// End input timing
    pub fn end_input(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.current.input_ms = start.elapsed().as_secs_f32() * 1000.0;
        }
    }

    /// Record widget count
    pub fn set_widget_count(&mut self, count: u32) {
        self.current.widget_count = count;
    }

    /// Record draw calls
    pub fn set_draw_calls(&mut self, count: u32) {
        self.current.draw_calls = count;
    }

    /// Record vertices
    pub fn set_vertices(&mut self, count: u32) {
        self.current.vertices = count;
    }

    /// Toggle visibility
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Show profiler
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hide profiler
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Check if visible
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Pause/resume data collection
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Get current FPS
    pub fn fps(&self) -> f32 {
        if let Some(last) = self.history.back() {
            if last.total_ms > 0.0 {
                return 1000.0 / last.total_ms;
            }
        }
        0.0
    }

    /// Get average FPS over history
    pub fn average_fps(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let total_time: f32 = self.history.iter().map(|f| f.total_ms).sum();
        if total_time > 0.0 {
            1000.0 * self.history.len() as f32 / total_time
        } else {
            0.0
        }
    }

    /// Get min/max/avg frame times
    pub fn frame_stats(&self) -> FrameStats {
        if self.history.is_empty() {
            return FrameStats::default();
        }

        let times: Vec<f32> = self.history.iter().map(|f| f.total_ms).collect();
        let min = times.iter().copied().fold(f32::INFINITY, f32::min);
        let max = times.iter().copied().fold(0.0f32, f32::max);
        let avg = times.iter().sum::<f32>() / times.len() as f32;

        FrameStats {
            min_ms: min,
            max_ms: max,
            avg_ms: avg,
        }
    }

    /// Get frame time history for graph
    pub fn frame_times(&self) -> Vec<f32> {
        self.history.iter().map(|f| f.total_ms).collect()
    }

    /// Get latest frame timing
    pub fn latest(&self) -> Option<&FrameTiming> {
        self.history.back()
    }

    /// Get performance level for current frame
    pub fn performance_level(&self) -> PerformanceLevel {
        if let Some(last) = self.history.back() {
            if last.total_ms >= self.config.critical_threshold_ms {
                PerformanceLevel::Critical
            } else if last.total_ms >= self.config.warning_threshold_ms {
                PerformanceLevel::Warning
            } else {
                PerformanceLevel::Good
            }
        } else {
            PerformanceLevel::Good
        }
    }

    /// Get config
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }

    /// Update config
    pub fn set_config(&mut self, config: ProfilerConfig) {
        self.config = config;
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Get breakdown percentages for latest frame
    pub fn breakdown(&self) -> FrameBreakdown {
        if let Some(last) = self.history.back() {
            let total = last.total_ms.max(0.001);
            FrameBreakdown {
                layout_pct: last.layout_ms / total * 100.0,
                render_pct: last.render_ms / total * 100.0,
                input_pct: last.input_ms / total * 100.0,
                other_pct: 100.0
                    - (last.layout_ms + last.render_ms + last.input_ms) / total * 100.0,
            }
        } else {
            FrameBreakdown::default()
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    pub min_ms: f32,
    pub max_ms: f32,
    pub avg_ms: f32,
}

/// Frame breakdown percentages
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameBreakdown {
    pub layout_pct: f32,
    pub render_pct: f32,
    pub input_pct: f32,
    pub other_pct: f32,
}

/// Performance warning level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceLevel {
    Good,
    Warning,
    Critical,
}

/// Macro for timing a scope
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _timer = $crate::devtools::profiler::ScopeTimer::new($name);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basics() {
        let mut profiler = Profiler::new();

        profiler.begin_frame();
        std::thread::sleep(Duration::from_millis(1));
        profiler.end_frame();

        assert!(profiler.fps() > 0.0);
        assert!(profiler.latest().is_some());
    }

    #[test]
    fn test_frame_stats() {
        let mut profiler = Profiler::new();

        for _ in 0..10 {
            profiler.begin_frame();
            profiler.end_frame();
        }

        let stats = profiler.frame_stats();
        assert!(stats.avg_ms >= 0.0);
    }
}
