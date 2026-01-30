//! Engine context and state management
//! Ported from context.hpp

use super::{FrameArena, Rectangle, Vec2, ID};
use crate::core::persistence::PersistenceManager;
use crate::core::types::WindowID;
use std::collections::HashMap;

/// Input context - raw facts from OS
/// Per Iron Philosophy XI: "Input は事実であれ、意味になるな"
/// NO interpretation methods here. Just raw facts.
#[derive(Default)]
pub struct InputContext {
    // Mouse position (facts)
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_dx: f32,
    pub mouse_dy: f32,

    // Mouse button state (facts)
    pub mouse_down: bool,     // Current frame: is button held?
    pub mouse_was_down: bool, // Previous frame: was button held?

    // Wheel delta (fact)
    pub mouse_wheel: f32,

    // Keyboard (facts)
    pub chars: Vec<char>,
    pub keys: Vec<u32>,

    // IME (facts)
    pub ime_active: bool,
    pub ime_composition: String,
    pub ime_cursor: i32,
    pub ime_result: String,
}

impl InputContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frame update (copy current to previous)
    pub fn advance_frame(&mut self) {
        self.mouse_was_down = self.mouse_down;
        self.mouse_wheel = 0.0;
        self.chars.clear();
        self.keys.clear();
        self.ime_result.clear();
    }

    /// Mouse just pressed this frame
    pub fn mouse_just_pressed(&self) -> bool {
        self.mouse_down && !self.mouse_was_down
    }

    /// Mouse just released this frame
    pub fn mouse_just_released(&self) -> bool {
        !self.mouse_down && self.mouse_was_down
    }
}

/// Interaction state - ephemeral per-frame
/// Per Iron Philosophy XII: "RuntimeState は揮発性メモ"
#[derive(Default)]
pub struct InteractionState {
    pub hot_id: ID,    // Currently hovered (this frame)
    pub active_id: ID, // Currently pressed/captured
    pub focus_id: ID,  // Keyboard focus candidate

    // Capture state (for dragging)
    pub captured_id: ID,
}

impl InteractionState {
    pub fn reset_frame(&mut self) {
        self.hot_id = ID::NONE;
        // active_id and focus_id persist within frame
    }

    pub fn is_hot(&self, id: ID) -> bool {
        self.hot_id == id
    }

    pub fn is_active(&self, id: ID) -> bool {
        self.active_id == id
    }

    pub fn is_focused(&self, id: ID) -> bool {
        self.focus_id == id
    }

    pub fn is_captured(&self, id: ID) -> bool {
        self.captured_id == id
    }

    pub fn set_hot(&mut self, id: ID) {
        self.hot_id = id;
    }

    pub fn set_active(&mut self, id: ID) {
        self.active_id = id;
    }

    pub fn set_focus(&mut self, id: ID) {
        self.focus_id = id;
    }

    pub fn capture(&mut self, id: ID) {
        self.captured_id = id;
    }

    pub fn release(&mut self) {
        self.captured_id = ID::NONE;
    }
}

/// Persistent state - survives frames
#[derive(Default)]
pub struct PersistentState {
    /// ID -> Last frame's computed rect (for 1-frame-delayed hit test)
    pub last_frame_rects: HashMap<u64, Rectangle>,

    /// ID -> Scroll offset (for scroll containers)
    pub scroll_offsets: HashMap<u64, Vec2>,

    /// ID -> Animation state (for transitions)
    pub animations: HashMap<u64, AnimationState>,

    /// Currently focused element
    pub focused_id: ID,
}

/// Animation state for smooth transitions
#[derive(Clone, Default)]
pub struct AnimationState {
    pub value: f32,
    pub target: f32,
    pub velocity: f32,
}

impl PersistentState {
    pub fn store_rect(&mut self, id: ID, rect: Rectangle) {
        self.last_frame_rects.insert(id.0, rect);
    }

    pub fn get_rect(&self, id: ID) -> Rectangle {
        self.last_frame_rects
            .get(&id.0)
            .copied()
            .unwrap_or(Rectangle::ZERO)
    }

    pub fn get_scroll(&self, id: ID) -> Vec2 {
        self.scroll_offsets
            .get(&id.0)
            .copied()
            .unwrap_or(Vec2::ZERO)
    }

    pub fn set_scroll(&mut self, id: ID, offset: Vec2) {
        self.scroll_offsets.insert(id.0, offset);
    }
}

/// Debug configuration
#[derive(Default)]
pub struct DebugConfig {
    pub show_debug_overlay: bool,
    pub deterministic_mode: bool,
    pub fixed_dt: f32,
}

/// Frame state - per-frame transient
pub struct FrameState {
    pub arena: FrameArena,
    pub dt: f32,
    pub time: f64,
}

impl Default for FrameState {
    fn default() -> Self {
        Self {
            arena: FrameArena::new(),
            dt: 1.0 / 60.0,
            time: 0.0,
        }
    }
}

impl FrameState {
    pub fn reset(&mut self) {
        self.arena.reset();
    }
}

/// Per-Window Context
/// Holds all state relevant to a single OS window
pub struct WindowContext {
    pub id: WindowID,
    pub frame: FrameState,
    pub input: InputContext,
    pub interaction: InteractionState,
    pub persistent: PersistentState,
    pub width: u32,
    pub height: u32,
}

impl WindowContext {
    pub fn new(id: WindowID, width: u32, height: u32) -> Self {
        Self {
            id,
            frame: FrameState::default(),
            input: InputContext::default(),
            interaction: InteractionState::default(),
            persistent: PersistentState::default(),
            width,
            height,
        }
    }

    pub fn begin_frame(&mut self, dt: f32) {
        self.frame.reset();
        self.frame.dt = dt;
        self.frame.time += dt as f64;
        self.interaction.reset_frame();
    }

    pub fn end_frame(&mut self) {
        self.input.advance_frame();
    }
}

/// Main engine context
/// Manages multiple windows and shared configuration
pub struct EngineContext {
    pub config: DebugConfig,
    pub windows: HashMap<WindowID, WindowContext>,
    pub persistence: PersistenceManager,
    // Kept for backward compatibility/default convenience
    // This points to the "Main" window (ID=0)
    // NOTE: In Rust we can't easily return a ref from a getter that calculates,
    // so we might need to change how we access the "current" window.
    // For now, we'll auto-create window 0 on new()
}

impl Default for EngineContext {
    fn default() -> Self {
        let mut windows = HashMap::new();
        windows.insert(
            WindowID::MAIN,
            WindowContext::new(WindowID::MAIN, 1280, 720),
        );

        Self {
            config: DebugConfig::default(),
            windows,
            persistence: PersistenceManager::default(),
        }
    }
}

impl EngineContext {
    pub fn new(width: u32, height: u32) -> Self {
        let mut windows = HashMap::new();
        windows.insert(
            WindowID::MAIN,
            WindowContext::new(WindowID::MAIN, width, height),
        );

        Self {
            windows,
            persistence: PersistenceManager::default(),
            ..Default::default()
        }
    }

    /// Get a window context by ID
    pub fn get_window(&mut self, id: WindowID) -> Option<&mut WindowContext> {
        self.windows.get_mut(&id)
    }

    /// Get the main window (ID=0) - For backward compatibility
    pub fn main_window(&mut self) -> &mut WindowContext {
        self.windows
            .entry(WindowID::MAIN)
            .or_insert_with(|| WindowContext::new(WindowID::MAIN, 1280, 720))
    }

    /// Register or ensure a window exists
    pub fn ensure_window(&mut self, id: WindowID, width: u32, height: u32) -> &mut WindowContext {
        self.windows
            .entry(id)
            .or_insert_with(|| WindowContext::new(id, width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_mouse_events() {
        let mut input = InputContext::new();

        // Initial state
        assert!(!input.mouse_just_pressed());
        assert!(!input.mouse_just_released());

        // Press
        input.mouse_down = true;
        assert!(input.mouse_just_pressed());

        // Hold
        input.advance_frame();
        input.mouse_down = true;
        assert!(!input.mouse_just_pressed());

        // Release
        input.advance_frame();
        input.mouse_down = false;
        assert!(input.mouse_just_released());
    }

    #[test]
    fn test_interaction_state() {
        let mut state = InteractionState::default();
        let id = ID::from_str("button1");

        state.set_hot(id);
        assert!(state.is_hot(id));

        state.set_active(id);
        assert!(state.is_active(id));

        state.capture(id);
        assert!(state.is_captured(id));

        state.release();
        assert!(!state.is_captured(id));
    }
}
