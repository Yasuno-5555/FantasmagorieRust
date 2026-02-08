use std::collections::{HashMap, HashSet};
use winit::event::{ElementState, Event, WindowEvent, MouseButton, DeviceEvent};
use winit::keyboard::{PhysicalKey, KeyCode};
use crate::input::action_map::ActionMap;
use crate::input::types::{InputBinding, InputState};
use crate::core::Vec2;

pub struct InputManager {
    action_map: ActionMap,
    
    // Raw state
    pressed_keys: HashSet<KeyCode>,
    pressed_mouse_buttons: HashSet<MouseButton>,
    mouse_position: Vec2,
    mouse_delta: Vec2,
    scroll_delta: Vec2,

    // Action state (Strings for now, could be optimization targets later)
    action_states: HashMap<String, InputState>,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            action_map: ActionMap::new(),
            pressed_keys: HashSet::new(),
            pressed_mouse_buttons: HashSet::new(),
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            scroll_delta: Vec2::ZERO,
            action_states: HashMap::new(),
        }
    }

    pub fn get_action_map_mut(&mut self) -> &mut ActionMap {
        &mut self.action_map
    }

    /// Process winit events to update raw state
    pub fn process_event<T>(&mut self, event: &Event<T>) -> bool {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    if let PhysicalKey::Code(keycode) = key_event.physical_key {
                        match key_event.state {
                            ElementState::Pressed => {
                                self.pressed_keys.insert(keycode);
                            }
                            ElementState::Released => {
                                self.pressed_keys.remove(&keycode);
                            }
                        }
                    }
                    true
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    match state {
                        ElementState::Pressed => {
                            self.pressed_mouse_buttons.insert(*button);
                        }
                        ElementState::Released => {
                            self.pressed_mouse_buttons.remove(button);
                        }
                    }
                    true
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.mouse_position = Vec2::new(position.x as f32, position.y as f32);
                    true
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => {
                            self.scroll_delta = Vec2::new(*x, *y);
                        }
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            self.scroll_delta = Vec2::new(pos.x as f32, pos.y as f32);
                        }
                    }
                    true
                }
                _ => false,
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                self.mouse_delta = Vec2::new(delta.0 as f32, delta.1 as f32);
                true
            }
            _ => false,
        }
    }

    /// Should be called at the beginning of each frame to update Action States based on Raw State
    pub fn update(&mut self) {
        // Reset per-frame deltas if needed (unless we want to accumulate them)
        // self.mouse_delta = Vec2::ZERO; // Depending on usage
        // self.scroll_delta = Vec2::ZERO;

        // Update High-Level Actions
        // In a real optimized system, we might iterate over active bindings rather than tracking every string every frame.
        // For now, we iterate over all defined actions in the map.
        // NOTE: This requires exposing keys from ActionMap, or we just compute on demand.
        // Let's iterate through the map's keys if possible, or just re-evaluate known actions.
        
        // Actually, to correctly support "JustPressed" / "JustReleased", we need previous state.
        // For simplicity in this iteration, "update" resolves the current state based on raw inputs.
        // "JustPressed" logic would require tracking `prev_action_states`.
    }

    pub fn is_action_active(&self, action: &str) -> bool {
        if let Some(bindings) = self.action_map.get_bindings(action) {
            for binding in bindings {
                match binding {
                    InputBinding::Keyboard(key) => {
                        if self.pressed_keys.contains(key) {
                            return true;
                        }
                    }
                    InputBinding::Mouse(btn) => {
                        if self.pressed_mouse_buttons.contains(btn) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    
    pub fn get_mouse_position(&self) -> Vec2 {
        self.mouse_position
    }
}
