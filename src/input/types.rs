use winit::event::MouseButton;
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputBinding {
    Keyboard(KeyCode),
    Mouse(MouseButton),
    // Gamepad support can be added later
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputState {
    Pressed,
    Held,
    Released,
    None,
}

impl InputState {
    pub fn is_active(&self) -> bool {
        match self {
            InputState::Pressed | InputState::Held => true,
            _ => false,
        }
    }

    pub fn is_just_pressed(&self) -> bool {
        matches!(self, InputState::Pressed)
    }

    pub fn is_just_released(&self) -> bool {
        matches!(self, InputState::Released)
    }
}
