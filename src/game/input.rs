use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Semantic actions in the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    Jump,
    Attack,
    Interact,
}

/// Abstract mapping of inputs to actions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionMap {
    /// Keyboard mapping: code -> Action
    pub keyboard: HashMap<u32, Action>,
    /// Controller mapping: button -> Action (Mock)
    pub controller: HashMap<u32, Action>,
}

impl ActionMap {
    pub fn new_default() -> Self {
        let mut map = Self::default();
        // Default WASD + Space
        map.keyboard.insert(17, Action::MoveUp);    // W (Approx code)
        map.keyboard.insert(31, Action::MoveDown);  // S
        map.keyboard.insert(30, Action::MoveLeft);  // A
        map.keyboard.insert(32, Action::MoveRight); // D
        map.keyboard.insert(57, Action::Jump);      // Space
        map.keyboard.insert(28, Action::Interact);  // Enter
        map
    }
}

/// The state of an action in the current frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionState {
    pub active_actions: HashMap<Action, bool>,
}

impl ActionState {
    pub fn is_active(&self, action: Action) -> bool {
        *self.active_actions.get(&action).unwrap_or(&false)
    }

    pub fn set_active(&mut self, action: Action, active: bool) {
        self.active_actions.insert(action, active);
    }
}
