use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A simple state identifier.
pub type StateId = String;

/// Condition for transitioning between states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    Timer(f32),
    Signal(String),
    Custom(String), // Placeholder for custom logic
}

/// A transition from one state to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub target: StateId,
    pub condition: Condition,
}

/// A single state in the FSM.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct State {
    pub name: StateId,
    pub transitions: Vec<Transition>,
}

/// The StateMachine component.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateMachine {
    pub current_state: StateId,
    pub states: HashMap<StateId, State>,
    pub timer: f32,
}

impl StateMachine {
    pub fn new(initial: &str) -> Self {
        Self {
            current_state: initial.to_string(),
            states: HashMap::new(),
            timer: 0.0,
        }
    }

    pub fn add_state(&mut self, state: State) {
        self.states.insert(state.name.clone(), state);
    }

    pub fn update(&mut self, dt: f32) -> Option<StateId> {
        self.timer += dt;
        
        let mut next_state = None;
        if let Some(state) = self.states.get(&self.current_state) {
            for trans in &state.transitions {
                match trans.condition {
                    Condition::Timer(t) => {
                        if self.timer >= t {
                            next_state = Some(trans.target.clone());
                            break;
                        }
                    }
                    _ => {} // Signal and Custom are handled by the controller
                }
            }
        }

        if let Some(next) = next_state {
            self.current_state = next;
            self.timer = 0.0;
            return Some(self.current_state.clone());
        }

        None
    }
    
    pub fn force_transition(&mut self, target: &str) {
        if self.states.contains_key(target) {
            self.current_state = target.to_string();
            self.timer = 0.0;
        }
    }
}
