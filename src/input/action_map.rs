use std::collections::HashMap;
use crate::input::types::InputBinding;

#[derive(Default)]
pub struct ActionMap {
    bindings: HashMap<String, Vec<InputBinding>>,
}

impl ActionMap {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    pub fn bind(&mut self, action: &str, binding: InputBinding) {
        self.bindings
            .entry(action.to_string())
            .or_default()
            .push(binding);
    }

    pub fn unbind(&mut self, action: &str, binding: InputBinding) {
        if let Some(links) = self.bindings.get_mut(action) {
            links.retain(|b| *b != binding);
        }
    }

    pub fn get_bindings(&self, action: &str) -> Option<&Vec<InputBinding>> {
        self.bindings.get(action)
    }
}
