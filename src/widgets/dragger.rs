//! ValueDragger widget - Professional numeric input with drag and type support
use crate::core::{ColorF, Vec2, ID};
use crate::view::header::{ViewHeader, ViewType};
use std::cell::Cell;

/// ValueDragger builder
pub struct ValueDraggerBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub value: &'a mut f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

impl<'a> ValueDraggerBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn step(mut self, step: f32) -> Self {
        self.step = step;
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let id = self.view.id.get();

        // Handle Interaction
        if self.view.is_editing.get() {
            // Hybrid Mode: TextInput logic
            // In immediate mode, the widget itself could handle the input transition
            // or we expect the user to handle it.
            // For Specialty Store parity, we handle it internally if possible.

            if crate::view::interaction::is_key_pressed(winit::keyboard::KeyCode::Enter) {
                self.view.is_editing.set(false);
                // Parsing would happen here or in render pass?
                // Usually we want to parse it now.
                let text = self.view.text.get();
                if let Ok(val) = text.parse::<f32>() {
                    *self.value = val.clamp(self.min, self.max);
                }
            }
        } else {
            if crate::view::interaction::is_clicked(id) {
                // Click to edit
                self.view.is_editing.set(true);
                // Initialize text with current value
                // self.view.text.set(...) - Need formatted string.
            } else if crate::view::interaction::is_active(id) {
                // Drag to change
                crate::view::interaction::request_cursor(None);
                let (dx, _dy) = crate::view::interaction::mouse_delta();
                if dx != 0.0 {
                    let modifiers = crate::view::interaction::modifiers();
                    let is_shift = (modifiers & 1) != 0;
                    let sensitivity = if is_shift { 0.1 } else { 1.0 };

                    let range = self.max - self.min;
                    let delta = dx * (range / 400.0) * sensitivity;

                    *self.value = (*self.value + delta).clamp(self.min, self.max);
                }
            }
        }

        // Sync
        self.view.value.set(*self.value);
        self.view.min.set(self.min);
        self.view.max.set(self.max);

        self.view
    }
}
