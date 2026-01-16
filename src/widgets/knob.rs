//! Knob widget - Rotary control with infinite turn and bloom
use crate::core::{ColorF, Vec2, ID};
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext; // Needed? UIContext returns the builder
use std::cell::Cell;

/// Knob builder
pub struct KnobBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub value: &'a mut f32,
    pub min: f32,
    pub max: f32,
    pub label: Option<&'a str>,
}

impl<'a> KnobBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, radius: f32) -> Self {
        self.view.width.set(radius * 2.0);
        self.view.height.set(radius * 2.0);
        self
    }

    /// Set main color (bloom color)
    pub fn color(self, color: ColorF) -> Self {
        self.view.fg_color.set(color);
        self.view.glow_color.set(color);
        self.view.glow_strength.set(1.5); // Default bloom for knob
        self
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self.view.text.set(label);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        // Here we might need to store min/max somewhere if renderer needs normalization?
        // But renderer just draws. Interaction logic happens in UIContext or specialized logic.
        // Wait, Knob needs interaction logic.
        // Usually `view.view_type` triggers specific interaction handling in `interaction.rs`.
        // We'll add ViewType::Knob later.
        // For now, let's reuse Slider logic or just define properties?
        // Actually, we need to store value, min, max references or state.
        // ViewHeader is for layout/render properties. 
        // Logic usually lives in user code (immediate mode), but `binding.rs` needs to handle it.
        // Since this is immediate mode, the interaction logic is:
        // 1. Detect drag.
        // 2. Update value.
        // 3. Render returns.
        
        let id = self.view.id.get();
        
        // Handle interaction (Immediate Mode Logic)
        if crate::view::interaction::is_active(id) {
             crate::view::interaction::request_cursor(None);
             let (_dx, dy) = crate::view::interaction::mouse_delta();
             if dy != 0.0 {
                 let range = self.max - self.min;
                 let modifiers = crate::view::interaction::modifiers();
                 let is_shift = (modifiers & 1) != 0;
                 
                 // Vertical drag: UP (negative dy) increases value
                 let sensitivity = if is_shift { 0.1 } else { 1.0 };
                 let delta = -dy * (range / 200.0) * sensitivity; // 200px for full range
                 
                 *self.value = (*self.value + delta).clamp(self.min, self.max);
             }
        }
        
        // Sync value to view for rendering
        self.view.value.set(*self.value);
        self.view.min.set(self.min);
        self.view.max.set(self.max);
        
        self.view
    }
}
