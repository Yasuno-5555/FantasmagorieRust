//! Fader widget - Professional vertical slider
use crate::core::{ColorF, Vec2, ID};
use crate::view::header::{ViewHeader, ViewType};
use std::cell::Cell;

/// Fader builder
pub struct FaderBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub value: &'a mut f32,
    pub min: f32,
    pub max: f32,
}

impl<'a> FaderBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn bipolar(self, enabled: bool) -> Self {
        self.view.is_bipolar.set(enabled);
        self
    }

    pub fn logarithmic(self, enabled: bool) -> Self {
        self.view.is_logarithmic.set(enabled);
        self
    }

    pub fn color(self, color: ColorF) -> Self {
        self.view.fg_color.set(color);
        self.view.glow_color.set(color);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let id = self.view.id.get();

        // Handle interaction
        if crate::view::interaction::is_active(id) {
            let (_dx, dy) = crate::view::interaction::mouse_delta();
            if dy != 0.0 {
                let range = self.max - self.min;
                let modifiers = crate::view::interaction::modifiers();
                let is_shift = (modifiers & 1) != 0;

                let sensitivity = if is_shift { 0.1 } else { 1.0 };

                // Fader is vertical: Drag UP (decreasing dy) increases value
                // Pixel height is view.height
                let h = self.view.height.get().max(10.0);
                let delta = -dy * (range / h) * sensitivity;

                *self.value = (*self.value + delta).clamp(self.min, self.max);
            }
        }

        // Sync
        self.view.value.set(*self.value);
        self.view.min.set(self.min);
        self.view.max.set(self.max);

        self.view
    }
}
