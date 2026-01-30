//! Micro-Interactions - Premium UI polish
//! Ghost Scrollbar, Glass Toast, Laser Tooltip
use crate::core::{ColorF, Vec2, ID};
use crate::view::header::ViewHeader;

/// Toast notification state
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToastType {
    Info,
    Success,
    Warning,
    Error,
}

/// Glass Toast builder - slide/blur notification
pub struct ToastBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub message: &'a str,
    pub toast_type: ToastType,
}

impl<'a> ToastBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn duration(self, seconds: f32) -> Self {
        self.view.max.set(seconds); // Store duration in max
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.message);

        // Set color based on type
        let color = match self.toast_type {
            ToastType::Info => ColorF::new(0.3, 0.6, 1.0, 1.0),
            ToastType::Success => ColorF::new(0.3, 0.8, 0.4, 1.0),
            ToastType::Warning => ColorF::new(1.0, 0.7, 0.2, 1.0),
            ToastType::Error => ColorF::new(1.0, 0.3, 0.3, 1.0),
        };
        self.view.glow_color.set(color);

        self.view
    }
}

/// Laser Tooltip builder - glowing panel that follows cursor
pub struct TooltipBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub text: &'a str,
}

impl<'a> TooltipBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn glow(self, strength: f32) -> Self {
        self.view.glow_strength.set(strength);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.text);
        self.view
    }
}
