//! Collapsible Container widget - Expandable panel with spring animation
use crate::core::{ID, ColorF};
use crate::view::header::ViewHeader;

/// Collapsible container builder
pub struct CollapsibleBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub title: &'a str,
    pub initial_open: bool,
}

impl<'a> CollapsibleBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, _h: f32) -> Self {
        self.view.width.set(w);
        // Height is dynamic based on content and open state
        self
    }

    pub fn bg(self, color: ColorF) -> Self {
        self.view.bg_color.set(color);
        self
    }

    pub fn header_height(self, h: f32) -> Self {
        self.view.min.set(h); // Store header height in min field
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.title);
        self.view.is_expanded.set(self.initial_open);
        self.view
    }
}
