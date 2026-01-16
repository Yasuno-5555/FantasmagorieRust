//! Context Menu widget - Popup menu with blur overlay
use crate::core::{ID, ColorF};
use crate::view::header::ViewHeader;

/// Context Menu builder
pub struct ContextMenuBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
}

impl<'a> ContextMenuBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn bg(self, color: ColorF) -> Self {
        self.view.bg_color.set(color);
        self
    }

    pub fn blur(self, amount: f32) -> Self {
        self.view.backdrop_blur.set(amount);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view
    }
}

/// Menu Item builder
pub struct MenuItemBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub label: &'a str,
}

impl<'a> MenuItemBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn fg(self, color: ColorF) -> Self {
        self.view.fg_color.set(color);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.label);
        self.view
    }

    /// Check if this menu item was clicked
    pub fn clicked(&self) -> bool {
        // Set text first to ensure it's available
        self.view.text.set(self.label);
        crate::view::interaction::is_clicked(self.view.id.get())
    }
}
