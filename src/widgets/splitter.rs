//! Splitter widget - Draggable panel resizing
use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};

/// Splitter builder
pub struct SplitterBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
}

impl<'a> SplitterBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn ratio(self, r: f32) -> Self {
        self.view.ratio.set(r);
        self
    }

    pub fn vertical(self) -> Self {
        self.view.is_vertical.set(true);
        self
    }

    pub fn horizontal(self) -> Self {
        self.view.is_vertical.set(false);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let id = self.view.id.get();
        let is_active = crate::view::interaction::is_active(id);
        
        if is_active {
            // Get last frame rect to calculate percentage
            if let Some(rect) = crate::view::interaction::get_rect(id) {
                let mouse = crate::view::interaction::get_mouse_pos();
                let is_vertical = self.view.is_vertical.get();
                
                let new_ratio = if is_vertical {
                    ((mouse.y - rect.y) / rect.h).clamp(0.05, 0.95)
                } else {
                    ((mouse.x - rect.x) / rect.w).clamp(0.05, 0.95)
                };
                
                self.view.ratio.set(new_ratio);
                crate::view::interaction::request_cursor(Some(if is_vertical {
                    winit::window::CursorIcon::RowResize
                } else {
                    winit::window::CursorIcon::ColResize
                }));
            }
        } else if crate::view::interaction::is_hot(id) {
             let is_vertical = self.view.is_vertical.get();
             crate::view::interaction::request_cursor(Some(if is_vertical {
                    winit::window::CursorIcon::RowResize
                } else {
                    winit::window::CursorIcon::ColResize
                }));
        }

        self.view
    }

    pub fn changed(&self) -> bool {
        crate::view::interaction::is_changed(self.view.id.get())
    }
}
