//! Canvas widget - Infinite workspace with panning and zooming
use crate::core::{Vec2, ID};
use crate::view::header::{ViewHeader, ViewType};
use std::cell::Cell;

/// Canvas builder
pub struct CanvasBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
}

impl<'a> CanvasBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn bg(self, color: crate::core::ColorF) -> Self {
        self.view.bg_color.set(color);
        self
    }

    pub fn border_color(self, color: crate::core::ColorF) -> Self {
        self.view.border_color.set(color);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let id = self.view.id.get();
        let (mut offset, mut zoom) = crate::view::interaction::get_canvas_transform(id);

        // Panning: Right mouse or Middle mouse
        if crate::view::interaction::is_hot(id) {
            if crate::view::interaction::is_right_mouse_down()
                || crate::view::interaction::is_middle_mouse_down()
            {
                let delta = crate::view::interaction::get_mouse_delta();
                offset.x += delta.x;
                offset.y += delta.y;
                crate::view::interaction::set_canvas_transform(id, offset, zoom);
            }

            // Zooming: Mouse wheel
            let (scroll_x, scroll_y) = crate::view::interaction::get_scroll_delta();
            if scroll_y != 0.0 {
                let old_zoom = zoom;
                zoom *= 1.1f32.powf(scroll_y / 30.0);
                zoom = zoom.clamp(0.1, 5.0);

                // Adjust offset to zoom towards mouse pos
                let mouse_pos = crate::view::interaction::get_mouse_pos();
                let rect = self.view.computed_rect.get();
                let local_mouse = mouse_pos - Vec2::new(rect.x, rect.y);

                // New offset = Mouse - (Mouse - OldOffset) * (NewZoom / OldZoom)
                // Simplified: offset update based on pivot
                offset = local_mouse - (local_mouse - offset) * (zoom / old_zoom);

                crate::view::interaction::set_canvas_transform(id, offset, zoom);
            }
        }

        self.view
    }

    /// Convenience for immediate mode closure
    pub fn build_with<F>(self, f: F) -> &'a ViewHeader<'a>
    where
        F: FnOnce(),
    {
        // Actually UIContext handles the closure/nesting.
        // This is just a marker.
        self.view
    }
}
