//! Node and Socket widgets - Visual programming elements
use crate::core::{Vec2, ID, ColorF};
use crate::view::header::{ViewHeader, ViewType};
use std::cell::Cell;

/// Node builder
pub struct NodeBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub title: &'a str,
}

impl<'a> NodeBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn pos(self, x: f32, y: f32) -> Self {
        self.view.pos_x.set(x);
        self.view.pos_y.set(y);
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.title);
        
        // Node dragging logic
        let id = self.view.id.get();
        if crate::view::interaction::is_active(id) {
            let delta = crate::view::interaction::get_mouse_delta();
            self.view.pos_x.set(self.view.pos_x.get() + delta.x);
            self.view.pos_y.set(self.view.pos_y.get() + delta.y);
        }
        
        self.view
    }
}

/// Socket builder
pub struct SocketBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub name: &'a str,
    pub is_input: bool,
}

impl<'a> SocketBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn color(self, color: ColorF) -> Self {
        self.view.fg_color.set(color);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.name);
        self.view.is_bipolar.set(self.is_input); // Reuse for input flag
        
        // Socket wire interaction logic
        let id = self.view.id.get();
        if crate::view::interaction::is_clicked(id) {
             // Start wire drag?
             // For now, let's just log or set a placeholder state
             let pos = crate::view::interaction::get_mouse_pos();
             // We'd need PortId here, but for now we'll just track start pos
             crate::view::interaction::set_wire_state(crate::core::wire::WireState::Dragging {
                 start_port: crate::core::wire::PortId::new(0, 0, if self.is_input { crate::core::wire::PortType::Input } else { crate::core::wire::PortType::Output }),
                 start_pos: pos,
                 end_pos: pos,
             });
        }
        
        self.view
    }
}
