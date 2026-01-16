use crate::core::context::UIContext;
use crate::core::types::NodeID;
use crate::widgets::{WidgetKind, ButtonData};
use super::traits::WidgetBuilder;

pub struct ButtonBuilder<'a> {
    ctx: &'a mut UIContext,
    id: NodeID,
    clicked: bool,
}

impl<'a> ButtonBuilder<'a> {
    pub fn new(ctx: &'a mut UIContext, label: &str) -> Self {
        let stable_id = ctx.get_id(label);
        
        // Interaction Check (using prev frame layout)
        let mut clicked = false;
        if let Some(layout) = ctx.prev_layout.get(&stable_id) {
            // Simple AABB check
            if ctx.mouse_pos.x >= layout.x && ctx.mouse_pos.x <= layout.x + layout.w &&
               ctx.mouse_pos.y >= layout.y && ctx.mouse_pos.y <= layout.y + layout.h 
            {
                 if ctx.mouse_down {
                     clicked = true;
                 }
            }
        }

        let kind = WidgetKind::Button(ButtonData {
            label: label.to_string(),
            is_primary: false,
            is_danger: false,
        });

        let id = ctx.begin_node(label, kind);
        
        // Style
        {
            let style = &mut ctx.store.style[id];
            style.bg = crate::core::types::Color::hex(0x555555FF);
            style.corner_radius = 4.0;
            style.text = crate::core::types::Color::WHITE;
            
            let constraints = &mut ctx.store.constraints[id];
            constraints.width = -1.0; // Auto
            constraints.height = 30.0;
            constraints.padding = 4.0;
        }

        Self { ctx, id, clicked }
    }

    pub fn primary(self) -> Self {
        let id = self.id;
        if let WidgetKind::Button(data) = &mut self.ctx.store.widget[id] {
            data.is_primary = true;
        }
        self.ctx.store.style[id].bg = crate::core::types::Color::hex(0x007ACCFF);
        self
    }

    pub fn danger(self) -> Self {
        let id = self.id;
        if let WidgetKind::Button(data) = &mut self.ctx.store.widget[id] {
            data.is_danger = true;
        }
        self.ctx.store.style[id].bg = crate::core::types::Color::hex(0xE51400FF);
        self
    }

    pub fn clicked(self) -> bool {
        self.clicked
    }
}

impl<'a> WidgetBuilder for ButtonBuilder<'a> {
    fn ctx(&mut self) -> &mut UIContext {
        self.ctx
    }
    fn id(&self) -> NodeID {
        self.id
    }
}

impl<'a> Drop for ButtonBuilder<'a> {
    fn drop(&mut self) {
        self.ctx.end_node();
    }
}
