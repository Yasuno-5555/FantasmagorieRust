use crate::core::context::UIContext;
use crate::core::types::NodeID;
use crate::widgets::{WidgetKind, WindowData};
use super::traits::WidgetBuilder;

pub struct WindowBuilder<'a> {
    ctx: &'a mut UIContext,
    id: NodeID,
}

impl<'a> WindowBuilder<'a> {
    pub fn new(ctx: &'a mut UIContext, title: &str) -> Self {
        let kind = WidgetKind::Window(WindowData {
            title: title.to_string(),
            closable: true,
            resizable: false,
            draggable: true,
        });

        // Use title as ID for now
        let id = ctx.begin_node(title, kind);
        
        // Window default styling
        {
            let style = &mut ctx.store.style[id];
            style.bg = crate::core::types::Color::hex(0x1A1A1AFF);
            style.corner_radius = 8.0;
            
            let constraints = &mut ctx.store.constraints[id];
            constraints.width = 400.0;
            constraints.height = 300.0;
            constraints.padding = 8.0;
        }

        Self { ctx, id }
    }

    pub fn children<F>(self, f: F) -> Self 
    where F: FnOnce(&mut UIContext) 
    {
        // Execute children closure
        f(self.ctx);
        self
    }
}

impl<'a> WidgetBuilder for WindowBuilder<'a> {
    fn ctx(&mut self) -> &mut UIContext {
        self.ctx
    }

    fn id(&self) -> NodeID {
        self.id
    }
}

// Drop implementation calls end_node
impl<'a> Drop for WindowBuilder<'a> {
    fn drop(&mut self) {
        self.ctx.end_node();
    }
}
