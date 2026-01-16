use crate::core::context::UIContext;
use crate::core::types::NodeID;
use crate::widgets::{WidgetKind, LabelData};
use super::traits::WidgetBuilder;

pub struct LabelBuilder<'a> {
    ctx: &'a mut UIContext,
    id: NodeID,
}

impl<'a> LabelBuilder<'a> {
    pub fn new(ctx: &'a mut UIContext, text: &str) -> Self {
        let kind = WidgetKind::Label(LabelData {
            text: text.to_string(),
            bold: false,
        });

        // Use mixed ID potentially? For now text.
        let id = ctx.begin_node(text, kind); 
        
        {
             let render = &mut ctx.store.render[id];
             render.is_text = true;
             render.text = text.to_string();
             
             let constraints = &mut ctx.store.constraints[id];
             constraints.width = -1.0;
             constraints.height = 20.0;
        }
        
        Self { ctx, id }
    }

    pub fn bold(self) -> Self {
        let id = self.id;
        if let WidgetKind::Label(data) = &mut self.ctx.store.widget[id] {
            data.bold = true;
        }
        self
    }
}

impl<'a> WidgetBuilder for LabelBuilder<'a> {
    fn ctx(&mut self) -> &mut UIContext {
        self.ctx
    }
    fn id(&self) -> NodeID {
        self.id
    }
}

impl<'a> Drop for LabelBuilder<'a> {
    fn drop(&mut self) {
        self.ctx.end_node();
    }
}
