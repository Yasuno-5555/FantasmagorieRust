use crate::core::context::UIContext;
use crate::core::types::NodeID;
use crate::widgets::{WidgetKind, SliderData};
use super::traits::WidgetBuilder;

pub struct SliderBuilder<'a> {
    ctx: &'a mut UIContext,
    id: NodeID,
}

impl<'a> SliderBuilder<'a> {
    pub fn new(ctx: &'a mut UIContext, label: &str, value: &mut f32, min: f32, max: f32) -> Self {
        let stable_id = ctx.get_id(label);
        
        // Interaction
        if let Some(layout) = ctx.prev_layout.get(&stable_id) {
             if ctx.mouse_down {
                  if ctx.mouse_pos.x >= layout.x && ctx.mouse_pos.x <= layout.x + layout.w &&
                     ctx.mouse_pos.y >= layout.y && ctx.mouse_pos.y <= layout.y + layout.h 
                  {
                      if layout.w > 0.0 {
                        let ratio = (ctx.mouse_pos.x - layout.x) / layout.w;
                        let ratio = ratio.clamp(0.0, 1.0);
                        *value = min + (max - min) * ratio;
                      }
                  }
             }
        }

        let kind = WidgetKind::Slider(SliderData {
            label: label.to_string(),
            min,
            max,
            value: *value,
        });

        let id = ctx.begin_node(label, kind);
        
        {
             let constraints = &mut ctx.store.constraints[id];
             constraints.width = 200.0;
             constraints.height = 24.0;
             
             let style = &mut ctx.store.style[id];
             style.bg = crate::core::types::Color::hex(0x333333FF);
             style.corner_radius = 2.0;
        }
        
        Self { ctx, id }
    }
}

impl<'a> WidgetBuilder for SliderBuilder<'a> {
    fn ctx(&mut self) -> &mut UIContext {
        self.ctx
    }
    fn id(&self) -> NodeID {
        self.id
    }
}

impl<'a> Drop for SliderBuilder<'a> {
    fn drop(&mut self) {
        self.ctx.end_node();
    }
}
