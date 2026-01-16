use crate::core::context::UIContext;
use crate::core::types::NodeID;

pub trait WidgetBuilder {
    fn ctx(&mut self) -> &mut UIContext;
    fn id(&self) -> NodeID;

    fn width(mut self, w: f32) -> Self where Self: Sized {
        let id = self.id();
        self.ctx().store.constraints[id].width = w;
        self
    }

    fn height(mut self, h: f32) -> Self where Self: Sized {
        let id = self.id();
        self.ctx().store.constraints[id].height = h;
        self
    }

    fn grow(mut self, g: f32) -> Self where Self: Sized {
        let id = self.id();
        self.ctx().store.constraints[id].grow = g;
        self
    }

    fn padding(mut self, p: f32) -> Self where Self: Sized {
        let id = self.id();
        self.ctx().store.constraints[id].padding = p;
        self
    }
}
