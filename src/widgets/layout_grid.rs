use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct LayoutGridBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    view: &'a ViewHeader<'a>,
}

impl<'b, 'a> LayoutGridBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, cols: usize) -> Self {
        let id = ID::from_u64(ui.next_id());
        let view = ui.arena.alloc(ViewHeader {
            view_type: ViewType::LayoutGrid,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });
        
        // Store column count in value (as f32)
        view.value.set(cols as f32);
        
        // Default to stretch
        view.width.set(0.0); // 0 means auto/stretch in some contexts, or we set flex_grow
        view.flex_grow.set(1.0); 

        ui.push_child(view);

        Self {
            ui,
            view,
        }
    }
    
    pub fn new_scroll(ui: &'b mut UIContext<'a>, cols: usize) -> Self {
        let s = Self::new(ui, cols);
        s.view.scroll_y.set(true);
        s
    }

    pub fn gap(self, gap: f32) -> Self {
        // We can reuse padding/margin for gap? Or simple logic: padding = gap?
        // Let's use wobble_x/y for gap x/y if unused? Hacky.
        // Or reuse `points` (Array of Vec2) for generic storage?
        // ViewHeader has `wobble_x`. 
        // Let's just use `margin` for gap for now (uniform).
        self.view.margin.set(gap);
        self
    }
    
    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view
    }
}
