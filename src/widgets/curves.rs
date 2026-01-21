use crate::core::ID;
use crate::view::curves::CurveEditorData;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct CurveEditorBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    id: ID,
    height: f32,
}

impl<'b, 'a> CurveEditorBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, id: &str) -> Self {
        Self {
            ui,
            id: ID::from_str(id),
            height: 250.0,
        }
    }

    pub fn height(mut self, h: f32) -> Self {
        self.height = h;
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let view = self.ui.arena.alloc(ViewHeader {
            view_type: ViewType::CurveEditor,
            id: std::cell::Cell::new(self.id),
            ..Default::default()
        });

        view.height.set(self.height);
        view.flex_grow.set(1.0);

        // Ensure persistent state exists
        crate::view::curves::ensure_curve_data_exists(self.id);

        self.ui.push_child(view);
        view
    }
}
