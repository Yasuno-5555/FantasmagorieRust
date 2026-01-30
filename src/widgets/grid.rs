use crate::core::ID;
use crate::view::grid::GridData;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct GridBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    id: ID,
    step_x: f32,
    step_y: f32,
    offset_x: f32,
    offset_y: f32,
    scale: f32,
}

impl<'b, 'a> GridBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, id: &str) -> Self {
        Self {
            ui,
            id: ID::from_str(id),
            step_x: 50.0,
            step_y: 50.0,
            offset_x: 0.0,
            offset_y: 0.0,
            scale: 1.0,
        }
    }

    pub fn step(mut self, x: f32, y: f32) -> Self {
        self.step_x = x;
        self.step_y = y;
        self
    }

    pub fn offset(mut self, x: f32, y: f32) -> Self {
        self.offset_x = x;
        self.offset_y = y;
        self
    }

    pub fn scale(mut self, s: f32) -> Self {
        self.scale = s;
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let view = self.ui.arena.alloc(ViewHeader {
            view_type: ViewType::Grid,
            id: std::cell::Cell::new(self.id),
            ..Default::default()
        });

        // Grid usually fills available space
        view.width.set(f32::NAN);
        view.height.set(f32::NAN);
        view.flex_grow.set(1.0);

        let data = self.ui.arena.alloc(GridData::new());
        data.step_x.set(self.step_x);
        data.step_y.set(self.step_y);
        data.offset_x.set(self.offset_x);
        data.offset_y.set(self.offset_y);
        data.scale.set(self.scale);

        view.grid_data.set(Some(data));

        self.ui.push_child(view);
        view
    }
}
