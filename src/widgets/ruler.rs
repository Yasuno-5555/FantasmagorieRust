use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};
use crate::view::ruler::{RulerData, RulerOrientation};
use crate::widgets::UIContext;

pub struct RulerBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    id: ID,
    orientation: RulerOrientation,
    start: f32,
    scale: f32,
}

impl<'b, 'a> RulerBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, id: &str) -> Self {
        Self {
            ui,
            id: ID::from_str(id),
            orientation: RulerOrientation::Horizontal,
            start: 0.0,
            scale: 1.0,
        }
    }

    pub fn vertical(mut self) -> Self {
        self.orientation = RulerOrientation::Vertical;
        self
    }

    pub fn horizontal(mut self) -> Self {
        self.orientation = RulerOrientation::Horizontal;
        self
    }

    pub fn start(mut self, s: f32) -> Self {
        self.start = s;
        self
    }

    pub fn scale(mut self, s: f32) -> Self {
        self.scale = s;
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let view = self.ui.arena.alloc(ViewHeader {
            view_type: ViewType::Ruler,
            id: std::cell::Cell::new(self.id),
            ..Default::default()
        });

        if self.orientation == RulerOrientation::Horizontal {
            view.width.set(f32::NAN); // Fill width
            view.height.set(20.0);
            view.flex_grow.set(1.0);
        } else {
            view.width.set(20.0);
            view.height.set(f32::NAN); // Fill height
            view.flex_grow.set(1.0);
        }

        let data = self.ui.arena.alloc(RulerData::new());
        data.orientation.set(self.orientation);
        data.start.set(self.start);
        data.scale.set(self.scale);

        view.ruler_data.set(Some(data));

        self.ui.push_child(view);
        view
    }
}
