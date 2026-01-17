use crate::core::{ColorF, Vec2};
use crate::view::plot::{Colormap, PlotData, PlotItem};
use crate::widgets::UIContext;

pub struct PlotBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    title: Option<&'a str>,
    items: Vec<PlotItem<'a>>,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    height: f32,
    width_fill: bool,
}

impl<'b, 'a> PlotBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>) -> Self {
        Self {
            ui,
            title: None,
            items: Vec::new(),
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
            height: 200.0,
            width_fill: true,
        }
    }

    pub fn title(mut self, title: &'a str) -> Self {
        self.title = Some(title);
        self
    }

    pub fn height(mut self, h: f32) -> Self {
        self.height = h;
        self
    }

    pub fn x_range(mut self, min: f32, max: f32) -> Self {
        self.x_min = min;
        self.x_max = max;
        self
    }

    pub fn y_range(mut self, min: f32, max: f32) -> Self {
        self.y_min = min;
        self.y_max = max;
        self
    }

    pub fn line(mut self, points: &'a [Vec2], color: ColorF) -> Self {
        self.items.push(PlotItem::Line {
            points,
            color,
            width: 2.0,
        });
        self
    }

    pub fn fast_line(
        mut self,
        y_data: &'a [f32],
        x_start: f32,
        x_step: f32,
        color: ColorF,
    ) -> Self {
        self.items.push(PlotItem::FastLine {
            y_data,
            x_start,
            x_step,
            color,
            width: 1.5,
        });
        self
    }

    pub fn heatmap(
        mut self,
        data: &'a [f32],
        width: usize,
        height: usize,
        colormap: Colormap,
    ) -> Self {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        if data.is_empty() {
            min = 0.0;
            max = 1.0;
        } else {
            for v in data {
                if *v < min {
                    min = *v;
                }
                if *v > max {
                    max = *v;
                }
            }
        }

        self.items.push(PlotItem::Heatmap {
            data,
            width,
            height,
            colormap,
            min,
            max,
        });
        self
    }

    pub fn heatmap_with_range(
        mut self,
        data: &'a [f32],
        width: usize,
        height: usize,
        colormap: Colormap,
        min: f32,
        max: f32,
    ) -> Self {
        self.items.push(PlotItem::Heatmap {
            data,
            width,
            height,
            colormap,
            min,
            max,
        });
        self
    }

    pub fn build(self) {
        // Create container view
        let view = self.ui.arena.alloc(crate::view::header::ViewHeader {
            view_type: crate::view::header::ViewType::Plot,
            id: std::cell::Cell::new(crate::core::ID::from_u64(self.ui.next_id())),
            ..Default::default()
        });

        view.height.set(self.height);
        if self.width_fill {
            view.align.set(crate::view::header::Align::Stretch);
            view.width.set(f32::NAN);
        } else {
            view.width.set(self.height); // Square default if not fill?
        }

        // Allocate items slice on arena
        let items_slice = self.ui.arena.alloc_slice(self.items.as_slice());

        // Allocate PlotData on arena
        let plot_data = self.ui.arena.alloc(PlotData {
            items: items_slice,
            x_min: self.x_min,
            x_max: self.x_max,
            y_min: self.y_min,
            y_max: self.y_max,
            title: self.title,
        });

        // Assign to view
        view.plot_data.set(Some(plot_data));

        self.ui.push_child(view);
    }
}
