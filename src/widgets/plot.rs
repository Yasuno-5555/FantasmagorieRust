//! Plot widget - Real-time data visualization
use crate::core::{ColorF, Vec2, ID};
use crate::view::header::{ViewHeader, ViewType};
use std::cell::Cell;

/// Plot builder
pub struct PlotBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub data: &'a [f32],
    pub min: f32,
    pub max: f32,
}

impl<'a> PlotBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn color(self, color: ColorF) -> Self {
        self.view.fg_color.set(color);
        self.view.glow_color.set(color);
        self
    }
    
    pub fn build(self) -> &'a ViewHeader<'a> {
         self.view.plot_data.set(Some(self.data));
         self.view.min.set(self.min);
         self.view.max.set(self.max);
         self.view
    }
}
