use crate::core::{ColorF, Vec2};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Colormap {
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Grayscale,
}

impl Colormap {
    pub fn get_color(&self, t: f32) -> ColorF {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Grayscale => ColorF::new(t, t, t, 1.0),
            Colormap::Viridis => {
                // Simplified Viridis approximation
                // Yellow: (1.0, 0.9, 0.1) at t=1.0
                // Purple: (0.2, 0.0, 0.3) at t=0.0
                if t < 0.25 {
                    let s = t * 4.0;
                    ColorF::new(0.2 + 0.1 * s, 0.0 + 0.2 * s, 0.3 + 0.2 * s, 1.0)
                } else if t < 0.5 {
                    let s = (t - 0.25) * 4.0;
                    ColorF::new(0.3 + 0.0 * s, 0.2 + 0.4 * s, 0.5 - 0.1 * s, 1.0)
                } else if t < 0.75 {
                    let s = (t - 0.5) * 4.0;
                    ColorF::new(0.3 + 0.4 * s, 0.6 + 0.2 * s, 0.4 - 0.2 * s, 1.0)
                } else {
                    let s = (t - 0.75) * 4.0;
                    ColorF::new(0.7 + 0.3 * s, 0.8 + 0.1 * s, 0.2 - 0.1 * s, 1.0)
                }
            }
            Colormap::Plasma => {
                // Simple Plasma approximation (Purple -> Pink -> Yellow)
                if t < 0.5 {
                    let s = t * 2.0;
                    ColorF::new(0.4 + 0.5 * s, 0.1 + 0.1 * s, 0.6 - 0.2 * s, 1.0)
                } else {
                    let s = (t - 0.5) * 2.0;
                    ColorF::new(0.9 + 0.1 * s, 0.2 + 0.6 * s, 0.4 - 0.3 * s, 1.0)
                }
            }
            _ => ColorF::new(t, t, t, 1.0), // TODO: Magma/Inferno
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlotItem<'a> {
    Line {
        points: &'a [Vec2],
        color: ColorF,
        width: f32,
    },
    FastLine {
        y_data: &'a [f32],
        x_start: f32,
        x_step: f32,
        color: ColorF,
        width: f32,
    },
    Heatmap {
        data: &'a [f32],
        width: usize,
        height: usize,
        colormap: Colormap,
        min: f32,
        max: f32,
    },
}

pub struct PlotData<'a> {
    pub items: &'a [PlotItem<'a>],
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub title: Option<&'a str>,
}
