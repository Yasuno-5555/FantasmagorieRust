//! Concrete view types
//! Ported from definitions.hpp

use super::header::{ViewHeader, ViewType};
use crate::core::ColorF;
use std::cell::Cell;

/// Box view - basic container
pub struct BoxView<'a> {
    pub header: ViewHeader<'a>,
}

impl<'a> Default for BoxView<'a> {
    fn default() -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Box,
                bg_color: Cell::new(ColorF::new(0.15, 0.15, 0.18, 1.0)),
                ..Default::default()
            },
        }
    }
}

/// Text view - displays text
pub struct TextView<'a> {
    pub header: ViewHeader<'a>,
    pub text: &'a str,
}

impl<'a> TextView<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Text,
                ..Default::default()
            },
            text,
        }
    }
}

/// Button view - interactive button
pub struct ButtonView<'a> {
    pub header: ViewHeader<'a>,
    pub label: &'a str,
    pub hover_color: ColorF,
    pub active_color: ColorF,
}

impl<'a> ButtonView<'a> {
    pub fn new(label: &'a str) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Button,
                bg_color: Cell::new(ColorF::new(0.25, 0.25, 0.3, 1.0)),
                elevation: Cell::new(2.0),
                border_radius_tl: Cell::new(6.0),
                border_radius_tr: Cell::new(6.0),
                border_radius_br: Cell::new(6.0),
                border_radius_bl: Cell::new(6.0),
                ..Default::default()
            },
            label,
            hover_color: ColorF::new(0.35, 0.35, 0.4, 1.0),
            active_color: ColorF::new(0.2, 0.2, 0.25, 1.0),
        }
    }
}

/// Scroll view - scrollable container
pub struct ScrollView<'a> {
    pub header: ViewHeader<'a>,
    pub show_scrollbar: bool,
    pub scrollbar_color: ColorF,
}

impl<'a> Default for ScrollView<'a> {
    fn default() -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Scroll,
                ..Default::default()
            },
            show_scrollbar: true,
            scrollbar_color: ColorF::new(0.3, 0.3, 0.35, 0.6),
        }
    }
}

/// Text input view - single line input
pub struct TextInputView<'a> {
    pub header: ViewHeader<'a>,
    pub buffer: &'a mut str,
    pub placeholder: Option<&'a str>,
}

/// Text area view - multi-line input
pub struct TextAreaView<'a> {
    pub header: ViewHeader<'a>,
    pub buffer: &'a mut str,
    pub placeholder: Option<&'a str>,
    pub line_height: f32,
}

impl<'a> TextAreaView<'a> {
    pub fn new(buffer: &'a mut str) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::TextArea,
                ..Default::default()
            },
            buffer,
            placeholder: None,
            line_height: 14.0 * 1.4,
        }
    }
}

/// Toggle view - boolean switch
pub struct ToggleView<'a> {
    pub header: ViewHeader<'a>,
    pub value: &'a mut bool,
    pub label: Option<&'a str>,
}

/// Slider view - numeric slider
pub struct SliderView<'a> {
    pub header: ViewHeader<'a>,
    pub value: &'a mut f32,
    pub min: f32,
    pub max: f32,
    pub label: Option<&'a str>,
}

impl<'a> SliderView<'a> {
    pub fn new(value: &'a mut f32, min: f32, max: f32) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Slider,
                ..Default::default()
            },
            value,
            min,
            max,
            label: None,
        }
    }
}

/// Splitter view - resizable split container
pub struct SplitterView<'a> {
    pub header: ViewHeader<'a>,
    pub split_ratio: &'a mut f32,
    pub is_vertical: bool,
    pub handle_thickness: f32,
}

impl<'a> SplitterView<'a> {
    pub fn new(split_ratio: &'a mut f32, is_vertical: bool) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Splitter,
                ..Default::default()
            },
            split_ratio,
            is_vertical,
            handle_thickness: 8.0,
        }
    }
}

/// Context menu item
#[derive(Clone, Copy)]
pub struct ContextMenuItem<'a> {
    pub label: &'a str,
    pub triggered: bool,
}

/// Context menu view
pub struct ContextMenuView<'a> {
    pub header: ViewHeader<'a>,
    pub items: &'a [ContextMenuItem<'a>],
    pub is_open: &'a mut bool,
    pub anchor_x: f32,
    pub anchor_y: f32,
}

/// Plot types
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlotType {
    #[default]
    Line,
    Bar,
    Scatter,
}

/// Plot view - data visualization
pub struct PlotView<'a> {
    pub header: ViewHeader<'a>,
    pub data: &'a [f32],
    pub plot_type: PlotType,
    pub line_color: ColorF,
    pub min_val: f32,
    pub max_val: f32,
}

impl<'a> PlotView<'a> {
    pub fn new(data: &'a [f32]) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Plot,
                ..Default::default()
            },
            data,
            plot_type: PlotType::Line,
            line_color: ColorF::new(0.3, 0.6, 1.0, 1.0),
            min_val: 0.0,
            max_val: 1.0,
        }
    }
}

/// Table row builder function type
pub type RowBuilderFn = fn(row_index: usize, user_data: *mut ());

/// Table view - tabular data
pub struct TableView<'a> {
    pub header: ViewHeader<'a>,
    pub row_count: usize,
    pub row_height: f32,
    pub row_builder: Option<RowBuilderFn>,
    pub user_data: *mut (),
}

impl<'a> Default for TableView<'a> {
    fn default() -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Table,
                ..Default::default()
            },
            row_count: 0,
            row_height: 24.0,
            row_builder: None,
            user_data: std::ptr::null_mut(),
        }
    }
}

/// Tree node view - collapsible tree item
pub struct TreeNodeView<'a> {
    pub header: ViewHeader<'a>,
    pub label: &'a str,
    pub expanded: &'a mut bool,
    pub depth: i32,
    pub indent_per_level: f32,
}

impl<'a> TreeNodeView<'a> {
    pub fn new(label: &'a str, expanded: &'a mut bool) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::TreeNode,
                ..Default::default()
            },
            label,
            expanded,
            depth: 0,
            indent_per_level: 20.0,
        }
    }
}

/// Color picker view
pub struct ColorPickerView<'a> {
    pub header: ViewHeader<'a>,
    pub color: &'a mut ColorF,
    pub sv_size: f32,
    pub hue_bar_width: f32,
}

impl<'a> ColorPickerView<'a> {
    pub fn new(color: &'a mut ColorF) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::ColorPicker,
                ..Default::default()
            },
            color,
            sv_size: 150.0,
            hue_bar_width: 20.0,
        }
    }
}

/// Menu bar view
pub struct MenuBarView<'a> {
    pub header: ViewHeader<'a>,
    pub bar_height: f32,
    pub active_menu: i32,
}

impl<'a> Default for MenuBarView<'a> {
    fn default() -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::MenuBar,
                ..Default::default()
            },
            bar_height: 26.0,
            active_menu: -1,
        }
    }
}

/// Menu item view
pub struct MenuItemView<'a> {
    pub header: ViewHeader<'a>,
    pub label: &'a str,
    pub shortcut: Option<&'a str>,
    pub triggered: bool,
    pub is_separator: bool,
}

/// Bezier curve view
pub struct BezierView<'a> {
    pub header: ViewHeader<'a>,
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
    pub thickness: f32,
}

impl<'a> BezierView<'a> {
    pub fn new(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], thickness: f32) -> Self {
        Self {
            header: ViewHeader {
                view_type: ViewType::Bezier,
                ..Default::default()
            },
            p0,
            p1,
            p2,
            p3,
            thickness,
        }
    }
}

/// Drag source view
pub struct DragSourceView<'a> {
    pub header: ViewHeader<'a>,
    pub payload_type: &'a str,
    pub payload_data: *const (),
    pub payload_size: usize,
}

/// Drop target view
pub struct DropTargetView<'a> {
    pub header: ViewHeader<'a>,
    pub accept_type: &'a str,
    pub dropped: bool,
    pub received_data: Option<*const ()>,
}
