//! ViewHeader - Common structure for all views
//! Ported from definitions.hpp
//!
//! Uses Cell<T> for layout outputs to enable interior mutability
//! during layout pass (solving Rust borrow checker constraints)

use crate::core::{ColorF, Rectangle, Vec2, ID};
use std::cell::Cell;

/// View type enum
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ViewType {
    #[default]
    Box,
    Text,
    Button,
    Scroll,
    TextInput,
    Toggle,
    Slider,
    ContextMenu,
    Plot,
    Node,
    Socket,
    Canvas,
    Wire,
    Table,
    TreeNode,
    ColorPicker,
    MenuBar,
    MenuItem,
    DragSource,
    DropTarget,
    TextArea,
    Splitter,
    Bezier,
    Image,
    Markdown,
    Path,
    Knob,
    Fader,
    ValueDragger,
    Collapsible,
    Toast,
    Tooltip,
    Scene3D,
    CurveEditor,
    Ruler,
    Grid,
    TransformGizmo,
    Math,

    _MAX,
}

/// Alignment for cross-axis
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Align {
    Start,
    Center,
    End,
    #[default]
    Stretch,
}

/// Size specification for layout
#[derive(Clone, Copy, Debug, Default)]
pub struct Size {
    pub w: f32,
    pub h: f32,
}

impl Size {
    pub const ZERO: Size = Size { w: 0.0, h: 0.0 };

    pub const fn new(w: f32, h: f32) -> Self {
        Self { w, h }
    }
}

/// Common header for all views
/// Universal Masquerade: All common properties live here.
///
/// Uses Cell<T> for layout outputs to enable interior mutability.
/// This allows the layout engine to write computed values while
/// holding shared references to the tree.
#[derive(Clone)]
pub struct ViewHeader<'a> {
    // --- Identity ---
    pub view_type: ViewType,
    pub id: Cell<ID>,

    // --- Tree Structure (intrusive linked list) ---
    /// Next sibling in the linked list
    pub next_sibling: Cell<Option<&'a ViewHeader<'a>>>,
    /// First child in the linked list
    pub first_child: Cell<Option<&'a ViewHeader<'a>>>,

    // --- Layout Inputs (Cell for interior mutability during build/layout) ---
    pub width: Cell<f32>,  // 0 = Auto
    pub height: Cell<f32>, // 0 = Auto
    pub pos_x: Cell<f32>,
    pub pos_y: Cell<f32>,
    pub padding: Cell<f32>,
    pub margin: Cell<f32>,
    pub flex_grow: Cell<f32>,
    pub flex_shrink: Cell<f32>,
    pub is_row: Cell<bool>,
    pub wrap: Cell<bool>,
    pub is_squircle: Cell<bool>,
    pub is_bipolar: Cell<bool>,
    pub is_logarithmic: Cell<bool>,
    pub is_editing: Cell<bool>,
    pub clip: Cell<bool>,
    pub align: Cell<Align>,
    pub scroll_x: Cell<bool>,
    pub scroll_y: Cell<bool>,

    // --- Style Inputs (Cell for interior mutability) ---
    // Note: Cell makes them mutable via shared reference
    pub bg_color: Cell<ColorF>,
    pub fg_color: Cell<ColorF>,
    pub border_radius_tl: Cell<f32>,
    pub border_radius_tr: Cell<f32>,
    pub border_radius_br: Cell<f32>,
    pub border_radius_bl: Cell<f32>,
    pub border_width: Cell<f32>,
    pub border_color: Cell<ColorF>,
    pub bg_hover: Cell<Option<ColorF>>,
    pub bg_active: Cell<Option<ColorF>>,
    pub elevation: Cell<f32>,
    pub backdrop_blur: Cell<f32>,
    pub glow_strength: Cell<f32>,
    pub glow_color: Cell<ColorF>,
    pub wobble_x: Cell<f32>,
    pub wobble_y: Cell<f32>,
    pub font_size: Cell<f32>,

    // String refs are Copy (impl Copy for &str), Cell requires Copy.
    // &str is Copy.
    pub text: Cell<&'a str>,
    pub icon: Cell<&'a str>,
    pub icon_size: Cell<f32>, // 0 = same as font_size

    // --- Slider/Toggle value ---
    pub value: Cell<f32>,
    pub min: Cell<f32>,
    pub max: Cell<f32>,

    // --- Text Content ---


    // --- Bezier ---
    pub points: Cell<[Vec2; 4]>,
    pub thickness: Cell<f32>,

    // --- Image ---
    pub texture_id: Cell<Option<u64>>,

    // --- Splitter ---
    pub ratio: Cell<f32>,
    pub is_vertical: Cell<bool>,

    // --- Collapsible ---
    pub is_expanded: Cell<bool>,

    // --- ColorPicker ---
    pub color_hsv: Cell<[f32; 3]>, // H, S, V

    // --- Path ---
    pub content_height: Cell<f32>, // Target height for animation
    pub path: Cell<Option<&'a crate::draw::path::Path>>,

    // --- Plot ---
    pub plot_data: Cell<Option<&'a crate::view::plot::PlotData<'a>>>,

    // --- Scene3D ---
    pub scene_data: Cell<Option<&'a crate::view::scene3d::Scene3DData>>,

    // --- CurveEditor ---
    pub curve_data: Cell<Option<&'a crate::view::curves::CurveEditorData>>,
    pub ruler_data: Cell<Option<&'a crate::view::ruler::RulerData>>,
    pub grid_data: Cell<Option<&'a crate::view::grid::GridData>>,
    pub gizmo_data: Cell<Option<&'a crate::view::gizmo::GizmoData>>,

    // --- Interaction Output ---
    // --- Layout Outputs (Cell for interior mutability) ---
    pub measured_size: Cell<Size>,
    pub content_size: Cell<Size>,
    pub computed_rect: Cell<Rectangle>,
}

impl<'a> Default for ViewHeader<'a> {
    fn default() -> Self {
        Self {
            view_type: ViewType::Box,
            id: Cell::new(ID::NONE),
            next_sibling: Cell::new(None),
            first_child: Cell::new(None),

            // Layout
            width: Cell::new(0.0),
            height: Cell::new(0.0),
            pos_x: Cell::new(0.0),
            pos_y: Cell::new(0.0),
            padding: Cell::new(0.0),
            margin: Cell::new(0.0),
            flex_grow: Cell::new(0.0),
            flex_shrink: Cell::new(1.0),
            is_row: Cell::new(false),
            wrap: Cell::new(false),
            is_squircle: Cell::new(false),
            is_bipolar: Cell::new(false),
            is_logarithmic: Cell::new(false),
            is_editing: Cell::new(false),
            clip: Cell::new(false),
            align: Cell::new(Align::Stretch),
            scroll_x: Cell::new(false),
            scroll_y: Cell::new(false),

            // Style
            bg_color: Cell::new(ColorF::TRANSPARENT),
            fg_color: Cell::new(ColorF::WHITE),
            border_radius_tl: Cell::new(0.0),
            border_radius_tr: Cell::new(0.0),
            border_radius_br: Cell::new(0.0),
            border_radius_bl: Cell::new(0.0),
            border_width: Cell::new(0.0),
            border_color: Cell::new(ColorF::TRANSPARENT),
            bg_hover: Cell::new(None),
            bg_active: Cell::new(None),
            elevation: Cell::new(0.0),
            backdrop_blur: Cell::new(0.0),
            glow_strength: Cell::new(0.0),
            glow_color: Cell::new(ColorF::TRANSPARENT),
            wobble_x: Cell::new(0.0),
            wobble_y: Cell::new(0.0),
            font_size: Cell::new(14.0),
            text: Cell::new(""),
            icon: Cell::new(""),
            icon_size: Cell::new(0.0),

            // Values
            value: Cell::new(0.0),
            min: Cell::new(0.0),
            max: Cell::new(1.0),

            // Bezier
            points: Cell::new([Vec2::ZERO; 4]),
            thickness: Cell::new(1.0),



            // Image
            texture_id: Cell::new(None),

            // Splitter
            ratio: Cell::new(0.5),
            is_vertical: Cell::new(false),

            // Collapsible
            is_expanded: Cell::new(true),
            content_height: Cell::new(0.0),

            // ColorPicker
            color_hsv: Cell::new([0.0, 1.0, 1.0]),

            // Path
            path: Cell::new(None),

            // Plot
            plot_data: Cell::new(None),

            // Scene3D
            scene_data: Cell::new(None),

            // CurveEditor
            curve_data: Cell::new(None),
            ruler_data: Cell::new(None),
            grid_data: Cell::new(None),
            gizmo_data: Cell::new(None),

            // Outputs
            measured_size: Cell::new(Size::ZERO),
            content_size: Cell::new(Size::ZERO),
            computed_rect: Cell::new(Rectangle::ZERO),
        }
    }
}

impl<'a> ViewHeader<'a> {
    /// Add a child to this view
    pub fn add_child(&self, child: &'a ViewHeader<'a>) {
        if self.first_child.get().is_none() {
            self.first_child.set(Some(child));
        } else {
            // Find last child
            let mut current = self.first_child.get();
            while let Some(c) = current {
                if c.next_sibling.get().is_none() {
                    c.next_sibling.set(Some(child));
                    return;
                }
                current = c.next_sibling.get();
            }
        }
    }

    /// Iterate over children
    pub fn children(&self) -> ChildIter<'a> {
        ChildIter {
            current: self.first_child.get(),
        }
    }

    /// Get computed rectangle
    pub fn rect(&self) -> Rectangle {
        self.computed_rect.get()
    }

    /// Set float property by name (for inspector/scripting)
    /// Note: Uses Cell for mutability via shared reference
    pub fn set_property_float(&self, name: &str, val: f32) {
        match name {
            "width" => self.width.set(val),
            "height" => self.height.set(val),
            "radius" | "border_radius" => {
                self.border_radius_tl.set(val);
                self.border_radius_tr.set(val);
                self.border_radius_br.set(val);
                self.border_radius_bl.set(val);
            }
            "radius_tl" => self.border_radius_tl.set(val),
            "radius_tr" => self.border_radius_tr.set(val),
            "radius_br" => self.border_radius_br.set(val),
            "radius_bl" => self.border_radius_bl.set(val),
            "border_width" => self.border_width.set(val),
            "padding" => self.padding.set(val),
            "margin" => self.margin.set(val),
            "flex" | "flex_grow" => self.flex_grow.set(val),
            "shadow" | "elevation" => self.elevation.set(val),
            "blur" | "backdrop_blur" => self.backdrop_blur.set(val),
            "font_size" => self.font_size.set(val),
            "icon_size" => self.icon_size.set(val),
            "value" => self.value.set(val),
            "thickness" => self.thickness.set(val),
            "ratio" => self.ratio.set(val),
            _ => {}
        }
    }
}

/// Iterator over children
pub struct ChildIter<'a> {
    current: Option<&'a ViewHeader<'a>>,
}

impl<'a> Iterator for ChildIter<'a> {
    type Item = &'a ViewHeader<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.current;
        if let Some(c) = self.current {
            self.current = c.next_sibling.get();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FrameArena;

    #[test]
    fn test_view_header_default() {
        let header = ViewHeader::default();
        assert_eq!(header.view_type, ViewType::Box);
        assert!(header.first_child.get().is_none());
        assert_eq!(header.flex_shrink.get(), 1.0);
    }

    #[test]
    fn test_add_children() {
        let arena = FrameArena::new();

        let parent = arena.alloc(ViewHeader::default());
        let child1 = arena.alloc(ViewHeader {
            id: ID::from_str("c1"),
            ..Default::default()
        });
        let child2 = arena.alloc(ViewHeader {
            id: ID::from_str("c2"),
            ..Default::default()
        });

        parent.add_child(child1);
        parent.add_child(child2);

        let children: Vec<_> = parent.children().collect();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].id, ID::from_str("c1"));
        assert_eq!(children[1].id, ID::from_str("c2"));
    }
}
