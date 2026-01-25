//! Widget builders - Fluent API for view construction
//! Ported from api.hpp
//!
//! Uses immutable references with Cell for interior mutability
//! to avoid borrow checker conflicts with tree structure.

pub mod canvas;
pub mod collapsible;
pub mod context_menu;
pub mod curves;
pub mod dragger;
pub mod fader;
pub mod gizmo;
pub mod grid;
pub mod knob;
pub mod math;
pub mod micro_interactions;
pub mod node;
pub mod plot;
pub mod checkbox;
pub mod dropdown;
pub mod layout_grid;
pub mod radio;
pub mod prelude;
pub mod ruler;
pub mod scene3d;
pub mod toggle;
pub mod virtual_list;

use crate::core::persistence::PersistenceManager;
use crate::core::{ColorF, FrameArena, Theme, ID};
use crate::view::header::{ViewHeader, ViewType};

/// Box builder - uses immutable ref since ViewHeader uses Cell for mutable fields
pub struct BoxBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
}

impl<'a> BoxBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn width(self, w: f32) -> Self {
        self.view.width.set(w);
        self
    }

    pub fn height(self, h: f32) -> Self {
        self.view.height.set(h);
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn padding(self, p: f32) -> Self {
        self.view.padding.set(p);
        self
    }

    pub fn margin(self, m: f32) -> Self {
        self.view.margin.set(m);
        self
    }

    pub fn flex_grow(self, grow: f32) -> Self {
        self.view.flex_grow.set(grow);
        self
    }

    pub fn scroll(self) -> Self {
        self.view.scroll_x.set(true);
        self.view.scroll_y.set(true);
        self
    }

    pub fn row(self) -> Self {
        self.view.is_row.set(true);
        self
    }

    pub fn column(self) -> Self {
        self.view.is_row.set(false);
        self
    }

    pub fn bg(self, color: ColorF) -> Self {
        self.view.bg_color.set(color);
        self
    }

    pub fn fg(self, color: ColorF) -> Self {
        self.view.fg_color.set(color);
        self
    }

    pub fn radius(self, r: f32) -> Self {
        self.view.border_radius_tl.set(r);
        self.view.border_radius_tr.set(r);
        self.view.border_radius_br.set(r);
        self.view.border_radius_bl.set(r);
        self
    }

    pub fn squircle(self, r: f32) -> Self {
        self.view.is_squircle.set(true);
        self.radius(r)
    }

    pub fn elevation(self, e: f32) -> Self {
        self.view.elevation.set(e);
        self
    }

    pub fn align(self, a: crate::view::header::Align) -> Self {
        self.view.align.set(a);
        self
    }

    pub fn font_size(self, size: f32) -> Self {
        self.view.font_size.set(size);
        self
    }

    pub fn backdrop_blur(self, blur: f32) -> Self {
        self.view.backdrop_blur.set(blur);
        self
    }

    pub fn border(self, width: f32, color: ColorF) -> Self {
        self.view.border_width.set(width);
        self.view.border_color.set(color);
        self
    }

    pub fn glow(self, strength: f32, color: ColorF) -> Self {
        self.view.glow_strength.set(strength);
        self.view.glow_color.set(color);
        self
    }

    pub fn aurora(self) -> Self {
        self.view.aurora.set(true);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view
    }
}

/// Text builder
pub struct TextBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub text: &'a str,
}

impl<'a> TextBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn font_size(self, size: f32) -> Self {
        self.view.font_size.set(size);
        self
    }

    pub fn fg(self, color: ColorF) -> Self {
        self.view.fg_color.set(color);
        self
    }

    pub fn layout_margin(self, m: f32) -> Self {
        self.view.margin.set(m);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.text);
        self.view
    }
}

/// Button builder
pub struct ButtonBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub label: &'a str,
}

impl<'a> ButtonBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn radius(self, r: f32) -> Self {
        self.view.border_radius_tl.set(r);
        self.view.border_radius_tr.set(r);
        self.view.border_radius_br.set(r);
        self.view.border_radius_bl.set(r);
        self
    }

    pub fn squircle(self, r: f32) -> Self {
        self.view.is_squircle.set(true);
        self.radius(r)
    }

    pub fn bg(self, color: ColorF) -> Self {
        self.view.bg_color.set(color);
        self
    }

    pub fn hover(self, c: ColorF) -> Self {
        self.view.bg_hover.set(Some(c));
        self
    }

    pub fn active(self, c: ColorF) -> Self {
        self.view.bg_active.set(Some(c));
        self
    }

    pub fn glow(self, strength: f32, color: ColorF) -> Self {
        self.view.glow_strength.set(strength);
        self.view.glow_color.set(color);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.label);
        self.view
    }

    /// Check if clicked (convenience for immediate mode)
    pub fn clicked(&self) -> bool {
        crate::view::interaction::is_clicked(self.view.id.get())
    }
}

/// UI context for building views
pub struct UIContext<'a> {
    pub arena: &'a FrameArena,
    parent_stack: Vec<&'a ViewHeader<'a>>,
    root: Option<&'a ViewHeader<'a>>,
    pub theme: Theme,
    next_id: u64,
    pub persistence: Option<&'a PersistenceManager>,
}

impl<'a> UIContext<'a> {
    pub fn new(arena: &'a FrameArena) -> Self {
        Self {
            arena,
            parent_stack: Vec::new(),
            root: None,
            theme: Theme::default(),
            next_id: 1,
            persistence: None,
        }
    }

    pub fn with_persistence(mut self, p: &'a PersistenceManager) -> Self {
        self.persistence = Some(p);
        self
    }

    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Internal helper to push a view to the parent stack or set it as root
    fn push_child(&mut self, view: &'a ViewHeader<'a>) {
        if self.root.is_none() {
            self.root = Some(view);
        }
        if let Some(parent) = self.parent_stack.last() {
            parent.add_child(view);
        }
    }

    /// Create a box container
    pub fn r#box(&mut self) -> BoxBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Box,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Apply default colors via Cell
        view.bg_color.set(self.theme.panel);

        self.push_child(view);

        BoxBuilder { view }
    }

    /// Create a row container
    pub fn row(&mut self) -> BoxBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Box,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        view.is_row.set(true);
        view.bg_color.set(ColorF::TRANSPARENT);

        self.push_child(view);

        BoxBuilder { view }
    }

    /// Create a column container
    pub fn column(&mut self) -> BoxBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Box,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        view.is_row.set(false);
        view.bg_color.set(ColorF::TRANSPARENT);

        self.push_child(view);

        BoxBuilder { view }
    }

    /// Create text label
    pub fn text(&mut self, text: &'a str) -> TextBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Text,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });
        view.fg_color.set(self.theme.text);

        self.push_child(view);

        TextBuilder { view, text }
    }

    /// Create button
    pub fn button(&mut self, label: &'a str) -> ButtonBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Button,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Defaults
        view.bg_color.set(self.theme.panel);
        view.elevation.set(2.0);
        view.border_radius_tl.set(6.0);
        view.border_radius_tr.set(6.0);
        view.border_radius_br.set(6.0);
        view.border_radius_bl.set(6.0);

        // Button hover/active from theme
        // We can derive hover/active from accent or panel
        // For Cyberpunk: panel is dark, accent is Cyan
        // Let's use accent for active, and a lighter panel for hover
        view.bg_hover.set(Some(self.theme.panel.lighten(0.1)));
        view.bg_active.set(Some(self.theme.accent.with_alpha(0.3)));

        self.push_child(view);

        ButtonBuilder { view, label }
    }

    /// Create markdown view
    pub fn markdown(&mut self, text: &'a str) -> MarkdownBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Markdown,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        self.push_child(view);

        MarkdownBuilder { view, text }
    }

    /// Begin a container (push to parent stack)
    pub fn begin(&mut self, view: &'a ViewHeader<'a>) {
        if self.root.is_none() {
            self.root = Some(view);
        }
        self.parent_stack.push(view);
    }

    /// End a container (pop from parent stack)
    pub fn end(&mut self) {
        self.parent_stack.pop();
    }

    /// Create knob
    pub fn knob(
        &mut self,
        value: &'a mut f32,
        min: f32,
        max: f32,
    ) -> crate::widgets::knob::KnobBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Knob,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Default Knob Style (Theme aware)
        view.width.set(50.0);
        view.height.set(50.0);
        view.bg_color.set(self.theme.panel);
        view.fg_color.set(self.theme.accent); // Active arc color
        view.border_color.set(self.theme.border);
        view.border_width.set(2.0);
        view.is_squircle.set(true); // Base shape
        view.border_radius_tl.set(25.0); // Circle by default

        // Interaction ID needed? Yes.
        // view.id is set by arena? No, arena just allocs. ID is 0 by default.
        // Builder usually sets ID from caller.

        self.push_child(view);

        crate::widgets::knob::KnobBuilder {
            view,
            value,
            min,
            max,
            label: None,
        }
    }

    /// Create toggle switch
    pub fn toggle(
        &mut self,
        value: &'a mut bool,
    ) -> crate::widgets::toggle::ToggleBuilder<'_, 'a> {
        crate::widgets::toggle::ToggleBuilder::new(self, value)
    }

    /// Create checkbox
    pub fn checkbox(
        &mut self,
        value: &'a mut bool,
    ) -> crate::widgets::checkbox::CheckboxBuilder<'_, 'a> {
        crate::widgets::checkbox::CheckboxBuilder::new(self, value)
    }

    /// Create radio button
    pub fn radio(
        &mut self,
        value: &'a mut i32,
        my_value: i32,
    ) -> crate::widgets::radio::RadioBuilder<'_, 'a> {
        crate::widgets::radio::RadioBuilder::new(self, value, my_value)
    }

    /// Create dropdown
    pub fn dropdown(
        &mut self,
        items: Vec<String>,
        index: &'a mut usize,
    ) -> crate::widgets::dropdown::DropdownBuilder<'_, 'a> {
        crate::widgets::dropdown::DropdownBuilder::new(self, items, index)
    }

    /// Create layout grid
    pub fn layout_grid(
        &mut self,
        cols: usize,
    ) -> crate::widgets::layout_grid::LayoutGridBuilder<'_, 'a> {
        crate::widgets::layout_grid::LayoutGridBuilder::new(self, cols)
    }

    /// Create fader
    pub fn fader(
        &mut self,
        value: &'a mut f32,
        min: f32,
        max: f32,
    ) -> crate::widgets::fader::FaderBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Fader,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Default Fader Style
        view.width.set(30.0);
        view.height.set(150.0);
        view.bg_color.set(self.theme.panel.darken(0.2)); // Trough color
        view.fg_color.set(self.theme.accent);
        view.border_color.set(self.theme.border);
        view.border_width.set(1.0);
        view.border_radius_tl.set(4.0);
        view.border_radius_tr.set(4.0);
        view.border_radius_br.set(4.0);
        view.border_radius_bl.set(4.0);

        self.push_child(view);

        crate::widgets::fader::FaderBuilder {
            view,
            value,
            min,
            max,
        }
    }

    /// Create value dragger
    pub fn value_dragger(
        &mut self,
        value: &'a mut f32,
        min: f32,
        max: f32,
    ) -> crate::widgets::dragger::ValueDraggerBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::ValueDragger,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Default Style
        view.width.set(80.0);
        view.height.set(24.0);
        view.bg_color.set(self.theme.panel);
        view.fg_color.set(self.theme.text);
        view.border_color.set(self.theme.border);
        view.border_width.set(1.0);
        view.border_radius_tl.set(2.0);
        view.border_radius_tr.set(2.0);
        view.border_radius_br.set(2.0);
        view.border_radius_bl.set(2.0);

        self.push_child(view);

        crate::widgets::dragger::ValueDraggerBuilder {
            view,
            value,
            min,
            max,
            step: 0.01,
        }
    }

    pub fn canvas(&mut self) -> canvas::CanvasBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Canvas,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });
        self.push_child(view);
        canvas::CanvasBuilder { view }
    }

    pub fn node(&mut self, title: &'a str) -> node::NodeBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Node,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });
        self.push_child(view);
        // Pass shared persistence reference
        node::NodeBuilder {
            view,
            title,
            persistence: self.persistence,
            should_persist: false,
        }
    }

    pub fn socket(&mut self, name: &'a str, is_input: bool) -> node::SocketBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Socket,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });
        self.push_child(view);
        node::SocketBuilder {
            view,
            name,
            is_input,
        }
    }

    /// Get root view
    pub fn root(&self) -> Option<&'a ViewHeader<'a>> {
        self.root
    }

    /// Create a context menu
    pub fn context_menu(&mut self) -> context_menu::ContextMenuBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::ContextMenu,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Context menu default style
        view.bg_color.set(self.theme.panel.with_alpha(0.95));
        view.border_color.set(self.theme.border);
        view.border_width.set(1.0);
        view.border_radius_tl.set(8.0);
        view.border_radius_tr.set(8.0);
        view.border_radius_br.set(8.0);
        view.border_radius_bl.set(8.0);
        view.backdrop_blur.set(20.0);
        view.elevation.set(10.0);

        self.push_child(view);
        context_menu::ContextMenuBuilder { view }
    }

    /// Create a menu item
    pub fn menu_item(&mut self, label: &'a str) -> context_menu::MenuItemBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::MenuItem,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Menu item default style
        view.width.set(200.0);
        view.height.set(32.0);
        view.fg_color.set(self.theme.text);
        view.padding.set(8.0);

        self.push_child(view);
        context_menu::MenuItemBuilder { view, label }
    }

    /// Create a collapsible container
    pub fn collapsible(
        &mut self,
        title: &'a str,
        initial_open: bool,
    ) -> collapsible::CollapsibleBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Collapsible,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Collapsible default style
        view.bg_color.set(self.theme.panel);
        view.border_color.set(self.theme.border);
        view.border_width.set(1.0);
        view.border_radius_tl.set(6.0);
        view.border_radius_tr.set(6.0);
        view.border_radius_br.set(6.0);
        view.border_radius_bl.set(6.0);
        view.fg_color.set(self.theme.text);
        view.min.set(32.0); // Header height

        self.push_child(view);
        collapsible::CollapsibleBuilder {
            view,
            title,
            initial_open,
        }
    }

    /// Create a toast notification
    pub fn toast(
        &mut self,
        message: &'a str,
        toast_type: micro_interactions::ToastType,
    ) -> micro_interactions::ToastBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Toast,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Toast default style - glass effect
        view.bg_color.set(self.theme.panel.with_alpha(0.85));
        view.border_color.set(self.theme.border);
        view.border_width.set(1.0);
        view.border_radius_tl.set(8.0);
        view.border_radius_tr.set(8.0);
        view.border_radius_br.set(8.0);
        view.border_radius_bl.set(8.0);
        view.backdrop_blur.set(15.0);
        view.fg_color.set(self.theme.text);
        view.width.set(300.0);
        view.height.set(50.0);
        view.padding.set(12.0);
        view.max.set(3.0); // Default 3 second duration

        self.push_child(view);
        micro_interactions::ToastBuilder {
            view,
            message,
            toast_type,
        }
    }

    /// Create a laser tooltip
    pub fn tooltip(&mut self, text: &'a str) -> micro_interactions::TooltipBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::Tooltip,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Tooltip default style - laser/glow effect
        view.bg_color.set(self.theme.panel.with_alpha(0.95));
        view.border_color.set(self.theme.accent);
        view.border_width.set(1.0);
        view.border_radius_tl.set(4.0);
        view.border_radius_tr.set(4.0);
        view.border_radius_br.set(4.0);
        view.border_radius_bl.set(4.0);
        view.glow_strength.set(3.0);
        view.glow_color.set(self.theme.accent.with_alpha(0.5));
        view.fg_color.set(self.theme.text);
        view.padding.set(8.0);

        self.push_child(view);
        micro_interactions::TooltipBuilder { view, text }
    }

    /// Create text input
    pub fn text_input(&mut self, text: &'a str) -> TextInputBuilder<'a> {
        let id = ID::from_u64(self.next_id);
        self.next_id += 1;
        let view = self.arena.alloc(ViewHeader {
            view_type: ViewType::TextInput,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Defaults
        view.width.set(200.0);
        view.height.set(32.0);
        view.bg_color.set(self.theme.panel);
        view.fg_color.set(self.theme.text);
        view.border_width.set(1.0);
        view.border_color.set(self.theme.border);
        view.border_radius_tl.set(4.0);
        view.border_radius_tr.set(4.0);
        view.border_radius_br.set(4.0);
        view.border_radius_bl.set(4.0);
        view.is_editing.set(true); // Tag as editable

        self.push_child(view);

        TextInputBuilder { view, text }
    }

    /// Create a virtual list
    pub fn virtual_list<F>(
        &mut self,
        count: usize,
        item_height: f32,
        renderer: F,
    ) -> virtual_list::VirtualListBuilder<'_, 'a, F>
    where
        F: FnMut(&mut UIContext, usize),
    {
        virtual_list::VirtualListBuilder::new(self, count, item_height, renderer)
    }

    // Plotting
    pub fn plot(&mut self) -> plot::PlotBuilder<'_, 'a> {
        plot::PlotBuilder::new(self)
    }

    // 3D Scene
    pub fn scene_3d(&mut self, id: &str) -> scene3d::Scene3DBuilder<'_, 'a> {
        scene3d::Scene3DBuilder::new(self, id)
    }

    // Color Grading
    pub fn curve_editor(&mut self, id: &str) -> curves::CurveEditorBuilder<'_, 'a> {
        curves::CurveEditorBuilder::new(self, id)
    }

    // Precision
    pub fn ruler(&mut self, id: &str) -> ruler::RulerBuilder<'_, 'a> {
        ruler::RulerBuilder::new(self, id)
    }

    pub fn grid(&mut self, id: &str) -> grid::GridBuilder<'_, 'a> {
        grid::GridBuilder::new(self, id)
    }

    pub fn transform_gizmo(&mut self, id: &str) -> gizmo::TransformGizmoBuilder<'_, 'a> {
        gizmo::TransformGizmoBuilder::new(self, id)
    }

    pub fn math<'b>(&'b mut self, id: &str, text: &'b str) -> math::MathBuilder<'b, 'a> {
        math::MathBuilder::new(self, id, text)
    }
}

/// Text Input builder
pub struct TextInputBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub text: &'a str,
}

impl<'a> TextInputBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn bg(self, color: ColorF) -> Self {
        self.view.bg_color.set(color);
        self
    }

    pub fn font_size(self, size: f32) -> Self {
        self.view.font_size.set(size);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.text);
        self.view
    }
}

/// Markdown builder
pub struct MarkdownBuilder<'a> {
    pub view: &'a ViewHeader<'a>,
    pub text: &'a str,
}

impl<'a> MarkdownBuilder<'a> {
    pub fn id(self, id: impl Into<ID>) -> Self {
        self.view.id.set(id.into());
        self
    }

    pub fn size(self, w: f32, h: f32) -> Self {
        self.view.width.set(w);
        self.view.height.set(h);
        self
    }

    pub fn bg(self, color: ColorF) -> Self {
        self.view.bg_color.set(color);
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view.text.set(self.text);
        self.view
    }
}
