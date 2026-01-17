//! PyO3 bindings for Fantasmagorie
//!
//! Wraps Rust types for Python access, enabling the Ouroboros hot-reload workflow.
//!
//! Strategy: Python doesn't have lifetimes, so we use IDs to reference Arena-allocated views.
//! Builders are thin wrappers that modify views by ID lookup.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::core::{ColorF, FrameArena, Rectangle, Vec2, ID};
use crate::draw::DrawList;
use crate::view::animation::Easing;
use crate::view::header::{Align, ViewHeader, ViewType};
use crate::view::interaction::{
    animate, animate_ex, begin_interaction_pass, capture, drain_input_buffer, get_rect,
    get_scroll_delta, get_scroll_offset, handle_key_down, handle_key_up, handle_modifiers,
    handle_received_character, handle_scroll, is_active, is_any_captured, is_clicked, is_focused,
    is_hot, mouse_delta, mouse_pos, register_interactive, release, set_focus, set_scroll_offset,
    update_input, update_rect,
};
use crate::view::plot::{Colormap, PlotData, PlotItem};
use crate::view::render_ui;
use crate::view::scene3d::Scene3DData;

// Thread-local context for Python
// We essentially need a "per-window" context map now, but since PyContext object usually
// maps 1:1 to a window in the user's mind (ctx = Context(w, h)), we can keep the thread local
// as the "Current Active Context" or map ID -> Context.
//
// However, the `PyContext` struct in Python is just a handle. The actual data is in `PY_CONTEXT`.
// To support multiple windows, `PY_CONTEXT` should probably map `window_id -> PyContextInner`.
//
// OR, effectively, `PyContext` IS the window handle.
//
// Let's change `PY_CONTEXT` to be a Map of WindowID -> Inner.
// AND add a field `current_window_id` to know which one `with_view_mut` operates on.

thread_local! {
    pub static PY_CONTEXTS: RefCell<HashMap<u64, PyContextInner>> = RefCell::new(HashMap::new());
    pub static CURRENT_WINDOW: RefCell<u64> = RefCell::new(0);
}

pub struct PyContextInner {
    pub arena: FrameArena,
    pub views: HashMap<u64, *mut ViewHeader<'static>>,
    pub root_id: Option<u64>,
    pub parent_stack: Vec<u64>,
    pub next_id: u64,
    pub draw_list: DrawList,
    pub font_manager: crate::text::FontManager,
    pub width: u32,
    pub height: u32,
    pub width: u32,
    pub height: u32,
    pub plot_items: HashMap<u64, Vec<crate::view::plot::PlotItem<'static>>>,
}

impl PyContextInner {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            arena: FrameArena::new(),
            views: HashMap::new(),
            root_id: None,
            parent_stack: Vec::new(),
            next_id: 1,
            draw_list: DrawList::new(),
            font_manager: crate::text::FontManager::new(),
            width,
            height,
            plot_items: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.arena.reset();
        self.views.clear();
        self.root_id = None;
        self.parent_stack.clear();
        self.next_id = 1;
        self.draw_list.clear();
        self.plot_items.clear();
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Helper to modify view header in the CURRENT window
fn with_view_mut<F>(id: u64, f: F)
where
    F: FnOnce(&mut ViewHeader),
{
    PY_CONTEXTS.with(|contexts| {
        let mut contexts = contexts.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            let wid = *cw.borrow();
            if let Some(inner) = contexts.get_mut(&wid) {
                if let Some(&ptr) = inner.views.get(&id) {
                    unsafe {
                        f(&mut *ptr);
                    }
                }
            }
        });
    })
}

// ============================================================================
// Python Color type
// ============================================================================

#[pyclass(name = "Color")]
#[derive(Clone, Copy)]
pub struct PyColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[pymethods]
impl PyColor {
    #[new]
    #[pyo3(signature = (r=1.0, g=1.0, b=1.0, a=1.0))]
    fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        PyColor { r, g, b, a }
    }

    #[staticmethod]
    fn white() -> Self {
        PyColor {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    }

    #[staticmethod]
    fn black() -> Self {
        PyColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }

    #[staticmethod]
    fn red() -> Self {
        PyColor {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }

    #[staticmethod]
    fn green() -> Self {
        PyColor {
            r: 0.0,
            g: 1.0,
            b: 0.0,
            a: 1.0,
        }
    }

    #[staticmethod]
    fn blue() -> Self {
        PyColor {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        }
    }

    #[staticmethod]
    fn hex(val: u32) -> Self {
        let c = ColorF::from_hex(val);
        PyColor {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        }
    }

    fn alpha(&self, a: f32) -> Self {
        PyColor {
            r: self.r,
            g: self.g,
            b: self.b,
            a,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Color({:.2}, {:.2}, {:.2}, {:.2})",
            self.r, self.g, self.b, self.a
        )
    }
}

impl From<PyColor> for ColorF {
    fn from(c: PyColor) -> ColorF {
        ColorF::new(c.r, c.g, c.b, c.a)
    }
}

// --- Easing ---

#[pyclass(name = "Easing")]
#[derive(Clone, Copy)]
pub enum PyEasing {
    Linear,
    QuadIn,
    QuadOut,
    QuadInOut,
    CubicIn,
    CubicOut,
    CubicInOut,
    ExpoIn,
    ExpoOut,
    ExpoInOut,
    ElasticIn,
    ElasticOut,
    ElasticInOut,
    BackIn,
    BackOut,
    BackInOut,
    Spring,
}

impl From<PyEasing> for Easing {
    fn from(e: PyEasing) -> Self {
        match e {
            PyEasing::Linear => Easing::Linear,
            PyEasing::QuadIn => Easing::QuadIn,
            PyEasing::QuadOut => Easing::QuadOut,
            PyEasing::QuadInOut => Easing::QuadInOut,
            PyEasing::CubicIn => Easing::CubicIn,
            PyEasing::CubicOut => Easing::CubicOut,
            PyEasing::CubicInOut => Easing::CubicInOut,
            PyEasing::ExpoIn => Easing::ExpoIn,
            PyEasing::ExpoOut => Easing::ExpoOut,
            PyEasing::ExpoInOut => Easing::ExpoInOut,
            PyEasing::ElasticIn => Easing::ElasticIn,
            PyEasing::ElasticOut => Easing::ElasticOut,
            PyEasing::ElasticInOut => Easing::ElasticInOut,
            PyEasing::BackIn => Easing::BackIn,
            PyEasing::BackOut => Easing::BackOut,
            PyEasing::BackInOut => Easing::BackInOut,
            PyEasing::Spring => Easing::Spring,
        }
    }
}

// ============================================================================
// Python Context (Engine)
// ============================================================================

#[pyclass(name = "Context", unsendable)]
pub struct PyContext {
    window_id: u64,
    width: u32,
    height: u32,
}

#[pymethods]
impl PyContext {
    #[new]
    #[pyo3(signature = (width=1280, height=720, window_id=0))]
    fn new(width: u32, height: u32, window_id: u64) -> Self {
        PY_CONTEXTS.with(|ctx| {
            ctx.borrow_mut()
                .insert(window_id, PyContextInner::new(width, height));
        });
        PyContext {
            width,
            height,
            window_id,
        }
    }

    fn begin_frame(&self) {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            if let Some(inner) = contexts.get_mut(&self.window_id) {
                inner.reset();
            }
        });
        // Set this as current for subsequent builder calls
        CURRENT_WINDOW.with(|cw| *cw.borrow_mut() = self.window_id);
    }

    fn end_frame(&self) -> PyResult<usize> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            if let Some(inner) = contexts.get_mut(&self.window_id) {
                // Get root and render
                if let Some(root_id) = inner.root_id {
                    if let Some(&ptr) = inner.views.get(&root_id) {
                        unsafe {
                            let root = &*ptr;
                            inner.draw_list.clear();
                            render_ui(
                                root,
                                inner.width as f32,
                                inner.height as f32,
                                &mut inner.draw_list,
                            );
                        }
                        return Ok(inner.draw_list.len());
                    }
                }
                Ok(0)
            } else {
                Err(PyRuntimeError::new_err("Context not initialized"))
            }
        })
    }

    fn get_width(&self) -> u32 {
        self.width
    }

    fn get_height(&self) -> u32 {
        self.height
    }

    fn draw_command_count(&self) -> usize {
        PY_CONTEXTS.with(|ctx| {
            ctx.borrow()
                .get(&self.window_id)
                .map(|i| i.draw_list.len())
                .unwrap_or(0)
        })
    }

    /// Explicitly set this context as active (for manual thread hopping if needed)
    fn make_current(&self) {
        CURRENT_WINDOW.with(|cw| *cw.borrow_mut() = self.window_id);
    }
}

// ============================================================================
// Widget Builders
// ============================================================================

/// Box builder returned to Python
#[pyclass(name = "BoxBuilder", unsendable)]
#[derive(Clone, Copy)]
pub struct PyBoxBuilder {
    view_id: u64,
}

#[pymethods]
impl PyBoxBuilder {
    fn width(&self, w: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.width.set(w));
        Ok(*self)
    }

    fn height(&self, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.height.set(h));
        Ok(*self)
    }

    fn size(&self, w: f32, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.width.set(w);
            v.height.set(h);
        });
        Ok(*self)
    }

    fn padding(&self, p: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.padding.set(p));
        Ok(*self)
    }

    fn margin(&self, m: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.margin.set(m));
        Ok(*self)
    }

    fn bg(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.bg_color.set(c.into()));
        Ok(*self)
    }

    fn radius(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_radius_tl.set(r);
            v.border_radius_tr.set(r);
            v.border_radius_br.set(r);
            v.border_radius_bl.set(r);
        });
        Ok(*self)
    }

    fn radius_tl(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_tl.set(r));
        Ok(*self)
    }
    fn radius_tr(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_tr.set(r));
        Ok(*self)
    }
    fn radius_br(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_br.set(r));
        Ok(*self)
    }
    fn radius_bl(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_bl.set(r));
        Ok(*self)
    }

    fn icon(&self, icon: String) -> PyResult<Self> {
        // Need access to arena string allocator for current window
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                    if let Some(&ptr) = inner.views.get(&self.view_id) {
                        unsafe {
                            let s = inner.arena.alloc_str(&icon);
                            (*ptr).icon = std::mem::transmute::<&str, &'static str>(s).into();
                        }
                    }
                }
            })
        });
        Ok(*self)
    }

    fn icon_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.icon_size.set(size));
        Ok(*self)
    }

    fn border(&self, width: f32, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_width.set(width);
            let c: ColorF = color.into();
            v.border_color.set(c);
        });
        Ok(*self)
    }

    fn shadow(&self, elevation: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.elevation.set(elevation));
        Ok(*self)
    }

    fn flex(&self, grow: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.flex_grow.set(grow));
        Ok(*self)
    }

    fn blur(&self, b: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.backdrop_blur.set(b));
        Ok(*self)
    }

    fn hover(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.bg_hover.set(Some(c)));
        Ok(*self)
    }

    fn active(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.bg_active.set(Some(c)));
        Ok(*self)
    }

    #[pyo3(signature = (property, target, duration=None, easing=None))]
    fn animate(
        &self,
        property: String,
        target: f32,
        duration: Option<f32>,
        easing: Option<PyEasing>,
    ) -> PyResult<Self> {
        let id = ID::from_u64(self.view_id);
        let val = if let Some(d) = duration {
            animate_ex(
                id,
                &property,
                target,
                d,
                easing.map(Into::into).unwrap_or(Easing::ExpoOut),
            )
        } else {
            animate(id, &property, target, 10.0)
        };

        with_view_mut(self.view_id, |v| v.set_property_float(&property, val));
        Ok(*self)
    }

    fn hovered(&self) -> bool {
        crate::view::interaction::is_hot(ID::from_u64(self.view_id))
    }

    fn clicked(&self) -> bool {
        crate::view::interaction::is_clicked(ID::from_u64(self.view_id))
    }
}

/// Text builder
#[pyclass(name = "TextBuilder", unsendable)]
#[derive(Clone, Copy)]
pub struct PyTextBuilder {
    view_id: u64,
}

#[pymethods]
impl PyTextBuilder {
    fn color(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.fg_color.set(c));
        Ok(*self)
    }

    #[pyo3(signature = (property, target, duration=None, easing=None))]
    fn animate(
        &self,
        property: String,
        target: f32,
        duration: Option<f32>,
        easing: Option<PyEasing>,
    ) -> PyResult<Self> {
        let id = ID::from_u64(self.view_id);
        let val = if let Some(d) = duration {
            animate_ex(
                id,
                &property,
                target,
                d,
                easing.map(Into::into).unwrap_or(Easing::ExpoOut),
            )
        } else {
            animate(id, &property, target, 10.0)
        };

        with_view_mut(self.view_id, |v| v.set_property_float(&property, val));
        Ok(self.clone())
    }

    fn font_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.font_size.set(size));
        Ok(*self)
    }

    fn icon(&self, icon: String) -> PyResult<Self> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                    if let Some(&ptr) = inner.views.get(&self.view_id) {
                        unsafe {
                            let s = inner.arena.alloc_str(&icon);
                            (*ptr).icon = std::mem::transmute::<&str, &'static str>(s).into();
                        }
                    }
                }
            })
        });
        Ok(*self)
    }

    fn icon_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.icon_size.set(size));
        Ok(*self)
    }
}

/// Button builder
#[pyclass(name = "ButtonBuilder", unsendable)]
#[derive(Clone, Copy)]
pub struct PyButtonBuilder {
    view_id: u64,
}

#[pymethods]
impl PyButtonBuilder {
    fn width(&self, w: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.width.set(w));
        Ok(*self)
    }

    fn height(&self, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.height.set(h));
        Ok(*self)
    }

    fn bg(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.bg_color.set(c.into()));
        Ok(*self)
    }

    fn radius(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_radius_tl.set(r);
            v.border_radius_tr.set(r);
            v.border_radius_br.set(r);
            v.border_radius_bl.set(r);
        });
        Ok(*self)
    }

    fn radius_tl(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_tl.set(r));
        Ok(*self)
    }
    fn radius_tr(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_tr.set(r));
        Ok(*self)
    }
    fn radius_br(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_br.set(r));
        Ok(*self)
    }
    fn radius_bl(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.border_radius_bl.set(r));
        Ok(*self)
    }

    fn border(&self, width: f32, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_width.set(width);
            let c: ColorF = color.into();
            v.border_color.set(c);
        });
        Ok(*self)
    }

    fn shadow(&self, elevation: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.elevation.set(elevation));
        Ok(*self)
    }

    fn fg(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.fg_color.set(c));
        Ok(*self)
    }

    fn icon(&self, icon: String) -> PyResult<Self> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                    if let Some(&ptr) = inner.views.get(&self.view_id) {
                        unsafe {
                            let s = inner.arena.alloc_str(&icon);
                            (*ptr).icon = std::mem::transmute::<&str, &'static str>(s).into();
                        }
                    }
                }
            })
        });
        Ok(*self)
    }

    fn icon_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.icon_size.set(size));
        Ok(*self)
    }

    fn hover(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.bg_hover.set(Some(c)));
        Ok(*self)
    }

    fn active(&self, color: PyColor) -> PyResult<Self> {
        let c: ColorF = color.into();
        with_view_mut(self.view_id, |v| v.bg_active.set(Some(c)));
        Ok(*self)
    }

    fn font_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.font_size.set(size));
        Ok(*self)
    }

    fn clicked(&self) -> bool {
        use crate::view::interaction;
        interaction::is_clicked(ID::from_u64(self.view_id))
    }

    #[pyo3(signature = (property, target, duration=None, easing=None))]
    fn animate(
        &self,
        property: String,
        target: f32,
        duration: Option<f32>,
        easing: Option<PyEasing>,
    ) -> PyResult<Self> {
        let id = ID::from_u64(self.view_id);
        let val = if let Some(d) = duration {
            animate_ex(
                id,
                &property,
                target,
                d,
                easing.map(Into::into).unwrap_or(Easing::ExpoOut),
            )
        } else {
            animate(id, &property, target, 10.0)
        };

        with_view_mut(self.view_id, |v| v.set_property_float(&property, val));
        Ok(*self)
    }
}

// --- Plot Builder ---

#[pyclass]
#[derive(Clone)]
pub struct PyPlotBuilder {
    pub view_id: u64,
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub title: Option<String>,
}

#[pymethods]
impl PyPlotBuilder {
    fn title(&mut self, title: String) -> PyResult<Self> {
        self.title = Some(title);
        Ok(self.clone())
    }

    fn x_range(&mut self, min: f32, max: f32) -> PyResult<Self> {
        self.x_min = min;
        self.x_max = max;
        Ok(self.clone())
    }

    fn y_range(&mut self, min: f32, max: f32) -> PyResult<Self> {
        self.y_min = min;
        self.y_max = max;
        Ok(self.clone())
    }

    fn height(&self, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.height.set(h));
        Ok(self.clone())
    }

    fn line(
        &self,
        py: Python<'_>,
        y_data: PyObject,
        color: PyColor,
        width: Option<f32>,
    ) -> PyResult<Self> {
        PY_CONTEXTS.with(|ctx| -> PyResult<()> {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| -> PyResult<()> {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();

                // Extract f32 slice from Python object (NumPy array)
                let buffer = pyo3::buffer::PyBuffer::<f32>::get(y_data.as_ref(py))?;
                let slice = buffer
                    .as_slice(py)
                    .ok_or_else(|| PyRuntimeError::new_err("Failed to get buffer slice"))?;

                // Copy to Vec first (to avoid ReadOnlyCell issues with alloc_slice)
                let data: Vec<f32> = slice.iter().map(|c| c.get()).collect();
                let arena_slice = inner.arena.alloc_slice(&data);

                // Create item
                let item = PlotItem::FastLine {
                    y_data: unsafe { std::mem::transmute(arena_slice) },
                    x_start: 0.0,
                    x_step: 1.0,
                    color: color.into(),
                    width: width.unwrap_or(2.0),
                };

                inner.plot_items.entry(self.view_id).or_default().push(item);
                Ok(())
            })
        })?;
        Ok(self.clone())
    }

    fn heatmap(
        &self,
        py: Python<'_>,
        data: PyObject,
        width: usize,
        height: usize,
        colormap: String,
        min: f32,
        max: f32,
    ) -> PyResult<Self> {
        let cm = match colormap.as_str() {
            "viridis" => Colormap::Viridis,
            "plasma" => Colormap::Plasma,
            "magma" => Colormap::Magma,
            "inferno" => Colormap::Inferno,
            _ => Colormap::Grayscale,
        };

        PY_CONTEXTS.with(|ctx| -> PyResult<()> {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| -> PyResult<()> {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();

                let buffer = pyo3::buffer::PyBuffer::<f32>::get(data.as_ref(py))?;
                let slice = buffer
                    .as_slice(py)
                    .ok_or_else(|| PyRuntimeError::new_err("Failed to get buffer slice"))?;

                let data: Vec<f32> = slice.iter().map(|c| c.get()).collect();
                let arena_slice = inner.arena.alloc_slice(&data);

                let item = PlotItem::Heatmap {
                    data: unsafe { std::mem::transmute(arena_slice) },
                    width,
                    height,
                    colormap: cm,
                    min,
                    max,
                };

                inner.plot_items.entry(self.view_id).or_default().push(item);
                Ok(())
            })
        })?;
        Ok(self.clone())
    }

    fn build(&self) -> PyResult<()> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let items = inner.plot_items.remove(&self.view_id).unwrap_or_default();
                        let items_slice = inner.arena.alloc_slice(&items);

                        let plot_data = inner.arena.alloc(PlotData {
                            items: std::mem::transmute(items_slice),
                            x_min: self.x_min,
                            x_max: self.x_max,
                            y_min: self.y_min,
                            y_max: self.y_max,
                            title: self.title.as_ref().map(|s| {
                                let arena_s = inner.arena.alloc_str(s);
                                std::mem::transmute::<&str, &'static str>(arena_s)
                            }),
                        });

                        (*ptr).plot_data.set(Some(std::mem::transmute(plot_data)));
                    }
                }
            })
        });
        Ok(())
    }
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct PyVec3(pub crate::core::Vec3);

#[pymethods]
impl PyVec3 {
    #[new]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self(crate::core::Vec3::new(x, y, z))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyScene3DBuilder {
    view_id: u64,
    camera_pos: crate::core::Vec3,
    camera_target: crate::core::Vec3,
    fov: f32,
    texture_id: u64,
    lut_id: u64,
    lut_intensity: f32,
}

#[pymethods]
impl PyScene3DBuilder {
    fn camera(&mut self, pos: (f32, f32, f32), target: (f32, f32, f32)) -> Self {
        self.camera_pos = crate::core::Vec3::new(pos.0, pos.1, pos.2);
        self.camera_target = crate::core::Vec3::new(target.0, target.1, target.2);
        self.clone()
    }

    fn texture(&mut self, id: u64) -> Self {
        self.texture_id = id;
        self.clone()
    }

    fn lut(&mut self, id: u64, intensity: f32) -> Self {
        self.lut_id = id;
        self.lut_intensity = intensity;
        self.clone()
    }

    fn fov(&mut self, f: f32) -> Self {
        self.fov = f;
        self.clone()
    }

    fn height(&mut self, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.height.set(h));
        Ok(self.clone())
    }

    fn build(&self) -> PyResult<()> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let data = inner
                            .arena
                            .alloc(Scene3DData::new(ID::from_u64(self.view_id)));
                        data.texture_id.set(self.texture_id);
                        data.camera_pos.set(self.camera_pos);
                        data.camera_target.set(self.camera_target);
                        data.fov.set(self.fov);
                        data.lut_id.set(self.lut_id);
                        data.lut_intensity.set(self.lut_intensity);

                        (*ptr).scene_data.set(Some(std::mem::transmute(data)));
                    }
                }
            })
        });
        Ok(())
    }

    fn hovered(&self) -> bool {
        is_hot(ID::from_u64(self.view_id))
    }

    fn active(&self) -> bool {
        is_active(ID::from_u64(self.view_id))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyRulerBuilder {
    view_id: u64,
    orientation: i32, // 0 = Horizontal, 1 = Vertical
    start: f32,
    scale: f32,
}

#[pymethods]
impl PyRulerBuilder {
    fn vertical(&mut self) -> Self {
        self.orientation = 1;
        self.clone()
    }

    fn horizontal(&mut self) -> Self {
        self.orientation = 0;
        self.clone()
    }

    fn start(&mut self, val: f32) -> Self {
        self.start = val;
        self.clone()
    }

    fn scale(&mut self, val: f32) -> Self {
        self.scale = val;
        self.clone()
    }

    fn build(&self) -> PyResult<()> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let data = inner.arena.alloc(crate::view::ruler::RulerData::new());
                        data.orientation.set(if self.orientation == 1 {
                            crate::view::ruler::RulerOrientation::Vertical
                        } else {
                            crate::view::ruler::RulerOrientation::Horizontal
                        });
                        data.start.set(self.start);
                        data.scale.set(self.scale);

                        (*ptr).ruler_data.set(Some(std::mem::transmute(data)));
                    }
                }
            })
        });
        Ok(())
    }

    fn width(&self, w: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.width.set(w));
        Ok(self.clone())
    }

    fn height(&self, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.height.set(h));
        Ok(self.clone())
    }

    fn flex_grow(&self, g: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.flex_grow.set(g));
        Ok(self.clone())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyGridBuilder {
    view_id: u64,
    step_x: f32,
    step_y: f32,
    scale: f32,
}

#[pymethods]
impl PyGridBuilder {
    fn step(&mut self, x: f32, y: f32) -> Self {
        self.step_x = x;
        self.step_y = y;
        self.clone()
    }

    fn scale(&mut self, s: f32) -> Self {
        self.scale = s;
        self.clone()
    }

    fn build(&self) -> PyResult<()> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let data = inner.arena.alloc(crate::view::grid::GridData::new());
                        data.step_x.set(self.step_x);
                        data.step_y.set(self.step_y);
                        data.scale.set(self.scale);

                        (*ptr).grid_data.set(Some(std::mem::transmute(data)));
                    }
                }
            })
        });
        Ok(())
    }

    fn flex_grow(&self, g: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.flex_grow.set(g));
        Ok(self.clone())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCurveEditorBuilder {
    view_id: u64,
}

#[pymethods]
impl PyCurveEditorBuilder {
    fn height(&mut self, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.height.set(h));
        Ok(self.clone())
    }

    fn build(&self) -> PyResult<()> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let data = inner
                            .arena
                            .alloc(crate::view::curves::CurveEditorData::new());
                        (*ptr).curve_data.set(Some(std::mem::transmute(data)));
                    }
                }
            })
        });
        Ok(())
    }
}

// ============================================================================
// Free Functions (Widget Constructors)
// ============================================================================

fn alloc_view_in_current_window(is_row: bool, view_type: ViewType) -> PyResult<u64> {
    PY_CONTEXTS.with(|ctx| {
        let mut contexts = ctx.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            let wid = *cw.borrow();
            let inner = contexts.get_mut(&wid).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Context not initialized for window {}", wid))
            })?;

            let view_id = inner.alloc_id();

            let view = inner.arena.alloc(ViewHeader {
                view_type,
                id: ID::from_u64(view_id).into(),
                is_row: is_row.into(),
                bg_color: if matches!(view_type, ViewType::Box) {
                    ColorF::transparent().into()
                } else {
                    ColorF::transparent().into()
                },
                ..Default::default()
            });

            // Default BG for Box if not row/column logic overridden by callers?
            // Original code had defaults in py_box etc. Let's replicate those in wrappers.

            let ptr = view as *mut ViewHeader;
            inner
                .views
                .insert(view_id, unsafe { std::mem::transmute(ptr) });

            if inner.root_id.is_none() {
                inner.root_id = Some(view_id);
            }

            if let Some(&parent_id) = inner.parent_stack.last() {
                if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                    unsafe {
                        (*parent_ptr).add_child(&*ptr);
                    }
                }
            }

            // Only push to stack if container? Handled by caller.
            Ok(view_id)
        })
    })
}

/// Create a Box container
#[pyfunction]
#[pyo3(name = "Box")]
fn py_box() -> PyResult<PyBoxBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Box)?;
    with_view_mut(view_id, |v| {
        v.bg_color.set(ColorF::new(0.15, 0.15, 0.18, 1.0))
    });
    Ok(PyBoxBuilder { view_id })
}

/// Create a Row container
#[pyfunction]
#[pyo3(name = "Row")]
fn py_row() -> PyResult<PyBoxBuilder> {
    let view_id = alloc_view_in_current_window(true, ViewType::Box)?;
    with_view_mut(view_id, |v| v.bg_color.set(ColorF::transparent()));

    // Push to stack
    PY_CONTEXTS.with(|ctx| {
        let mut contexts = ctx.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                inner.parent_stack.push(view_id);
            }
        });
    });

    Ok(PyBoxBuilder { view_id })
}

/// Create a Column container
#[pyfunction]
#[pyo3(name = "Column")]
fn py_column() -> PyResult<PyBoxBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Box)?;
    with_view_mut(view_id, |v| {
        v.bg_color.set(ColorF::transparent());
        v.is_row.set(false);
    });

    PY_CONTEXTS.with(|ctx| {
        let mut contexts = ctx.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                inner.parent_stack.push(view_id);
            }
        });
    });

    Ok(PyBoxBuilder { view_id })
}

/// Create a Text label
#[pyfunction]
#[pyo3(name = "Text")]
fn py_text(text: &str) -> PyResult<PyTextBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Text)?;

    PY_CONTEXTS.with(|ctx| {
        let mut contexts = ctx.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                unsafe {
                    let s = inner.arena.alloc_str(text);
                    if let Some(&ptr) = inner.views.get(&view_id) {
                        (*ptr).text = std::mem::transmute::<&str, &'static str>(s).into();
                        (*ptr).fg_color.set(ColorF::white());
                        (*ptr).font_size.set(14.0);
                    }
                }
            }
        });
    });

    Ok(PyTextBuilder { view_id })
}

/// Create a Button
#[pyfunction]
#[pyo3(name = "Button")]
fn py_button(label: &str) -> PyResult<PyButtonBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Button)?;

    // Set text and defaults
    PY_CONTEXTS.with(|ctx| {
        let mut contexts = ctx.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                unsafe {
                    let s = inner.arena.alloc_str(label);
                    if let Some(&ptr) = inner.views.get(&view_id) {
                        (*ptr).text = std::mem::transmute::<&str, &'static str>(s).into();
                        (*ptr).bg_color.set(ColorF::new(0.2, 0.2, 0.25, 1.0));
                        (*ptr).fg_color.set(ColorF::white());
                        (*ptr).font_size.set(14.0);
                        (*ptr).border_radius_tl.set(4.0);
                        (*ptr).border_radius_tr.set(4.0);
                        (*ptr).border_radius_br.set(4.0);
                        (*ptr).border_radius_bl.set(4.0);
                        (*ptr).padding.set(8.0);
                    }
                }
            }
        });
    });

    Ok(PyButtonBuilder { view_id })
}

/// End the current container
#[pyfunction]
#[pyo3(name = "End")]
fn py_end() -> PyResult<()> {
    PY_CONTEXTS.with(|ctx| {
        let mut contexts = ctx.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                inner.parent_stack.pop();
            }
        });
    });
    Ok(())
}

/// Create a Plot
#[pyfunction]
#[pyo3(name = "Plot")]
fn py_plot() -> PyResult<PyPlotBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Plot)?;

    // Set default styling for plots
    with_view_mut(view_id, |v| {
        v.height.set(300.0);
        v.bg_color.set(ColorF::new(0.05, 0.05, 0.07, 1.0));
        v.border_width.set(1.0);
        v.border_color.set(ColorF::new(0.3, 0.3, 0.35, 1.0));
    });

    Ok(PyPlotBuilder {
        view_id,
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
        title: None,
    })
}

/// Create a 3D Scene Viewport
#[pyfunction]
#[pyo3(name = "Scene3D")]
fn py_scene_3d() -> PyResult<PyScene3DBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Scene3D)?;

    with_view_mut(view_id, |v| {
        v.height.set(400.0);
        v.bg_color.set(ColorF::new(0.01, 0.01, 0.02, 1.0));
    });

    Ok(PyScene3DBuilder {
        view_id,
        camera_pos: crate::core::Vec3::new(0.0, 0.0, 5.0),
        camera_target: crate::core::Vec3::ZERO,
        fov: 45.0,
        texture_id: 0,
        lut_id: 0,
        lut_intensity: 1.0,
    })
}

/// Create a Curve Editor
#[pyfunction]
#[pyo3(name = "CurveEditor")]
fn py_curve_editor() -> PyResult<PyCurveEditorBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::CurveEditor)?;

    with_view_mut(view_id, |v| {
        v.height.set(250.0);
        v.bg_color.set(ColorF::new(0.05, 0.05, 0.07, 1.0));
    });

    Ok(PyCurveEditorBuilder { view_id })
}

#[pyclass]
#[derive(Clone)]
pub struct PyTransformGizmoBuilder {
    view_id: u64,
    snap_enabled: bool,
    mode: i32,
}

#[pymethods]
impl PyTransformGizmoBuilder {
    fn snap(&mut self, enabled: bool) -> Self {
        self.snap_enabled = enabled;
        self.clone()
    }

    fn mode(&mut self, m: i32) -> Self {
        self.mode = m;
        self.clone()
    }

    fn build(&self) -> PyResult<()> {
        // ... (Similar construction as others)
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let data = inner.arena.alloc(crate::view::gizmo::GizmoData::new());

                        let mode = match self.mode {
                            0 => crate::view::gizmo::GizmoMode::Translate,
                            1 => crate::view::gizmo::GizmoMode::Rotate,
                            2 => crate::view::gizmo::GizmoMode::Scale,
                            _ => crate::view::gizmo::GizmoMode::Translate,
                        };
                        data.mode.set(mode);

                        let mut snap = data.snap_context.get();
                        snap.enabled = self.snap_enabled;
                        data.snap_context.set(snap);

                        (*ptr).gizmo_data.set(Some(std::mem::transmute(data)));
                    }
                }
            })
        });
        Ok(())
    }
}

/// Create a Transform Gizmo
#[pyfunction]
#[pyo3(name = "TransformGizmo")]
fn py_transform_gizmo() -> PyResult<PyTransformGizmoBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::TransformGizmo)?;
    Ok(PyTransformGizmoBuilder {
        view_id,
        snap_enabled: false,
        mode: 0,
    })
}

/// Create a Ruler
#[pyfunction]
#[pyo3(name = "Ruler")]
fn py_ruler() -> PyResult<PyRulerBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Ruler)?;

    // Default style
    with_view_mut(view_id, |v| {
        v.bg_color.set(ColorF::new(0.08, 0.08, 0.08, 1.0));
        v.border_width.set(1.0);
        v.border_color.set(ColorF::new(0.2, 0.2, 0.2, 1.0));
    });

    Ok(PyRulerBuilder {
        view_id,
        orientation: 0,
        start: 0.0,
        scale: 1.0,
    })
}

/// Create a Grid
#[pyfunction]
#[pyo3(name = "Grid")]
fn py_grid() -> PyResult<PyGridBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Grid)?;

    Ok(PyGridBuilder {
        view_id,
        step_x: 50.0,
        step_y: 50.0,
        scale: 1.0,
    })
}

#[pyclass]
#[derive(Clone)]
pub struct PyMathBuilder {
    view_id: u64,
    text: String,
    font_size: f32,
}

#[pymethods]
impl PyMathBuilder {
    fn font_size(&mut self, size: f32) -> Self {
        self.font_size = size;
        self.clone()
    }

    // color, etc?

    fn build(&self) -> PyResult<()> {
        PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                let inner = contexts.get_mut(&*cw.borrow()).unwrap();
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        (*ptr).font_size.set(self.font_size);
                        *(*ptr).text.borrow_mut() = self.text.clone();
                    }
                }
            })
        });
        Ok(())
    }
}

/// Create a Math view
#[pyfunction]
#[pyo3(name = "Math")]
fn py_math(text: String) -> PyResult<PyMathBuilder> {
    let view_id = alloc_view_in_current_window(false, ViewType::Math)?;

    Ok(PyMathBuilder {
        view_id,
        text,
        font_size: 24.0,
    })
}

/// Register the Python module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyContext>()?;
    m.add_class::<PyColor>()?;
    m.add_class::<PyEasing>()?;
    m.add_class::<PyBoxBuilder>()?;
    m.add_class::<PyTextBuilder>()?;
    m.add_class::<PyButtonBuilder>()?;
    #[cfg(feature = "opengl")]
    m.add_class::<super::buffer::BufferView>()?;

    m.add_class::<PyPlotBuilder>()?;
    m.add_class::<PyScene3DBuilder>()?;
    m.add_class::<PyCurveEditorBuilder>()?;
    m.add_class::<PyRulerBuilder>()?;
    m.add_class::<PyGridBuilder>()?;
    m.add_class::<PyTransformGizmoBuilder>()?;
    m.add_class::<PyMathBuilder>()?;
    m.add_class::<PyVec3>()?;

    m.add_function(wrap_pyfunction!(py_box, m)?)?;
    m.add_function(wrap_pyfunction!(py_row, m)?)?;
    m.add_function(wrap_pyfunction!(py_column, m)?)?;
    m.add_function(wrap_pyfunction!(py_text, m)?)?;
    m.add_function(wrap_pyfunction!(py_button, m)?)?;
    m.add_function(wrap_pyfunction!(py_plot, m)?)?;
    m.add_function(wrap_pyfunction!(py_scene_3d, m)?)?;
    m.add_function(wrap_pyfunction!(py_curve_editor, m)?)?;
    m.add_function(wrap_pyfunction!(py_ruler, m)?)?;
    m.add_function(wrap_pyfunction!(py_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_transform_gizmo, m)?)?;
    m.add_function(wrap_pyfunction!(py_math, m)?)?;
    m.add_function(wrap_pyfunction!(py_end, m)?)?;

    Ok(())
}
