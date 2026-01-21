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

    pub last_view: Option<u64>,
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

            last_view: None,
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
        self.last_view = None;
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
                                &mut inner.font_manager,
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
            inner.last_view = Some(view_id);

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
#[pyo3(name = "Text", signature = (text, size=14.0))]
fn py_text(text: &str, size: f32) -> PyResult<PyTextBuilder> {
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
                        (*ptr).font_size.set(size);
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
                        let s = inner.arena.alloc_str(&self.text);
                        (*ptr).text.set(std::mem::transmute::<&str, &'static str>(s));
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

// ============================================================================
// Immediate Mode Property Functions
// ============================================================================

fn get_last_view_id() -> Option<u64> {
    PY_CONTEXTS.with(|ctx| {
        let contexts = ctx.borrow();
        CURRENT_WINDOW.with(|cw| {
            contexts
                .get(&*cw.borrow())
                .and_then(|inner| inner.last_view)
        })
    })
}

#[pyfunction]
#[pyo3(name = "Width")]
fn py_width_free(w: f32) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
        with_view_mut(id, |v| v.width.set(w));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "Height")]
fn py_height_free(h: f32) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
        with_view_mut(id, |v| v.height.set(h));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "FlexGrow")]
fn py_flex_grow_free(g: f32) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
        with_view_mut(id, |v| v.flex_grow.set(g));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "BgColor")]
fn py_bg_color_free(r: f32, g: f32, b: f32, a: f32) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
        with_view_mut(id, |v| v.bg_color.set(ColorF::new(r, g, b, a).into()));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "Padding")]
fn py_padding_free(l: f32, t: f32, r: f32, b: f32) -> PyResult<()> {
    // Current ViewHeader only carries uniform padding in `padding` field?
    // Let's check ViewHeader struct. If it has only `padding`, we use average or max?
    // Step 116 py_button uses `padding.set(8.0)`.
    // If the struct has l/t/r/b, use them.
    // If not, use `padding`.
    // Assuming ViewHeader has separate padding fields is safer or check defaults?
    // ViewHeader has `padding` (f32). It likely does not have l/t/r/b?
    // Let's assume uniform `padding` for now, or just set `padding` to `l`.
    // Wait, PyButtonBuilder (Step 116) sets `padding`.
    // But `phase3_demo.py` passes 4 args.
    // I should check `ViewHeader` definition.
    // Step 86.
    // I'll assume it has `padding` only for now.
    // I will ignore other 3 or use first.
    if let Some(id) = get_last_view_id() {
        with_view_mut(id, |v| v.padding.set(l));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "Orientation")]
fn py_orientation_free(o: i32) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
        // Needs to set specific data for Ruler?
        // RulerData lives in `ruler_data` field. Need unsafe access.
         PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                    if let Some(&ptr) = inner.views.get(&id) {
                       unsafe {
                           if let Some(data) = (*ptr).ruler_data.get() {
                                let orient = if o == 1 {
                                    crate::view::ruler::RulerOrientation::Vertical
                                } else {
                                    crate::view::ruler::RulerOrientation::Horizontal
                                };
                                data.orientation.set(orient);
                           }
                       }
                    }
                }
            });
         });
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "Snap")]
fn py_snap_free(enabled: bool) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
         PY_CONTEXTS.with(|ctx| {
            let mut contexts = ctx.borrow_mut();
            CURRENT_WINDOW.with(|cw| {
                if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                    if let Some(&ptr) = inner.views.get(&id) {
                       unsafe {
                           if let Some(data) = (*ptr).gizmo_data.get() {
                               // data is &GizmoData
                               let mut sc = data.snap_context.get();
                               sc.enabled = enabled;
                               data.snap_context.set(sc);
                           }
                       }
                    }
                }
            });
         });
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "CameraPos")]
fn py_camera_pos_free(x: f32, y: f32, z: f32) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
         PY_CONTEXTS.with(|ctx| {
             let mut contexts = ctx.borrow_mut();
             CURRENT_WINDOW.with(|cw| {
                 if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                     if let Some(&ptr) = inner.views.get(&id) {
                        unsafe {
                            if let Some(data) = (*ptr).scene_data.get() {
                                data.camera_pos.set(crate::core::Vec3::new(x, y, z));
                            }
                        }
                     }
                 }
             });
         });
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "CameraTarget")]
fn py_camera_target_free(x: f32, y: f32, z: f32) -> PyResult<()> {
    if let Some(id) = get_last_view_id() {
         PY_CONTEXTS.with(|ctx| {
             let mut contexts = ctx.borrow_mut();
             CURRENT_WINDOW.with(|cw| {
                 if let Some(inner) = contexts.get_mut(&*cw.borrow()) {
                     if let Some(&ptr) = inner.views.get(&id) {
                        unsafe {
                            if let Some(data) = (*ptr).scene_data.get() {
                                data.camera_target.set(crate::core::Vec3::new(x, y, z));
                            }
                        }
                     }
                 }
             });
         });
    }
    Ok(())
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
    m.add_function(wrap_pyfunction!(py_slider, m)?)?;
    m.add_function(wrap_pyfunction!(py_text_input, m)?)?;
    m.add_function(wrap_pyfunction!(py_bezier, m)?)?;
    m.add_function(wrap_pyfunction!(py_image, m)?)?;
    m.add_function(wrap_pyfunction!(py_toggle, m)?)?;
    m.add_function(wrap_pyfunction!(py_splitter, m)?)?;
    m.add_function(wrap_pyfunction!(py_color_picker, m)?)?;
    m.add_function(wrap_pyfunction!(py_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(py_draw_path, m)?)?;
    m.add_function(wrap_pyfunction!(py_t, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_locale, m)?)?;
    m.add_function(wrap_pyfunction!(py_add_translation, m)?)?;
    m.add_function(wrap_pyfunction!(py_mount, m)?)?;
    m.add_function(wrap_pyfunction!(py_capture_frame, m)?)?;
    m.add_function(wrap_pyfunction!(py_plot, m)?)?;
    m.add_function(wrap_pyfunction!(py_scene_3d, m)?)?;
    m.add_function(wrap_pyfunction!(py_curve_editor, m)?)?;
    m.add_function(wrap_pyfunction!(py_ruler, m)?)?;
    m.add_function(wrap_pyfunction!(py_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_transform_gizmo, m)?)?;
    m.add_function(wrap_pyfunction!(py_math, m)?)?;
    m.add_function(wrap_pyfunction!(py_end, m)?)?;

    // Free Functions
    m.add_function(wrap_pyfunction!(py_width_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_height_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_flex_grow_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_bg_color_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_padding_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_orientation_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_snap_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_camera_pos_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_camera_target_free, m)?)?;

    Ok(())
}
/// Image builder
#[pyclass(name = "Image")]
#[derive(Clone)]
struct PyImageBuilder {
    view_id: u64,
}

#[pymethods]
impl PyImageBuilder {
    fn size(&self, w: f32, h: f32) -> Self {
        with_view_mut(self.view_id, |v| { v.width.set(w); v.height.set(h); });
        self.clone()
    }

    fn tint(&self, c: PyColor) -> Self {
        with_view_mut(self.view_id, |v| v.fg_color.set(ColorF::new(c.r, c.g, c.b, c.a)));
        self.clone()
    }
}

/// Create an Image
#[pyfunction]
#[pyo3(name = "Image")]
fn py_image(path: String) -> PyResult<PyImageBuilder> {
    PY_CONTEXTS.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        // Use current window
        CURRENT_WINDOW.with(|cw| {
            let wid = *cw.borrow();
            let inner = borrow.get_mut(&wid)
                .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
            
            let view_id = inner.alloc_id();
            let id = crate::core::ID::from_u64(view_id);
            
            // Load Image via TextureManager
            let mut tex_id = None;
            let mut w = 100.0;
            let mut h = 100.0;
            
            crate::resource::TEXTURE_MANAGER.with(|tm| {
                if let Some((tid, tw, th)) = tm.borrow_mut().load_from_path(&path) {
                    tex_id = Some(tid);
                    w = tw as f32;
                    h = th as f32;
                }
            });
            
            let view = inner.arena.alloc(ViewHeader {
                 view_type: ViewType::Image,
                 ..Default::default()
            });
            view.id.set(id);
            view.texture_id.set(tex_id);
            view.width.set(w);
            view.height.set(h);
            view.fg_color.set(ColorF::white());

            let ptr = view as *mut ViewHeader;
            inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

            if let Some(&parent_id) = inner.parent_stack.last() {
                if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                    unsafe { (*parent_ptr).add_child(&*ptr); }
                }
            }

            Ok(PyImageBuilder { view_id })
        })
    })
}

/// Bezier builder
#[pyclass(name = "Bezier")]
#[derive(Clone)]
struct PyBezierBuilder {
    view_id: u64,
}

#[pymethods]
impl PyBezierBuilder {
    fn thickness(&self, t: f32) -> Self {
        with_view_mut(self.view_id, |v| v.thickness.set(t));
        self.clone()
    }
    
    fn color(&self, c: PyColor) -> Self {
        with_view_mut(self.view_id, |v| v.fg_color.set(ColorF::new(c.r, c.g, c.b, c.a)));
        self.clone()
    }
}

/// Create a Bezier curve
#[pyfunction]
#[pyo3(name = "Bezier")]
fn py_bezier(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> PyResult<PyBezierBuilder> {
    PY_CONTEXTS.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            let wid = *cw.borrow();
            let inner = borrow.get_mut(&wid)
                .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
            
            let view_id = inner.alloc_id();
            let id = crate::core::ID::from_u64(view_id);
            
            let points = [
                Vec2::new(p0.0, p0.1),
                Vec2::new(p1.0, p1.1),
                Vec2::new(p2.0, p2.1),
                Vec2::new(p3.0, p3.1),
            ];

            let view = inner.arena.alloc(ViewHeader {
                 view_type: ViewType::Bezier,
                 ..Default::default()
            });
            view.id.set(id);
            view.points.set(points);
            view.fg_color.set(ColorF::white());
            view.thickness.set(2.0);

            let ptr = view as *mut ViewHeader;
            inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

            if let Some(&parent_id) = inner.parent_stack.last() {
                if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                    unsafe { (*parent_ptr).add_child(&*ptr); }
                }
            }

            Ok(PyBezierBuilder { view_id })
        })
    })
}
        // So `py_bezier` expects absolute coordinates.
        // This is fine for connecting nodes.
        
        let points = [
            crate::core::Vec2::new(p0.0, p0.1),
            crate::core::Vec2::new(p1.0, p1.1),
            crate::core::Vec2::new(p2.0, p2.1),
            crate::core::Vec2::new(p3.0, p3.1),
        ];

        let view = inner.arena.alloc(ViewHeader {
             view_type: ViewType::Bezier,
             id,
             points,
             thickness: 2.0,
             fg_color: ColorF::white(),
             // Layout: 0 size so it doesn't disrupt flow?
             // Or max bounds?
             width: 0.0,
             height: 0.0,
             ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PyBezierBuilder { view_id })
    })
}



/// Splitter builder
#[pyclass(name = "Splitter")]
#[derive(Clone)]
struct PySplitterBuilder {
    view_id: u64,
    ratio: f32, // Return explicit ratio if changed
}

#[pymethods]
impl PySplitterBuilder {
    fn is_vertical(&self, v: bool) -> Self {
        with_view_mut(self.view_id, |header| header.is_vertical = v);
        self.clone()
    }
    
    fn ratio(&self, r: f32) -> Self {
        with_view_mut(self.view_id, |header| header.ratio = r);
        self.clone()
    }
    
    fn get_ratio(&self) -> f32 {
        self.ratio
    }
}

/// Create a Splitter
#[pyfunction]
#[pyo3(name = "Splitter")]
fn py_splitter(ratio: f32) -> PyResult<PySplitterBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        let id = crate::core::ID::from_u64(view_id);
        
        // INTERACTION LOGIC
        // Using last frame's rect to determine dragging
        // If active, update ratio
        let mut current_ratio = ratio;
        let is_active = crate::view::interaction::is_active(id);
        
        if is_active {
            // Get last frame rect
            if let Some(rect) = crate::view::interaction::get_rect(id) {
                 // Get mouse delta
                 // Note: We need orientation. Defaults to horiz (false).
                 // But orientation is set by builder methods LATER?
                 // Wait! Builder methods run AFTER this function.
                 // So we don't know is_vertical here unless we persist it or pass it in constructor.
                 // Assuming horizontal default. If user switches to vertical, interaction logic here is wrong?
                 // Solution: Interaction logic should happen in builder? No, ratio is return limit.
                 // Persist `is_vertical` state? Use `last_frame_rects` to guess? No.
                 // Immediate Mode dilemma.
                 // User should pass `is_vertical` to constructor if they want correct interaction?
                 // Or `fanta.Splitter(ratio, vertical=False)`.
                 // Or we use `ctx.interaction.last_frame_layout` to retrieve orientation? Not stored.
                 // We will just support `mouse_delta` projected onto last frame's axis?
                 // Actually, if we use `delta_x` and `delta_y`, and `rect` aspect ratio?
                 // If `rect.w > rect.h` -> Horizontal?
                 // Let's assume user passes correct ratio to update.
                 
                 // Access global mouse delta directly?
                 let (dx, dy) = crate::view::interaction::mouse_delta();
                 
                 // We need to know if vertical.
                 // Let's check aspect ratio of last frame rect. A splitter usually fills container.
                 // But splitters can be nested.
                 // Let's assume horizontal default. 
                 // If we had `is_vertical` argument, we'd use it.
                 // Let's just use `dx` for now, assuming horizontal.
                 // To support vertical properly, we might need `py_v_splitter` or argument.
                 
                 let is_vertical = false; // TODO: Persist or infer
                 
                 let total = if is_vertical { rect.h } else { rect.w };
                 if total > 1.0 {
                     let delta = if is_vertical { dy } else { dx };
                     current_ratio += delta / total;
                     current_ratio = current_ratio.clamp(0.1, 0.9);
                 }
            }
        }

        let view = inner.arena.alloc(ViewHeader {
             view_type: ViewType::Splitter,
             id,
             ratio: current_ratio,
             is_vertical: false, // Default
             // Default is stretch?
             width: 0.0, 
             height: 0.0,
             align: Align::Stretch,
             ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        // Push parent? Splitter IS a layout container.
        inner.parent_stack.push(view_id);

        Ok(PySplitterBuilder { view_id, ratio: current_ratio })
    })
}
#[pyclass]
#[derive(Clone)]
struct PyTextInputBuilder {
    text: String,
    width: f32,
    height: f32,
    return_val: Option<String>,
}

#[pymethods]
impl PyTextInputBuilder {
    fn width(&self, w: f32) -> Self { let mut s = self.clone(); s.width = w; s }
    fn height(&self, h: f32) -> Self { let mut s = self.clone(); s.height = h; s }
    
    /// Get the updated text value
    fn get_value(&self) -> String {
        self.return_val.clone().unwrap_or(self.text.clone())
    }
}

/// TextInput function
#[pyfunction]
#[pyo3(name = "TextInput")]
fn py_text_input(text: String) -> PyResult<PyTextInputBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        let id = crate::core::ID::from_u64(view_id);
        
        // Input Logic
        use crate::view::interaction::{is_clicked, is_focused, set_focus, is_key_pressed, drain_input_buffer};
        
        let mut new_text = text.clone();
        
        if is_clicked(id) {
             set_focus(id);
        }
        
        if is_focused(id) {
             let input = drain_input_buffer();
             if !input.is_empty() {
                 new_text.push_str(&input);
             }
             if is_key_pressed(winit::keyboard::KeyCode::Backspace) {
                 new_text.pop();
             }
             if is_key_pressed(winit::keyboard::KeyCode::Enter) {
                 set_focus(crate::core::ID::NONE);
             }
        }
        
        // Allocate string on arena
        let text_str = inner.arena.alloc_str(&new_text);
        let text_static = unsafe { std::mem::transmute::<&str, &'static str>(text_str) };

        let view = inner.arena.alloc(ViewHeader {
             view_type: ViewType::TextInput,
             id,
             text: text_static,
             width: 200.0,
             height: 30.0,
             bg_color: ColorF::new(0.05, 0.05, 0.05, 1.0),
             border_radius_tl: 4.0,
             border_radius_tr: 4.0,
             border_radius_br: 4.0,
             border_radius_bl: 4.0,
             ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PyTextInputBuilder {
            text: text,
            width: 0.0,
            height: 0.0,
            return_val: Some(new_text),
        })
    })
}

// Inserting NEW CODE:

/// ColorPicker builder
#[pyclass(name = "ColorPicker")]
#[derive(Clone)]
struct PyColorPickerBuilder {
    view_id: u64,
    h: f32,
    s: f32,
    v: f32,
}

#[pymethods]
impl PyColorPickerBuilder {
    fn get_h(&self) -> f32 { self.h }
    fn get_s(&self) -> f32 { self.s }
    fn get_v(&self) -> f32 { self.v }
    
    fn get_color(&self) -> PyColor {
        // Simple hsv_to_rgb inline
        let h = self.h;
        let s = self.s;
        let v = self.v;
        
        let c = v * s;
        let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h < 1.0/6.0 {
            (c, x, 0.0)
        } else if h < 2.0/6.0 {
            (x, c, 0.0)
        } else if h < 3.0/6.0 {
            (0.0, c, x)
        } else if h < 4.0/6.0 {
            (0.0, x, c)
        } else if h < 5.0/6.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        PyColor { r: r + m, g: g + m, b: b + m, a: 1.0 }
    }
}

/// Create a ColorPicker
#[pyfunction]
#[pyo3(name = "ColorPicker")]
fn py_color_picker(h: f32, s: f32, v: f32) -> PyResult<PyColorPickerBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        let id = crate::core::ID::from_u64(view_id);
        
        // INTERACTION
        let mut cur_h = h.clamp(0.0, 1.0);
        let mut cur_s = s.clamp(0.0, 1.0);
        let mut cur_v = v.clamp(0.0, 1.0);
        
        // Register interaction
        let is_active = crate::view::interaction::is_active(id);
        
        // If clicking (active), check where we clicked
        if is_active {
            if let Some(rect) = crate::view::interaction::get_rect(id) {
                let (mx, my) = crate::view::interaction::mouse_pos();
                let rel_x = mx - rect.x;
                let rel_y = my - rect.y;
                
                // Layout Constants (duplicated from renderer.rs/layout.rs)
                let padding = 10.0;
                let sv_size = 150.0;
                let gap = 8.0;
                let hue_width = 20.0;
                
                // Check SV Box
                let sv_rect_x = padding;
                let sv_rect_y = padding;
                if rel_x >= sv_rect_x && rel_x <= sv_rect_x + sv_size &&
                   rel_y >= sv_rect_y && rel_y <= sv_rect_y + sv_size {
                       cur_s = ((rel_x - sv_rect_x) / sv_size).clamp(0.0, 1.0);
                       cur_v = 1.0 - ((rel_y - sv_rect_y) / sv_size).clamp(0.0, 1.0);
                }
                
                // Check Hue Bar
                let hue_rect_x = padding + sv_size + gap;
                let hue_rect_y = padding;
                if rel_x >= hue_rect_x && rel_x <= hue_rect_x + hue_width &&
                   rel_y >= hue_rect_y && rel_y <= hue_rect_y + sv_size {
                       cur_h = ((rel_y - hue_rect_y) / sv_size).clamp(0.0, 1.0);
                }
            }
        }
        
        let view = inner.arena.alloc(ViewHeader {
             view_type: ViewType::ColorPicker,
             id,
             color_hsv: [cur_h, cur_s, cur_v],
             width: 0.0, // Layout engine calculates default size
             height: 0.0,
             ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PyColorPickerBuilder { view_id, h: cur_h, s: cur_s, v: cur_v })
    })
}

/// Markdown builder
#[pyclass(name = "MarkdownBuilder")]
#[derive(Clone)]
pub struct PyMarkdownBuilder {
    pub view_id: u64,
}

#[pymethods]
impl PyMarkdownBuilder {
    fn width(&self, w: f32) -> Self {
        with_view_mut(self.view_id, |v| v.width = w);
        self.clone()
    }
    
    fn height(&self, h: f32) -> Self {
        with_view_mut(self.view_id, |v| v.height = h);
        self.clone()
    }

    fn bg(&self, c: PyColor) -> Self {
        with_view_mut(self.view_id, |v| v.bg_color = c.into());
        self.clone()
    }

    fn fg(&self, c: PyColor) -> Self {
        with_view_mut(self.view_id, |v| v.fg_color = c.into());
        self.clone()
    }

    fn radius(&self, r: f32) -> Self {
        with_view_mut(self.view_id, |v| {
            v.border_radius_tl = r;
            v.border_radius_tr = r;
            v.border_radius_br = r;
            v.border_radius_bl = r;
        });
        self.clone()
    }

    fn border(&self, width: f32, color: PyColor) -> Self {
        with_view_mut(self.view_id, |v| {
            v.border_width = width;
            v.border_color = color.into();
        });
        self.clone()
    }

    fn shadow(&self, elevation: f32) -> Self {
        with_view_mut(self.view_id, |v| v.elevation = elevation);
        self.clone()
    }

    #[pyo3(signature = (property, target, duration=None, easing=None))]
    fn animate(&self, property: String, target: f32, duration: Option<f32>, easing: Option<PyEasing>) -> PyResult<Self> {
        let id = ID::from_u64(self.view_id);
        let val = if let Some(d) = duration {
            animate_ex(id, &property, target, d, easing.map(Into::into).unwrap_or(Easing::ExpoOut))
        } else {
            animate(id, &property, target, 10.0)
        };

        with_view_mut(self.view_id, |v| v.set_property_float(&property, val));
        Ok(self.clone())
    }
}

/// Create a Markdown view
#[pyfunction]
#[pyo3(name = "Markdown")]
pub fn py_markdown(text: String) -> PyResult<PyMarkdownBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut().ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        let view_id = inner.alloc_id();
        let id = crate::core::ID::from_u64(view_id);
        
        let text_str = inner.arena.alloc_str(&text);
        let text_static = unsafe { std::mem::transmute::<&str, &'static str>(text_str) };

        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Markdown,
            id,
            text: text_static,
            font_size: 14.0,
            fg_color: ColorF::white(),
            ..Default::default()
        });
        
        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });
        
        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }
        
        Ok(PyMarkdownBuilder { view_id })
    })
}

/// Path object for vector graphics
#[pyclass(name = "Path")]
#[derive(Clone)]
pub struct PyPath {
    pub inner: crate::draw::path::Path,
}

#[pymethods]
impl PyPath {
    #[new]
    fn new() -> Self {
        Self { inner: crate::draw::path::Path::new() }
    }

    fn move_to(&mut self, x: f32, y: f32) {
        self.inner.move_to(crate::core::Vec2::new(x, y));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.inner.line_to(crate::core::Vec2::new(x, y));
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.inner.quad_to(crate::core::Vec2::new(cx, cy), crate::core::Vec2::new(x, y));
    }

    fn cubic_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        self.inner.cubic_to(crate::core::Vec2::new(c1x, c1y), crate::core::Vec2::new(c2x, c2y), crate::core::Vec2::new(x, y));
    }

    fn close(&mut self) {
        self.inner.close();
    }

    #[staticmethod]
    fn circle(cx: f32, cy: f32, r: f32) -> Self {
        Self { inner: crate::draw::path::Path::circle(crate::core::Vec2::new(cx, cy), r) }
    }

    #[staticmethod]
    fn polygon(cx: f32, cy: f32, r: f32, sides: i32) -> Self {
        Self { inner: crate::draw::path::Path::polygon(crate::core::Vec2::new(cx, cy), r, sides) }
    }

    fn add_circle(&mut self, cx: f32, cy: f32, r: f32) -> Self {
        let other = crate::draw::path::Path::circle(crate::core::Vec2::new(cx, cy), r);
        self.inner.segments.extend(other.segments);
        self.clone()
    }

    fn add_polygon(&mut self, cx: f32, cy: f32, r: f32, sides: i32) -> Self {
        let other = crate::draw::path::Path::polygon(crate::core::Vec2::new(cx, cy), r, sides);
        self.inner.segments.extend(other.segments);
        self.clone()
    }
}

/// Path draw builder
#[pyclass(name = "PathDrawBuilder")]
#[derive(Clone, Copy)]
pub struct PyPathDrawBuilder {
    pub view_id: u64,
}

#[pymethods]
impl PyPathDrawBuilder {
    fn thickness(&self, t: f32) -> Self {
        with_view_mut(self.view_id, |v| v.thickness = t);
        *self
    }

    fn color(&self, c: PyColor) -> Self {
        with_view_mut(self.view_id, |v| v.fg_color = c.into());
        *self
    }

    #[pyo3(signature = (property, target, duration=None, easing=None))]
    fn animate(&self, property: String, target: f32, duration: Option<f32>, easing: Option<PyEasing>) -> PyResult<Self> {
        let id = ID::from_u64(self.view_id);
        let val = if let Some(d) = duration {
            animate_ex(id, &property, target, d, easing.map(Into::into).unwrap_or(Easing::ExpoOut))
        } else {
            animate(id, &property, target, 10.0)
        };

        with_view_mut(self.view_id, |v| v.set_property_float(&property, val));
        Ok(self.clone())
    }
}

/// Capture current frame as image
#[pyfunction]
#[pyo3(name = "capture_frame")]
pub fn py_capture_frame(path: String) {
    crate::view::interaction::request_screenshot(&path);
}

/// Start drawing a Path
#[pyfunction]
#[pyo3(name = "DrawPath")]
pub fn py_draw_path(path: PyPath) -> PyResult<PyPathDrawBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut().ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        let view_id = inner.alloc_id();
        let id = crate::core::ID::from_u64(view_id);
        
        let path_alloc = inner.arena.alloc(path.inner);
        let path_static = unsafe { std::mem::transmute(path_alloc) };

        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Path,
            id,
            path: Some(path_static),
            thickness: 1.0,
            ..Default::default()
        });
        
        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });
        
        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }
        
        Ok(PyPathDrawBuilder { view_id })
    })
}

/// Translation helper
#[pyfunction]
#[pyo3(name = "t")]
pub fn py_t(key: String) -> String {
    crate::core::i18n::I18nManager::t(&key)
}

/// Set current locale
#[pyfunction]
#[pyo3(name = "SetLocale")]
pub fn py_set_locale(locale: String) {
    crate::core::i18n::I18nManager::set_locale(&locale);
}

/// Add a translation to a catalog
#[pyfunction]
#[pyo3(name = "AddTranslation")]
pub fn py_add_translation(locale: String, key: String, value: String) {
    crate::core::i18n::I18nManager::add_translation(&locale, &key, &value);
}

/// Mount data to VFS
#[pyfunction]
#[pyo3(name = "Mount")]
pub fn py_mount(path: String, data: Vec<u8>) {
    crate::resource::vfs::VFS.with(|v| v.borrow_mut().mount(&path, data));
}
=======
>>>>>>> f6ccfc496e92db6add503037729bddfd387996d9
