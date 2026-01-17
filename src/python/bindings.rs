//! PyO3 bindings for Fantasmagorie
//!
//! Wraps Rust types for Python access, enabling the Ouroboros hot-reload workflow.
//! 
//! Strategy: Python doesn't have lifetimes, so we use IDs to reference Arena-allocated views.
//! Builders are thin wrappers that modify views by ID lookup.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::core::{ColorF, ID, FrameArena, Rectangle, Vec2};
use crate::view::header::{Align, ViewHeader, ViewType};
use crate::view::interaction::{animate, animate_ex, begin_interaction_pass, capture, drain_input_buffer, get_rect, get_scroll_delta, get_scroll_offset, handle_key_down, handle_key_up, handle_modifiers, handle_received_character, handle_scroll, is_active, is_any_captured, is_clicked, is_focused, is_hot, mouse_delta, mouse_pos, register_interactive, release, set_focus, set_scroll_offset, update_input, update_rect};
use crate::view::animation::Easing;
use crate::draw::DrawList;
use crate::view::render_ui;

// Thread-local context for Python
thread_local! {
    pub static PY_CONTEXT: RefCell<Option<PyContextInner>> = RefCell::new(None);
}

pub struct PyContextInner {
    pub arena: FrameArena,
    pub views: HashMap<u64, *mut ViewHeader<'static>>,
    pub root_id: Option<u64>,
    pub parent_stack: Vec<u64>,
    pub next_id: u64,
    pub draw_list: DrawList,
    pub width: u32,
    pub height: u32,
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
            width,
            height,
        }
    }

    pub fn reset(&mut self) {
        self.arena.reset();
        self.views.clear();
        self.root_id = None;
        self.parent_stack.clear();
        self.next_id = 1;
        self.draw_list.clear();
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Helper to modify view header
fn with_view_mut<F>(id: u64, f: F)
where F: FnOnce(&mut ViewHeader)
{
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        if let Some(inner) = borrow.as_mut() {
            if let Some(&ptr) = inner.views.get(&id) {
                unsafe { f(&mut *ptr); }
            }
        }
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
        PyColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
    }

    #[staticmethod]
    fn black() -> Self {
        PyColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }

    #[staticmethod]
    fn red() -> Self {
        PyColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }
    }

    #[staticmethod]
    fn green() -> Self {
        PyColor { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }
    }

    #[staticmethod]
    fn blue() -> Self {
        PyColor { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }
    }

    #[staticmethod]
    fn hex(val: u32) -> Self {
        let c = ColorF::from_hex(val);
        PyColor { r: c.r, g: c.g, b: c.b, a: c.a }
    }

    fn alpha(&self, a: f32) -> Self {
        PyColor { r: self.r, g: self.g, b: self.b, a }
    }

    fn __repr__(&self) -> String {
        format!("Color({:.2}, {:.2}, {:.2}, {:.2})", self.r, self.g, self.b, self.a)
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
    QuadIn, QuadOut, QuadInOut,
    CubicIn, CubicOut, CubicInOut,
    ExpoIn, ExpoOut, ExpoInOut,
    ElasticIn, ElasticOut, ElasticInOut,
    BackIn, BackOut, BackInOut,
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
    width: u32,
    height: u32,
}

#[pymethods]
impl PyContext {
    #[new]
    #[pyo3(signature = (width=1280, height=720))]
    fn new(width: u32, height: u32) -> Self {
        PY_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = Some(PyContextInner::new(width, height));
        });
        PyContext { width, height }
    }

    fn begin_frame(&self) {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                inner.reset();
            }
        });
    }

    fn end_frame(&self) -> PyResult<usize> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                // Get root and render
                if let Some(root_id) = inner.root_id {
                    if let Some(&ptr) = inner.views.get(&root_id) {
                        unsafe {
                            let root = &*ptr;
                            inner.draw_list.clear();
                            render_ui(root, inner.width as f32, inner.height as f32, &mut inner.draw_list);
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
        PY_CONTEXT.with(|ctx| {
            ctx.borrow().as_ref().map(|i| i.draw_list.len()).unwrap_or(0)
        })
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
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { (*ptr).width = w; }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn height(&self, h: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { (*ptr).height = h; }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn size(&self, w: f32, h: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { 
                        (*ptr).width = w;
                        (*ptr).height = h;
                    }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn padding(&self, p: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { 
                        (*ptr).padding = p;
                    }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn margin(&self, m: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { 
                        (*ptr).margin = m;
                    }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn bg(&self, color: PyColor) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { 
                        (*ptr).bg_color = ColorF::new(color.r, color.g, color.b, color.a);
                    }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn radius(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_radius_tl = r;
            v.border_radius_tr = r;
            v.border_radius_br = r;
            v.border_radius_bl = r;
        });
        Ok(*self)
    }

    fn radius_tl(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_tl = r); Ok(*self) }
    fn radius_tr(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_tr = r); Ok(*self) }
    fn radius_br(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_br = r); Ok(*self) }
    fn radius_bl(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_bl = r); Ok(*self) }

    fn icon(&self, icon: String) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let s = inner.arena.alloc_str(&icon);
                        (*ptr).icon = std::mem::transmute::<&str, &'static str>(s);
                    }
                }
            }
        });
        Ok(*self)
    }

    fn icon_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.icon_size = size);
        Ok(*self)
    }

    fn border(&self, width: f32, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_width = width;
            v.border_color = color.into();
        });
        Ok(*self)
    }

    fn shadow(&self, elevation: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { 
                        (*ptr).elevation = elevation; 
                        // println!("DEBUG: PyBoxBuilder set elevation to {}", elevation);
                    }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn flex(&self, grow: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { (*ptr).flex_grow = grow; }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }
    fn blur(&self, b: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { (*ptr).backdrop_blur = b; }
                }
            }
        });
        Ok(PyBoxBuilder { view_id: self.view_id })
    }

    fn icon(&self, icon: String) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let s = inner.arena.alloc_str(&icon);
                        (*ptr).icon = std::mem::transmute::<&str, &'static str>(s);
                    }
                }
            }
        });
        Ok(*self)
    }

    fn icon_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.icon_size = size);
        Ok(*self)
    }

    fn hover(&self, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.bg_hover = Some(color.into()));
        Ok(*self)
    }

    fn active(&self, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.bg_active = Some(color.into()));
        Ok(*self)
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
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                     unsafe { 
                        (*ptr).fg_color = ColorF::new(color.r, color.g, color.b, color.a);
                    }
                }
            }
        });
        Ok(PyTextBuilder { view_id: self.view_id })
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

    fn font_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.font_size = size);
        Ok(*self)
    }

    fn icon(&self, icon: String) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let s = inner.arena.alloc_str(&icon);
                        (*ptr).icon = std::mem::transmute::<&str, &'static str>(s);
                    }
                }
            }
        });
        Ok(*self)
    }

    fn icon_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.icon_size = size);
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
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { (*ptr).width = w; }
                }
            }
        });
        Ok(PyButtonBuilder { view_id: self.view_id })
    }

    fn height(&self, h: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { (*ptr).height = h; }
                }
            }
        });
        Ok(PyButtonBuilder { view_id: self.view_id })
    }

    fn bg(&self, color: PyColor) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                     unsafe { 
                        (*ptr).bg_color = ColorF::new(color.r, color.g, color.b, color.a);
                    }
                }
            }
        });
        Ok(PyButtonBuilder { view_id: self.view_id })
    }

    fn radius(&self, r: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_radius_tl = r;
            v.border_radius_tr = r;
            v.border_radius_br = r;
            v.border_radius_bl = r;
        });
        Ok(self.clone())
    }

    fn radius_tl(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_tl = r); Ok(self.clone()) }
    fn radius_tr(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_tr = r); Ok(self.clone()) }
    fn radius_br(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_br = r); Ok(self.clone()) }
    fn radius_bl(&self, r: f32) -> PyResult<Self> { with_view_mut(self.view_id, |v| v.border_radius_bl = r); Ok(self.clone()) }

    fn border(&self, width: f32, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| {
            v.border_width = width;
            v.border_color = color.into();
        });
        Ok(self.clone())
    }

    fn shadow(&self, elevation: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.elevation = elevation);
        Ok(self.clone())
    }

    fn fg(&self, color: PyColor) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                     unsafe { 
                        (*ptr).fg_color = ColorF::new(color.r, color.g, color.b, color.a);
                    }
                }
            }
        });
        Ok(*self)
    }

    fn icon(&self, icon: String) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe {
                        let s = inner.arena.alloc_str(&icon);
                        (*ptr).icon = std::mem::transmute::<&str, &'static str>(s);
                    }
                }
            }
        });
        Ok(*self)
    }

    fn icon_size(&self, size: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.icon_size = size);
        Ok(*self)
    }

    fn hover(&self, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.bg_hover = Some(color.into()));
        Ok(*self)
    }

    fn active(&self, color: PyColor) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.bg_active = Some(color.into()));
        Ok(*self)
    }
        
    fn font_size(&self, size: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                    unsafe { (*ptr).font_size = size; }
                }
            }
        });
        Ok(PyButtonBuilder { view_id: self.view_id })
    }

    fn clicked(&self) -> bool {
        use crate::view::interaction;
        interaction::is_clicked(ID::from_u64(self.view_id))
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

// ============================================================================
// Free Functions (Widget Constructors)
// ============================================================================

/// Create a Box container
#[pyfunction]
#[pyo3(name = "Box")]
fn py_box() -> PyResult<PyBoxBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        
        // Allocate view on arena
        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Box,
            id: ID::from_u64(view_id),
            bg_color: ColorF::new(0.15, 0.15, 0.18, 1.0),
            ..Default::default()
        });

        // Store raw pointer (arena guarantees lifetime until reset)
        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        // Set as root if first view
        if inner.root_id.is_none() {
            inner.root_id = Some(view_id);
        }

        // Add to parent if in stack
        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe {
                    (*parent_ptr).add_child(&*ptr);
                }
            }
        }

        Ok(PyBoxBuilder { view_id })
    })
}

/// Create a Row container
#[pyfunction]
#[pyo3(name = "Row")]
fn py_row() -> PyResult<PyBoxBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        
        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Box,
            id: ID::from_u64(view_id),
            is_row: true,
            bg_color: ColorF::transparent(),
            ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if inner.root_id.is_none() {
            inner.root_id = Some(view_id);
        }

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        // Push onto parent stack
        inner.parent_stack.push(view_id);

        Ok(PyBoxBuilder { view_id })
    })
}

/// Create a Column container
#[pyfunction]
#[pyo3(name = "Column")]
fn py_column() -> PyResult<PyBoxBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        
        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Box,
            id: ID::from_u64(view_id),
            is_row: false,
            bg_color: ColorF::transparent(),
            ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if inner.root_id.is_none() {
            inner.root_id = Some(view_id);
        }

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        inner.parent_stack.push(view_id);

        Ok(PyBoxBuilder { view_id })
    })
}

/// Create a Text label
#[pyfunction]
#[pyo3(name = "Text")]
fn py_text(text: &str) -> PyResult<PyTextBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        
        // Allocate string on arena
        let text_str = inner.arena.alloc_str(text);
        let text_static = unsafe { std::mem::transmute::<&str, &'static str>(text_str) };
        
        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Text,
            id: ID::from_u64(view_id),
            text: text_static,
            ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PyTextBuilder { view_id })
    })
}

/// Create a Button
#[pyfunction]
#[pyo3(name = "Button")]
fn py_button(label: &str) -> PyResult<PyButtonBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();

        // Allocate label string on arena
        let label_str = inner.arena.alloc_str(label);
        let label_static = unsafe { std::mem::transmute::<&str, &'static str>(label_str) };
        
        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Button,
            id: ID::from_u64(view_id),
            text: label_static,
            bg_color: ColorF::new(0.25, 0.25, 0.3, 1.0),
            elevation: 2.0,
             border_radius_tl: 6.0,
             border_radius_tr: 6.0,
             border_radius_br: 6.0,
             border_radius_bl: 6.0,
            ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PyButtonBuilder { view_id })
    })
}

/// Slider builder
#[pyclass]
pub struct PySliderBuilder {
    view_id: u64,
    value: f32,
}

#[pymethods]
impl PySliderBuilder {
    fn width(&self, w: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                     unsafe { (*ptr).width = w; }
                }
            }
        });
        Ok(PySliderBuilder { view_id: self.view_id, value: self.value })
    }

    fn height(&self, h: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                     unsafe { (*ptr).height = h; }
                }
            }
        });
        Ok(PySliderBuilder { view_id: self.view_id, value: self.value })
    }

    fn get_value(&self) -> f32 {
        self.value
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

/// Create a Slider
#[pyfunction]
#[pyo3(name = "Slider")]
fn py_slider(value: f32, min: f32, max: f32) -> PyResult<PySliderBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        
        let mut new_value = value;
        let id_obj = ID::from_u64(view_id);

        // Interaction Logic: Value Update
        use crate::view::interaction;
        if interaction::is_active(id_obj) {
             let (dx, _) = interaction::mouse_delta();
             // Sensitivity: 1% of range per pixel? 
             // Or based on width? Current frame we don't know computed width. 
             // Assume 200px width?
             // Use generic sensitivity.
             let range = max - min;
             let sensitivity = range / 200.0; 
             new_value += dx * sensitivity;
             new_value = new_value.clamp(min, max);
        }

        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Slider,
            id: id_obj,
            value: new_value,
            min,
            max,
            ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PySliderBuilder { view_id, value: new_value })
    })
}

/// Toggle builder
#[pyclass]
pub struct PyToggleBuilder {
    view_id: u64,
    value: bool,
}

#[pymethods]
impl PyToggleBuilder {
    fn height(&self, h: f32) -> PyResult<Self> {
        PY_CONTEXT.with(|ctx| {
            if let Some(inner) = ctx.borrow_mut().as_mut() {
                if let Some(&ptr) = inner.views.get(&self.view_id) {
                     unsafe { (*ptr).height = h; }
                }
            }
        });
        Ok(PyToggleBuilder { view_id: self.view_id, value: self.value })
    }

    fn get_value(&self) -> bool {
        self.value
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

/// Create a Toggle
#[pyfunction]
#[pyo3(name = "Toggle")]
fn py_toggle(value: bool) -> PyResult<PyToggleBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        let id_obj = ID::from_u64(view_id);
        
        // Interaction Logic: Toggle Value
        use crate::view::interaction;
        let mut new_value = value;
        if interaction::is_clicked(id_obj) {
             new_value = !new_value;
        }

        let view = inner.arena.alloc(ViewHeader {
            view_type: ViewType::Toggle,
            id: id_obj,
            value: if new_value { 1.0 } else { 0.0 }, // Store boolean as f32
            ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PyToggleBuilder { view_id, value: new_value })
    })
}

/// End current container
#[pyfunction]
#[pyo3(name = "End")]
fn py_end() -> PyResult<()> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        inner.parent_stack.pop();
        Ok(())
    })
}

// ============================================================================
// Module Registration
// ============================================================================

/// Register all Python bindings
pub fn register(py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PyContext>()?;
    m.add_class::<PyColor>()?;
    m.add_class::<PyBoxBuilder>()?;
    m.add_class::<PyTextBuilder>()?;
    m.add_class::<PyButtonBuilder>()?;
    m.add_class::<PySliderBuilder>()?;
    m.add_class::<PyTextInputBuilder>()?;
    m.add_class::<PySplitterBuilder>()?;
    m.add_class::<PyColorPickerBuilder>()?;
    
    // Functions
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
    m.add_function(wrap_pyfunction!(py_end, m)?)?;
    
    // Builders
    m.add_class::<PyMarkdownBuilder>()?;
    m.add_class::<PyPath>()?;
    m.add_class::<PyPathDrawBuilder>()?;
    m.add_class::<PyEasing>()?;
    
    Ok(())
}

// ...

/// Image builder
#[pyclass(name = "Image")]
#[derive(Clone)]
struct PyImageBuilder {
    view_id: u64,
}

#[pymethods]
impl PyImageBuilder {
    fn size(&self, w: f32, h: f32) -> Self {
        with_view_mut(self.view_id, |v| { v.width = w; v.height = h; });
        self.clone()
    }

    fn tint(&self, c: PyColor) -> Self {
        with_view_mut(self.view_id, |v| v.fg_color = ColorF::new(c.r, c.g, c.b, c.a));
        self.clone()
    }
}

/// Create an Image
#[pyfunction]
#[pyo3(name = "Image")]
fn py_image(path: String) -> PyResult<PyImageBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        let id = crate::core::ID::from_u64(view_id);
        
        // Load Image via TextureManager
        // TextureManager is thread_local in resource module
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
        
        // If failed to load, we can throw error or return placeholder?
        // Let's assume placeholder rect if no texture.
        
        let view = inner.arena.alloc(ViewHeader {
             view_type: ViewType::Image,
             id,
             texture_id: tex_id,
             width: w,
             height: h,
             fg_color: ColorF::white(), // Default no tint
             ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }

        Ok(PyImageBuilder { view_id })
    })
}

// ... (previous structs)

/// Bezier builder
#[pyclass(name = "Bezier")]
#[derive(Clone)]
struct PyBezierBuilder {
    view_id: u64,
}

#[pymethods]
impl PyBezierBuilder {
    fn thickness(&self, t: f32) -> Self {
        with_view_mut(self.view_id, |v| v.thickness = t);
        self.clone()
    }
    
    fn color(&self, c: PyColor) -> Self {
        with_view_mut(self.view_id, |v| v.fg_color = ColorF::new(c.r, c.g, c.b, c.a));
        self.clone()
    }
}

/// Create a Bezier curve
#[pyfunction]
#[pyo3(name = "Bezier")]
fn py_bezier(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> PyResult<PyBezierBuilder> {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let inner = borrow.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Context not initialized"))?;
        
        let view_id = inner.alloc_id();
        let id = crate::core::ID::from_u64(view_id);
        
        // Calculate bounding box for layout (approximate)
        // Or just let user manage layout? Bezier is usually absolute or relative to container.
        // Let's set auto size? No, Bezier might not affect flow layout much if it's overlay.
        // But for "Wire", it might be absolute position.
        // We set points relative to what?
        // If we draw raw coordinates, we assume screen space or local space?
        // Renderer uses `view.points` directly in `add_bezier`. `add_bezier` uses `draw_command`.
        // `DrawCommand::Bezier` coords are in screen space (if backend uses them as pos).
        // Wait, `opengl.rs` uses `p0` directly.
        // If `p0` is (0,0), it's top-left of window.
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
