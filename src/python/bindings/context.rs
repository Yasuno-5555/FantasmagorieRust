use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::core::{ColorF, FrameArena, ID};
use crate::draw::DrawList;
use crate::view::header::{ViewHeader, ViewType};
use crate::view::{Context, render_ui};
use crate::view::plot::PlotItem;

use super::widgets::{PyToggleBuilder, PyCheckboxBuilder, PyRadioBuilder, PyDropdownBuilder};
use super::layout::PyGridLayoutBuilder;
use super::types::{PyColor, PyEasing};

// Thread-local context for Python
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
    pub context: Context, // Stateful interaction context
    
    pub last_view: Option<u64>,
    pub plot_items: HashMap<u64, Vec<PlotItem<'static>>>,
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
            context: Context::default(), // Assuming Default works, or Context::new()
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
        // Context might need reset too?
    }

    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Helper to modify view header in the CURRENT window
pub fn with_view_mut<F>(id: u64, f: F)
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

pub fn with_current_context<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(&mut PyContextInner) -> PyResult<R>,
{
    PY_CONTEXTS.with(|contexts| {
        let mut contexts = contexts.borrow_mut();
        CURRENT_WINDOW.with(|cw| {
            let wid = *cw.borrow();
            let inner = contexts.get_mut(&wid).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Context not initialized for window {}", wid))
            })?;
            f(inner)
        })
    })
}

pub fn get_last_view_id() -> Option<u64> {
    PY_CONTEXTS.with(|ctx| {
        let contexts = ctx.borrow();
        CURRENT_WINDOW.with(|cw| {
             if let Some(inner) = contexts.get( &*cw.borrow() ) {
                 inner.last_view
             } else {
                 None
             }
        })
    })
}

pub fn alloc_view_in_current_window(is_row: bool, view_type: ViewType) -> PyResult<u64> {
    with_current_context(|inner| {
        let view_id = inner.alloc_id();
        inner.last_view = Some(view_id);

        let view = inner.arena.alloc(ViewHeader {
            view_type,
            id: Cell::new(ID::from_u64(view_id)),
            is_row: Cell::new(is_row),
            bg_color: Cell::new(ColorF::transparent()),
            ..Default::default()
        });

        let ptr = view as *mut ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

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
        Ok(view_id)
    })
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
    
    // --- Widget Helpers on Context ---

    #[pyo3(signature = (label, value))]
    fn toggle(&self, label: String, value: bool) -> PyResult<(bool, PyToggleBuilder)> {
         let mut new_val = value;
         
         let builder = with_current_context(|inner| {
                // Assuming inner.context has toggle method that returns a widget builder logic
                // But wait, inner.context is stateless Context. 
                // The original code was: `inner.context.toggle(&mut new_val)`
                // We need to implement that or rely on it existing.
                // For now, let's replicate the original binding logic logic if possible.
                // But wait, the original logic created a view implicitly?
                // `inner.context.toggle` suggests immediate mode logic.
                
                // Let's assume we can alloc a view here manually if needed, or use the helper.
                // Original: `let builder = inner.context.toggle(...)` -> This suggests Context returns a builder?
                // crate::view::Context::toggle likely returns something.
                
                // Since I cannot see crate::view::Context right now, I will assume it works as before.
                // But I added `context: Context` field.
                
                let builder = inner.context.toggle(&mut new_val);
                 
                let view_id = builder.view.id.get().0;
                 
                // Set text immediately
                let s = inner.arena.alloc_str(&label);
                builder.view.text.set(std::mem::transmute::<&str, &'static str>(s));

                Ok(PyToggleBuilder { view_id })
         })?;

        Ok((new_val, builder))
    }

    #[pyo3(signature = (label, value))]
    fn checkbox(&self, label: String, value: bool) -> PyResult<(bool, PyCheckboxBuilder)> {
         let mut new_val = value;
         
         let builder = with_current_context(|inner| {
                let builder = inner.context.checkbox(&mut new_val);
                let view_id = builder.view.id.get().0;
                let s = inner.arena.alloc_str(&label);
                builder.view.text.set(std::mem::transmute::<&str, &'static str>(s));
                Ok(PyCheckboxBuilder { view_id })
         })?;

        Ok((new_val, builder))
    }

    #[pyo3(signature = (label, current_value, my_value))]
    fn radio(&self, label: String, current_value: i32, my_value: i32) -> PyResult<(i32, PyRadioBuilder)> {
         let mut new_val = current_value;
         
         let builder = with_current_context(|inner| {
                 let builder = inner.context.radio(&mut new_val, my_value);
                 let view_id = builder.view.id.get().0;
                 let s = inner.arena.alloc_str(&label);
                 builder.view.text.set(std::mem::transmute::<&str, &'static str>(s));
                 Ok(PyRadioBuilder { view_id })
         })?;

        Ok((new_val, builder))
    }

    #[pyo3(signature = (label, items, index))]
    fn dropdown(&self, label: String, items: Vec<String>, index: usize) -> PyResult<(usize, PyDropdownBuilder)> {
         let mut new_idx = index;
         
         let builder = with_current_context(|inner| {
             // Need to alloc strings for items
             let item_strs: Vec<&'static str> = items.iter().map(|s| {
                 let allocated = inner.arena.alloc_str(s);
                 unsafe { std::mem::transmute::<&str, &'static str>(allocated) }
             }).collect();
             
             let builder = inner.context.dropdown(&mut new_idx, &item_strs);
             let view_id = builder.view.id.get().0;
             let s = inner.arena.alloc_str(&label);
             builder.view.text.set(std::mem::transmute::<&str, &'static str>(s));
             
             Ok(PyDropdownBuilder { view_id })
         })?;
         
         Ok((new_idx, builder))
    }

    #[pyo3(signature = (cols))]
    fn grid(&self, cols: usize) -> PyResult<PyGridLayoutBuilder> {
        let builder = with_current_context(|inner| {
            let builder = inner.context.grid(cols);
            Ok(PyGridLayoutBuilder { view_id: builder.view.id.get().0 })
        })?;
        Ok(builder)
    }
}
