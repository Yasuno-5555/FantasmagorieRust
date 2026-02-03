use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::core::{ColorF, FrameArena, ID};
use crate::draw::DrawList;
use crate::view::header::{ViewHeader, ViewType};
use crate::view::render_ui;
use crate::widgets::UIContext;
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
    // pub context: UIContext, // Removed to avoid self-reference
    
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
            // context: UIContext::default(), 
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
                // Create temp UIContext
                // ...
                let arena = unsafe { std::mem::transmute::<&FrameArena, &'static FrameArena>(&inner.arena) };
                let views = &mut inner.views;
                let parent_stack = &inner.parent_stack;
                let current_next_id = inner.next_id;
                
                let (view_id, final_next_id) = {
                    let mut ui = UIContext::new(arena);
                    ui.next_id = current_next_id;
                    for &id in parent_stack {
                        if let Some(&ptr) = views.get(&id) { unsafe { ui.begin(&*ptr); } }
                    }
                    
                    let builder = ui.toggle(&mut new_val);
                    let view_id = builder.view.id.get().0;
                    let view_ptr = unsafe { std::mem::transmute::<&ViewHeader, *mut ViewHeader<'static>>(builder.view) };
                    
                    let s = arena.alloc_str(&label);
                    builder.view.text.set(unsafe{ std::mem::transmute::<&str, &'static str>(s) });

                    views.insert(view_id, view_ptr);
                    
                    (view_id, ui.next_id)
                }; 
                inner.next_id = final_next_id;
                
                Ok(PyToggleBuilder { view_id })
         })?;

        Ok((new_val, builder))
    }

    #[pyo3(signature = (label, value))]
    fn checkbox(&self, label: String, value: bool) -> PyResult<(bool, PyCheckboxBuilder)> {
         let mut new_val = value;
         
         let builder = with_current_context(|inner| {
                let arena = unsafe { std::mem::transmute::<&FrameArena, &'static FrameArena>(&inner.arena) };
                let views = &mut inner.views;
                let parent_stack = &inner.parent_stack;
                let current_next_id = inner.next_id;

                let (view_id, final_next_id) = {
                    let mut ui = UIContext::new(arena);
                    ui.next_id = current_next_id;
                    for &id in parent_stack {
                        if let Some(&ptr) = views.get(&id) { unsafe { ui.begin(&*ptr); } }
                    }
                    
                    let builder = ui.checkbox(&mut new_val);
                    let view_id = builder.view.id.get().0;
                    let view_ptr = unsafe { std::mem::transmute::<&ViewHeader, *mut ViewHeader<'static>>(builder.view) };
                    let s = arena.alloc_str(&label);
                    builder.view.text.set(unsafe{ std::mem::transmute::<&str, &'static str>(s) });
                    
                    views.insert(view_id, view_ptr);
                    (view_id, ui.next_id)
                };
                inner.next_id = final_next_id;
                
                Ok(PyCheckboxBuilder { view_id })
         })?;

        Ok((new_val, builder))
    }

    #[pyo3(signature = (label, current_value, my_value))]
    fn radio(&self, label: String, current_value: i32, my_value: i32) -> PyResult<(i32, PyRadioBuilder)> {
         let mut new_val = current_value;
         
         let builder = with_current_context(|inner| {
                 let arena = unsafe { std::mem::transmute::<&FrameArena, &'static FrameArena>(&inner.arena) };
                 let views = &mut inner.views;
                 let parent_stack = &inner.parent_stack;
                 let current_next_id = inner.next_id;

                 let (view_id, final_next_id) = {
                     let mut ui = UIContext::new(arena);
                     ui.next_id = current_next_id;
                     for &id in parent_stack {
                         if let Some(&ptr) = views.get(&id) { unsafe { ui.begin(&*ptr); } }
                     }
    
                     let builder = ui.radio(&mut new_val, my_value);
                     let view_id = builder.view.id.get().0;
                     let view_ptr = unsafe { std::mem::transmute::<&ViewHeader, *mut ViewHeader<'static>>(builder.view) };
                     let s = arena.alloc_str(&label);
                     builder.view.text.set(unsafe{ std::mem::transmute::<&str, &'static str>(s) });
                     
                     views.insert(view_id, view_ptr);
                     (view_id, ui.next_id)
                 };
                 inner.next_id = final_next_id;
                 
                 Ok(PyRadioBuilder { view_id })
         })?;

        Ok((new_val, builder))
    }

    #[pyo3(signature = (label, items, index))]
    fn dropdown(&self, label: String, items: Vec<String>, index: usize) -> PyResult<(usize, PyDropdownBuilder)> {
         let mut new_idx = index;
         
         let builder = with_current_context(|inner| {
             let arena = unsafe { std::mem::transmute::<&FrameArena, &'static FrameArena>(&inner.arena) };
             let views = &mut inner.views;
             let parent_stack = &inner.parent_stack;
             let current_next_id = inner.next_id;

             let (view_id, final_next_id) = {
                 let mut ui = UIContext::new(arena);
                 ui.next_id = current_next_id;
                 for &id in parent_stack {
                     if let Some(&ptr) = views.get(&id) { unsafe { ui.begin(&*ptr); } }
                 }
    
                 let builder = ui.dropdown(items, &mut new_idx);
                 let view_id = builder.view.id.get().0;
                 let view_ptr = unsafe { std::mem::transmute::<&ViewHeader, *mut ViewHeader<'static>>(builder.view) };
                 let s = arena.alloc_str(&label);
                 builder.view.text.set(unsafe{ std::mem::transmute::<&str, &'static str>(s) });
                 
                 views.insert(view_id, view_ptr);
                 (view_id, ui.next_id)
             };
             inner.next_id = final_next_id;
    
             Ok(PyDropdownBuilder { view_id })
         })?;
         
         Ok((new_idx, builder))
    }

    #[pyo3(signature = (cols))]
    fn grid(&self, cols: usize) -> PyResult<PyGridLayoutBuilder> {
        let builder = with_current_context(|inner| {
            let arena = unsafe { std::mem::transmute::<&FrameArena, &'static FrameArena>(&inner.arena) };
            let views = &mut inner.views;
            let parent_stack = &inner.parent_stack;
            let current_next_id = inner.next_id;

            let (view_id, final_next_id) = {
                let mut ui = UIContext::new(arena);
                ui.next_id = current_next_id;
                for &id in parent_stack {
                    if let Some(&ptr) = views.get(&id) { unsafe { ui.begin(&*ptr); } }
                }
   
               let builder = ui.layout_grid(cols);
               let view_id = builder.view.id.get().0;
               let view_ptr = unsafe { std::mem::transmute::<&ViewHeader, *mut ViewHeader<'static>>(builder.view) };
               
               views.insert(view_id, view_ptr);
               (view_id, ui.next_id)
            };
            inner.next_id = final_next_id;
    
            Ok(PyGridLayoutBuilder { view_id })
        })?;
        Ok(builder)
    }
}
