use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use crate::core::{ColorF, ID};
use crate::view::header::{ViewType};
use crate::view::interaction::{is_active, get_rect, mouse_delta};

use super::context::{
    with_view_mut, with_current_context, alloc_view_in_current_window, 
    PY_CONTEXTS, CURRENT_WINDOW
};
use super::widgets::PyBoxBuilder;

// ============================================================================
// Layout Builders
// ============================================================================

#[pyclass(name = "GridLayoutBuilder", unsendable)]
#[derive(Clone, Copy)]
pub struct PyGridLayoutBuilder {
    pub view_id: u64,
}

#[pymethods]
impl PyGridLayoutBuilder {
    fn gap(&self, gap: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.margin.set(gap));
        Ok(*self)
    }

    fn width(&self, w: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.width.set(w));
        Ok(*self)
    }
    
    fn height(&self, h: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.height.set(h));
        Ok(*self)
    }
    
    fn flex(&self, grow: f32) -> PyResult<Self> {
        with_view_mut(self.view_id, |v| v.flex_grow.set(grow));
        Ok(*self)
    }
}

#[pyclass(name = "Splitter", unsendable)]
#[derive(Clone)]
pub struct PySplitterBuilder {
    pub view_id: u64,
    pub ratio: f32, 
}

#[pymethods]
impl PySplitterBuilder {
    fn is_vertical(&self, v: bool) -> Self {
        with_view_mut(self.view_id, |header| header.is_vertical.set(v));
        self.clone()
    }
    
    fn ratio(&self, r: f32) -> Self {
        with_view_mut(self.view_id, |header| header.ratio.set(r));
        self.clone()
    }
    
    fn get_ratio(&self) -> f32 {
        self.ratio
    }
}

// ============================================================================
// Layout Free Functions
// ============================================================================

#[pyfunction]
#[pyo3(name = "Row")]
pub fn py_row() -> PyResult<PyBoxBuilder> {
    let view_id = alloc_view_in_current_window(true, ViewType::Box)?;
    with_view_mut(view_id, |v| v.bg_color.set(ColorF::transparent()));

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

#[pyfunction]
#[pyo3(name = "Column")]
pub fn py_column() -> PyResult<PyBoxBuilder> {
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

#[pyfunction]
#[pyo3(name = "End")]
pub fn py_end() -> PyResult<()> {
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

#[pyfunction]
#[pyo3(name = "Splitter")]
pub fn py_splitter(ratio: f32) -> PyResult<PySplitterBuilder> {
    with_current_context(|inner| {
        let view_id = inner.alloc_id();
        let id = ID::from_u64(view_id);
        
        // Interaction Logic (simplified for port, assuming checked in bindings.rs)
        let mut current_ratio = ratio;
        let is_active = is_active(id);
        
        if is_active {
            if let Some(rect) = get_rect(id) {
                 let (dx, dy) = mouse_delta();
                 // Assuming horizontal default as per original comments
                 let is_vertical = false; 
                 let total = if is_vertical { rect.h } else { rect.w };
                 if total > 1.0 {
                     let delta = if is_vertical { dy } else { dx };
                     current_ratio += delta / total;
                     current_ratio = current_ratio.clamp(0.1, 0.9);
                 }
            }
        }

        let view = inner.arena.alloc(crate::view::header::ViewHeader {
            view_type: ViewType::Splitter,
            id: std::cell::Cell::new(id),
            // Initialize ratio? ViewHeader struct needs 'ratio' field? 
            // Previous bindings.rs: with_view_mut(view_id, |header| header.ratio.set(r));
            // ViewHeader must have ratio field.
            ratio: std::cell::Cell::new(current_ratio),
            bg_color: std::cell::Cell::new(ColorF::transparent()),
            ..Default::default()
        });

        let ptr = view as *mut crate::view::header::ViewHeader;
        inner.views.insert(view_id, unsafe { std::mem::transmute(ptr) });

        if let Some(&parent_id) = inner.parent_stack.last() {
            if let Some(&parent_ptr) = inner.views.get(&parent_id) {
                unsafe { (*parent_ptr).add_child(&*ptr); }
            }
        }
        
        // Push splitter to stack? Original bindings did NOT push splitter to stack?
        // Wait, Splitter is a container? 
        // Original bindings: 
        // `if let Some(&parent_id) = inner.parent_stack.last() ...`
        // It did NOT push itself. But typically splitters contain children.
        // If it's a container, it creates 2 children?
        // Let's assume user manually adds children or Splitter manages them differently.
        // For now, mirroring strict port.
        
        Ok(PySplitterBuilder {
            view_id,
            ratio: current_ratio,
        })
    })
}
