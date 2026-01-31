use pyo3::prelude::*;
use super::context::PyContext;
use super::types::{PyColor, PyEasing};
use super::widgets::*;
use super::layout::*;

/// Register the Python module
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyContext>()?;
    m.add_class::<PyColor>()?;
    m.add_class::<PyEasing>()?;
    
    // Widgets
    m.add_class::<PyBoxBuilder>()?;
    m.add_class::<PyTextBuilder>()?;
    m.add_class::<PyButtonBuilder>()?;
    m.add_class::<PyToggleBuilder>()?;
    m.add_class::<PyCheckboxBuilder>()?;
    m.add_class::<PyRadioBuilder>()?;
    m.add_class::<PySliderBuilder>()?;
    m.add_class::<PyDropdownBuilder>()?;
    m.add_class::<PyImageBuilder>()?;
    m.add_class::<PyBezierBuilder>()?;
    m.add_class::<PyMarkdownBuilder>()?;
    m.add_class::<PyPathDrawBuilder>()?;
    m.add_class::<PyGridBuilder>()?;
    m.add_class::<PyRulerBuilder>()?;
    m.add_class::<PyTransformGizmoBuilder>()?;
    m.add_class::<PyMathBuilder>()?;
    m.add_class::<PyScene3DBuilder>()?;
    m.add_class::<PyCurveEditorBuilder>()?;
    m.add_class::<PyPlotBuilder>()?;
    m.add_class::<PyColorPickerBuilder>()?;
    m.add_class::<PyVec3>()?;
    m.add_class::<PyPath>()?;

    // Layout
    m.add_class::<PyGridLayoutBuilder>()?;
    m.add_class::<PySplitterBuilder>()?;

    // Free Functions
    m.add_function(wrap_pyfunction!(py_box, m)?)?;
    m.add_function(wrap_pyfunction!(py_text, m)?)?;
    m.add_function(wrap_pyfunction!(py_button, m)?)?;
    m.add_function(wrap_pyfunction!(py_toggle, m)?)?;
    m.add_function(wrap_pyfunction!(py_checkbox, m)?)?;
    m.add_function(wrap_pyfunction!(py_radio, m)?)?;
    m.add_function(wrap_pyfunction!(py_slider, m)?)?;
    m.add_function(wrap_pyfunction!(py_dropdown, m)?)?; // I need to add py_dropdown to widgets.rs?
    m.add_function(wrap_pyfunction!(py_image, m)?)?;
    m.add_function(wrap_pyfunction!(py_bezier, m)?)?;
    m.add_function(wrap_pyfunction!(py_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(py_draw_path, m)?)?;
    m.add_function(wrap_pyfunction!(py_capture_frame, m)?)?;
    m.add_function(wrap_pyfunction!(py_t, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_locale, m)?)?;
    m.add_function(wrap_pyfunction!(py_add_translation, m)?)?;
    m.add_function(wrap_pyfunction!(py_mount, m)?)?;
    m.add_function(wrap_pyfunction!(py_plot, m)?)?;
    m.add_function(wrap_pyfunction!(py_scene_3d, m)?)?;
    m.add_function(wrap_pyfunction!(py_curve_editor, m)?)?;
    m.add_function(wrap_pyfunction!(py_ruler, m)?)?;
    m.add_function(wrap_pyfunction!(py_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_transform_gizmo, m)?)?;
    m.add_function(wrap_pyfunction!(py_math, m)?)?;
    m.add_function(wrap_pyfunction!(py_color_picker, m)?)?;
    m.add_function(wrap_pyfunction!(py_text_input, m)?)?;

    // Layout Functions
    m.add_function(wrap_pyfunction!(py_row, m)?)?;
    m.add_function(wrap_pyfunction!(py_column, m)?)?;
    m.add_function(wrap_pyfunction!(py_end, m)?)?;
    m.add_function(wrap_pyfunction!(py_splitter, m)?)?;

    // Immediate Mode Props
    m.add_function(wrap_pyfunction!(py_width_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_height_free, m)?)?;

    Ok(())
}
