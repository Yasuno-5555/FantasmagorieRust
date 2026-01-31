use pyo3::prelude::*;
use crate::core::ColorF;
use crate::view::animation::Easing;

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
