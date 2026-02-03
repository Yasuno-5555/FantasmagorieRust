use crate::draw::DrawList;
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Clone)]
pub enum BlendMode {
    Alpha,
    Additive,
    Multiply,
    Screen,
}

#[derive(Clone)]
pub enum TimeProperty<T> {
    Static(T),
    Dynamic(Arc<dyn Fn(f64) -> T + Send + Sync>),
}

impl<T: Copy> TimeProperty<T> {
    pub fn evaluate(&self, time: f64) -> T {
        match self {
            Self::Static(v) => *v,
            Self::Dynamic(f) => f(time),
        }
    }
}

impl<T> From<T> for TimeProperty<T> {
    fn from(v: T) -> Self {
        Self::Static(v)
    }
}

#[derive(Clone)]
pub struct ShaderSlot {
    pub source: String,
    pub entry_point: String,
    pub parameters: HashMap<String, f32>,
}

pub enum LayerSource {
    DrawList(DrawList),
    Shader(ShaderSlot),
}

pub struct Layer {
    pub name: String,
    pub source: LayerSource,
    pub blend_mode: BlendMode,
    pub opacity: TimeProperty<f32>,
    pub visible: bool,
}

impl Layer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source: LayerSource::DrawList(DrawList::new()),
            blend_mode: BlendMode::Alpha,
            opacity: TimeProperty::Static(1.0),
            visible: true,
        }
    }
    
    pub fn with_shader(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source: LayerSource::Shader(ShaderSlot {
                source: source.into(),
                entry_point: "fs_main".to_string(),
                parameters: HashMap::new(),
            }),
            blend_mode: BlendMode::Alpha,
            opacity: TimeProperty::Static(1.0),
            visible: true,
        }
    }
}

pub struct Composition {
    pub layers: Vec<Layer>,
}

impl Composition {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }
}
