use super::core::types::Color;

pub mod traits;
pub mod window;
pub mod button;
pub mod label;
pub mod slider;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WindowData {
    pub title: String,
    pub closable: bool,
    pub resizable: bool,
    pub draggable: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ButtonData {
    pub label: String,
    pub is_primary: bool,
    pub is_danger: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LabelData {
    pub text: String,
    pub bold: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SliderData {
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub value: f32, // Display value
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContainerData {
    // Just a container
}

#[derive(Clone, Debug, PartialEq)]
pub enum WidgetKind {
    None,
    Window(WindowData),
    Button(ButtonData),
    Label(LabelData),
    Slider(SliderData),
    Container(ContainerData),
}

impl Default for WidgetKind {
    fn default() -> Self {
        Self::None
    }
}
