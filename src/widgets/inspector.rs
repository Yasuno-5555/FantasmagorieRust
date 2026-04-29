use crate::widgets::UIContext;
use crate::core::ColorF;

/// Trait for types that can be displayed and edited in the Inspector
pub trait Inspectable<'b> {
    fn inspect(&mut self, ui: &mut UIContext<'b>, label: &'b str);
}

impl<'b> Inspectable<'b> for f32 {
    fn inspect(&mut self, ui: &mut UIContext<'b>, label: &'b str) {
        ui.row().height(32.0).build();
        ui.text(label).width(100.0).build();
        ui.value_dragger(self, -1000.0, 1000.0).flex_grow(1.0).build();
        ui.end();
    }
}

impl<'b> Inspectable<'b> for bool {
    fn inspect(&mut self, ui: &mut UIContext<'b>, label: &'b str) {
        ui.row().height(32.0).build();
        ui.text(label).width(100.0).build();
        ui.toggle(self).build();
        ui.end();
    }
}

impl<'b> Inspectable<'b> for ColorF {
    fn inspect(&mut self, ui: &mut UIContext<'b>, label: &'b str) {
        ui.row().height(32.0).build();
        ui.text(label).width(100.0).build();
        // Placeholder for color picker interaction
        ui.r#box().width(20.0).height(20.0).bg(*self).radius(4.0).build();
        ui.end();
    }
}

/// Inspector widget builder
pub struct InspectorBuilder<'a, 'b> {
    ui: &'a mut UIContext<'b>,
}

impl<'a, 'b> InspectorBuilder<'a, 'b> {
    pub fn new(ui: &'a mut UIContext<'b>) -> Self {
        Self { ui }
    }

    pub fn field<T: Inspectable<'b>>(&mut self, label: &'b str, value: &mut T) -> &mut Self {
        value.inspect(self.ui, label);
        self
    }

    pub fn section(&mut self, title: &'b str) -> &mut Self {
        self.ui.row().height(24.0).bg(self.ui.theme.panel.darken(0.1)).build();
        self.ui.text(title).font_size(12.0).fg(self.ui.theme.text_dim).build();
        self.ui.end();
        self
    }
}

impl<'b> UIContext<'b> {
    pub fn inspector(&mut self) -> InspectorBuilder<'_, 'b> {
        self.column().padding(8.0).spacing(4.0).build();
        InspectorBuilder::new(self)
    }

    pub fn end_inspector(&mut self) {
        self.end();
    }
}
