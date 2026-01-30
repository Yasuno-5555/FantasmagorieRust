use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct CheckboxBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    view: &'a ViewHeader<'a>,
    value: &'a mut bool,
    label: Option<&'a str>,
}

impl<'b, 'a> CheckboxBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, value: &'a mut bool) -> Self {
        let id = ID::from_u64(ui.next_id());
        let view = ui.arena.alloc(ViewHeader {
            view_type: ViewType::Checkbox,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Default Style
        view.width.set(20.0);
        view.height.set(20.0);
        view.bg_color.set(ui.theme.panel.darken(0.1));
        view.fg_color.set(ui.theme.accent);
        view.border_color.set(ui.theme.border);
        view.border_width.set(1.0);
        view.border_radius_tl.set(4.0);
        view.border_radius_tr.set(4.0);
        view.border_radius_br.set(4.0);
        view.border_radius_bl.set(4.0);
        view.is_squircle.set(false);

        // Store value
        view.value.set(if *value { 1.0 } else { 0.0 });

        ui.push_child(view);

        Self {
            ui,
            view,
            value,
            label: None,
        }
    }

    pub fn label(mut self, text: &'a str) -> Self {
        self.label = Some(text);
        self.view.text.set(text.into());
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        // Simple click interaction
        if crate::view::interaction::is_clicked(self.view.id.get()) {
            *self.value = !*self.value;
            self.view.value.set(if *self.value { 1.0 } else { 0.0 });
        }
        self.view
    }
}
