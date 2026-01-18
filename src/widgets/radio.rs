use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct RadioBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    view: &'a ViewHeader<'a>,
    selected_value: &'a mut i32, // Simplified: integer based selection for now
    my_value: i32,
    label: Option<&'a str>,
}

impl<'b, 'a> RadioBuilder<'b, 'a> {
    // We'll use i32 for selection group value. 
    // Usage: ui.radio("Label", current_value, my_id_value) -> returns new current_value
    pub fn new(ui: &'b mut UIContext<'a>, selected_value: &'a mut i32, my_value: i32) -> Self {
        let id = ID::from_u64(ui.next_id());
        let view = ui.arena.alloc(ViewHeader {
            view_type: ViewType::Radio,
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
        view.border_radius_tl.set(10.0); // Circular
        view.border_radius_tr.set(10.0);
        view.border_radius_br.set(10.0);
        view.border_radius_bl.set(10.0);
        view.is_squircle.set(false);

        // Store state: 1.0 if selected, 0.0 if not
        let is_selected = *selected_value == my_value;
        view.value.set(if is_selected { 1.0 } else { 0.0 });

        ui.push_child(view);

        Self {
            ui,
            view,
            selected_value,
            my_value,
            label: None,
        }
    }

    pub fn label(mut self, text: &'a str) -> Self {
        self.label = Some(text);
        self.view.text.set(text.into());
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        if crate::view::interaction::is_clicked(self.view.id.get()) {
            *self.selected_value = self.my_value;
            // self.view.value.set(1.0); // Will update next frame or immediately?
            // Since we updated *selected_value, next rebuild or frame will reflect it.
        }
        self.view
    }
}
