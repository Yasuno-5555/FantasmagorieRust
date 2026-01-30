use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct ToggleBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    view: &'a ViewHeader<'a>,
    value: &'a mut bool,
    label: Option<&'a str>,
}

impl<'b, 'a> ToggleBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, value: &'a mut bool) -> Self {
        let id = ID::from_u64(ui.next_id());
        let view = ui.arena.alloc(ViewHeader {
            view_type: ViewType::Toggle,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Default Style
        view.width.set(40.0);
        view.height.set(20.0);
        view.bg_color.set(ui.theme.panel.darken(0.1));
        view.fg_color.set(ui.theme.accent);
        view.border_radius_tl.set(10.0);
        view.border_radius_tr.set(10.0);
        view.border_radius_br.set(10.0);
        view.border_radius_bl.set(10.0);
        view.is_squircle.set(true);

        // Store initial value
        // We use view.value (f32) to store boolean state (0.0 or 1.0) for the renderer
        view.value.set(if *value { 1.0 } else { 0.0 });
        view.min.set(0.0);
        view.max.set(1.0);

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
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        // Handle interaction
        if crate::view::interaction::is_clicked(self.view.id.get()) {
            *self.value = !*self.value;
            self.view.value.set(if *self.value { 1.0 } else { 0.0 });
        }
        
        // If label exists, we might want to wrap this in a Row with Text?
        // Or just let the user handle layout. 
        // For simplicity, let's just properly set the view properties.
        
        self.view
    }
}
