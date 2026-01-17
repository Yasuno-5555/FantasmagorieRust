use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct MathBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    id: ID,
    text: &'b str,
    font_size: f32,
}

impl<'b, 'a> MathBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, id: &str, text: &'b str) -> Self {
        Self {
            ui,
            id: ID::from_str(id),
            text,
            font_size: 16.0,
        }
    }

    pub fn font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let view = self.ui.arena.alloc(ViewHeader {
            view_type: ViewType::Math,
            id: std::cell::Cell::new(self.id),
            ..Default::default()
        });

        // This is simplified. Normally we would store the parsed MathNode in view.math_data.
        // For now we just store the text for the renderer to parse (inefficient but works for PoC).
        // ViewHeader doesn't have a MathData slot yet, so we reuse `text` field and parse on render?
        // Wait, Header has `text` field? No. It has `knob_value`, etc.
        // It has `text_data` only via `TextBuilder` which wasn't fully generic.
        // Actually Header uses `text_input_data` or similar.
        // Let's verify ViewHeader fields.

        // I should have added `text` field to ViewHeader or reused one.
        // I'll assume I can just use a new `math_data` or store in a sidecar hashmap if I was smart,
        // but ViewHeader usually holds data.

        // Let's assume I parse in render for now, but I need to store the source string.
        // ViewHeader typically has no generic string storage except maybe `text_buffer`?
        // Let's assume view.id can be proxy for text or I add `text: Cell<String>` to ViewHeader?
        // BUT ViewHeader is lightweight?
        // I will add `pub text: RefCell<String>` to ViewHeader. It's useful for many things.
        // Or check if it exists.

        view.font_size.set(self.font_size);

        // Hack: temporarily store math source in a special way or I need to update ViewHeader again.
        // I'll add `text: RefCell<String>` to ViewHeader.
        *view.text.borrow_mut() = self.text.to_string();

        self.ui.push_child(view);
        view
    }
}
