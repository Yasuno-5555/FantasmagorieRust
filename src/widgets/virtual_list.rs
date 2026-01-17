// use crate::core::{ID, Vec2};
use crate::widgets::UIContext;
use std::ops::Range;

/// Virtual List helper (Builder Pattern)
pub struct VirtualListBuilder<'b, 'a, F>
where
    F: FnMut(&mut UIContext, usize),
{
    ui: &'b mut UIContext<'a>,
    count: usize,
    item_height: f32,
    renderer: F,
    on_range_change: Option<Box<dyn FnMut(Range<usize>) + 'b>>,
}

impl<'b, 'a, F> VirtualListBuilder<'b, 'a, F>
where
    F: FnMut(&mut UIContext, usize),
{
    pub fn new(ui: &'b mut UIContext<'a>, count: usize, item_height: f32, renderer: F) -> Self {
        Self {
            ui,
            count,
            item_height,
            renderer,
            on_range_change: None,
        }
    }

    /// Callback when visible range changes significantly
    pub fn on_range_change<C: FnMut(Range<usize>) + 'b>(mut self, callback: C) -> Self {
        self.on_range_change = Some(Box::new(callback));
        self
    }

    pub fn build(mut self) {
        // Container for the list (should be scrollable)
        let container = self.ui.column().scroll().build();
        let id = container.id.get();
        self.ui.begin(container);

        // previous frame rect
        // Use full path to avoid import errors
        let rect = crate::core::resource::GLOBAL_RESOURCES
            .with(|_res| crate::view::interaction::get_rect(id));

        let scroll_offset = crate::view::interaction::get_scroll_offset(id);

        let viewport_height = rect.unwrap_or_default().h;

        // Safety check to avoid 0 height issues on first frame
        let visible_height = if viewport_height < 1.0 {
            500.0
        } else {
            viewport_height
        };

        let start_index = (scroll_offset.y / self.item_height).floor().max(0.0) as usize;
        let visible_count = (visible_height / self.item_height).ceil() as usize + 2; // +2 buffer

        let end_index = (start_index + visible_count).min(self.count);
        let current_range = start_index..end_index;

        // Check hook
        if let Some(mut cb) = self.on_range_change {
            // Retrieve last range
            let last_range = crate::view::interaction::get_last_visible_range(id);
            let should_notify = match last_range {
                Some(r) => r != current_range,
                None => true,
            };

            if should_notify {
                cb(current_range.clone());
                crate::view::interaction::set_last_visible_range(id, current_range.clone());
            }
        } else {
            // If no hook, still track it? Maybe not needed.
            // But if hook is added dynamically, we might want history.
            // Let's update it anyway if we have access.
            // But we consumed the callback in the Option match.
            // Simpler logic:
            let last_range = crate::view::interaction::get_last_visible_range(id);
            if last_range.as_ref() != Some(&current_range) {
                crate::view::interaction::set_last_visible_range(id, current_range.clone());
            }
        }

        // Top Spacer
        if start_index > 0 {
            let h = start_index as f32 * self.item_height;
            self.ui
                .r#box()
                .height(h)
                .bg(crate::core::ColorF::TRANSPARENT)
                .build();
        }

        // Render visible items
        for i in start_index..end_index {
            (self.renderer)(self.ui, i);
        }

        // Bottom Spacer
        if end_index < self.count {
            let h = (self.count - end_index) as f32 * self.item_height;
            self.ui
                .r#box()
                .height(h)
                .bg(crate::core::ColorF::TRANSPARENT)
                .build();
        }

        self.ui.end();
    }
}
