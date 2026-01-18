use crate::core::ID;
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct DropdownBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    view: &'a ViewHeader<'a>,
    selected_index: &'a mut usize,
    items: Vec<String>,
}

impl<'b, 'a> DropdownBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, items: Vec<String>, selected_index: &'a mut usize) -> Self {
        let id = ID::from_u64(ui.next_id());
        
        // 1. Check persistence for Open state
        let mut is_open = false;
        if let Some(pm) = &ui.persistence {
            if let Some(state) = pm.borrow().load::<bool>(id) {
                is_open = state;
            }
        }

        // 2. Create Header View
        let view = ui.arena.alloc(ViewHeader {
            view_type: ViewType::Dropdown,
            id: std::cell::Cell::new(id),
            ..Default::default()
        });

        // Default Style
        view.width.set(120.0);
        view.height.set(24.0);
        view.bg_color.set(ui.theme.panel.darken(0.05));
        view.fg_color.set(ui.theme.text);
        view.border_color.set(ui.theme.border);
        view.border_width.set(1.0);
        view.border_radius_tl.set(4.0);
        view.border_radius_tr.set(4.0);
        view.border_radius_br.set(4.0);
        view.border_radius_bl.set(4.0);
        view.is_squircle.set(false);

        // Store open state as value (1.0 = open)
        view.value.set(if is_open { 1.0 } else { 0.0 });
        
        // Set label to selected item
        if *selected_index < items.len() {
             let s = ui.arena.alloc_str(&items[*selected_index]);
             view.text.set(unsafe{ std::mem::transmute::<&str, &'static str>(s) });
        } else {
             view.text.set("Select...");
        }

        ui.push_child(view);
        
        // 3. If Open, create children (Items)
        // Note: In immediate mode, we usually return a builder that allows adding content OR we handle it here.
        // Since we passed `items`, we can handle it here.
        // HOWEVER, we need to handle clicks on children.
        // If we add children now, they will be part of the tree.
        
        if is_open {
            // We need to add children to `view`. `ui.push_child` added `view` to current stack top.
            // To add children to `view`, we must push `view` to parent stack.
             ui.parent_stack.push(view);
             
             for (i, item) in items.iter().enumerate() {
                  let item_id = ID::from_u64(ui.next_id());
                  let item_view = ui.arena.alloc(ViewHeader {
                      view_type: ViewType::Button, // Use Button for items
                      id: std::cell::Cell::new(item_id),
                      ..Default::default()
                  });
                  
                  // Item Style
                  item_view.width.set(0.0); // Auto fill?
                  item_view.height.set(24.0);
                  item_view.bg_color.set(ui.theme.panel.lighten(0.05));
                  item_view.fg_color.set(ui.theme.text);
                  item_view.text.set( unsafe{ std::mem::transmute::<&str, &'static str>(ui.arena.alloc_str(item)) } );
                  item_view.flex_grow.set(1.0);
                  
                  ui.push_child(item_view);
                  
                  // Check click
                  if crate::view::interaction::is_clicked(item_id) {
                      *selected_index = i;
                      is_open = false; // Close on selection
                  }
             }
             
             ui.parent_stack.pop();
        }

        // 4. Check interaction on Header (Toggle open/close)
        if crate::view::interaction::is_clicked(id) {
            is_open = !is_open;
        }
        
        // Save new state if changed (or just always save)
        if let Some(pm) = &ui.persistence {
            pm.borrow().save(id, &is_open);
        }
        
        // Update view value in case it changed this frame
        view.value.set(if is_open { 1.0 } else { 0.0 });

        Self {
            ui,
            view,
            selected_index,
            items,
        }
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        self.view
    }
}
