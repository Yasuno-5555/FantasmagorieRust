use fanta_rust::core::FrameArena;
use fanta_rust::core::{Rectangle, Vec2, ID};
use fanta_rust::widgets::UIContext;

#[test]
fn test_virtual_list_rendering() {
    let arena = FrameArena::new();
    let mut ui = UIContext::new(&arena);

    // 1. Initial Setup
    // We need to simulate the ID that VirtualList will use.
    // UIContext starts next_id at 1.
    // ui.column().scroll() calls:
    //   column -> ID(1), next_id=2
    //   scroll -> modifies view 1
    // So the container ID will be 1.
    let container_id = ID::from_u64(1);

    // Mock interaction state using public API
    // Viewport height: 500.0
    // Item height: 20.0
    // Capacity at 0 scroll: 500/20 = 25 items (+ buffer)

    fanta_rust::view::interaction::update_rect(
        container_id,
        Rectangle {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 500.0,
        },
    );

    fanta_rust::view::interaction::set_scroll_offset(container_id, Vec2::new(0.0, 0.0));

    let count = 1000;

    // Render
    ui.virtual_list(count, 20.0, |ui, index| {
        let text = ui.arena.alloc_str(&format!("Item {}", index));
        let text = ui.arena.alloc_str(&format!("Item {}", index));
        ui.text(text).build();
    })
    .build();

    // Verify children count.
    // Container is child 1 (root's first child is None? No, we didn't use `begin()`, we pushed child).
    // UIContext pushes to root stack.
    // `virtual_list` creates a container.
    // container has children: Spacer, Items..., Spacer.

    // Since we didn't call `ui.begin()`, the container is just pushed to the context's list (if it has a parent).
    // But UIContext default has no root parent initially unless `window` or `begin`.
    // Wait, `UIContext::new` has `root: None`.
    // `push_child` adds to `root` or `parent_stack.last()`.
    // If no parent, it might panic or do nothing?
    // Let's check `push_child` impl.
    // If it requires a parent, we need a root.

    // `UIContext::push_child` logic:
    // if let Some(parent) = self.parent_stack.last() { ... }
    // else if let Some(root) = self.root { ... }
    // else { self.root = Some(view); }

    // So first child becomes root. Good.

    let root = ui.root().expect("Root should exist");
    assert_eq!(root.id.get(), container_id);

    let children: Vec<_> = root.children().collect();

    // Count should be roughly visible_count + 1 or 2 spacers.
    // at scroll 0:
    // Top spacer: height 0 -> might be skipped? Logic says `if start_index > 0`.
    // Bottom spacer: height (1000 - end) * 20.
    // Items: 0 to ~27 (25 + 2 buffer).

    // So children count: 27 items + 1 bottom spacer = 28.
    // Or close to it.

    println!("Children count: {}", children.len());
    // Assert strictly
    assert!(
        children.len() < 50,
        "Should handle 1000 items by virtualizing (got {})",
        children.len()
    );
    assert!(children.len() > 20, "Should render visible items");

    // Check first item text
    // The children include spacers which are Boxes. Items are Text.
    // First child at index 0 is Item 0 (no top spacer).
    // Let's verify text content requires downcasting or checking text field.
    // ViewHeader has `text` cell.
    assert_eq!(children[0].text.get(), "Item 0");
}

#[test]
fn test_virtual_list_scrolled() {
    let arena = FrameArena::new();
    let mut ui = UIContext::new(&arena);
    let container_id = ID::from_u64(1);

    fanta_rust::view::interaction::update_rect(
        container_id,
        Rectangle {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 500.0,
        },
    );

    // Scroll down 2000px (100 items * 20px)
    fanta_rust::view::interaction::set_scroll_offset(container_id, Vec2::new(0.0, 2000.0));

    let count = 1000;
    ui.virtual_list(count, 20.0, |ui, index| {
        let text = ui.arena.alloc_str(&format!("Item {}", index));
        let text = ui.arena.alloc_str(&format!("Item {}", index));
        ui.text(text).build();
    })
    .build();

    let root = ui.root().expect("Root should exist");
    let children: Vec<_> = root.children().collect();

    // Should have Top Spacer, Items 100..127, Bottom Spacer.
    println!("Children count with scroll: {}", children.len());
    assert!(children.len() < 50);

    // First child should be Top Spacer (Box with height 2000)
    // Check height
    let first = children[0];
    assert_eq!(first.view_type, fanta_rust::view::header::ViewType::Box);
    assert_eq!(first.height.get(), 2000.0);

    // Second child should be Item 100
    let second = children[1];
    assert_eq!(second.text.get(), "Item 100");
}

#[test]
fn test_virtual_list_hooks() {
    let arena = FrameArena::new();
    let mut ui = UIContext::new(&arena);
    let container_id = ID::from_u64(1);

    fanta_rust::view::interaction::update_rect(
        container_id,
        Rectangle {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 500.0,
        },
    );

    // Initial state: no range stored

    let count = 1000;

    // We need to capture the range that was passed to the callback.
    // std::cell::Cell allows interior mutability in closure
    let range_captured = std::cell::RefCell::new(None);

    ui.virtual_list(count, 20.0, |ui, index| {
        let text = ui.arena.alloc_str(&format!("Item {}", index));
        ui.text(text).build();
    })
    .on_range_change(|range| {
        *range_captured.borrow_mut() = Some(range);
    })
    .build();

    // First frame: should trigger hook with visible range (0..27 ish)
    {
        let r = range_captured.borrow();
        assert!(r.is_some(), "Hook should be called on first frame");
        let range = r.as_ref().unwrap();
        assert_eq!(range.start, 0);
        assert!(range.end > 20);
    }

    // Second frame: same scroll, same range. Should NOT trigger hook.
    *range_captured.borrow_mut() = None;

    // In unit test, we reuse context but we need to reset arena/hierarchy if simulating frames.
    // UIContext doesn't have explicit reset for frame, usually we make a new one.
    // But persistence is static thread_local in interaction.rs. So it persists!

    let arena2 = FrameArena::new();
    let mut ui2 = UIContext::new(&arena2);

    ui2.virtual_list(count, 20.0, |ui, index| {
        let text = ui.arena.alloc_str(&format!("Item {}", index));
        ui.text(text).build();
    })
    .on_range_change(|range| {
        *range_captured.borrow_mut() = Some(range);
    })
    .build();

    {
        let r = range_captured.borrow();
        assert!(
            r.is_none(),
            "Hook should NOT be called if range hasn't changed"
        );
    }

    // Third frame: Scroll down.
    fanta_rust::view::interaction::set_scroll_offset(container_id, Vec2::new(0.0, 100.0)); // 5 items down

    let arena3 = FrameArena::new();
    let mut ui3 = UIContext::new(&arena3);

    ui3.virtual_list(count, 20.0, |ui, index| {
        let text = ui.arena.alloc_str(&format!("Item {}", index));
        ui.text(text).build();
    })
    .on_range_change(|range| {
        *range_captured.borrow_mut() = Some(range);
    })
    .build();

    {
        let r = range_captured.borrow();
        assert!(r.is_some(), "Hook SHOULD be called after scroll");
        let range = r.as_ref().unwrap();
        // 100 / 20 = 5. Start index should be 5.
        assert_eq!(range.start, 5);
    }
}
