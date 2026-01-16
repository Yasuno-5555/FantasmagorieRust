use crate::core::node::NodeStore;
use crate::core::types::{NodeID, LayoutDir};

pub struct LayoutEngine;

impl LayoutEngine {
    pub fn solve(store: &mut NodeStore, root: NodeID, width: f32, height: f32) {
        if root >= store.layout.len() { return; }

        // Stack: id, x, y, w, h
        let mut stack = vec![(root, 0.0, 0.0, width, height)];

        while let Some((id, x, y, w, h)) = stack.pop() {
            // Update this node
            if id >= store.layout.len() { continue; }
            store.layout[id].x = x;
            store.layout[id].y = y;
            store.layout[id].w = w;
            store.layout[id].h = h;

            // Get children (clone to avoid borrow)
            let children = if id < store.children.len() {
                store.children[id].clone()
            } else {
                continue;
            };

            if children.is_empty() { continue; }

            // Constraints
            let padding = if id < store.constraints.len() { store.constraints[id].padding } else { 0.0 };
            
            let content_x = x + padding;
            let content_y = y + padding;
            let content_w = (w - padding * 2.0).max(0.0);
            let content_h = (h - padding * 2.0).max(0.0);

            // Access LayoutDir
            let dir = store.layout[id].dir.unwrap_or(LayoutDir::Column); // Default column
            let is_row = dir == LayoutDir::Row;
            
            let main_axis = if is_row { content_w } else { content_h };
            let cross_axis = if is_row { content_h } else { content_w };

            // Phase 1: Measure
            let mut total_fixed = 0.0;
            let mut total_grow = 0.0;

            for &child_id in &children {
                if child_id >= store.constraints.len() { continue; }
                let c = &store.constraints[child_id];
                
                let fixed = if is_row { c.width } else { c.height };
                if fixed >= 0.0 {
                    total_fixed += fixed;
                }
                total_grow += c.grow;
            }

            // Phase 2: Distribute
            let remaining = (main_axis - total_fixed).max(0.0);
            let mut cursor = if is_row { content_x } else { content_y };

            // We iterate in REVERSE order if pushing to stack to process FIRST child first?
            // Stack is LIFO. 
            // If I push Child 1, then Child 2. Pop -> Child 2.
            // Layout order usually matters for Z-order rendering but for calculation it's independent.
            // BUT `cursor` update depends on order.
            // The recursions are independent.
            // So I can push in normal order ?
            // No, the loop calculates `child_x/y`.
            // So I calculate all children positions, THEN push them.
            // Order of pushing to stack determines order of processing.
            // It doesn't affect correctness of THIS node's children layout, only the order we visit grandchildren.
            // Since `cursor` is local, we calculate correct (x,y) for all children here.
            
            for &child_id in &children {
                if child_id >= store.constraints.len() { continue; }
                
                let c = &store.constraints[child_id];
                
                 // Main axis size
                let fixed = if is_row { c.width } else { c.height };
                let main_size = if fixed >= 0.0 {
                    fixed
                } else if total_grow > 0.0 && c.grow > 0.0 {
                    remaining * (c.grow / total_grow)
                } else {
                    0.0
                };

                 // Cross axis size
                let cross_fixed = if is_row { c.height } else { c.width };
                let cross_size = if cross_fixed >= 0.0 { cross_fixed } else { cross_axis };

                let (child_x, child_y, child_w, child_h) = if is_row {
                    let cx = cursor;
                    let cy = content_y;
                    cursor += main_size;
                    (cx, cy, main_size, cross_size)
                } else {
                    let cx = content_x;
                    let cy = cursor;
                    cursor += main_size;
                    (cx, cy, cross_size, main_size)
                };

                stack.push((child_id, child_x, child_y, child_w, child_h));
            }
        }
    }
}
