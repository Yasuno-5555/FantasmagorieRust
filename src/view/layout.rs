//! 2-pass Flexbox Layout Engine
//! Ported from layout.cpp
//!
//! Pass 1: Measure (Bottom-Up) - Children tell parent their size
//! Pass 2: Arrange (Top-Down) - Parent assigns positions to children

use super::header::{Align, Size, ViewHeader, ViewType};
use crate::core::Rectangle;

/// Public entry point for layout computation
pub fn compute_flex_layout(root: &ViewHeader, screen_w: f32, screen_h: f32, fm: &mut crate::text::FontManager) {
    measure_recursive(root, fm);
    arrange_recursive(root, 0.0, 0.0, screen_w, screen_h);
}

/// Pass 1: Measure (Bottom-Up)
/// Each node determines its "intrinsic" or "desired" size
fn measure_recursive(node: &ViewHeader, fm: &mut crate::text::FontManager) {
    let mut content_w: f32 = 0.0;
    let mut content_h: f32 = 0.0;

    let node_is_row = node.is_row.get();
    let node_padding = node.padding.get();

    // Measure children first (bottom-up)
    for child in node.children() {
        measure_recursive(child, fm);

        let measured = child.measured_size.get();
        let child_margin = child.margin.get();
        let child_w = measured.w + child_margin * 2.0;
        let child_h = measured.h + child_margin * 2.0;

        if node_is_row {
            // Row: sum widths, max height
            content_w += child_w;
            content_h = content_h.max(child_h);
        } else {
            // Column: max width, sum heights
            content_w = content_w.max(child_w);
            content_h += child_h;
        }
    }

    // Special handling for Grid Measure
    if node.view_type == ViewType::LayoutGrid {
        measure_grid(node);
        return;
    }

    // Add own padding
    content_w += node_padding * 2.0;
    content_h += node_padding * 2.0;

    // Special handling for Grid Measure
    if node.view_type == ViewType::LayoutGrid {
        measure_grid(node);
        return;
    }

    // Type-specific sizing
    match node.view_type {
        ViewType::Text => {
            if fm.fonts.is_empty() {
                fm.load_system_font();
            }
            let measured = fm.measure_text(node.text.get(), node.font_size.get());
            content_w = measured.x;
            content_h = measured.y;
        }
        ViewType::Button => {
            if fm.fonts.is_empty() {
                fm.load_system_font();
            }
            let measured = fm.measure_text(node.text.get(), node.font_size.get());
            // Button: add padding for label
            content_w = content_w.max(measured.x + 32.0);
            content_h = content_h.max(measured.y + 16.0);
        }
        ViewType::TextInput => {
            content_w = content_w.max(200.0);
            content_h = content_h.max(node.font_size.get() * 1.2 + 12.0);
        }
        ViewType::TextArea => {
            content_w = content_w.max(300.0);
            content_h = content_h.max(node.font_size.get() * 1.4 * 3.0 + 24.0);
        }
        ViewType::Toggle => {
            content_w = content_w.max(100.0);
            content_h = content_h.max(24.0);
        }
        ViewType::Slider => {
            content_w = content_w.max(150.0);
            content_h = content_h.max(30.0);
        }
        ViewType::Plot => {
            content_w = content_w.max(200.0);
            content_h = content_h.max(120.0);
        }
        ViewType::ColorPicker => {
            // 150 (sv) + 8 (gap) + 20 (hue bar)
            content_w = content_w.max(178.0);
            content_h = content_h.max(188.0);
        }
        ViewType::MenuBar => {
            content_h = content_h.max(26.0);
        }
        ViewType::MenuItem => {
            content_w = content_w.max(150.0);
            content_h = content_h.max(24.0);
        }
        ViewType::Socket => {
            content_w = 16.0;
            content_h = 16.0;
        }
        ViewType::Node => {
            content_w = content_w.max(120.0);
            content_h = content_h.max(40.0);
        }
        ViewType::Canvas => {
            // Usually fills parent, but let's give it a min size
            content_w = content_w.max(400.0);
            content_h = content_h.max(300.0);
        }
        _ => {}
    }

    // Store content size
    node.content_size.set(Size::new(content_w, content_h));

    // Apply explicit size constraints
    let width = node.width.get();
    let height = node.height.get();
    let final_w = if width > 0.0 { width } else { content_w };
    let final_h = if height > 0.0 { height } else { content_h };
    node.measured_size.set(Size::new(final_w, final_h));
}

/// Pass 2: Arrange (Top-Down)
/// Parent assigns final rect to each child based on flex_grow and layout direction
fn arrange_recursive(node: &ViewHeader, x: f32, y: f32, avail_w: f32, avail_h: f32) {
    // Set own computed rect
    node.computed_rect
        .set(Rectangle::new(x, y, avail_w, avail_h));

    // No children? Done.
    let _first = match node.first_child.get() {
        Some(c) => c,
        None => return,
    };

    let padding = node.padding.get();

    // Calculate inner area (after padding)
    let inner_x = x + padding;
    let inner_y = y + padding;
    let inner_w = avail_w - padding * 2.0;
    let inner_h = avail_h - padding * 2.0;

    // Wrap mode
    if node.wrap.get() {
        arrange_wrap(node, inner_x, inner_y, inner_w, inner_h);
        return;
    }

    // Splitter special handling
    if node.view_type == ViewType::Splitter {
        arrange_splitter(node, inner_x, inner_y, inner_w, inner_h);
        return;
    }

    // Canvas special handling: Absolute children
    if node.view_type == ViewType::Canvas {
        for child in node.children() {
            let cx = inner_x + child.pos_x.get();
            let cy = inner_y + child.pos_y.get();
            let cw = child.measured_size.get().w;
            let ch = child.measured_size.get().h;
            arrange_recursive(child, cx, cy, cw, ch);
        }
        return;
    }

    // Standard flexbox arrangement
    arrange_flex(node, inner_x, inner_y, inner_w, inner_h);
}

/// Arrange children with wrapping
fn arrange_wrap(node: &ViewHeader, inner_x: f32, inner_y: f32, inner_w: f32, inner_h: f32) {
    let is_row = node.is_row.get();
    let main_avail = if is_row { inner_w } else { inner_h };
    let mut line_cursor: f32 = 0.0;
    let mut cross_cursor: f32 = 0.0;
    let mut line_max_cross: f32 = 0.0;

    for child in node.children() {
        let measured = child.measured_size.get();
        let margin = child.margin.get();
        let c_main = if is_row { measured.w } else { measured.h } + margin * 2.0;
        let c_cross = if is_row { measured.h } else { measured.w } + margin * 2.0;

        // Wrap to next line?
        if line_cursor + c_main > main_avail && line_cursor > 0.0 {
            cross_cursor += line_max_cross;
            line_cursor = 0.0;
            line_max_cross = 0.0;
        }

        if is_row {
            arrange_recursive(
                child,
                inner_x + line_cursor + margin,
                inner_y + cross_cursor + margin,
                measured.w,
                measured.h,
            );
        } else {
            arrange_recursive(
                child,
                inner_x + cross_cursor + margin,
                inner_y + line_cursor + margin,
                measured.w,
                measured.h,
            );
        }

        line_cursor += c_main;
        line_max_cross = line_max_cross.max(c_cross);
    }
}

/// Arrange splitter children (exactly 2)
fn arrange_splitter(node: &ViewHeader, inner_x: f32, inner_y: f32, inner_w: f32, inner_h: f32) {
    // Get split ratio and orientation from node
    // Note: Ratio/Vertical are primitive values (not Cells) in some versions, but we changed them to Cell
    let ratio = node.ratio.get().clamp(0.0, 1.0);
    let handle = 8.0f32;

    let mut children = node.children();
    let child1 = match children.next() {
        Some(c) => c,
        None => return,
    };

    let child2_opt = children.next();

    let margin1 = child1.margin.get();

    if let Some(child2) = child2_opt {
        // We have 2 children
        let is_vertical = node.is_vertical.get();
        let margin2 = child2.margin.get();

        let avail = if is_vertical { inner_h } else { inner_w };
        let size1 = (avail - handle) * ratio;
        let size2 = avail - handle - size1;

        if is_vertical {
            arrange_recursive(
                child1,
                inner_x + margin1,
                inner_y + margin1,
                inner_w - margin1 * 2.0,
                size1 - margin1 * 2.0,
            );
            arrange_recursive(
                child2,
                inner_x + margin2,
                inner_y + size1 + handle + margin2,
                inner_w - margin2 * 2.0,
                size2 - margin2 * 2.0,
            );
        } else {
            arrange_recursive(
                child1,
                inner_x + margin1,
                inner_y + margin1,
                size1 - margin1 * 2.0,
                inner_h - margin1 * 2.0,
            );
            arrange_recursive(
                child2,
                inner_x + size1 + handle + margin2,
                inner_y + margin2,
                size2 - margin2 * 2.0,
                inner_h - margin2 * 2.0,
            );
        }
    } else if node.view_type == ViewType::LayoutGrid {
        arrange_grid(node, inner_x, inner_y, inner_w, inner_h);
    } else {
        // Just one child, fill
        arrange_recursive(
            child1,
            inner_x + margin1,
            inner_y + margin1,
            inner_w - margin1 * 2.0,
            inner_h - margin1 * 2.0,
        );
    }
}

/// Standard flexbox arrangement
fn arrange_flex(node: &ViewHeader, inner_x: f32, inner_y: f32, inner_w: f32, inner_h: f32) {
    let is_row = node.is_row.get();

    // Sum fixed sizes and flex totals
    let mut total_fixed: f32 = 0.0;
    let mut total_flex_grow: f32 = 0.0;
    let mut total_flex_shrink: f32 = 0.0;

    for child in node.children() {
        let measured = child.measured_size.get();
        let margin = child.margin.get();
        let c_main = if is_row { measured.w } else { measured.h } + margin * 2.0;
        total_fixed += c_main;
        total_flex_grow += child.flex_grow.get();
        total_flex_shrink += child.flex_shrink.get();
    }

    // Calculate remaining space
    let main_avail = if is_row { inner_w } else { inner_h };
    let remaining = main_avail - total_fixed;

    // Arrange children along main axis
    let mut cursor: f32 = 0.0;

    for child in node.children() {
        let measured = child.measured_size.get();
        let margin = child.margin.get();
        let c_measured = if is_row { measured.w } else { measured.h };
        let grow = child.flex_grow.get();
        let shrink = child.flex_shrink.get();

        let c_main = if remaining >= 0.0 && grow > 0.0 && total_flex_grow > 0.0 {
            // Grow: distribute extra space
            c_measured + (grow / total_flex_grow) * remaining
        } else if remaining < 0.0 && shrink > 0.0 && total_flex_shrink > 0.0 {
            // Shrink: reduce size proportionally
            let shrink_amount = (-remaining) * (shrink / total_flex_shrink);
            (c_measured - shrink_amount).max(0.0)
        } else {
            c_measured
        };

        // Cross axis size (with alignment)
        let cross_avail = if is_row {
            inner_h - margin * 2.0
        } else {
            inner_w - margin * 2.0
        };

        let c_measured_cross = if is_row { measured.h } else { measured.w };

        // Explicit size takes precedence
        let explicit_w = child.width.get();
        let explicit_h = child.height.get();
        let explicit_cross = if is_row { explicit_h } else { explicit_w };

        let child_align = child.align.get();
        let c_cross = if explicit_cross > 0.0 {
            explicit_cross
        } else if child_align == Align::Stretch {
            cross_avail
        } else {
            c_measured_cross
        };

        // Cross-axis offset based on alignment
        let cross_offset = match child_align {
            Align::Center => (cross_avail - c_cross) * 0.5,
            Align::End => cross_avail - c_cross,
            _ => 0.0,
        };

        if is_row {
            let c_x = inner_x + cursor + margin;
            let c_y = inner_y + margin + cross_offset;
            arrange_recursive(child, c_x, c_y, c_main, c_cross);
            cursor += c_main + margin * 2.0;
        } else {
            let c_x = inner_x + margin + cross_offset;
            let c_y = inner_y + cursor + margin;
            arrange_recursive(child, c_x, c_y, c_cross, c_main);
            cursor += c_main + margin * 2.0;
        }
    }
}

fn measure_grid(node: &ViewHeader) {
    let cols = node.value.get() as usize;
    if cols == 0 { return; } 
    
    let gap = node.margin.get();
    
    let mut max_child_w: f32 = 0.0;
    let mut row_h: f32 = 0.0;
    let mut total_h: f32 = 0.0;
    
    let children: Vec<&ViewHeader> = node.children().collect();
    
    for (i, child) in children.iter().enumerate() {
        let size = child.measured_size.get();
        let margin = child.margin.get();
        let w = size.w + margin * 2.0;
        let h = size.h + margin * 2.0;
        
        max_child_w = max_child_w.max(w);
        row_h = row_h.max(h);
        
        if (i + 1) % cols == 0 || i == children.len() - 1 {
            total_h += row_h + gap;
            row_h = 0.0;
        }
    }
    if total_h > 0.0 { total_h -= gap; }
    
    let total_w = max_child_w * (cols as f32) + (gap * (cols as f32 - 1.0));
    
    let padding = node.padding.get();
    node.measured_size.set(Size::new(
        total_w + padding * 2.0,
        total_h + padding * 2.0
    ));
}

fn arrange_grid(node: &ViewHeader, x: f32, y: f32, w: f32, _h: f32) {
    let padding = node.padding.get();
    let gap = node.margin.get();
    let cols = node.value.get() as usize;
    if cols == 0 { return; }

    let start_x = x + padding;
    let start_y = y + padding;
    let avail_w = w - padding * 2.0;
    
    let cell_w = (avail_w - ((cols as f32 - 1.0).max(0.0) * gap)) / (cols as f32);
    
    let children: Vec<&ViewHeader> = node.children().collect();
    let mut current_row_start_idx = 0;
    let mut current_y = start_y;
    
    while current_row_start_idx < children.len() {
        let end_idx = (current_row_start_idx + cols).min(children.len());
        let row_slice = &children[current_row_start_idx..end_idx];
        
        let mut row_max_h: f32 = 0.0;
        for child in row_slice {
            let size = child.measured_size.get();
            let margin = child.margin.get();
            row_max_h = row_max_h.max(size.h + margin * 2.0);
        }
        
        for (col_idx, child) in row_slice.iter().enumerate() {
            let child_x = start_x + (col_idx as f32) * (cell_w + gap);
            
            // Layout child
            arrange_recursive(child, child_x, current_y, cell_w, row_max_h);
        }
        
        current_y += row_max_h + gap;
        current_row_start_idx += cols;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameArena, ID};
    use std::cell::Cell;

    #[test]
    fn test_simple_column_layout() {
        let arena = FrameArena::new();

        let root = arena.alloc(ViewHeader {
            id: Cell::new(ID::from_str("root")),
            // Default ViewHeader fields are Cells, created via Default::default
            ..Default::default()
        });
        root.height.set(200.0);
        root.width.set(100.0);

        let child1 = arena.alloc(ViewHeader {
            id: Cell::new(ID::from_str("c1")),
            ..Default::default()
        });
        child1.height.set(50.0);

        let child2 = arena.alloc(ViewHeader {
            id: Cell::new(ID::from_str("c2")),
            ..Default::default()
        });
        child2.height.set(50.0);

        root.add_child(child1);
        root.add_child(child2);

        let mut fm = crate::text::FontManager::new();
        compute_flex_layout(root, 100.0, 200.0, &mut fm);

        let r1 = child1.computed_rect.get();
        let r2 = child2.computed_rect.get();

        assert_eq!(r1.y, 0.0);
        assert_eq!(r2.y, 50.0);
    }
}
