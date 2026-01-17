//! View renderer - converts View AST to DrawList commands
//! Ported from renderer.cpp

use super::header::{ViewHeader, ViewType};
use super::interaction;
use super::layout::compute_flex_layout;
use crate::core::{ColorF, Vec2};
use crate::draw::DrawList;
// use crate::widgets::{WindowRoot, UIContext};
// use crate::widgets::{WindowRoot, UIContext};
use crate::text::FontManager;

/// Render the UI tree to a DrawList
pub fn render_ui(
    root: &ViewHeader,
    width: f32,
    height: f32,
    dl: &mut DrawList,
    fm: &mut FontManager,
) {
    // Run interaction pass
    interaction::begin_interaction_pass();

    // Compute layout
    layout_ui(root, width, height);

    // Render tree
    render_view_recursive(root, dl, 0, fm);
}

fn layout_ui(root: &ViewHeader, width: f32, height: f32) {
    crate::view::layout::compute_flex_layout(root, width, height);
}

/// Recursive view renderer
fn render_view_recursive(view: &ViewHeader, dl: &mut DrawList, depth: i32, fm: &mut FontManager) {
    let rect = view.computed_rect.get();

    // 1. Universal Background rendering (Shadow, Blur, BG)
    let radii = [
        view.border_radius_tl.get(),
        view.border_radius_tr.get(),
        view.border_radius_br.get(),
        view.border_radius_bl.get(),
    ];

    if view.backdrop_blur.get() > 0.0 {
        dl.add_blur_rect_ex(
            Vec2::new(rect.x, rect.y),
            Vec2::new(rect.w, rect.h),
            radii,
            view.backdrop_blur.get(),
        );
    }

    let bg_color = view.bg_color.get();
    let elevation = view.elevation.get();
    let border_width = view.border_width.get();

    if bg_color.a > 0.0 || elevation > 0.0 || border_width > 0.0 {
        dl.add_rect_ex(
            Vec2::new(rect.x, rect.y),
            Vec2::new(rect.w, rect.h),
            radii,
            bg_color,
            elevation,
            view.is_squircle.get(),
            border_width,
            view.border_color.get(),
            Vec2::new(view.wobble_x.get(), view.wobble_y.get()),
            view.glow_strength.get(),
            view.glow_color.get(),
        );
    }

    // 2. Register for interaction
    let id = view.id.get();
    interaction::register_interactive(id, rect);

    // 3. Type-specific rendering
    match view.view_type {
        ViewType::Box => {
            render_label_and_icon_at(Vec2::new(rect.x, rect.y), view, dl, false);
        }
        ViewType::Text => {
            render_label_and_icon_at(Vec2::new(rect.x, rect.y), view, dl, false);
        }
        ViewType::Button => {
            render_button(view, dl);
        }
        ViewType::Toggle => {
            render_toggle(view, dl);
        }
        ViewType::Slider => {
            render_slider(view, dl);
        }
        ViewType::Scroll => {
            render_scroll(view, dl, depth, fm);
            return; // Scroll handles its own children
        }
        ViewType::TextInput => {
            render_text_input(view, dl);
        }
        ViewType::Bezier => {
            let points = view.points.get();
            dl.add_bezier(
                points[0],
                points[1],
                points[2],
                points[3],
                view.thickness.get(),
                view.fg_color.get(),
            );
        }
        ViewType::Image => {
            if let Some(tex_id) = view.texture_id.get() {
                dl.add_image_ex(
                    Vec2::new(rect.x, rect.y),
                    Vec2::new(rect.w, rect.h),
                    tex_id,
                    [0.0, 0.0, 1.0, 1.0], // Full UV
                    view.fg_color.get(),  // Use fg_color for tint
                    radii,
                );
            }
        }
        ViewType::Splitter => {
            render_splitter(view, dl);
        }
        ViewType::ColorPicker => {
            render_color_picker(view, dl);
        }
        ViewType::Markdown => {
            render_markdown(view, dl);
        }
        ViewType::Path => {
            if let Some(path) = view.path.get() {
                dl.add_path(path, view.fg_color.get(), view.thickness.get());
            }
        }
        ViewType::Knob => {
            render_knob(view, dl);
        }
        ViewType::Fader => {
            render_fader(view, dl);
        }
        ViewType::ValueDragger => {
            render_value_dragger(view, dl);
        }
        ViewType::Plot => render_plot(view, dl),
        ViewType::Scene3D => {
            if let Some(scene_data) = view.scene_data.get() {
                dl.add_viewport_3d(
                    Vec2::new(rect.x, rect.y),
                    Vec2::new(rect.w, rect.h),
                    scene_data.texture_id.get(),
                    scene_data.camera_pos.get(),
                    scene_data.camera_target.get(),
                    scene_data.fov.get(),
                    scene_data.lut_id.get(),
                    scene_data.lut_intensity.get(),
                );
            }
        }
        ViewType::Canvas => {
            render_canvas(view, dl, depth, fm);
            return; // Canvas handles its own recursion for transforms
        }
        ViewType::Node => render_node(view, dl, depth, fm),
        ViewType::Socket => render_socket(view, dl),
        ViewType::Wire => render_wire(view, dl),
        ViewType::ContextMenu => render_context_menu(view, dl, depth, fm),
        ViewType::MenuItem => render_menu_item(view, dl),
        ViewType::Collapsible => {
            render_collapsible(view, dl, depth, fm);
            return; // Collapsible handles its own child recursion
        }
        ViewType::Toast => render_toast(view, dl),
        ViewType::Tooltip => render_tooltip(view, dl),
        ViewType::CurveEditor => render_curve_editor(view, dl),
        ViewType::Ruler => render_ruler(view, dl),
        ViewType::Grid => render_grid(view, dl),
        ViewType::TransformGizmo => render_transform_gizmo(view, dl),
        ViewType::Math => render_math(view, dl, fm),
        _ => {}
    }

    // 4. Default child recursion
    if view.clip.get() {
        dl.push_clip(Vec2::new(rect.x, rect.y), Vec2::new(rect.w, rect.h));
    }

    for child in view.children() {
        render_view_recursive(child, dl, depth + 1, fm);
    }

    if view.clip.get() {
        dl.pop_clip();
    }
}

/// Render button with hover/active states
/// Render button with hover/active states
fn render_button(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let is_hot = interaction::is_hot(view.id.get());
    let is_active = interaction::is_active(view.id.get());

    // Determine color based on state
    let mut bg = view.bg_color.get();
    if is_active {
        bg = view
            .bg_active
            .get()
            .unwrap_or_else(|| ColorF::new(bg.r * 0.7, bg.g * 0.7, bg.b * 0.8, bg.a));
    } else if is_hot {
        bg = view
            .bg_hover
            .get()
            .unwrap_or_else(|| ColorF::new(bg.r * 1.2, bg.g * 1.2, bg.b * 1.3, bg.a));
    }

    let elevation = if is_active { 0.0 } else { view.elevation.get() };

    dl.add_rect_ex(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, rect.h),
        [
            view.border_radius_tl.get(),
            view.border_radius_tr.get(),
            view.border_radius_br.get(),
            view.border_radius_bl.get(),
        ],
        bg,
        elevation,
        view.is_squircle.get(),
        view.border_width.get(),
        view.border_color.get(),
        Vec2::ZERO,
        view.glow_strength.get(),
        view.glow_color.get(),
    );

    // Render Label and Icon (Centered)
    render_label_and_icon_at(Vec2::ZERO, view, dl, true);
}

fn render_label_and_icon_at(pos: Vec2, view: &ViewHeader, dl: &mut DrawList, centered: bool) {
    let icon = view.icon.get();
    let text = view.text.get();
    let has_icon = !icon.is_empty();
    let has_text = !text.is_empty();

    if has_icon || has_text {
        let i_size = if view.icon_size.get() > 0.0 {
            view.icon_size.get()
        } else {
            view.font_size.get()
        };
        let gap = 8.0;

        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            let icon_sz = if has_icon {
                fm.measure_text(icon, i_size)
            } else {
                Vec2::ZERO
            };
            let text_sz = if has_text {
                fm.measure_text(text, view.font_size.get())
            } else {
                Vec2::ZERO
            };

            let total_w = icon_sz.x + (if has_icon && has_text { gap } else { 0.0 }) + text_sz.x;

            let start_x = if centered {
                let rect = view.computed_rect.get();
                rect.x + (rect.w - total_w) * 0.5
            } else {
                pos.x
            };

            let start_y = if centered {
                let rect = view.computed_rect.get();
                // Visual Centering based on Cap Height approximation (Ascent + Descent)
                if let Some((ascent, descent, _gap)) = fm.vertical_metrics(view.font_size.get()) {
                    // Center roughly around middle of cap height:
                    // Text Height = ascent - descent (since descent is negative usually?)
                    // If descent is negative, then total height is ascent + abs(descent).
                    // Visual middle is baseline - (ascent + descent)/2
                    // Rect Middle is y + h/2.
                    // baseline - middle = y + h/2
                    // baseline = y + h/2 + (ascent + descent)/2 - desc_offset?
                    // Simpler:
                    // height of text body ~= ascent + descent (e.g. 10 + (-3) = 7)
                    // We want to center that body.
                    // top = baseline - ascent
                    // bottom = baseline - descent
                    // mid = (top + bottom) / 2 = baseline - (ascent + descent)/2
                    // We want mid = rect.cy
                    // baseline = rect.cy + (ascent + descent)/2

                    rect.y + rect.h * 0.5 + (ascent + descent) * 0.5 - ascent // Wait, render_text_at expects pos.y to be top-left?
                                                                              // No, render_text_at uses `baseline = pos.y + ascent`
                                                                              // So pos.y IS top-left used for calculation.
                                                                              // If we pass `start_y`, render_text will do `baseline = start_y + ascent`.
                                                                              // So we want `start_y + ascent = rect.cy + (ascent + descent)/2`
                                                                              // start_y = rect.cy + (ascent + descent)/2 - ascent
                                                                              //         = rect.cy + ascent/2 + descent/2 - ascent
                                                                              //         = rect.cy + descent/2 - ascent/2
                                                                              //         = rect.cy - (ascent - descent)/2
                } else {
                    // Fallback
                    let total_h = icon_sz.y.max(text_sz.y);
                    rect.y + (rect.h - total_h) * 0.5
                }
            } else {
                pos.y
            };

            let mut cur_x = start_x;

            if has_icon {
                // Render icon (Font index 1 if available, else 0)
                let f_idx = if fm.fonts.len() > 1 { 1 } else { 0 };
                render_text_at_special(
                    &mut *fm,
                    Vec2::new(cur_x, start_y),
                    icon,
                    i_size,
                    view.fg_color.get(),
                    f_idx,
                    dl,
                );
                cur_x += icon_sz.x + gap;
            }

            if has_text {
                render_text_at_special(
                    &mut *fm,
                    Vec2::new(cur_x, start_y),
                    text,
                    view.font_size.get(),
                    view.fg_color.get(),
                    0,
                    dl,
                );
            }
        });
    }
}

fn render_text_at_special(
    fm: &mut FontManager,
    pos: Vec2,
    text: &str,
    size: f32,
    color: ColorF,
    font_idx: usize,
    dl: &mut DrawList,
) {
    if text.is_empty() {
        return;
    }

    if fm.fonts.is_empty() {
        fm.init_fonts();
    }

    let f_idx = if font_idx < fm.fonts.len() {
        font_idx
    } else {
        0
    };

    // Use actual ascent if available
    let ascent = if let Some((a, _, _)) = fm.vertical_metrics(size) {
        a
    } else {
        size * 0.8 // Approximate ascent
    };

    let mut x = pos.x;
    let baseline = pos.y + ascent;

    for c in text.chars() {
        if let Some(glyph) = fm.get_glyph(f_idx, c, size) {
            let gx = x + glyph.bearing.x;
            // Correct Y-down calculation: Top = Baseline - (Ymin + Height)
            // glyph.bearing.y is ymin.
            let gy = baseline - (glyph.bearing.y + glyph.size.y);
            let w = glyph.size.x;
            let h = glyph.size.y;

            dl.add_text(
                Vec2::new(gx, gy),
                Vec2::new(w, h),
                [
                    glyph.uv.x,
                    glyph.uv.y,
                    glyph.uv.x + glyph.uv.w,
                    glyph.uv.y + glyph.uv.h,
                ],
                color,
            );
            x += glyph.advance;
        }
    }
}

fn render_text_at(
    fm: &mut FontManager,
    pos: Vec2,
    text: &str,
    size: f32,
    color: ColorF,
    dl: &mut DrawList,
) {
    if text.is_empty() {
        return;
    }

    if fm.fonts.is_empty() {
        fm.init_fonts();
    }

    // Use actual ascent if available
    let ascent = if let Some((a, _, _)) = fm.vertical_metrics(size) {
        a
    } else {
        size * 0.8 // Approximate ascent
    };
    let mut x = pos.x;
    let baseline = pos.y + ascent;

    for c in text.chars() {
        if let Some(glyph) = fm.get_glyph(0, c, size) {
            let gx = x + glyph.bearing.x;
            // Correct Y-down calculation: Top = Baseline - (Ymin + Height)
            let gy = baseline - (glyph.bearing.y + glyph.size.y);
            let w = glyph.size.x;
            let h = glyph.size.y;

            dl.add_text(
                Vec2::new(gx, gy),
                Vec2::new(w, h),
                [
                    glyph.uv.x,
                    glyph.uv.y,
                    glyph.uv.x + glyph.uv.w,
                    glyph.uv.y + glyph.uv.h,
                ],
                color,
            );
            x += glyph.advance;
        }
    }
}

fn render_markdown(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let text = view.text.get();
    if text.is_empty() {
        return;
    }

    use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

    let parser = Parser::new(text);
    let mut x = rect.x;
    let mut y = rect.y;
    let mut current_font_size = view.font_size.get();
    let _is_bold = false;
    let _is_italic = false;

    // Default font size if not set
    if current_font_size == 0.0 {
        current_font_size = 14.0;
    }

    let line_height = current_font_size * 1.5;

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                let scale = match level {
                    HeadingLevel::H1 => 2.0,
                    HeadingLevel::H2 => 1.5,
                    HeadingLevel::H3 => 1.2,
                    _ => 1.1,
                };
                current_font_size = view.font_size.get() * scale;
            }
            Event::End(TagEnd::Heading(..)) => {
                current_font_size = view.font_size.get();
                y += line_height * 1.2;
                x = rect.x;
            }
            Event::Start(Tag::Paragraph) => {}
            Event::End(TagEnd::Paragraph) => {
                y += line_height;
                x = rect.x;
            }
            Event::Text(t) => {
                // In a real impl, we'd use is_bold/is_italic to select different fonts
                crate::text::FONT_MANAGER.with(|fm| {
                    let mut fm = fm.borrow_mut();
                    render_text_at(
                        &mut fm,
                        Vec2::new(x, y),
                        &t,
                        current_font_size,
                        view.fg_color.get(),
                        dl,
                    );
                    let text_size = fm.measure_text(&t, current_font_size);
                    x += text_size.x;
                });
            }
            Event::SoftBreak | Event::HardBreak => {
                y += line_height;
                x = rect.x;
            }
            _ => {}
        }
    }
}

/// Render toggle switch
fn render_toggle(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    // Placeholder toggle rendering
    // Would need ToggleView cast for value

    let is_on = false; // Would come from ToggleView.value

    let track_color = if is_on {
        ColorF::new(0.2, 0.6, 0.4, 1.0)
    } else {
        ColorF::new(0.2, 0.2, 0.25, 1.0)
    };

    // Track
    dl.add_rounded_rect(
        Vec2::new(rect.x, rect.y + rect.h * 0.25),
        Vec2::new(40.0, rect.h * 0.5),
        10.0,
        track_color,
    );

    // Thumb
    let thumb_x = if is_on { rect.x + 22.0 } else { rect.x + 2.0 };
    dl.add_rounded_rect(
        Vec2::new(thumb_x, rect.y + rect.h * 0.25 + 2.0),
        Vec2::new(16.0, rect.h * 0.5 - 4.0),
        8.0,
        ColorF::new(0.9, 0.9, 1.0, 1.0),
    );
}

/// Render slider
fn render_slider(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    // Placeholder slider rendering
    // Would need SliderView cast for value/min/max

    let t = 0.5f32; // Would be (value - min) / (max - min)

    // Track
    dl.add_rounded_rect(
        Vec2::new(rect.x, rect.y + rect.h * 0.4),
        Vec2::new(rect.w, rect.h * 0.2),
        4.0,
        ColorF::new(0.15, 0.15, 0.2, 1.0),
    );

    // Filled portion
    dl.add_rounded_rect(
        Vec2::new(rect.x, rect.y + rect.h * 0.4),
        Vec2::new(rect.w * t, rect.h * 0.2),
        4.0,
        ColorF::new(0.3, 0.5, 0.8, 1.0),
    );

    // Thumb
    dl.add_rounded_rect_ex(
        Vec2::new(rect.x + rect.w * t - 6.0, rect.y + rect.h * 0.2),
        Vec2::new(12.0, rect.h * 0.6),
        4.0,
        ColorF::new(0.9, 0.9, 1.0, 1.0),
        4.0,
        false,
        0.0,
        ColorF::transparent(),
        Vec2::ZERO,
        0.0,
        ColorF::transparent(),
    );
}

/// Render text input
fn render_text_input(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let is_focused = interaction::is_focused(view.id.get());

    // Background (handled by generic pass, but we can reinforce or highlight)
    if is_focused {
        dl.add_rounded_rect_ex(
            Vec2::new(rect.x, rect.y),
            Vec2::new(rect.w, rect.h),
            view.border_radius_tl.get(),
            ColorF::transparent(),
            0.0,
            view.is_squircle.get(),
            1.0,                             // Border width
            ColorF::new(0.4, 0.6, 1.0, 1.0), // Focus highlight
            Vec2::ZERO,
            0.0,
            ColorF::transparent(),
        );
    }

    // Render text with padding
    let padding = 8.0;
    let mut text_pos = Vec2::new(
        rect.x + padding,
        rect.y + (rect.h - view.font_size.get()) * 0.5,
    );

    // Caret and IME
    crate::text::FONT_MANAGER.with(|fm| {
        let mut fm = fm.borrow_mut();

        let mut combined_text = view.text.get().to_string();
        let ime_preedit = interaction::get_ime_preedit();

        // If focused and has IME composition, inject it
        // If focused and has IME composition, inject it
        if is_focused {
            interaction::set_focused_text_input(view.id.get(), view.computed_rect.get());

            if !ime_preedit.is_empty() {
                combined_text.push_str(&ime_preedit);
            }

            // Simple caret at end of combined text
            let text_size = fm.measure_text(&combined_text, view.font_size.get());
            let caret_x = text_pos.x + text_size.x + 2.0;

            // Update OS IME Window Position (Screen Coordinates)
            // Note: We need window position offset if we want absolute screen coords.
            // For now, we assume window is at 0,0 locally, but winit SetImeCursorArea expects window-relative?
            // Yes, winit expects physical position relative to top-left of content area.
            interaction::set_ime_cursor_area(
                Vec2::new(caret_x, text_pos.y),
                Vec2::new(2.0, view.font_size.get()),
            );

            // Draw Caret
            dl.add_rounded_rect(
                Vec2::new(caret_x, text_pos.y),
                Vec2::new(2.0, view.font_size.get()),
                0.0,
                ColorF::white(),
            );

            // Draw Underline for Preedit
            if !ime_preedit.is_empty() {
                let original_size = fm.measure_text(view.text.get(), view.font_size.get());
                let preedit_size = fm.measure_text(&ime_preedit, view.font_size.get());
                let ul_start = text_pos.x + original_size.x;
                let ul_end = ul_start + preedit_size.x;

                dl.add_line(
                    Vec2::new(ul_start, text_pos.y + view.font_size.get() + 2.0),
                    Vec2::new(ul_end, text_pos.y + view.font_size.get() + 2.0),
                    1.0,
                    ColorF::new(1.0, 1.0, 1.0, 0.8),
                );
            }
        }

        render_text_at(
            &mut fm,
            text_pos,
            &combined_text,
            view.font_size.get(),
            view.fg_color.get(),
            dl,
        );
    });
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> ColorF {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    ColorF::new(r + m, g + m, b + m, 1.0)
}

/// Render ColorPicker
fn render_color_picker(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();

    // Layout Constants matching layout.rs
    let sv_size = 150.0;
    let hue_width = 20.0;
    let gap = 8.0;

    let padding = 10.0; // Internal padding
    let start_x = rect.x + padding;
    let start_y = rect.y + padding;

    // --- SV Box ---
    let sv_pos = Vec2::new(start_x, start_y);
    let sv_rect_size = Vec2::new(sv_size, sv_size);

    let hsv = view.color_hsv.get();
    let h = hsv[0];
    let s = hsv[1];
    let v = hsv[2];

    let pure_hue = hsv_to_rgb(h, 1.0, 1.0);
    let white = ColorF::white();
    let black = ColorF::black();

    // Gradient: TL=White, TR=Hue, BL=Black, BR=Black
    dl.add_gradient_rect(sv_pos, sv_rect_size, [white, pure_hue, black, black]);

    // Selection Circle (S, V)
    let sel_x = sv_pos.x + s * sv_size;
    let sel_y = sv_pos.y + (1.0 - v) * sv_size; // V=1 at top(0)

    dl.add_circle(Vec2::new(sel_x, sel_y), 5.0, ColorF::white(), false);
    dl.add_circle(Vec2::new(sel_x, sel_y), 4.0, ColorF::black(), false);

    // --- Hue Bar ---
    let hue_pos = Vec2::new(start_x + sv_size + gap, start_y);
    // ...

    // 6 segments
    let seg_h = sv_size / 6.0;
    let colors = [
        (1.0, 0.0, 0.0), // R
        (1.0, 1.0, 0.0), // Y
        (0.0, 1.0, 0.0), // G
        (0.0, 1.0, 1.0), // C
        (0.0, 0.0, 1.0), // B
        (1.0, 0.0, 1.0), // M
        (1.0, 0.0, 0.0), // R
    ];

    for i in 0..6 {
        let y0 = hue_pos.y + i as f32 * seg_h;
        let c1 = colors[i];
        let c2 = colors[i + 1];
        let col1 = ColorF::new(c1.0, c1.1, c1.2, 1.0);
        let col2 = ColorF::new(c2.0, c2.1, c2.2, 1.0);

        dl.add_gradient_rect(
            Vec2::new(hue_pos.x, y0),
            Vec2::new(hue_width, seg_h),
            [col1, col1, col2, col2],
        );
    }

    // Selection Indicator (Hue)
    let hue_sel_y = hue_pos.y + h * sv_size;
    dl.add_rounded_rect(
        Vec2::new(hue_pos.x - 2.0, hue_sel_y - 3.0),
        Vec2::new(hue_width + 4.0, 6.0),
        2.0,
        ColorF::white(),
    );
}

/// Render splitter handle
fn render_splitter(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let handle_size = 8.0;

    // Calculate handle position matching layout.rs
    let ratio = view.ratio.get().clamp(0.0, 1.0);

    let handle_rect = if view.is_vertical.get() {
        let avail = rect.h;
        let size1 = (avail - handle_size) * ratio;
        crate::core::Rectangle::new(rect.x, rect.y + size1, rect.w, handle_size)
    } else {
        let avail = rect.w;
        let size1 = (avail - handle_size) * ratio;
        crate::core::Rectangle::new(rect.x + size1, rect.y, handle_size, rect.h)
    };

    // Draw handle visual (centered line or small rect)
    let is_hot = interaction::is_hot(view.id.get());
    let is_active = interaction::is_active(view.id.get());

    let color = if is_active {
        ColorF::new(0.4, 0.6, 1.0, 1.0)
    } else if is_hot {
        ColorF::new(0.3, 0.5, 0.8, 1.0)
    } else {
        ColorF::new(0.2, 0.2, 0.25, 1.0)
    };

    dl.add_rounded_rect(
        Vec2::new(handle_rect.x, handle_rect.y),
        Vec2::new(handle_rect.w, handle_rect.h),
        2.0,
        color,
    );
}

/// Render scroll container
fn render_scroll(view: &ViewHeader, dl: &mut DrawList, depth: i32, fm: &mut FontManager) {
    let rect = view.computed_rect.get();
    let content_size = view.content_size.get();

    // 1. Get current scroll state
    let mut offset = interaction::get_scroll_offset(view.id.get());

    // 2. Handle Input (if hovered)
    if interaction::is_hot(view.id.get()) {
        let (_dx, dy) = interaction::get_scroll_delta();
        // Mouse wheel: dy > 0 usually means scroll UP (content moves down), so we DECREASE offset.
        // But usually wheel UP means we want to see content ABOVE.
        // Standard mapping: wheel down (negative) -> scroll down (increase offset).
        // Let's try: offset -= dy.
        if dy != 0.0 {
            offset.y -= dy; // Adjust sensitivity in window.rs if needed
        }
    }

    // 3. Clamp offset
    // Max scroll = content_height - view_height
    // Min scroll = 0
    let max_scroll_y = (content_size.h - rect.h).max(0.0);
    offset.y = offset.y.clamp(0.0, max_scroll_y);

    // 4. Save state
    interaction::set_scroll_offset(view.id.get(), offset);

    // 5. Push clip and transform
    dl.push_clip(Vec2::new(rect.x, rect.y), Vec2::new(rect.w, rect.h));

    // Offset translates content UP (negative y)
    dl.push_transform(Vec2::new(-offset.x, -offset.y), 1.0);

    // Render children
    for child in view.children() {
        render_view_recursive(child, dl, depth + 1, fm);
    }

    dl.pop_transform();

    // 6. Draw Scrollbar (Overlay)
    if max_scroll_y > 0.0 {
        let bar_width = 6.0;
        let view_h = rect.h;
        let content_h = content_size.h;

        let ratio = view_h / content_h;
        let bar_h = (view_h * ratio).max(20.0); // Min height

        // Progress 0..1
        let progress = offset.y / max_scroll_y;

        // Available track for bar top
        let track_h = view_h - bar_h;
        let bar_y = rect.y + progress * track_h;
        let bar_x = rect.x + rect.w - bar_width - 2.0; // Right padding

        let bar_color = if interaction::is_hot(view.id.get()) {
            ColorF::new(0.6, 0.6, 0.6, 0.8)
        } else {
            ColorF::new(0.5, 0.5, 0.5, 0.4)
        };

        dl.add_rounded_rect(
            Vec2::new(bar_x, bar_y),
            Vec2::new(bar_width, bar_h),
            3.0,
            bar_color,
        );
    }

    dl.pop_clip();
}

fn render_curve_editor(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let id = view.id.get();

    use crate::view::curves::with_curve_data_mut;

    with_curve_data_mut(id, |data| {
        let is_active = interaction::is_active(id);
        let (mx, my) = interaction::mouse_pos();

        let curve_idx = data.active_curve_idx;
        let mut curve = &mut data.curves[curve_idx];

        // Interaction
        thread_local! {
            static DRAG_POINT: std::cell::Cell<Option<usize>> = std::cell::Cell::new(None);
        }

        if is_active {
            let idx = DRAG_POINT.with(|dp| {
                if dp.get().is_none() {
                    // Find closest point on first frame of activation
                    let mut closest = None;
                    let mut min_dist = 20.0 * 20.0; // 20px radius
                    for (i, p) in curve.points.iter().enumerate() {
                        let px = rect.x + rect.w * p.pos.x;
                        let py = rect.y + rect.h * (1.0 - p.pos.y);
                        let dist = (mx - px).powi(2) + (my - py).powi(2);
                        if dist < min_dist {
                            min_dist = dist;
                            closest = Some(i);
                        }
                    }
                    dp.set(closest);
                    closest
                } else {
                    dp.get()
                }
            });

            if let Some(i) = idx {
                if i < curve.points.len() {
                    let p = &mut curve.points[i];
                    if !p.locked {
                        p.pos.x = ((mx - rect.x) / rect.w).clamp(0.001, 0.999);
                    }
                    p.pos.y = (1.0 - (my - rect.y) / rect.h).clamp(0.0, 1.0);
                    curve
                        .points
                        .sort_by(|a, b| a.pos.x.partial_cmp(&b.pos.x).unwrap());
                }
            }
        } else {
            DRAG_POINT.with(|dp| dp.set(None));

            if interaction::is_clicked(id) {
                let mut hit = false;
                for p in &curve.points {
                    let px = rect.x + rect.w * p.pos.x;
                    let py = rect.y + rect.h * (1.0 - p.pos.y);
                    if (mx - px).powi(2) + (my - py).powi(2) < 8.0 * 8.0 {
                        hit = true;
                        break;
                    }
                }

                if !hit {
                    let nx = ((mx - rect.x) / rect.w).clamp(0.001, 0.999);
                    let ny = (1.0 - (my - rect.y) / rect.h).clamp(0.0, 1.0);
                    curve.points.push(crate::view::curves::CurvePoint {
                        pos: Vec2::new(nx, ny),
                        locked: false,
                    });
                    curve
                        .points
                        .sort_by(|a, b| a.pos.x.partial_cmp(&b.pos.x).unwrap());
                }
            }

            if interaction::is_right_clicked(id) {
                let mut to_remove = None;
                for (i, p) in curve.points.iter().enumerate() {
                    if p.locked {
                        continue;
                    }
                    let px = rect.x + rect.w * p.pos.x;
                    let py = rect.y + rect.h * (1.0 - p.pos.y);
                    if (mx - px).powi(2) + (my - py).powi(2) < 10.0 * 10.0 {
                        to_remove = Some(i);
                        break;
                    }
                }
                if let Some(i) = to_remove {
                    curve.points.remove(i);
                }
            }
        }

        // Draw background grid
        dl.add_rounded_rect(
            Vec2::new(rect.x, rect.y),
            Vec2::new(rect.w, rect.h),
            0.0,
            ColorF::new(0.05, 0.05, 0.07, 1.0),
        );

        let grid_color = ColorF::new(0.2, 0.2, 0.25, 0.5);
        for i in 1..4 {
            let t = i as f32 / 4.0;
            dl.add_line(
                Vec2::new(rect.x + rect.w * t, rect.y),
                Vec2::new(rect.x + rect.w * t, rect.y + rect.h),
                1.0,
                grid_color,
            );
            dl.add_line(
                Vec2::new(rect.x, rect.y + rect.h * t),
                Vec2::new(rect.x + rect.w, rect.y + rect.h * t),
                1.0,
                grid_color,
            );
        }

        // Draw active curve
        let mut path_points = Vec::with_capacity(100);
        for i in 0..=100 {
            let tx = i as f32 / 100.0;
            let ty = curve.evaluate(tx);
            path_points.push(Vec2::new(
                rect.x + rect.w * tx,
                rect.y + rect.h * (1.0 - ty),
            ));
        }

        dl.add_polyline(path_points, curve.color, 2.0, false);

        // Draw control points
        for p in &curve.points {
            let px = rect.x + rect.w * p.pos.x;
            let py = rect.y + rect.h * (1.0 - p.pos.y);
            dl.add_circle(Vec2::new(px, py), 4.0, ColorF::WHITE, true);
            if !p.locked {
                dl.add_circle(Vec2::new(px, py), 6.0, curve.color.with_alpha(0.5), false);
            }
        }
    });
}

/// Render rotary knob
fn render_knob(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let center = rect.center();
    let radius = rect.w.min(rect.h) * 0.4; // 80% diameter covers area

    // Value mapping
    let val = view.value.get();
    let min = view.min.get();
    let max = view.max.get();
    let t = ((val - min) / (max - min)).clamp(0.0, 1.0);

    // Angles: 135 deg (2.356 rad) to 405 deg (7.068 rad)
    // 0 is Right, 90 is Down.
    // 135 is Down-Left. 405 is Down-Right (45 deg).
    let start_angle = 2.3561945;
    let end_angle = 7.0685835;

    let current_angle = start_angle + (end_angle - start_angle) * t;

    let thickness = 4.0;

    // 1. Background Arc (Track)
    let track_color = view.border_color.get().with_alpha(0.3);
    dl.add_arc(
        center,
        radius,
        start_angle,
        end_angle,
        thickness,
        track_color,
    );

    // 2. Active Arc (Fill)
    let active_color = view.fg_color.get();
    dl.add_arc(
        center,
        radius,
        start_angle,
        current_angle,
        thickness,
        active_color,
    );

    // 3. Knob Cap / Indicator
    let ind_x = center.x + radius * current_angle.cos();
    let ind_y = center.y + radius * current_angle.sin();

    dl.add_circle(Vec2::new(ind_x, ind_y), thickness * 1.5, active_color, true);

    // 4. Label
    let text = view.text.get();
    if !text.is_empty() {
        let size = view.font_size.get() * 0.8;
        if size > 0.0 {
            crate::text::FONT_MANAGER.with(|fm| {
                let mut fm = fm.borrow_mut();
                let text_sz = fm.measure_text(text, size);
                // Position below knob
                let text_pos = Vec2::new(center.x - text_sz.x * 0.5, center.y + radius + 10.0);
                render_text_at(&mut fm, text_pos, text, size, view.fg_color.get(), dl);
            });
        }
    }
}

/// Render premium fader
fn render_fader(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let val = view.value.get();
    let min = view.min.get();
    let max = view.max.get();
    let is_bipolar = view.is_bipolar.get();

    let t = ((val - min) / (max - min)).clamp(0.0, 1.0);

    // 1. Trough (Background)
    // Already handled by universal background pass if bg_color is set in UIContext::fader
    // But we want "Inner Shadow" effect.
    // We can draw a slightly smaller darker rect inside
    let padding = 2.0;
    dl.add_rounded_rect(
        Vec2::new(rect.x + padding, rect.y + padding),
        Vec2::new(rect.w - padding * 2.0, rect.h - padding * 2.0),
        2.0,
        view.bg_color.get().darken(0.3),
    );

    // 2. Center Mark (for Bipolar)
    if is_bipolar {
        let mid_y = rect.y + rect.h * 0.5;
        dl.add_line(
            Vec2::new(rect.x, mid_y),
            Vec2::new(rect.x + rect.w, mid_y),
            1.0,
            view.border_color.get().with_alpha(0.5),
        );
    }

    // 3. Track (Active portion)
    let track_w = 4.0;
    let track_x = rect.x + (rect.w - track_w) * 0.5;
    let track_h = rect.h - 10.0;
    let track_y = rect.y + 5.0;

    // Background track line
    dl.add_rounded_rect(
        Vec2::new(track_x, track_y),
        Vec2::new(track_w, track_h),
        2.0,
        view.border_color.get().with_alpha(0.2),
    );

    // Active track line
    if is_bipolar {
        let mid_t = 0.5;
        let start_t = mid_t;
        let end_t = t;

        let y_start = track_y + track_h * (1.0 - start_t);
        let y_end = track_y + track_h * (1.0 - end_t);

        let (y_min, y_max) = if y_start < y_end {
            (y_start, y_end)
        } else {
            (y_end, y_start)
        };

        dl.add_rounded_rect(
            Vec2::new(track_x, y_min),
            Vec2::new(track_w, (y_max - y_min).max(1.0)),
            2.0,
            view.fg_color.get(),
        );
    } else {
        let fill_h = track_h * t;
        dl.add_rounded_rect(
            Vec2::new(track_x, track_y + track_h - fill_h),
            Vec2::new(track_w, fill_h.max(1.0)),
            2.0,
            view.fg_color.get(),
        );
    }

    // 4. Handle (Cap)
    let handle_h = 24.0;
    let handle_w = rect.w - 4.0;
    let handle_y = track_y + track_h * (1.0 - t) - handle_h * 0.5;
    let handle_x = rect.x + 2.0;

    // Handle Shadow
    dl.add_rounded_rect_ex(
        Vec2::new(handle_x, handle_y),
        Vec2::new(handle_w, handle_h),
        4.0,
        view.bg_color.get().lighten(0.1),
        4.0,  // Elevation
        true, // Squircle
        1.0,  // border
        view.border_color.get(),
        Vec2::ZERO,
        if interaction::is_active(view.id.get()) {
            2.0
        } else {
            0.0
        }, // Glow when dragging
        view.fg_color.get(),
    );

    // Handle Center Line (White)
    dl.add_line(
        Vec2::new(handle_x + 4.0, handle_y + handle_h * 0.5),
        Vec2::new(handle_x + handle_w - 4.0, handle_y + handle_h * 0.5),
        2.0,
        view.fg_color.get(),
    );
}

/// Render value dragger (numeric input)
fn render_value_dragger(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let is_editing = view.is_editing.get();

    // Background already rendered by recursive pass if bg_color is set

    if is_editing {
        // Render as TextInput
        render_text_input(view, dl);
    } else {
        // Render as Draggable Label
        let val = view.value.get();
        let text = format!("{:.2}", val); // Simplified formatting

        let color = if interaction::is_hot(view.id.get()) {
            view.fg_color.get().lighten(0.2)
        } else {
            view.fg_color.get()
        };

        // Use centering logic from render_label_and_icon_at but simplified
        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            let size = if view.font_size.get() > 0.0 {
                view.font_size.get()
            } else {
                12.0
            };
            let text_sz = fm.measure_text(&text, size);

            let pos = Vec2::new(
                rect.x + (rect.w - text_sz.x) * 0.5,
                rect.y + (rect.h - size) * 0.5,
            );

            render_text_at(&mut fm, pos, &text, size, color, dl);
        });

        // Draw blue underline if hot to indicate draggability
        if interaction::is_active(view.id.get()) {
            dl.add_line(
                Vec2::new(rect.x + 2.0, rect.y + rect.h - 2.0),
                Vec2::new(rect.x + rect.w - 2.0, rect.y + rect.h - 2.0),
                1.0,
                view.fg_color.get(),
            );
        }
    }
}

/// Render infinite canvas with grid
fn render_canvas(view: &ViewHeader, dl: &mut DrawList, depth: i32, fm: &mut FontManager) {
    let rect = view.computed_rect.get();

    // Register for hit testing
    interaction::update_rect(view.id.get(), rect);

    // Get current transform
    let (mut offset, mut zoom) = interaction::get_canvas_transform(view.id.get());

    // Panning with right/middle mouse when hot
    if interaction::is_hot(view.id.get()) {
        if interaction::is_right_mouse_down() || interaction::is_middle_mouse_down() {
            let delta = interaction::get_mouse_delta();
            offset.x += delta.x;
            offset.y += delta.y;
            interaction::set_canvas_transform(view.id.get(), offset, zoom);
        }

        // Zooming with scroll wheel
        let (_scroll_x, scroll_y) = interaction::get_scroll_delta();
        if scroll_y != 0.0 {
            let old_zoom = zoom;
            zoom *= 1.1f32.powf(scroll_y / 30.0);
            zoom = zoom.clamp(0.1, 5.0);

            // Adjust offset to zoom towards mouse position
            let mouse_pos = interaction::get_mouse_pos();
            let local_mouse = mouse_pos - Vec2::new(rect.x, rect.y);
            offset = local_mouse - (local_mouse - offset) * (zoom / old_zoom);

            interaction::set_canvas_transform(view.id.get(), offset, zoom);
        }
    }

    // 1. Background (Grid)
    dl.add_rounded_rect(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, rect.h),
        0.0,
        view.bg_color.get(),
    );

    // 2. Grid (Transformed)
    // We draw grid in the transformed space for simplicity, or just static grid?
    // Let's do static grid for now to avoid complexity in grid math.
    let grid_size = 50.0 * zoom;
    let grid_color = view.border_color.get().with_alpha(0.1);

    let ox = offset.x % grid_size;
    let oy = offset.y % grid_size;

    for i in -1..=(rect.w / grid_size) as i32 + 1 {
        let x = rect.x + i as f32 * grid_size + ox;
        dl.add_line(
            Vec2::new(x, rect.y),
            Vec2::new(x, rect.y + rect.h),
            1.0,
            grid_color,
        );
    }

    for i in -1..=(rect.h / grid_size) as i32 + 1 {
        let y = rect.y + i as f32 * grid_size + oy;
        dl.add_line(
            Vec2::new(rect.x, y),
            Vec2::new(rect.x + rect.w, y),
            1.0,
            grid_color,
        );
    }

    // 3. Transformation Phase
    dl.push_clip(Vec2::new(rect.x, rect.y), Vec2::new(rect.w, rect.h));
    dl.push_transform(Vec2::new(rect.x + offset.x, rect.y + offset.y), zoom);

    // 4. Render Nodes & Sub-widgets
    for child in view.children() {
        render_view_recursive(child, dl, depth + 1, fm);
    }

    dl.pop_transform();
    dl.pop_clip();

    // 5. Draw "Ghost Wire" (Active Dragging)
    let wire_state = interaction::get_wire_state();
    if let crate::core::wire::WireState::Dragging {
        start_pos, end_pos, ..
    } = wire_state
    {
        // Draw bezier from start to mouse
        let (cp1, cp2) = crate::core::wire::wire_control_points(start_pos, end_pos);
        dl.add_bezier(
            start_pos,
            cp1,
            cp2,
            end_pos,
            2.0,
            ColorF::new(0.5, 0.8, 1.0, 0.8),
        );

        // Interaction: if mouse released, reset wire state
        if !interaction::is_mouse_down() {
            interaction::set_wire_state(crate::core::wire::WireState::Idle);
        }
    }
}

/// Render draggable node
fn render_node(view: &ViewHeader, dl: &mut DrawList, _depth: i32, _fm: &mut FontManager) {
    let rect = view.computed_rect.get();

    // Register for hit testing
    interaction::update_rect(view.id.get(), rect);

    let is_hot = interaction::is_hot(view.id.get());
    let is_active = interaction::is_active(view.id.get());

    // Node dragging logic: update pos when active
    if is_active {
        let delta = interaction::get_mouse_delta();
        view.pos_x.set(view.pos_x.get() + delta.x);
        view.pos_y.set(view.pos_y.get() + delta.y);
    }

    // Header height
    let header_h = 24.0;

    // 1. Node Shadow/Body
    dl.add_rounded_rect_ex(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, rect.h),
        6.0,
        view.bg_color.get(),
        4.0,  // Elevation
        true, // Squircle
        1.0,
        view.border_color.get(),
        Vec2::ZERO,
        if is_active {
            4.0
        } else if is_hot {
            2.0
        } else {
            0.0
        },
        view.fg_color.get().with_alpha(0.5),
    );

    // 2. Header
    let header_color = view.bg_color.get().lighten(0.1);
    dl.add_rounded_rect_ex(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, header_h),
        6.0,
        header_color,
        0.0,
        true,
        0.0,
        ColorF::transparent(),
        Vec2::ZERO,
        0.0,
        ColorF::transparent(),
    );

    // 3. Title
    let title = view.text.get();
    if !title.is_empty() {
        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            let size = 12.0;
            let text_sz = fm.measure_text(title, size);
            let pos = Vec2::new(
                rect.x + (rect.w - text_sz.x) * 0.5,
                rect.y + (header_h - size) * 0.5,
            );
            render_text_at(&mut fm, pos, title, size, view.fg_color.get(), dl);
        });
    }
}

/// Render interactive socket
fn render_socket(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();

    // Register for hit testing
    interaction::update_rect(view.id.get(), rect);

    let is_hot = interaction::is_hot(view.id.get());
    let is_active = interaction::is_active(view.id.get());

    let center = rect.center();
    let radius = rect.w.min(rect.h) * 0.4;

    let color = view.fg_color.get();

    // Glow on interaction
    if is_active || is_hot {
        dl.add_circle(center, radius + 2.0, color.with_alpha(0.3), true);
    }

    dl.add_circle(center, radius, color, true);
    dl.add_circle(center, radius, view.border_color.get(), false);
}

/// Render connection wire (Bezier)
fn render_wire(view: &ViewHeader, dl: &mut DrawList) {
    let pts = view.points.get();
    let color = view.fg_color.get();
    let thickness = view.thickness.get().max(1.0);

    // Use the optimized path drawing for high-fidelity wires
    let mut path = crate::draw::Path::new();
    path.move_to(pts[0]);
    path.cubic_to(pts[1], pts[2], pts[3]);

    dl.add_path(&path, color, thickness);
}

/// Render context menu popup
fn render_context_menu(view: &ViewHeader, dl: &mut DrawList, depth: i32, fm: &mut FontManager) {
    let rect = view.computed_rect.get();

    // Register for hit testing
    interaction::update_rect(view.id.get(), rect);

    // 1. Backdrop blur effect (semi-transparent overlay)
    // In a real implementation, backdrop blur would be a separate pass
    // For now, just draw a semi-transparent background
    let blur = view.backdrop_blur.get();
    if blur > 0.0 {
        // This creates a visual "dimming" effect behind the menu
        // Actual blur would need shader support
    }

    // 2. Menu shadow
    let shadow_offset = Vec2::new(3.0, 3.0);
    let shadow_color = ColorF::new(0.0, 0.0, 0.0, 0.3);
    dl.add_rounded_rect(
        Vec2::new(rect.x + shadow_offset.x, rect.y + shadow_offset.y),
        Vec2::new(rect.w, rect.h),
        view.border_radius_tl.get(),
        shadow_color,
    );

    // 3. Menu body
    dl.add_rounded_rect_ex(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, rect.h),
        view.border_radius_tl.get(),
        view.bg_color.get(),
        0.0,
        false,
        view.border_width.get(),
        view.border_color.get(),
        Vec2::ZERO,
        view.glow_strength.get(),
        view.glow_color.get(),
    );

    // 4. Render children (menu items)
    for child in view.children() {
        render_view_recursive(child, dl, depth + 1, fm);
    }

    // 5. Close on click outside
    if interaction::is_mouse_down() && !interaction::is_hot(view.id.get()) {
        // Check if any child is hot, if not close
        let mut child_hot = false;
        for child in view.children() {
            if interaction::is_hot(child.id.get()) {
                child_hot = true;
                break;
            }
        }
        if !child_hot {
            interaction::close_context_menu();
        }
    }
}

/// Render menu item
fn render_menu_item(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();

    // Register for hit testing
    interaction::update_rect(view.id.get(), rect);

    let is_hot = interaction::is_hot(view.id.get());
    let is_active = interaction::is_active(view.id.get());

    // 1. Hover highlight
    if is_hot || is_active {
        let highlight_color = view.fg_color.get().with_alpha(0.15);
        dl.add_rounded_rect(
            Vec2::new(rect.x + 2.0, rect.y),
            Vec2::new(rect.w - 4.0, rect.h),
            4.0,
            highlight_color,
        );
    }

    // 2. Text
    let text = view.text.get();
    if !text.is_empty() {
        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            let size = view.font_size.get().max(14.0);
            let text_sz = fm.measure_text(text, size);
            let padding = view.padding.get();
            let pos = Vec2::new(rect.x + padding, rect.y + (rect.h - text_sz.y) * 0.5);
            render_text_at(&mut fm, pos, text, size, view.fg_color.get(), dl);
        });
    }

    // 3. Handle click
    if interaction::is_clicked(view.id.get()) {
        // Close menu on item click
        interaction::close_context_menu();
    }
}

/// Render collapsible container with spring-animated height
fn render_collapsible(view: &ViewHeader, dl: &mut DrawList, depth: i32, fm: &mut FontManager) {
    let rect = view.computed_rect.get();
    let header_h = view.min.get().max(24.0);
    let is_expanded = view.is_expanded.get();

    // Register for hit testing (header area)
    let header_rect = crate::core::Rectangle {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: header_h,
    };
    interaction::update_rect(view.id.get(), header_rect);

    let is_hot = interaction::is_hot(view.id.get());

    // Toggle on header click
    if interaction::is_clicked(view.id.get()) {
        view.is_expanded.set(!is_expanded);
    }

    // Spring-animated height
    let target_h = if is_expanded {
        header_h + view.content_height.get()
    } else {
        header_h
    };

    // Get animated height using lerp/smoothing for now
    let current_h = interaction::animate(view.id.get(), "height", target_h, 0.2);

    // 1. Container background
    dl.add_rounded_rect_ex(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, current_h),
        view.border_radius_tl.get(),
        view.bg_color.get(),
        0.0,
        false,
        view.border_width.get(),
        view.border_color.get(),
        Vec2::ZERO,
        0.0,
        ColorF::transparent(),
    );

    // 2. Header bar - use simple rounded rect
    let header_bg = if is_hot {
        view.bg_color.get().lighten(0.1)
    } else {
        view.bg_color.get().lighten(0.05)
    };
    dl.add_rounded_rect(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, header_h),
        view.border_radius_tl.get(),
        header_bg,
    );

    // 3. Arrow indicator (using lines instead of triangle)
    let arrow_x = rect.x + 12.0;
    let arrow_y = rect.y + header_h * 0.5;
    let arrow_size = 4.0;
    let arrow_color = view.fg_color.get();

    if is_expanded {
        // Down arrow: V shape
        dl.add_line(
            Vec2::new(arrow_x - arrow_size, arrow_y - arrow_size * 0.3),
            Vec2::new(arrow_x, arrow_y + arrow_size * 0.5),
            1.5,
            arrow_color,
        );
        dl.add_line(
            Vec2::new(arrow_x, arrow_y + arrow_size * 0.5),
            Vec2::new(arrow_x + arrow_size, arrow_y - arrow_size * 0.3),
            1.5,
            arrow_color,
        );
    } else {
        // Right arrow: > shape
        dl.add_line(
            Vec2::new(arrow_x - arrow_size * 0.3, arrow_y - arrow_size),
            Vec2::new(arrow_x + arrow_size * 0.5, arrow_y),
            1.5,
            arrow_color,
        );
        dl.add_line(
            Vec2::new(arrow_x + arrow_size * 0.5, arrow_y),
            Vec2::new(arrow_x - arrow_size * 0.3, arrow_y + arrow_size),
            1.5,
            arrow_color,
        );
    }

    // 4. Title text
    let title = view.text.get();
    if !title.is_empty() {
        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            let size = view.font_size.get().max(14.0);
            let text_sz = fm.measure_text(title, size);
            let pos = Vec2::new(
                rect.x + 28.0, // After arrow
                rect.y + (header_h - text_sz.y) * 0.5,
            );
            render_text_at(&mut fm, pos, title, size, view.fg_color.get(), dl);
        });
    }

    // 5. Render children if expanded (simple version without complex clipping)
    if current_h > header_h + 1.0 {
        // Push clip for content area
        let content_pos = Vec2::new(rect.x, rect.y + header_h);
        let content_size = Vec2::new(rect.w, current_h - header_h);
        dl.push_clip(content_pos, content_size);

        for child in view.children() {
            render_view_recursive(child, dl, depth + 1, fm);
        }

        dl.pop_clip();
    }
}

/// Render glass toast notification with slide animation
fn render_toast(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();

    // Animate slide-in from right
    let slide_offset = interaction::animate(view.id.get(), "slide", 0.0, 0.15);
    let actual_x = rect.x + (1.0 - slide_offset) * 50.0;

    // Fade alpha based on animation progress
    let alpha = slide_offset.min(1.0);

    // Background with glass effect
    dl.add_rounded_rect_ex(
        Vec2::new(actual_x, rect.y),
        Vec2::new(rect.w, rect.h),
        view.border_radius_tl.get(),
        view.bg_color
            .get()
            .with_alpha(view.bg_color.get().a * alpha),
        view.elevation.get(),
        false,
        view.border_width.get(),
        view.border_color.get().with_alpha(alpha),
        Vec2::ZERO,
        2.0, // Subtle glow
        view.glow_color.get().with_alpha(0.3 * alpha),
    );

    // Color indicator bar on left
    let indicator_w = 4.0;
    dl.add_rounded_rect(
        Vec2::new(actual_x, rect.y),
        Vec2::new(indicator_w, rect.h),
        view.border_radius_tl.get(),
        view.glow_color.get().with_alpha(alpha),
    );

    // Text
    let text = view.text.get();
    if !text.is_empty() {
        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            let size = view.font_size.get().max(14.0);
            let text_sz = fm.measure_text(text, size);
            let padding = view.padding.get();
            let pos = Vec2::new(
                actual_x + padding + indicator_w,
                rect.y + (rect.h - text_sz.y) * 0.5,
            );
            render_text_at(
                &mut fm,
                pos,
                text,
                size,
                view.fg_color.get().with_alpha(alpha),
                dl,
            );
        });
    }
}

/// Render laser tooltip with glow effect
fn render_tooltip(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();

    // Animate fade-in
    let alpha = interaction::animate(view.id.get(), "fade", 1.0, 0.1);

    // Glow effect (outer)
    let glow_strength = view.glow_strength.get();
    if glow_strength > 0.0 {
        dl.add_rounded_rect_ex(
            Vec2::new(rect.x - 2.0, rect.y - 2.0),
            Vec2::new(rect.w + 4.0, rect.h + 4.0),
            view.border_radius_tl.get() + 2.0,
            view.glow_color.get().with_alpha(0.3 * alpha),
            0.0,
            false,
            0.0,
            ColorF::transparent(),
            Vec2::ZERO,
            glow_strength,
            view.glow_color.get().with_alpha(0.5 * alpha),
        );
    }

    // Background
    dl.add_rounded_rect_ex(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, rect.h),
        view.border_radius_tl.get(),
        view.bg_color
            .get()
            .with_alpha(view.bg_color.get().a * alpha),
        0.0,
        false,
        view.border_width.get(),
        view.border_color.get().with_alpha(alpha),
        Vec2::ZERO,
        0.0,
        ColorF::transparent(),
    );

    // Text
    let text = view.text.get();
    if !text.is_empty() {
        crate::text::FONT_MANAGER.with(|fm| {
            let mut fm = fm.borrow_mut();
            let size = view.font_size.get().max(12.0);
            let text_sz = fm.measure_text(text, size);
            let padding = view.padding.get();
            let pos = Vec2::new(rect.x + padding, rect.y + (rect.h - text_sz.y) * 0.5);
            render_text_at(
                &mut fm,
                pos,
                text,
                size,
                view.fg_color.get().with_alpha(alpha),
                dl,
            );
        });
    }
}

fn render_plot(view: &ViewHeader, dl: &mut DrawList) {
    use crate::view::plot::PlotItem;
    let rect = view.computed_rect.get();

    // Data setup
    if let Some(data) = view.plot_data.get() {
        let x_range = data.x_max - data.x_min;
        let y_range = data.y_max - data.y_min;

        // Safety check for div by zero
        let x_range = if x_range == 0.0 { 1.0 } else { x_range };
        let y_range = if y_range == 0.0 { 1.0 } else { y_range };

        let map_x = |x: f32| -> f32 { rect.x + ((x - data.x_min) / x_range) * rect.w };

        let map_y = |y: f32| -> f32 { rect.y + rect.h - ((y - data.y_min) / y_range) * rect.h };

        // Clip to plot area
        dl.push_clip(Vec2::new(rect.x, rect.y), Vec2::new(rect.w, rect.h));

        for item in data.items {
            match item {
                PlotItem::Line {
                    points,
                    color,
                    width,
                } => {
                    let mut mapped = Vec::with_capacity(points.len());
                    for p in *points {
                        mapped.push(Vec2::new(map_x(p.x), map_y(p.y)));
                    }
                    dl.add_plot(
                        mapped,
                        *color,
                        ColorF::TRANSPARENT,
                        *width,
                        rect.y + rect.h, // baseline
                    );
                }
                PlotItem::FastLine {
                    y_data,
                    x_start,
                    x_step,
                    color,
                    width,
                } => {
                    let mut mapped = Vec::with_capacity(y_data.len());
                    for i in 0..y_data.len() {
                        let x = x_start + i as f32 * x_step;
                        let y = y_data[i];

                        // Simple AABB prune (optional here since add_plot handles clip, but better to prune early)
                        // But for high performance, we should only prune if we are far outside.
                        mapped.push(Vec2::new(map_x(x), map_y(y)));
                    }
                    dl.add_plot(mapped, *color, ColorF::TRANSPARENT, *width, rect.y + rect.h);
                }
                PlotItem::Heatmap {
                    data,
                    width,
                    height,
                    colormap,
                    min,
                    max,
                } => {
                    dl.add_heatmap(
                        Vec2::new(rect.x, rect.y),
                        Vec2::new(rect.w, rect.h),
                        data.to_vec(),
                        *width,
                        *height,
                        *colormap,
                        *min,
                        *max,
                    );
                }
            }
        }

        dl.pop_clip();
    }
}

fn render_ruler(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let data = match view.ruler_data.get() {
        Some(d) => d,
        None => return,
    };

    let orientation = data.orientation.get();
    let start_val = data.start.get();
    let scale = data.scale.get(); // pixels per unit
    let step = data.tick_step.get(); // units per tick
    let subs = data.subdivisions.get();

    // Background
    dl.add_rounded_rect(
        Vec2::new(rect.x, rect.y),
        Vec2::new(rect.w, rect.h),
        0.0,
        ColorF::new(0.08, 0.08, 0.08, 1.0),
    );
    // dl.add_rounded_rect_outline(Vec2::new(rect.x, rect.y), Vec2::new(rect.w, rect.h), 0.0, view.border_width.get(), view.border_color.get());
    // Since add_rounded_rect_outline might not exist, I'll check drawlist primitives.
    // Actually, let's just use lines for the border for now or just skip it as the bg helps.
    // Or assume add_rounded_rect draws both if configured?
    // In `render_box` we probably use something like `add_primitive`.
    // Let's just use `add_rounded_rect` for BG.
    // And for border, simple lines if needed.

    // Simplification for now:
    // dl.add_rounded_rect(..., 0.0, ...);

    let tick_color = ColorF::new(0.6, 0.6, 0.6, 1.0);
    let sub_tick_color = ColorF::new(0.4, 0.4, 0.4, 1.0);

    // Render Ticks
    match orientation {
        crate::view::ruler::RulerOrientation::Horizontal => {
            let start_pixel = rect.x;
            let end_pixel = rect.x + rect.w;

            // Calculate first tick value
            let start_unit = start_val;
            let end_unit = start_val + (rect.w / scale);

            // Align to step
            let first_tick_unit = (start_unit / step).floor() * step;
            let last_tick_unit = (end_unit / step).ceil() * step;

            let mut current_unit = first_tick_unit;
            while current_unit <= last_tick_unit {
                let px = rect.x + (current_unit - start_unit) * scale;
                if px >= start_pixel && px <= end_pixel {
                    // Major tick
                    dl.add_line(
                        Vec2::new(px, rect.y + rect.h * 0.5),
                        Vec2::new(px, rect.y + rect.h),
                        1.0,
                        tick_color,
                    );

                    // Subdivisions
                    let sub_step = step / (subs as f32);
                    for i in 1..subs {
                        let sub_unit = current_unit + (i as f32) * sub_step;
                        let sub_px = rect.x + (sub_unit - start_unit) * scale;
                        if sub_px >= start_pixel && sub_px <= end_pixel {
                            dl.add_line(
                                Vec2::new(sub_px, rect.y + rect.h * 0.75),
                                Vec2::new(sub_px, rect.y + rect.h),
                                1.0,
                                sub_tick_color,
                            );
                        }
                    }
                }
                current_unit += step;
            }
        }
        crate::view::ruler::RulerOrientation::Vertical => {
            let start_pixel = rect.y;
            let end_pixel = rect.y + rect.h;

            let start_unit = start_val;
            let end_unit = start_val + (rect.h / scale);

            let first_tick_unit = (start_unit / step).floor() * step;
            let last_tick_unit = (end_unit / step).ceil() * step;

            let mut current_unit = first_tick_unit;
            while current_unit <= last_tick_unit {
                let py = rect.y + (current_unit - start_unit) * scale;
                if py >= start_pixel && py <= end_pixel {
                    // Major tick
                    dl.add_line(
                        Vec2::new(rect.x + rect.w * 0.5, py),
                        Vec2::new(rect.x + rect.w, py),
                        1.0,
                        tick_color,
                    );

                    // Subdivisions
                    let sub_step = step / (subs as f32);
                    for i in 1..subs {
                        let sub_unit = current_unit + (i as f32) * sub_step;
                        let sub_py = rect.y + (sub_unit - start_unit) * scale;
                        if sub_py >= start_pixel && sub_py <= end_pixel {
                            dl.add_line(
                                Vec2::new(rect.x + rect.w * 0.75, sub_py),
                                Vec2::new(rect.x + rect.w, sub_py),
                                1.0,
                                sub_tick_color,
                            );
                        }
                    }
                }
                current_unit += step;
            }
        }
    }
}

fn render_grid(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let data = match view.grid_data.get() {
        Some(d) => d,
        None => return,
    };

    let step_x = data.step_x.get();
    let step_y = data.step_y.get();
    let offset_x = data.offset_x.get();
    let offset_y = data.offset_y.get();
    let scale = data.scale.get();
    let color = data.color.get();

    // Clip to rect
    dl.push_clip(Vec2::new(rect.x, rect.y), Vec2::new(rect.w, rect.h));

    // Calculate effective step
    let eff_step_x = step_x * scale;
    let eff_step_y = step_y * scale;

    if eff_step_x < 2.0 || eff_step_y < 2.0 {
        dl.pop_clip();
        return; // Too dense
    }

    // Vertical lines
    let start_x = (offset_x % eff_step_x) - eff_step_x;
    let mut x = start_x;
    while x < rect.w {
        if x >= 0.0 {
            dl.add_line(
                Vec2::new(rect.x + x, rect.y),
                Vec2::new(rect.x + x, rect.y + rect.h),
                1.0,
                color,
            );
        }
        x += eff_step_x;
    }

    // Horizontal lines
    let start_y = (offset_y % eff_step_y) - eff_step_y;
    let mut y = start_y;
    while y < rect.h {
        if y >= 0.0 {
            dl.add_line(
                Vec2::new(rect.x, rect.y + y),
                Vec2::new(rect.x + rect.w, rect.y + y),
                1.0,
                color,
            );
        }
        y += eff_step_y;
    }

    dl.pop_clip();
}

fn render_transform_gizmo(view: &ViewHeader, dl: &mut DrawList) {
    let rect = view.computed_rect.get();
    let _data = match view.gizmo_data.get() {
        Some(d) => d,
        None => return,
    };

    // For now, render a simple 2D representation of the 3D gizmo at the center of the rect
    // In a real implementation, we would project the 3D position to 2D using camera matrices.

    let center = Vec2::new(rect.x + rect.w * 0.5, rect.y + rect.h * 0.5);
    let size = 50.0f32;

    // X Axis (Red)
    dl.add_line(
        center,
        Vec2::new(center.x + size, center.y),
        3.0,
        ColorF::new(0.9, 0.2, 0.2, 1.0),
    );
    // Y Axis (Green) - Up is -Y in screen space usually, but let's draw it "Up"
    dl.add_line(
        center,
        Vec2::new(center.x, center.y - size),
        3.0,
        ColorF::new(0.2, 0.9, 0.2, 1.0),
    );
    // Z Axis (Blue) - In 2D, maybe diagonal?
    dl.add_line(
        center,
        Vec2::new(center.x - size * 0.5, center.y + size * 0.5),
        3.0,
        ColorF::new(0.2, 0.2, 0.9, 1.0),
    );

    // Handle center handle
    dl.add_circle(center, 5.0, ColorF::WHITE, true);

    // TODO: Implement actual 3D projection and interaction raycasting.
    // This requires access to the active Camera3D's view/projection matrices.
}

fn render_math(view: &ViewHeader, dl: &mut DrawList, fm: &mut FontManager) {
    let rect = view.computed_rect.get();
    let text_ref = view.text.get();
    if text_ref.is_empty() {
        return;
    }

    use crate::text::math::{measure, MathNode, Parser};

    // Parse
    let mut parser = Parser::new(&text_ref);
    let node = parser.parse();

    // Measure to center or align?
    // Let's assume left aligned at rect.x, rect.y for now.
    // Or vertically centered.
    let base_size = view.font_size.get();
    let size = measure(&node, fm, base_size);

    // Position: centered vertically in view
    let start_pos = Vec2::new(rect.x, rect.y + (rect.h - size.y) * 0.5);

    render_math_node(&node, fm, start_pos, base_size, view.fg_color.get(), dl);
}

fn render_math_node(
    node: &crate::text::math::MathNode,
    fm: &mut FontManager,
    pos: Vec2,
    size: f32,
    color: ColorF,
    dl: &mut DrawList,
) -> Vec2 {
    use crate::text::math::*;
    match node {
        MathNode::Text(t) => {
            render_text_at_special(fm, pos, t, size, color, 0, dl);
            // We need advanced width to return.
            // Inefficient to measure again but safe.
            let s = measure(node, fm, size);
            s
        }
        MathNode::Row(nodes) => {
            let mut cursor = pos;
            let mut max_h = 0.0f32;
            let mut total_w = 0.0;
            for n in nodes {
                let s = render_math_node(n, fm, cursor, size, color, dl);
                cursor.x += s.x;
                total_w += s.x;
                max_h = max_h.max(s.y);
            }
            Vec2::new(total_w, max_h)
        }
        MathNode::Sup(base, exp) => {
            let bs = render_math_node(base, fm, pos, size, color, dl);
            // Exp is smaller and raised
            let exp_size = size * 0.7;
            // Raise by some amount relative to base height? Or standard amount?
            // Raise by base_size * 0.4 usually.
            let raise = size * 0.4;
            let exp_pos = Vec2::new(pos.x + bs.x, pos.y - raise);
            let es = render_math_node(exp, fm, exp_pos, exp_size, color, dl);

            Vec2::new(bs.x + es.x, bs.y.max(es.y + raise)) // H is approximated
        }
        MathNode::Sub(base, sub) => {
            let bs = render_math_node(base, fm, pos, size, color, dl);
            let sub_size = size * 0.7;
            let drop = size * 0.2;
            let sub_pos = Vec2::new(pos.x + bs.x, pos.y + drop);
            let ss = render_math_node(sub, fm, sub_pos, sub_size, color, dl);

            Vec2::new(bs.x + ss.x, bs.y.max(ss.y + drop))
        }
        MathNode::Frac(num, den) => {
            let sub_size = size * 0.9;
            let padding = 4.0;

            let num_size = measure(num, fm, sub_size);
            let den_size = measure(den, fm, sub_size);

            let max_w = num_size.x.max(den_size.x);

            // Center num and den horizontally
            let num_x = pos.x + (max_w - num_size.x) * 0.5;
            let den_x = pos.x + (max_w - den_size.x) * 0.5;

            // Vertical layout:
            // Num bottom at baseline - something
            // Bar at baseline - something
            // Den top at baseline + something

            // Simplified:
            // Bar y = pos.y + size * 0.5;
            let bar_y = pos.y + size * 0.5;
            let bar_h = 1.0; // Thickness

            // Draw bar
            dl.add_rounded_rect(
                Vec2::new(pos.x, bar_y),
                Vec2::new(max_w + padding * 2.0, bar_h),
                0.0,
                color,
            );

            // Render Num (above bar)
            render_math_node(
                num,
                fm,
                Vec2::new(num_x + padding, bar_y - num_size.y - 2.0),
                sub_size,
                color,
                dl,
            );

            // Render Den (below bar)
            render_math_node(
                den,
                fm,
                Vec2::new(den_x + padding, bar_y + 4.0),
                sub_size,
                color,
                dl,
            );

            Vec2::new(max_w + padding * 2.0, num_size.y + den_size.y + 6.0)
        }
        MathNode::Sqrt(inner) => {
            let is = measure(inner, fm, size);
            let bar_h = 1.0;
            let roof_y = pos.y; // Top
                                // Draw root symbol (simplified as lines)
                                //  \/
                                //  /
                                // V shape roughly?
                                // Just draw a bracket for now "√ "
                                // We can use actual glyph if we had mapping.
                                // Line: (x, y+h) -> (x+5, y+h) -> (x+10, y+h-is.y) -> (x+10+is.x, y+h-is.y)
                                // Simplified box logic:
            let content_x = pos.x + 12.0;
            let content_y = pos.y + 4.0; // Padding

            // Render content
            render_math_node(inner, fm, Vec2::new(content_x, content_y), size, color, dl);

            // Draw root lines
            let h = is.y + 4.0;
            let w = is.x + 12.0;

            // Diagonal up
            dl.add_line(
                Vec2::new(pos.x, pos.y + h * 0.6),
                Vec2::new(pos.x + 4.0, pos.y + h),
                1.5,
                color,
            );
            // Diagonal down (up actually, from bottom to top right)
            dl.add_line(
                Vec2::new(pos.x + 4.0, pos.y + h),
                Vec2::new(pos.x + 8.0, pos.y),
                1.5,
                color,
            );
            // Roof
            dl.add_line(
                Vec2::new(pos.x + 8.0, pos.y),
                Vec2::new(pos.x + w, pos.y),
                1.5,
                color,
            );

            Vec2::new(w, h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{FrameArena, ID};

    #[test]
    fn test_render_basic() {
        let arena = FrameArena::new();
        let mut dl = DrawList::new();

        let root = arena.alloc(ViewHeader {
            id: std::cell::Cell::new(ID::from_str("root")),
            bg_color: ColorF::new(0.1, 0.1, 0.1, 1.0),
            width: 800.0,
            height: 600.0,
            ..Default::default()
        });

        render_ui(root, 800.0, 600.0, &mut dl);

        assert!(!dl.is_empty());
    }
}
