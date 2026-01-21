//! Interaction detection - hot/active/focus states
//! Ported from interaction.hpp

use crate::core::{Rectangle, Vec2, ID};
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    static CTX: RefCell<InteractionContext> = RefCell::new(InteractionContext::default());
}

// ...

struct InteractionContext {
    hot_id: ID,
    active_id: ID,
    focus_id: ID,
    captured_id: ID,
    mouse_x: f32,
    mouse_y: f32,
    mouse_delta_x: f32,
    mouse_delta_y: f32,
    mouse_down: bool,
    right_mouse_down: bool,
    middle_mouse_down: bool,
    mouse_was_down: bool,
    initialized: bool,

    // Keyboard
    keys_down: HashSet<winit::keyboard::KeyCode>,
    keys_pressed: HashSet<winit::keyboard::KeyCode>,
    modifiers: u32, // 1=Shift, 2=Ctrl, 4=Alt, 8=Super

    input_buffer: String,

    // Persistent layout for interaction (ID -> Rect)
    last_frame_rects: std::collections::HashMap<ID, Rectangle>,

    // Scroll state
    scroll_delta_x: f32,
    scroll_delta_y: f32,
    scroll_offsets: std::collections::HashMap<ID, Vec2>,
    pub virtual_list_ranges: std::collections::HashMap<ID, std::ops::Range<usize>>,

    // Animation state
    animation_states_ex:
        std::collections::HashMap<(ID, String), crate::view::animation::AnimationStateEx>,
    last_frame_time: std::time::Instant,
    dt: f32,

    /// Requested cursor state for this frame.
    cursor_requested: Option<Option<winit::window::CursorIcon>>,

    // Node & Canvas state
    canvas_transforms: std::collections::HashMap<ID, (Vec2, f32)>,
    pub wire_state: crate::core::wire::WireState,

    // Popup/Context Menu state
    active_menu_id: Option<ID>,
    popup_position: Vec2,
    popup_screen_size: Vec2,
    // Screenshot state
    screenshot_requested: Option<String>,

    // IME State
    ime_enabled: bool,
    ime_preedit: String,
    ime_cursor_range: Option<(usize, usize)>,
    ime_cursor_area: Vec2,
    focused_text_input: Option<ID>,
}

impl Default for InteractionContext {
    fn default() -> Self {
        Self {
            hot_id: ID::NONE,
            active_id: ID::NONE,
            focus_id: ID::NONE,
            captured_id: ID::NONE,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_delta_x: 0.0,
            mouse_delta_y: 0.0,
            mouse_down: false,
            right_mouse_down: false,
            middle_mouse_down: false,
            mouse_was_down: false,
            initialized: false,
            keys_down: HashSet::new(),
            keys_pressed: HashSet::new(),
            modifiers: 0,
            input_buffer: String::new(),
            last_frame_rects: std::collections::HashMap::new(),
            scroll_delta_x: 0.0,
            scroll_delta_y: 0.0,
            scroll_offsets: std::collections::HashMap::new(),
            virtual_list_ranges: std::collections::HashMap::new(),
            animation_states_ex: std::collections::HashMap::new(),
            last_frame_time: std::time::Instant::now(),
            dt: 1.0 / 60.0, // Default to 60fps
            cursor_requested: None,
            canvas_transforms: std::collections::HashMap::new(),
            wire_state: crate::core::wire::WireState::Idle,
            active_menu_id: None,
            popup_position: Vec2::ZERO,
            popup_screen_size: Vec2::new(1920.0, 1080.0),

            screenshot_requested: None,
            ime_enabled: false,
            ime_preedit: String::new(),
            ime_cursor_range: None,
            ime_cursor_area: Vec2::ZERO,
            focused_text_input: None,
        }
    }
}

// ... (existing begin_interaction_pass) ...

// ============ Screenshot Functions ============

/// Request a screenshot to be saved to the specified path
pub fn request_screenshot(path: &str) {
    CTX.with(|ctx| {
        ctx.borrow_mut().screenshot_requested = Some(path.to_string());
    });
}

/// Get and clear the screenshot request
pub fn get_screenshot_request() -> Option<String> {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.screenshot_requested.take()
    })
}

/// Begin interaction pass for this frame
pub fn begin_interaction_pass() {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();

        let now = std::time::Instant::now();
        ctx.dt = now.duration_since(ctx.last_frame_time).as_secs_f32();
        ctx.last_frame_time = now;

        // Clamp dt to avoid huge jumps on first frame or window hang
        if ctx.dt > 0.1 || ctx.dt <= 0.0 {
            ctx.dt = 1.0 / 60.0;
        }

        ctx.hot_id = ID::NONE;
        ctx.keys_pressed.clear();
        ctx.scroll_delta_x = 0.0;
        ctx.scroll_delta_y = 0.0;
        ctx.cursor_requested = None;
        // Don't clear virtual_list_ranges, they need to persist across frames
    });
}

/// Simple property animation
pub fn animate(id: ID, property: &str, target: f32, speed: f32) -> f32 {
    // Map legacy speed to duration
    animate_ex(
        id,
        property,
        target,
        1.0 / speed.max(0.1),
        crate::view::animation::Easing::ExpoOut,
    )
}

/// Advanced property animation
pub fn animate_ex(
    id: ID,
    property: &str,
    target: f32,
    duration: f32,
    easing: crate::view::animation::Easing,
) -> f32 {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        let key = (id, property.to_string());
        let dt = ctx.dt;

        let state = ctx.animation_states_ex.entry(key).or_insert(
            crate::view::animation::AnimationStateEx {
                value: target,
                start_value: target,
                target,
                duration,
                easing,
                ..Default::default()
            },
        );

        if (state.target - target).abs() > 0.001 {
            state.start_value = state.value;
            state.target = target;
            state.time = 0.0;
        }

        if state.easing == crate::view::animation::Easing::Spring {
            let mut spring = crate::view::animation::Spring::default();
            spring.velocity = state.velocity;
            state.value = spring.update(state.value, state.target, dt);
            state.velocity = spring.velocity;
        } else {
            state.time += dt;
            let t = (state.time / duration.max(0.001)).clamp(0.0, 1.0);
            let alpha = crate::view::animation::ease(t, state.easing);
            state.value = state.start_value + (state.target - state.start_value) * alpha;
        }

        state.value
    })
}

/// Update input state
pub fn update_input(
    mouse_x: f32,
    mouse_y: f32,
    mouse_down: bool,
    right_mouse_down: bool,
    middle_mouse_down: bool,
) {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();

        if !ctx.initialized {
            ctx.mouse_x = mouse_x;
            ctx.mouse_y = mouse_y;
            ctx.initialized = true;
        }

        let dx = mouse_x - ctx.mouse_x;
        let dy = mouse_y - ctx.mouse_y;

        ctx.mouse_delta_x = dx;
        ctx.mouse_delta_y = dy;

        ctx.mouse_was_down = ctx.mouse_down;
        ctx.mouse_x = mouse_x;
        ctx.mouse_y = mouse_y;
        ctx.mouse_down = mouse_down;
        ctx.right_mouse_down = right_mouse_down;
        ctx.middle_mouse_down = middle_mouse_down;
    });
}

pub fn is_mouse_down() -> bool {
    CTX.with(|ctx| ctx.borrow().mouse_down)
}

pub fn is_right_mouse_down() -> bool {
    CTX.with(|ctx| ctx.borrow().right_mouse_down)
}

pub fn is_middle_mouse_down() -> bool {
    CTX.with(|ctx| ctx.borrow().middle_mouse_down)
}

/// Get last frame's visible range for a virtual list
pub fn get_last_visible_range(id: ID) -> Option<std::ops::Range<usize>> {
    CTX.with(|ctx| ctx.borrow().virtual_list_ranges.get(&id).cloned())
}

/// Set visible range for a virtual list
pub fn set_last_visible_range(id: ID, range: std::ops::Range<usize>) {
    CTX.with(|ctx| {
        ctx.borrow_mut().virtual_list_ranges.insert(id, range);
    })
}

/// Check if point is in rectangle
fn hit_test(rect: Rectangle, px: f32, py: f32) -> bool {
    px >= rect.x && px <= rect.x + rect.w && py >= rect.y && py <= rect.y + rect.h
}

/// Register a widget for interaction
pub fn register_interactive(id: ID, rect: Rectangle) {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();

        // Hit test
        if hit_test(rect, ctx.mouse_x, ctx.mouse_y) {
            // Only set hot if nothing is captured or we are the captured element
            if ctx.captured_id.is_none() || ctx.captured_id == id {
                ctx.hot_id = id;
            }
        }

        // Handle click
        if ctx.hot_id == id {
            if ctx.mouse_down && !ctx.mouse_was_down {
                ctx.active_id = id;
                ctx.captured_id = id;
            }
        }

        // Release on mouse up
        if !ctx.mouse_down && ctx.captured_id == id {
            ctx.captured_id = ID::NONE;
            ctx.active_id = ID::NONE;
        }
    });
}

/// Check if widget is hot (hovered)
pub fn is_hot(id: ID) -> bool {
    id != ID::NONE && CTX.with(|ctx| ctx.borrow().hot_id == id)
}

/// Check if widget is active (pressed)
pub fn is_active(id: ID) -> bool {
    id != ID::NONE && CTX.with(|ctx| ctx.borrow().active_id == id)
}

/// Check if widget is focused
pub fn is_focused(id: ID) -> bool {
    id != ID::NONE && CTX.with(|ctx| ctx.borrow().focus_id == id)
}

/// Set focus to widget
pub fn set_focus(id: ID) {
    CTX.with(|ctx| {
        ctx.borrow_mut().focus_id = id;
    });
}

/// Check if widget was just clicked (released while hot)
pub fn is_clicked(id: ID) -> bool {
    id != ID::NONE
        && CTX.with(|ctx| {
            let ctx = ctx.borrow();
            ctx.hot_id == id && ctx.mouse_was_down && !ctx.mouse_down
        })
}

/// Check if widget was right clicked
pub fn is_right_clicked(id: ID) -> bool {
    id != ID::NONE
        && CTX.with(|ctx| {
            let ctx = ctx.borrow();
            ctx.hot_id == id && ctx.right_mouse_down && ctx.mouse_was_down // Simple approximation
        })
}

/// Capture mouse for dragging
pub fn capture(id: ID) {
    CTX.with(|ctx| {
        ctx.borrow_mut().captured_id = id;
    });
}

/// Release mouse capture
pub fn release() {
    CTX.with(|ctx| {
        ctx.borrow_mut().captured_id = ID::NONE;
    });
}

/// Check if any widget is captured
pub fn is_any_captured() -> bool {
    CTX.with(|ctx| !ctx.borrow().captured_id.is_none())
}

/// Get mouse position
pub fn mouse_pos() -> (f32, f32) {
    CTX.with(|ctx| {
        let ctx = ctx.borrow();
        (ctx.mouse_x, ctx.mouse_y)
    })
}

/// Get mouse delta for this frame
pub fn mouse_delta() -> (f32, f32) {
    CTX.with(|ctx| {
        let ctx = ctx.borrow();
        (ctx.mouse_delta_x, ctx.mouse_delta_y)
    })
}

/// Handle key down
pub fn handle_key_down(key: winit::keyboard::KeyCode) {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.keys_down.insert(key);
        ctx.keys_pressed.insert(key);
    });
}

/// Handle key up
pub fn handle_key_up(key: winit::keyboard::KeyCode) {
    CTX.with(|ctx| {
        ctx.borrow_mut().keys_down.remove(&key);
    });
}

/// Handle modifiers changes
pub fn handle_modifiers(mods: u32) {
    CTX.with(|ctx| {
        ctx.borrow_mut().modifiers = mods;
    });
}

/// Get current modifiers state
pub fn modifiers() -> u32 {
    CTX.with(|ctx| ctx.borrow().modifiers)
}

/// Handle character input (IME/Typing)
pub fn handle_received_character(c: char) {
    CTX.with(|ctx| {
        ctx.borrow_mut().input_buffer.push(c);
    });
}

/// Check if key is held down
pub fn is_key_down(key: winit::keyboard::KeyCode) -> bool {
    CTX.with(|ctx| ctx.borrow().keys_down.contains(&key))
}

/// Check if key was pressed this frame
pub fn is_key_pressed(key: winit::keyboard::KeyCode) -> bool {
    CTX.with(|ctx| ctx.borrow().keys_pressed.contains(&key))
}

/// Get and clear input buffer
pub fn drain_input_buffer() -> String {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        let buffer = ctx.input_buffer.clone();
        ctx.input_buffer.clear();
        buffer
    })
}

pub fn update_rect(id: ID, rect: Rectangle) {
    CTX.with(|ctx| {
        ctx.borrow_mut().last_frame_rects.insert(id, rect);
    })
}

pub fn get_rect(id: ID) -> Option<Rectangle> {
    CTX.with(|ctx| ctx.borrow().last_frame_rects.get(&id).cloned())
}

/// Handle mouse wheel scroll
pub fn handle_scroll(dx: f32, dy: f32) {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.scroll_delta_x += dx;
        ctx.scroll_delta_y += dy;
    });
}

/// Get scroll delta for this frame
pub fn get_scroll_delta() -> (f32, f32) {
    CTX.with(|ctx| {
        let ctx = ctx.borrow();
        (ctx.scroll_delta_x, ctx.scroll_delta_y)
    })
}

/// Get persistent scroll offset for a view
pub fn get_scroll_offset(id: ID) -> Vec2 {
    CTX.with(|ctx| {
        ctx.borrow()
            .scroll_offsets
            .get(&id)
            .cloned()
            .unwrap_or(Vec2::ZERO)
    })
}

/// Set persistent scroll offset for a view
pub fn set_scroll_offset(id: ID, offset: Vec2) {
    CTX.with(|ctx| {
        ctx.borrow_mut().scroll_offsets.insert(id, offset);
    })
}

/// Request a cursor state for this frame
pub fn request_cursor(icon: Option<winit::window::CursorIcon>) {
    CTX.with(|ctx| {
        ctx.borrow_mut().cursor_requested = Some(icon);
    })
}

/// Get the requested cursor state
pub fn get_requested_cursor() -> Option<Option<winit::window::CursorIcon>> {
    CTX.with(|ctx| ctx.borrow().cursor_requested)
}

/// Get canvas transform (offset, zoom)
pub fn get_canvas_transform(id: ID) -> (Vec2, f32) {
    CTX.with(|ctx| {
        ctx.borrow()
            .canvas_transforms
            .get(&id)
            .cloned()
            .unwrap_or((Vec2::ZERO, 1.0))
    })
}

/// Set canvas transform
pub fn set_canvas_transform(id: ID, offset: Vec2, zoom: f32) {
    CTX.with(|ctx| {
        ctx.borrow_mut()
            .canvas_transforms
            .insert(id, (offset, zoom));
    })
}

/// Get current mouse position
pub fn get_mouse_pos() -> Vec2 {
    CTX.with(|ctx| {
        let ctx = ctx.borrow();
        Vec2::new(ctx.mouse_x, ctx.mouse_y)
    })
}

/// Get mouse delta
pub fn get_mouse_delta() -> Vec2 {
    CTX.with(|ctx| {
        let ctx = ctx.borrow();
        Vec2::new(ctx.mouse_delta_x, ctx.mouse_delta_y)
    })
}

/// Get wire state
pub fn get_wire_state() -> crate::core::wire::WireState {
    CTX.with(|ctx| ctx.borrow().wire_state.clone())
}

/// Set wire state
pub fn set_wire_state(state: crate::core::wire::WireState) {
    CTX.with(|ctx| ctx.borrow_mut().wire_state = state);
}

// ============ Context Menu / Popup Functions ============

/// Open a context menu at the specified position
pub fn open_context_menu(id: ID, pos: Vec2) {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.active_menu_id = Some(id);
        ctx.popup_position = pos;
    });
}

/// Close the current context menu
pub fn close_context_menu() {
    CTX.with(|ctx| {
        ctx.borrow_mut().active_menu_id = None;
    });
}

/// Check if any context menu is currently open
pub fn is_context_menu_open() -> bool {
    CTX.with(|ctx| ctx.borrow().active_menu_id.is_some())
}

/// Get the ID of the currently active menu
pub fn get_active_menu_id() -> Option<ID> {
    CTX.with(|ctx| ctx.borrow().active_menu_id)
}

/// Get the popup position
pub fn get_popup_position() -> Vec2 {
    CTX.with(|ctx| ctx.borrow().popup_position)
}

/// Set screen size for popup positioning
pub fn set_popup_screen_size(size: Vec2) {
    CTX.with(|ctx| ctx.borrow_mut().popup_screen_size = size);
}

/// Get screen size for popup positioning
pub fn get_popup_screen_size() -> Vec2 {
    CTX.with(|ctx| ctx.borrow().popup_screen_size)
}

// ============ IME Functions ============


// ============ IME Functions ============

pub fn set_ime_enabled(enabled: bool) {
    CTX.with(|ctx| {
        ctx.borrow_mut().ime_enabled = enabled;
    });
}

pub fn is_ime_enabled() -> bool {
    CTX.with(|ctx| ctx.borrow().ime_enabled)
}

pub fn set_ime_preedit(text: String, range: Option<(usize, usize)>) {
    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.ime_preedit = text;
        ctx.ime_cursor_range = range;
    });
}

pub fn get_ime_preedit() -> String {
    CTX.with(|ctx| ctx.borrow().ime_preedit.clone())
}

pub fn get_ime_cursor_range() -> Option<(usize, usize)> {
    CTX.with(|ctx| ctx.borrow().ime_cursor_range)
}

pub fn set_ime_cursor_area(pos: Vec2) {
    CTX.with(|ctx| {
        ctx.borrow_mut().ime_cursor_area = pos;
    });
}

pub fn get_ime_cursor_area() -> Vec2 {
    CTX.with(|ctx| ctx.borrow().ime_cursor_area)
}

pub fn set_focused_text_input(id: Option<ID>) {
    CTX.with(|ctx| {
        ctx.borrow_mut().focused_text_input = id;
    });
}

pub fn get_focused_text_input() -> Option<ID> {
    CTX.with(|ctx| ctx.borrow().focused_text_input)
}
