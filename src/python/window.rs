//! Window runner - winit event loop integration
//!
//! Implements run_window() which:
//! 1. Creates a window with glutin/winit
//! 2. Initializes OpenGL context
//! 3. Runs the event loop:
//!    - Calls Python callback (builds AST)
//!    - Runs Layout pass (compute_flex_layout)
//!    - Runs Render pass (render_ui)
//!    - Draws with OpenGL backend

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit::window::WindowBuilder;
use std::sync::Arc;
use pyo3::exceptions::PyRuntimeError;

use crate::backend::{GraphicsBackend, WgpuBackend};
use crate::draw::DrawList;
use crate::view::render_ui;

use super::bindings::context::{PY_CONTEXTS, CURRENT_WINDOW, PyContextInner};

/// Initialize global context before each frame
fn init_frame(width: u32, height: u32) {
    PY_CONTEXTS.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        // Use ID 0 for the main window runner
        let inner = borrow
            .entry(0)
            .or_insert_with(|| PyContextInner::new(width, height));
        inner.width = width;
        inner.height = height;
        inner.reset();

        // Ensure CURRENT_WINDOW is 0
        CURRENT_WINDOW.with(|cw| *cw.borrow_mut() = 0);
    });
}

/// End frame: run layout and render passes on the AST
fn end_frame(width: u32, height: u32) -> DrawList {
    PY_CONTEXTS.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        if let Some(inner) = borrow.get_mut(&0) {
            // Get root view
            if let Some(root_id) = inner.root_id {
                if let Some(&root_ptr) = inner.views.get(&root_id) {
                    unsafe {
                        let root = &*root_ptr;

                        // Clear draw list
                        inner.draw_list.clear();
                        render_ui(
                            root,
                            width as f32,
                            height as f32,
                            &mut inner.draw_list,
                            &mut inner.font_manager,
                        );
                    }
                }
            }

            // Return a clone of the draw list
            inner.draw_list.clone()
        } else {
            DrawList::new()
        }
    })
}

/// Run the windowed application with Python callback
#[pyfunction]
#[pyo3(name = "run_window", signature = (callback, width=1280, height=720, title="Fantasmagorie".to_string()))]
pub fn py_run_window(
    py: Python,
    callback: PyObject,
    width: u32,
    height: u32,
    title: String,
) -> PyResult<()> {
    // Release the GIL while creating window (allows Python threads)
    py.allow_threads(|| run_window_impl(width, height, &title, callback))
}

fn run_window_impl(width: u32, height: u32, title: &str, callback: PyObject) -> PyResult<()> {
    // Create event loop
    let event_loop = EventLoop::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create event loop: {}", e)))?;

    // Window builder
    let window = Arc::new(WindowBuilder::new()
        .with_title(title)
        .with_inner_size(LogicalSize::new(width, height))
        .build(&event_loop)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create window: {}", e)))?);

    // Create WGPU backend
    let mut backend = WgpuBackend::new_async(window.clone(), width, height)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create WGPU backend: {}", e)))?;

    // State for the loop
    let mut current_width = width;
    let mut current_height = height;

    let mut frame_count = 0u64;

    // Input state
    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;
    let mut right_mouse_pressed = false;
    let mut middle_mouse_pressed = false;

    println!("🪟 Window created: {}x{}", width, height);
    println!("🎨 OpenGL backend initialized");
    println!("🐍 Entering event loop...");

    // Run event loop
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    println!("👋 Window closed. Goodbye!");
                    elwt.exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    if size.width > 0 && size.height > 0 {
                        current_width = size.width;
                        current_height = size.height;
                    if size.width > 0 && size.height > 0 {
                        current_width = size.width;
                        current_height = size.height;
                        // WGPU surface reconfiguration is handled inside backend.render or explicitly
                        backend.surface_config.width = size.width;
                        backend.surface_config.height = size.height;
                        backend.surface.configure(&backend.device, &backend.surface_config);
                    }
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position, .. },
                    ..
                } => {
                    cursor_x = position.x as f32;
                    cursor_y = position.y as f32;
                }
                Event::WindowEvent {
                    event: WindowEvent::MouseInput { state, button, .. },
                    ..
                } => match button {
                    MouseButton::Left => mouse_pressed = state == ElementState::Pressed,
                    MouseButton::Right => right_mouse_pressed = state == ElementState::Pressed,
                    MouseButton::Middle => middle_mouse_pressed = state == ElementState::Pressed,
                    _ => {}
                },
                Event::WindowEvent {
                    event: WindowEvent::MouseWheel { delta, .. },
                    ..
                } => {
                    let (dx, dy) = match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => (x * 30.0, y * 30.0),
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            (pos.x as f32, pos.y as f32)
                        }
                    };
                    crate::view::interaction::handle_scroll(dx, dy);
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { event, .. },
                    ..
                } => {
                    // Handle key state
                    if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                        if event.state == ElementState::Pressed {
                            crate::view::interaction::handle_key_down(code);

                            // Handle text input from the key event directly if available
                            if let Some(text) = event.text {
                                for c in text.chars() {
                                    if !c.is_control() {
                                        crate::view::interaction::handle_received_character(c);
                                    }
                                }
                            }
                        } else {
                            crate::view::interaction::handle_key_up(code);
                        }
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::Ime(ime_event),
                    ..
                } => match ime_event {
                    winit::event::Ime::Enabled => {
                        crate::view::interaction::set_ime_enabled(true);
                    }
                    winit::event::Ime::Preedit(text, cursor) => {
                        crate::view::interaction::set_ime_preedit(text, cursor);
                    }
                    winit::event::Ime::Commit(text) => {
                        crate::view::interaction::set_ime_preedit(String::new(), None);
                        for c in text.chars() {
                            crate::view::interaction::handle_received_character(c);
                        }
                    }
                    winit::event::Ime::Disabled => {
                        crate::view::interaction::set_ime_enabled(false);
                        crate::view::interaction::set_ime_preedit(String::new(), None);
                    }
                },
                // modifiers changed is the same
                Event::WindowEvent {
                    event: WindowEvent::ModifiersChanged(state),
                    ..
                } => {
                    let mut mods = 0;
                    let s = state.state();
                    if s.shift_key() {
                        mods |= 1;
                    }
                    if s.control_key() {
                        mods |= 2;
                    }
                    if s.alt_key() {
                        mods |= 4;
                    }
                    if s.super_key() {
                        mods |= 8;
                    }
                    crate::view::interaction::handle_modifiers(mods);
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    frame_count += 1;

                    // ═══════════════════════════════════════════════════════════════
                    // FRAME PIPELINE
                    // ═══════════════════════════════════════════════════════════════

                    // 0. INPUT: Update interaction state
                    crate::view::interaction::update_input(
                        cursor_x,
                        cursor_y,
                        mouse_pressed,
                        right_mouse_pressed,
                        middle_mouse_pressed,
                    );

                    // Update IME Cursor Position (OS Candidate Window)
                    let ime_pos = crate::view::interaction::get_ime_cursor_area();
                    let _ = window.set_ime_cursor_area(
                        winit::dpi::Position::Physical(winit::dpi::PhysicalPosition::new(
                            ime_pos.x as i32,
                            ime_pos.y as i32,
                        )),
                        winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(10.0, 20.0)),
                    );

                    // 1. BEGIN FRAME: Reset arena & context
                    init_frame(current_width, current_height);

                    // 2. PYTHON CALLBACK: Build AST (View tree)
                    Python::with_gil(|py| {
                        if let Err(e) = callback.call1(py, (current_width, current_height)) {
                            eprintln!("❌ Python callback error: {}", e);
                        }
                    });

                    // 3. END FRAME: Layout + Render (AST → DrawCommands)
                    let draw_list = end_frame(current_width, current_height);

                    // 4. BACKEND DRAW: DrawCommands → OpenGL
                    backend.render(&draw_list, current_width, current_height);

                    // Handle Screenshot Request
                    // Handle Screenshot Request
                    // (Disabled during Multi-Window verification to avoid gl ownership issues)
                    /*
                    if let Some(path) = crate::view::interaction::get_screenshot_request() {
                         // ...
                    }
                    */

                    // 5. SWAP BUFFERS
                    backend.present();
                    
                    // Check GL Error
                    /*
                    let err = unsafe { backend.get_error() }; // Need to expose get_error or just use gl context if I can access it. 
                    // Backend owns gl. I need to add get_error to Backend trait or just print it inside render.
                    */

                    // Debug output every 60 frames
                    if frame_count % 60 == 0 {
                        println!(
                            "   Frame {}: {} draw commands",
                            frame_count,
                            draw_list.len()
                        );
                    }
                }
                _ => {}
            }
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Event loop error: {}", e)))?;

    Ok(())
}
