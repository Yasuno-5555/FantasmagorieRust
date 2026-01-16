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

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use std::num::NonZeroU32;
use std::ffi::CString;

use winit::event::{Event, WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit::dpi::LogicalSize;

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{SurfaceAttributesBuilder, WindowSurface};

use glutin_winit::DisplayBuilder;
use raw_window_handle::HasRawWindowHandle;

use crate::draw::DrawList;
use crate::view::render_ui;
use crate::backend::{Backend, OpenGLBackend};

use super::bindings::PY_CONTEXT;

/// Initialize global context before each frame
fn init_frame(width: u32, height: u32) {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(super::bindings::PyContextInner::new(width, height));
        }
        if let Some(inner) = borrow.as_mut() {
            inner.width = width;
            inner.height = height;
            inner.reset();
        }
    });
}

/// End frame: run layout and render passes on the AST
fn end_frame(width: u32, height: u32) -> DrawList {
    PY_CONTEXT.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        if let Some(inner) = borrow.as_mut() {
            // Get root view
            if let Some(root_id) = inner.root_id {
                if let Some(&root_ptr) = inner.views.get(&root_id) {
                    unsafe {
                        let root = &*root_ptr;
                        
                        // Clear draw list
                        inner.draw_list.clear();
                        
                        // Layout + Render in one call (render_ui does both)
                        render_ui(root, width as f32, height as f32, &mut inner.draw_list);
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
#[pyo3(name = "run_window")]
pub fn py_run_window(
    py: Python,
    width: u32,
    height: u32,
    title: String,
    callback: PyObject,
) -> PyResult<()> {
    // Release the GIL while creating window (allows Python threads)
    py.allow_threads(|| {
        run_window_impl(width, height, &title, callback)
    })
}

fn run_window_impl(
    width: u32,
    height: u32,
    title: &str,
    callback: PyObject,
) -> PyResult<()> {
    // Create event loop
    let event_loop = EventLoop::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create event loop: {}", e)))?;

    // Window builder
    let window_builder = WindowBuilder::new()
        .with_title(title)
        .with_inner_size(LogicalSize::new(width, height));

    // Glutin config template
    let template = ConfigTemplateBuilder::new()
        .with_alpha_size(8);

    let display_builder = DisplayBuilder::new()
        .with_window_builder(Some(window_builder));

    // Build display and window
    let (window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            configs.reduce(|accum, config| {
                if config.num_samples() > accum.num_samples() {
                    config
                } else {
                    accum
                }
            }).unwrap()
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build display: {}", e)))?;

    let window = window.ok_or_else(|| PyRuntimeError::new_err("No window created"))?;
    let raw_window_handle = window.raw_window_handle();

    // Create OpenGL context
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(Some(raw_window_handle));

    let gl_display = gl_config.display();

    let not_current_gl_context = unsafe {
        gl_display.create_context(&gl_config, &context_attributes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create context: {}", e)))?
    };

    // Create surface
    let size = window.inner_size();
    let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new()
        .build(
            raw_window_handle,
            NonZeroU32::new(size.width).unwrap(),
            NonZeroU32::new(size.height).unwrap(),
        );

    let surface = unsafe {
        gl_display.create_window_surface(&gl_config, &surface_attributes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create surface: {}", e)))?
    };

    // Make context current
    let gl_context = not_current_gl_context.make_current(&surface)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make context current: {}", e)))?;

    // Load OpenGL functions
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            let c_str = CString::new(s).unwrap();
            gl_display.get_proc_address(&c_str) as *const _
        })
    };

    // Create OpenGL backend
    let backend: Box<dyn Backend> = unsafe {
        Box::new(OpenGLBackend::new(gl)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create backend: {}", e)))?)
    };

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
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                println!("👋 Window closed. Goodbye!");
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width > 0 && size.height > 0 {
                    current_width = size.width;
                    current_height = size.height;
                    surface.resize(
                        &gl_context,
                        NonZeroU32::new(size.width).unwrap(),
                        NonZeroU32::new(size.height).unwrap(),
                    );
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                cursor_x = position.x as f32;
                cursor_y = position.y as f32;
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                match button {
                    MouseButton::Left => mouse_pressed = state == ElementState::Pressed,
                    MouseButton::Right => right_mouse_pressed = state == ElementState::Pressed,
                    MouseButton::Middle => middle_mouse_pressed = state == ElementState::Pressed,
                    _ => {}
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseWheel { delta, .. }, .. } => {
                let (dx, dy) = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (x * 30.0, y * 30.0),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                };
                crate::view::interaction::handle_scroll(dx, dy);
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. } => {
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
            }
            Event::WindowEvent { event: WindowEvent::Ime(ime_event), .. } => {
                match ime_event {
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
                }
            }
            // modifiers changed is the same
            Event::WindowEvent { event: WindowEvent::ModifiersChanged(state), .. } => {
                let mut mods = 0;
                let s = state.state();
                if s.shift_key() { mods |= 1; }
                if s.control_key() { mods |= 2; }
                if s.alt_key() { mods |= 4; }
                if s.super_key() { mods |= 8; }
                crate::view::interaction::handle_modifiers(mods);
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                frame_count += 1;

                // ═══════════════════════════════════════════════════════════════
                // FRAME PIPELINE
                // ═══════════════════════════════════════════════════════════════

                // 0. INPUT: Update interaction state
                crate::view::interaction::update_input(cursor_x, cursor_y, mouse_pressed, right_mouse_pressed, middle_mouse_pressed);

                // Update IME Cursor Position (OS Candidate Window)
                let ime_pos = crate::view::interaction::get_ime_cursor_area();
                let _ = window.set_ime_cursor_area(winit::dpi::Position::Physical(winit::dpi::PhysicalPosition::new(
                    ime_pos.x as i32,
                    ime_pos.y as i32
                )), winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(10.0, 20.0)));


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
                if let Some(path) = crate::view::interaction::get_screenshot_request() {
                    println!("📸 Capturing screenshot to: {}", path);
                    // Read pixels
                    let mut pixels = vec![0u8; (current_width * current_height * 4) as usize];
                    unsafe {
                        gl.read_pixels(
                            0, 0,
                            current_width as i32, current_height as i32,
                            glow::RGBA,
                            glow::UNSIGNED_BYTE,
                            glow::PixelPackData::Slice(&mut pixels),
                        );
                    }
                    
                    // Flip Y (OpenGL is bottom-left origin)
                    let row_size = (current_width * 4) as usize;
                    let mut flipped_pixels = vec![0u8; pixels.len()];
                    for y in 0..current_height {
                        let src_row = (y * current_width * 4) as usize;
                        let dst_row = ((current_height - 1 - y) * current_width * 4) as usize;
                        flipped_pixels[dst_row..dst_row + row_size]
                            .copy_from_slice(&pixels[src_row..src_row + row_size]);
                    }

                    // Save using image crate
                    if let Err(e) = image::save_buffer(
                        &path,
                        &flipped_pixels,
                        current_width,
                        current_height,
                        image::ColorType::Rgba8,
                    ) {
                        eprintln!("❌ Failed to save screenshot: {}", e);
                    } else {
                        println!("✅ Screenshot saved!");
                    }
                }

                // 5. SWAP BUFFERS
                let _ = surface.swap_buffers(&gl_context);
                
                // Debug output every 60 frames
                if frame_count % 60 == 0 {
                    println!("   Frame {}: {} draw commands", frame_count, draw_list.len());
                }
            }
            _ => {}
        }
    }).map_err(|e| PyRuntimeError::new_err(format!("Event loop error: {}", e)))?;

    Ok(())
}
