//! Windowed demo to show off Visual Revolution features

use fanta_rust::prelude::*;
use fanta_rust::backend::{Backend, OpenGLBackend};

use winit::event::{Event, WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit::dpi::LogicalSize;
use winit::keyboard::{Key, NamedKey};

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{SurfaceAttributesBuilder, WindowSurface};

use glutin_winit::DisplayBuilder;
use raw_window_handle::HasRawWindowHandle;

use std::num::NonZeroU32;
use std::ffi::CString;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Window Demo");
    
    // Create event loop
    let event_loop = EventLoop::new()?;

    // Window builder
    let window_builder = WindowBuilder::new()
        .with_title("Fantasmagorie Visual Revolution")
        .with_inner_size(LogicalSize::new(1024, 768));

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
        })?;

    let window = window.ok_or("No window created")?;
    let raw_window_handle = window.raw_window_handle();

    // Create OpenGL context
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(Some(raw_window_handle));

    let gl_display = gl_config.display();

    let not_current_gl_context = unsafe {
        gl_display.create_context(&gl_config, &context_attributes)?
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
        gl_display.create_window_surface(&gl_config, &surface_attributes)?
    };

    // Make context current
    let gl_context = not_current_gl_context.make_current(&surface)?;

    // Load OpenGL functions
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            let c_str = CString::new(s).unwrap();
            gl_display.get_proc_address(&c_str) as *const _
        })
    };

    // Create OpenGL backend
    let mut backend = unsafe { OpenGLBackend::new(gl)? };

    // State
    let mut theme_idx = 0;
    let mut k1 = 0.5;
    let mut k2 = 0.2;
    let mut k3 = 0.8;
    let mut k4 = 0.0;
    let mut k5 = 1.0;
    let mut f1 = 0.5;
    let mut d1 = 440.0;
    let mut data = vec![0.0; 100];
    for i in 0..100 {
        data[i] = (i as f32 * 0.2).sin();
    }

    // Interaction state
    let mut current_width = size.width;
    let mut current_height = size.height;
    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;
    let mut right_mouse_pressed = false;
    let mut middle_mouse_pressed = false;

    // Run event loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
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
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. } => {
                if event.state == ElementState::Pressed {
                    if let Key::Named(NamedKey::Space) = event.logical_key {
                        theme_idx = (theme_idx + 1) % 3;
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // Update interaction
                fanta_rust::view::interaction::update_input(cursor_x, cursor_y, mouse_pressed, right_mouse_pressed, middle_mouse_pressed);
                fanta_rust::view::interaction::begin_interaction_pass();

                // Build UI
                let arena = FrameArena::new();
                let mut ctx = UIContext::new(&arena);

                // Set Theme
                let theme = match theme_idx {
                    0 => Theme::cyberpunk(),
                    1 => Theme::zen(),
                    2 => Theme::heat(),
                    _ => Theme::default(),
                };
                ctx.theme = theme;

                let root = ctx.column()
                    .size(current_width as f32, current_height as f32)
                    .bg(ctx.theme.atmosphere)
                    .padding(40.0)
                    .build();
                
                ctx.begin(root);
                
                let header = ctx.row().height(80.0).padding(20.0).align(Align::Center).build();
                ctx.begin(header);
                    ctx.text("THE SPECIALTY STORE").font_size(24.0).fg(ctx.theme.text).build();
                    ctx.r#box().width(20.0).build();
                    let t_name = match theme_idx { 0 => "CYBERPUNK", 1 => "ZEN", 2 => "HEAT", _ => "" };
                    ctx.text(t_name).font_size(18.0).fg(ctx.theme.accent).build();
                    ctx.r#box().flex_grow(1.0).build();
                    ctx.text("Press SPACE to Switch Theme").font_size(14.0).fg(ctx.theme.text.with_alpha(0.5)).build();
                ctx.end();
                
                let content = ctx.row().flex_grow(1.0).padding(40.0).align(Align::Center).build();
                ctx.begin(content);
                    let col1 = ctx.column().flex_grow(1.0).align(Align::Center).build();
                    ctx.begin(col1);
                        ctx.knob(&mut k1, 0.0, 1.0).size(60.0).label("Volume").build();
                        ctx.r#box().height(40.0).build();
                        ctx.knob(&mut k2, 0.0, 1.0).size(80.0).label("Drive").build();
                    ctx.end();

                    let col2 = ctx.column().flex_grow(1.0).align(Align::Center).build();
                    ctx.begin(col2);
                        let hero = ctx.knob(&mut k3, 0.0, 1.0).size(120.0).label("MASTER").build();
                        hero.glow_strength.set(3.0);
                    ctx.end();

                    let col3 = ctx.column().flex_grow(1.0).align(Align::Center).build();
                    ctx.begin(col3);
                        ctx.knob(&mut k4, 0.0, 1.0).size(50.0).label("Pan").build();
                        ctx.r#box().height(20.0).build();
                        ctx.knob(&mut k5, 0.0, 1.0).size(50.0).label("Send").build();
                    ctx.end();
                    
                    let col4 = ctx.column().width(150.0).padding(20.0).align(Align::Center).build();
                    ctx.begin(col4);
                         ctx.fader(&mut f1, 0.0, 1.0).build();
                         ctx.r#box().height(20.0).build();
                         ctx.value_dragger(&mut d1, 20.0, 2000.0).build();
                         ctx.text("Freq (Hz)").font_size(12.0).fg(ctx.theme.text.with_alpha(0.7)).build();
                    ctx.end();
                    
                    let col5 = ctx.column().flex_grow(1.0).padding(20.0).build();
                    ctx.begin(col5);
                        ctx.text("Waveform").font_size(14.0).fg(ctx.theme.text).build();
                        ctx.r#box().height(10.0).build();
                        ctx.plot(&data, -1.0, 1.0).size(300.0, 150.0).build();
                    ctx.end();
                ctx.end();
                ctx.end();

                // Layout & Render
                if let Some(root) = ctx.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::view::renderer::render_ui(root, current_width as f32, current_height as f32, &mut dl);
                    backend.render(&dl, current_width, current_height);
                }
                
                // Apply cursor requests
                if let Some(cursor) = fanta_rust::view::interaction::get_requested_cursor() {
                    match cursor {
                        None => {
                            window.set_cursor_visible(false);
                        }
                        Some(icon) => {
                            window.set_cursor_visible(true);
                            window.set_cursor_icon(icon);
                        }
                    }
                } else {
                    window.set_cursor_visible(true);
                    window.set_cursor_icon(winit::window::CursorIcon::Default);
                }

                let _ = surface.swap_buffers(&gl_context);
            }
            _ => {}
        }
    })?;

    Ok(())
}
