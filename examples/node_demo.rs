//! Node Editor Demo - showcases Canvas, Node, and Socket widgets

use fanta_rust::backend::{GraphicsBackend, OpenGLBackend};
use fanta_rust::prelude::*;

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{SurfaceAttributesBuilder, WindowSurface};

use glutin_winit::DisplayBuilder;
use raw_window_handle::HasRawWindowHandle;

use std::ffi::CString;
use std::num::NonZeroU32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Node Editor Demo");

    // Create event loop
    let event_loop = EventLoop::new()?;

    // Window builder
    let window_builder = WindowBuilder::new()
        .with_title("Fantasmagorie Node Editor")
        .with_inner_size(LogicalSize::new(1280, 720));

    // Glutin config template
    let template = ConfigTemplateBuilder::new().with_alpha_size(8);

    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

    // Build display and window
    let (window, gl_config) = display_builder.build(&event_loop, template, |configs| {
        configs
            .reduce(|accum, config| {
                if config.num_samples() > accum.num_samples() {
                    config
                } else {
                    accum
                }
            })
            .unwrap()
    })?;

    let window = window.ok_or("No window created")?;
    let raw_window_handle = window.raw_window_handle();

    // Create OpenGL context
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(Some(raw_window_handle));

    let gl_display = gl_config.display();

    let not_current_gl_context =
        unsafe { gl_display.create_context(&gl_config, &context_attributes)? };

    // Create surface
    let size = window.inner_size();
    let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        raw_window_handle,
        NonZeroU32::new(size.width).unwrap(),
        NonZeroU32::new(size.height).unwrap(),
    );

    let surface = unsafe { gl_display.create_window_surface(&gl_config, &surface_attributes)? };

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
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
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
                let (_dx, dy) = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (x * 30.0, y * 30.0),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                };
                fanta_rust::view::interaction::handle_scroll(0.0, dy);
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                // Update interaction
                fanta_rust::view::interaction::update_input(
                    cursor_x,
                    cursor_y,
                    mouse_pressed,
                    right_mouse_pressed,
                    middle_mouse_pressed,
                );
                fanta_rust::view::interaction::begin_interaction_pass();

                // Build UI
                let arena = FrameArena::new();
                let mut ctx = UIContext::new(&arena);
                ctx.theme = Theme::cyberpunk();

                // Root container
                let root_box = ctx
                    .r#box()
                    .column()
                    .width(current_width as f32)
                    .height(current_height as f32)
                    .padding(20.0)
                    .bg(ctx.theme.bg)
                    .build();
                ctx.begin(root_box);
                {
                    ctx.text("Node Editor Demo (Right-drag to pan, Scroll to zoom)")
                        .font_size(18.0)
                        .fg(ctx.theme.accent)
                        .build();

                    // Canvas area
                    let canvas = ctx
                        .canvas()
                        .size(current_width as f32 - 40.0, current_height as f32 - 80.0)
                        .bg(ctx.theme.bg.darken(0.3))
                        .build();
                    ctx.begin(canvas);
                    {
                        // Node 1: Generator
                        let node1 = ctx
                            .node("Generator")
                            .pos(100.0, 50.0)
                            .size(150.0, 80.0)
                            .build();
                        ctx.begin(node1);
                        {
                            let row = ctx.row().build();
                            ctx.begin(row);
                            {
                                ctx.socket("Out", false)
                                    .color(ColorF::new(1.0, 0.4, 0.4, 1.0))
                                    .build();
                            }
                            ctx.end();
                        }
                        ctx.end();

                        // Node 2: Filter
                        let node2 = ctx
                            .node("Filter")
                            .pos(350.0, 100.0)
                            .size(150.0, 100.0)
                            .build();
                        ctx.begin(node2);
                        {
                            let row = ctx.row().build();
                            ctx.begin(row);
                            {
                                ctx.socket("In", true)
                                    .color(ColorF::new(1.0, 0.4, 0.4, 1.0))
                                    .build();
                                ctx.socket("Mod", true)
                                    .color(ColorF::new(0.4, 1.0, 0.4, 1.0))
                                    .build();
                                ctx.socket("Out", false)
                                    .color(ColorF::new(1.0, 1.0, 0.4, 1.0))
                                    .build();
                            }
                            ctx.end();
                        }
                        ctx.end();

                        // Node 3: Output
                        let node3 = ctx
                            .node("Output")
                            .pos(600.0, 80.0)
                            .size(130.0, 70.0)
                            .build();
                        ctx.begin(node3);
                        {
                            let row = ctx.row().build();
                            ctx.begin(row);
                            {
                                ctx.socket("In", true)
                                    .color(ColorF::new(1.0, 1.0, 0.4, 1.0))
                                    .build();
                            }
                            ctx.end();
                        }
                        ctx.end();
                    }
                    ctx.end();
                }
                ctx.end();

                // Layout & Render
                let root = ctx.root().unwrap();
                let mut draw_list = fanta_rust::DrawList::new();
                fanta_rust::text::FONT_MANAGER.with(|fm| {
                    fanta_rust::view::renderer::render_ui(
                        root,
                        current_width as f32,
                        current_height as f32,
                        &mut draw_list,
                        &mut fm.borrow_mut(),
                    );
                });

                backend.render(&draw_list, current_width, current_height);

                surface.swap_buffers(&gl_context).unwrap();
            }
            _ => {}
        }
    })?;

    Ok(())
}
