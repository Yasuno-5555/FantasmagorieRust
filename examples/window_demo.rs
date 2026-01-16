//! Windowed demo to show off Visual Revolution features

use fanta_rust::prelude::*;
use fanta_rust::backend::{Backend, OpenGLBackend};

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
    let mut current_width = size.width;
    let mut current_height = size.height;
    
    // Interaction state
    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;

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
                if button == MouseButton::Left {
                    mouse_pressed = state == ElementState::Pressed;
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // Update interaction
                fanta_rust::view::interaction::update_input(cursor_x, cursor_y, mouse_pressed, false, false);
                fanta_rust::view::interaction::begin_interaction_pass();

                // Build UI
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // Check out the Mesh Gradient Background (Mode 5) - it's automatic in render!
                
                // Root container with Glass effect
                // Note: We don't set bg here to let the Mesh Gradient show through, 
                // OR we set a semi-transparent BG for the glass effect.
                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .padding(40.0)
                    .build();
                
                ui.begin(root);
                
                // Title
                ui.text("Fantasmagorie V5: Visual Revolution")
                    .font_size(32.0)
                    .fg(ColorF::white())
                    .layout_margin(10.0);
                
                // Glass Panel
                let panel = ui.column()
                    .size(500.0, 400.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.1)) // 10% white for glass base
                    .backdrop_blur(20.0) // BLUR!
                    .squircle(24.0) // SQUIRCLE! (via is_squircle below)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.3)) // Hairline
                    .elevation(10.0) // Shadow
                    .padding(20.0)
                    .build();
                    
                ui.begin(panel);
                
                ui.text("Cinematic Glass & Glow")
                    .font_size(24.0)
                    .fg(ColorF::white());

                ui.text("This panel features Mipmap Blur, Noise, and Squircle geometry.")
                    .font_size(16.0)
                    .fg(ColorF::new(0.8, 0.8, 0.8, 1.0))
                    .layout_margin(10.0);

                // Glowy Buttons
                let btn1 = ui.button("Glowing Button")
                     .size(200.0, 50.0)
                     .bg(ColorF::new(0.3, 0.6, 1.0, 1.0))
                     .radius(12.0)
                     .glow(0.8, ColorF::new(0.3, 0.6, 1.0, 1.0)) // GLOW!
                     .build();
                
                ui.text("The background is the new 'Aurora' mesh gradient.")
                    .font_size(14.0)
                    .fg(ColorF::new(0.7, 0.7, 0.7, 1.0))
                    .layout_margin(20.0);
                    
                ui.end(); // End Panel
                
                ui.end(); // End Root

                // Layout & Render
                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::view::render_ui(root, current_width as f32, current_height as f32, &mut dl);
                    
                    backend.render(&dl, current_width, current_height);
                }
                
                let _ = surface.swap_buffers(&gl_context);
            }
            _ => {}
        }
    })?;

    Ok(())
}
