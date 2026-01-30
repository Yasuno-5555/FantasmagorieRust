//! minimal.rs - Minimal "Hello World" example for Fantasmagorie
//! Simple button and text demo with basic event handling.

use fanta_rust::prelude::*;
use fanta_rust::backend::{OpenGLBackend, GraphicsBackend};

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
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
    println!("Fantasmagorie Minimal Demo");

    let event_loop = EventLoop::new()?;
    let window_builder = WindowBuilder::new()
        .with_title("Fantasmagorie: Hello World")
        .with_inner_size(LogicalSize::new(800, 600));

    let template = ConfigTemplateBuilder::new().with_alpha_size(8);
    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

    let (window, gl_config) = display_builder.build(&event_loop, template, |configs| {
        configs.reduce(|accum, config| if config.num_samples() > accum.num_samples() { config } else { accum }).unwrap()
    })?;

    let window = window.ok_or("No window created")?;
    let raw_window_handle = window.raw_window_handle();
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(Some(raw_window_handle));

    let gl_display = gl_config.display();
    let not_current_gl_context = unsafe { gl_display.create_context(&gl_config, &context_attributes)? };
    let size = window.inner_size();
    let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        raw_window_handle,
        NonZeroU32::new(size.width).unwrap(),
        NonZeroU32::new(size.height).unwrap(),
    );
    let surface = unsafe { gl_display.create_window_surface(&gl_config, &surface_attributes)? };
    let gl_context = not_current_gl_context.make_current(&surface)?;
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            let c_str = CString::new(s).unwrap();
            gl_display.get_proc_address(&c_str) as *const _
        })
    };

    let mut backend = unsafe { OpenGLBackend::new(gl)? };
    let mut current_width = size.width;
    let mut current_height = size.height;

    // --- APP STATE ---
    let mut counter = 0;
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width > 0 && size.height > 0 {
                    current_width = size.width;
                    current_height = size.height;
                    surface.resize(&gl_context, NonZeroU32::new(size.width).unwrap(), NonZeroU32::new(size.height).unwrap());
                }
            }
            Event::WindowEvent { event: ref win_event, .. } => {
                fanta_rust::view::interaction::handle_event(win_event);
            }
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                fanta_rust::view::interaction::begin_interaction_pass();
                
                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // --- UI CONSTRUCTION ---
                let bg_color = ui.theme.bg;
                
                // Root Column
                let root_header = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .bg(bg_color)
                    .align(Align::Center)
                    .padding(50.0)
                    .build();
                
                ui.begin(root_header);
                    ui.text("Welcome to Fantasmagorie").font_size(32.0).build();
                    
                    ui.r#box().height(20.0).build(); // Spacer
                    
                    let btn = ui.button("Click Me!").size(200.0, 50.0);
                    if btn.clicked() {
                        counter += 1;
                        println!("Button clicked! Count: {}", counter);
                    }
                    btn.build();
                    
                    ui.r#box().height(20.0).build(); // Spacer
                    
                    let counter_text = ui.arena.alloc_str(&format!("Counter: {}", counter));
                    ui.text(counter_text).font_size(24.0).build();
                    
                    // Theme Switcher Example
                    let row_header = ui.row().height(60.0).spacing(10.0).padding(10.0).build();
                    ui.begin(row_header);
                        let b1 = ui.button("Cyberpunk").size(100.0, 30.0);
                        if b1.clicked() {
                           println!("Switching to Cyberpunk");
                        }
                        b1.build();

                        let b2 = ui.button("Zen").size(100.0, 30.0);
                        if b2.clicked() {
                           println!("Switching to Zen");
                        }
                        b2.build();
                    ui.end();

                ui.end();

                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        fanta_rust::view::render_ui(root, current_width as f32, current_height as f32, &mut dl, &mut fm.borrow_mut());
                    });
                    backend.render(&dl, current_width, current_height);
                }
                let _ = surface.swap_buffers(&gl_context);
            }
            _ => {}
        }
    })?;

    Ok(())
}
