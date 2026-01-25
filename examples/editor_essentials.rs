//! Phase 3: Editor Essentials Demo
//! Showcases Interactive Splitters, Drag & Drop, and Simple Reactivity correctly.

use fanta_rust::backend::{GraphicsBackend, OpenGLBackend};
use fanta_rust::prelude::*;

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
    println!("Fantasmagorie Phase 3 - Editor Essentials Demo");

    let event_loop = EventLoop::new()?;
    let window_builder = WindowBuilder::new()
        .with_title("Fantasmagorie Phase 3: Editor Essentials")
        .with_inner_size(LogicalSize::new(1280, 720));

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
    let mut split_ratio = 0.3f32;
    let mut dropped_files = Vec::new();

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
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                fanta_rust::view::interaction::begin_interaction_pass();
                
                // DRAG & DROP DETECTION
                let current_dropped = fanta_rust::view::interaction::get_dropped_files();
                if !current_dropped.is_empty() {
                    dropped_files = current_dropped;
                }

                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // --- UI CONSTRUCTION ---
                let root = ui.column().size(current_width as f32, current_height as f32).build();
                ui.begin(root);

                // Header
                ui.row().height(60.0).bg(ColorF::new(0.04, 0.04, 0.06, 1.0)).padding(15.0);
                ui.text("FANTASMAGORIE Editor V3").font_size(24.0).fg(ColorF::new(0.0, 1.0, 0.8, 1.0));
                ui.end();

                // Main Panel with Splitter
                let main = ui.row().flex_grow(1.0).build();
                ui.begin(main);

                let sp_builder = ui.splitter().ratio(split_ratio).horizontal();
                let sp_changed = sp_builder.changed();
                let splitter = sp_builder.build();
                
                ui.begin(splitter);
                
                // Sidebar
                let sidebar = ui.column().padding(20.0).bg(ColorF::new(0.08, 0.08, 0.1, 1.0)).build();
                ui.begin(sidebar);
                ui.text("PROJECT FILES").font_size(14.0).fg(ColorF::new(0.5, 0.5, 0.5, 1.0));
                ui.text("Drop files here!").font_size(12.0).fg(ColorF::new(0.3, 0.6, 1.0, 0.8));
                
                for path in &dropped_files {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("Unknown");
                    ui.text(ui.arena.alloc_str(&format!("📄 {}", name))).font_size(14.0).fg(ColorF::white()).margin(4.0);
                }
                
                ui.end();

                // Editor
                let editor = ui.column().padding(40.0).bg(ColorF::new(0.12, 0.12, 0.15, 1.0)).build();
                ui.begin(editor);
                ui.text("INTERACTIVE PANELS").font_size(18.0).fg(ColorF::white());
                
                if sp_changed {
                     ui.text("DRAGGING SPLITTER...").font_size(12.0).fg(ColorF::new(1.0, 0.8, 0.0, 1.0)).margin(10.0);
                }
                
                // Update local ratio (Simple Reactivity)
                split_ratio = splitter.ratio.get();
                
                if ui.button("RESET VIEW").size(150.0, 40.0).margin(20.0).clicked() {
                    split_ratio = 0.3;
                    dropped_files.clear();
                }
                
                ui.end();

                ui.end(); // End Splitter
                ui.end(); // End Main Panel
                ui.end(); // End Root

                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        fanta_rust::view::render_ui(root, current_width as f32, current_height as f32, &mut dl, &mut fm.borrow_mut());
                    });
                    
                    if let Some(cursor) = fanta_rust::view::interaction::get_requested_cursor() {
                        if let Some(icon) = cursor { window.set_cursor_icon(icon); }
                    }

                    backend.render(&dl, current_width, current_height);
                }
                let _ = surface.swap_buffers(&gl_context);
            }
            Event::WindowEvent { event: ref win_event, .. } => {
                // UNIFIED EVENT HANDLING
                fanta_rust::view::interaction::handle_event(win_event);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
