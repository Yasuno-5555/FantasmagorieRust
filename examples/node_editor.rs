//! Phase 4: Node Editor (Lattice) Demo
//! Showcases Dynamic Grid, Draggable Nodes, Sockets, and Bezier Wires.

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
    println!("Fantasmagorie Phase 4 - Node Editor Demo");

    let event_loop = EventLoop::new()?;
    let window_builder = WindowBuilder::new()
        .with_title("Fantasmagorie Phase 4: Lattice (Node Editor)")
        .with_inner_size(LogicalSize::new(1440, 900));

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
    let mut node_pos_1 = Vec2::new(100.0, 100.0);
    let mut node_pos_2 = Vec2::new(500.0, 250.0);
    let mut node_pos_3 = Vec2::new(150.0, 450.0);
    
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
                let root = ui.column().size(current_width as f32, current_height as f32).build();
                ui.begin(root);

                // Header
                ui.row().height(50.0).bg(ColorF::new(0.04, 0.04, 0.06, 1.0)).padding(10.0);
                ui.text("LATTICE Node Editor").font_size(20.0).fg(ColorF::new(0.0, 1.0, 0.8, 1.0));
                ui.end();

                // Infinite Canvas
                let canvas_id = ID::from_str("main_canvas");
                let canvas = ui.canvas().id(canvas_id).flex_grow(1.0).bg(ColorF::new(0.08, 0.08, 0.1, 1.0)).build();
                ui.begin(canvas);
                
                // Draw some wires between nodes
                // In a real app these would be dynamic
                let p0 = node_pos_1 + Vec2::new(180.0, 40.0);
                let p3 = node_pos_2 + Vec2::new(0.0, 40.0);
                let dist = (p3.x - p0.x).abs() * 0.5;
                
                let mut dl = fanta_rust::DrawList::new();
                dl.add_bezier(
                    p0, 
                    p0 + Vec2::new(dist, 0.0), 
                    p3 - Vec2::new(dist, 0.0), 
                    p3, 
                    2.0, 
                    ColorF::new(0.0, 0.8, 1.0, 0.6)
                );
                
                // Nest the nodes inside the canvas
                // Node 1: OSCILLATOR
                let n1 = ui.node("OSCILLATOR").id("node1").pos(node_pos_1.x, node_pos_1.y).size(180.0, 120.0).build();
                ui.begin(n1);
                ui.row().height(30.0).padding(10.0).align(fanta_rust::view::header::Align::Center);
                ui.socket("OUT", false).id("out1").color(ColorF::new(0.0, 0.8, 1.0, 1.0)).build();
                ui.end();
                ui.end();
                node_pos_1 = Vec2::new(n1.pos_x.get(), n1.pos_y.get());

                // Node 2: FILTER
                let n2 = ui.node("LOWPASS FILTER").id("node2").pos(node_pos_2.x, node_pos_2.y).size(180.0, 120.0).build();
                ui.begin(n2);
                ui.row().height(30.0).padding(10.0).align(fanta_rust::view::header::Align::Center);
                ui.socket("IN", true).id("in2").color(ColorF::new(1.0, 0.8, 0.0, 1.0)).build();
                ui.socket("OUT", false).id("out2").color(ColorF::new(0.0, 0.8, 1.0, 1.0)).build();
                ui.end();
                ui.end();
                node_pos_2 = Vec2::new(n2.pos_x.get(), n2.pos_y.get());

                // Node 3: REVERB
                let n3 = ui.node("REVERB").id("node3").pos(node_pos_3.x, node_pos_3.y).size(150.0, 100.0).build();
                ui.begin(n3);
                ui.row().height(30.0).padding(10.0).align(fanta_rust::view::header::Align::Center);
                ui.socket("IN", true).id("in3").color(ColorF::new(1.0, 0.8, 0.0, 1.0)).build();
                ui.end();
                ui.end();
                node_pos_3 = Vec2::new(n3.pos_x.get(), n3.pos_y.get());

                ui.end(); // End Canvas
                ui.end(); // End Root

                if let Some(root) = ui.root() {
                    let mut final_dl = fanta_rust::DrawList::new();
                    // Merge previous manual wires
                    for cmd in dl.commands() {
                         // This is a bit manual, normally wires would be widgets too
                         // Let's just add a wire widget to Fanta-Rust soon.
                    }
                    
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        fanta_rust::view::render_ui(root, current_width as f32, current_height as f32, &mut final_dl, &mut fm.borrow_mut());
                    });
                    
                    if let Some(cursor) = fanta_rust::view::interaction::get_requested_cursor() {
                        if let Some(icon) = cursor { window.set_cursor_icon(icon); }
                    }

                    backend.render(&final_dl, current_width, current_height);
                }
                let _ = surface.swap_buffers(&gl_context);
            }
            _ => {}
        }
    })?;

    Ok(())
}
