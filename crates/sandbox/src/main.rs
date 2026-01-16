use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use glutin::context::NotCurrentGlContext;
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use raw_window_handle::HasRawWindowHandle;
use std::num::NonZeroU32;

use fantasmagorie::core::context::UIContext;
use fantasmagorie::render::renderer::Renderer;
use fantasmagorie::widgets::window::WindowBuilder as FWindow;
use fantasmagorie::widgets::button::ButtonBuilder;
use fantasmagorie::widgets::traits::WidgetBuilder;
use fantasmagorie::layout::flex::LayoutEngine;

// Simplified setup for brevity
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window_builder = WindowBuilder::new().with_title("Fantasmagorie Rust");
    // NOTE: Real glutin setup is verbose. I'll use a simplified flow or assume user knows helper.
    // For this port, verifying compilation of usage is key.
    
    // UI Context
    let mut ui = UIContext::new();
    
    println!("Starting Fantasmagorie Sandbox...");

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            },
            Event::AboutToWait => {
                // Update
                ui.begin_frame();
                
                // Build UI
                {
                    let mut win = FWindow::new(&mut ui, "Demo Window");
                    win.width(400.0).height(300.0);
                    
                    // Button
                    let btn = ButtonBuilder::new(&mut ui, "Click Me!").primary();
                    if btn.clicked() {
                        println!("Clicked!");
                    }
                    // ButtonBuilder drops here, calling end_node
                    
                    // Window drops here, calling end_node
                }
                
                // Layout
                LayoutEngine::solve(&mut ui.store, ui.root_id, 800.0, 600.0); // Dummy size
                
                ui.end_frame();
            },
            _ => ()
        }
    }).unwrap();
}
