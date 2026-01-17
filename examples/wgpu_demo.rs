use fanta_rust::backend::{Backend, WgpuBackend};
use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - WGPU Demo");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie Visual Revolution (WGPU)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let window = Arc::new(window);

    // WGPU Setup
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let surface = unsafe { instance.create_surface(window.clone()) }?;

    // Block on async creation
    let size = window.inner_size();
    let mut backend = pollster::block_on(WgpuBackend::new_async(
        &instance,
        &surface,
        size.width,
        size.height,
    ))
    .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut current_width = size.width;
    let mut current_height = size.height;

    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;

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
                    // WGPU resize needs surface reconfiguration usually,
                    // or just creating a new swapchain.
                    // The backend typically handles it or we re-configure surface.
                    // But WgpuBackend::new_async configured it once.
                    // Ideally we'd call backend.resize(w, h).
                    // But for now let's hope it works or just restart.
                    // Actually, WGPU requires explicit re-configure.
                    // Let's assume fixed size for this verification demo.
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
            } => {
                if button == MouseButton::Left {
                    mouse_pressed = state == ElementState::Pressed;
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                fanta_rust::view::interaction::update_input(
                    cursor_x,
                    cursor_y,
                    mouse_pressed,
                    false,
                    false,
                );
                fanta_rust::view::interaction::begin_interaction_pass();

                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // UI Definition (Same as window_demo)
                let root = ui
                    .column()
                    .size(current_width as f32, current_height as f32)
                    .padding(40.0)
                    .build();

                ui.begin(root);

                ui.text("Fantasmagorie V5: Visual Revolution (WGPU)")
                    .font_size(32.0)
                    .fg(ColorF::white())
                    .layout_margin(10.0);

                let panel = ui
                    .column()
                    .size(500.0, 400.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.1))
                    .squircle(24.0)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.3))
                    .elevation(10.0)
                    .padding(20.0)
                    .build();

                ui.begin(panel);

                ui.text("WGPU Backend Verification")
                    .font_size(24.0)
                    .fg(ColorF::white());

                ui.text("Verify: Squircle, Mesh Gradient, Glow correctly rendered.")
                    .font_size(16.0)
                    .fg(ColorF::new(0.8, 0.8, 0.8, 1.0))
                    .layout_margin(10.0);

                let btn1 = ui
                    .button("Glowing Button (WGPU)")
                    .size(200.0, 50.0)
                    .bg(ColorF::new(0.3, 0.6, 1.0, 1.0))
                    .radius(12.0)
                    .glow(0.8, ColorF::new(0.3, 0.6, 1.0, 1.0))
                    .build();

                ui.end();
                ui.end();

                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::view::render_ui(
                        root,
                        current_width as f32,
                        current_height as f32,
                        &mut dl,
                    );

                    // Use specific WGPU method
                    backend.render_to_surface(&dl, &surface, current_width, current_height);
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
