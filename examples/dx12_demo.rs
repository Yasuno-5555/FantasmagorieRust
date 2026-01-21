use fanta_rust::backend::{Backend, Dx12Backend};
use fanta_rust::prelude::*;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;
use windows::Win32::Foundation::HWND;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - DirectX 12 Demo");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie Visual Revolution (DX12)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let window = Arc::new(window);

    // Get HWND
    let hwnd = match window.raw_window_handle() {
        RawWindowHandle::Win32(handle) => HWND(handle.hwnd as isize),
        _ => panic!("Not running on Windows/Win32"),
    };

    let size = window.inner_size();

    // Initialize DX12 Backend
    let mut backend = unsafe { Dx12Backend::new(hwnd, size.width, size.height)? };

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
                    // DX12 resize handling (skipped for demo simplicity, might crash or stretch)
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

                // UI Definition
                let root = ui
                    .column()
                    .size(current_width as f32, current_height as f32)
                    .padding(40.0)
                    .build();

                ui.begin(root);

                ui.text("Fantasmagorie V5: Visual Revolution (DX12)")
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

                ui.text("DirectX 12 Backend Verification")
                    .font_size(24.0)
                    .fg(ColorF::white());

                ui.text("HLSL Shader Logic: Squircle, Glow, Mesh Gradient via Root Constants.")
                    .font_size(16.0)
                    .fg(ColorF::new(0.8, 0.8, 0.8, 1.0))
                    .layout_margin(10.0);

                let btn1 = ui
                    .button("Glowing DX12 Button")
                    .size(200.0, 50.0)
                    .bg(ColorF::new(1.0, 0.4, 0.4, 1.0))
                    .radius(12.0)
                    .glow(1.0, ColorF::new(1.0, 0.4, 0.4, 1.0))
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
                    backend.render(&dl, current_width, current_height);
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
