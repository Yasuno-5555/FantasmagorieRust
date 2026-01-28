use fanta_rust::prelude::*;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

#[cfg(feature = "wgpu")]
use fanta_rust::backend::WgpuBackend;
#[cfg(feature = "vulkan")]
use fanta_rust::backend::VulkanBackend;
#[cfg(feature = "dx12")]
use fanta_rust::backend::{Dx12Backend, GraphicsBackend as _};
#[cfg(feature = "opengl")]
use fanta_rust::backend::OpenGLBackend;

#[cfg(feature = "opengl")]
use glutin::{
    config::ConfigTemplateBuilder,
    context::{ContextApi, ContextAttributesBuilder, Version},
    display::GetGlDisplay,
    prelude::*,
    surface::{SurfaceAttributesBuilder, WindowSurface},
};
#[cfg(feature = "opengl")]
use glutin_winit::DisplayBuilder;
#[cfg(feature = "opengl")]
use raw_window_handle::HasRawWindowHandle;
#[cfg(feature = "opengl")]
use std::ffi::CString;
#[cfg(feature = "opengl")]
use std::num::NonZeroU32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let backend_name = if args.len() > 1 {
        args[1].to_lowercase()
    } else {
        let mut name = "none";
        #[cfg(feature = "wgpu")] { name = "wgpu"; }
        #[cfg(all(not(feature = "wgpu"), feature = "vulkan"))] { name = "vulkan"; }
        #[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), feature = "opengl"))] { name = "opengl"; }
        #[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "opengl"), feature = "dx12"))] { name = "dx12"; }
        name.to_string()
    };

    println!("🚀 Fantasmagorie Ultimate Demo - Backend: {}", backend_name);

    let event_loop = EventLoop::new()?;
    let window_builder = winit::window::WindowBuilder::new()
        .with_title(format!("Fantasmagorie Ultimate Demo ({})", backend_name))
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

    // --- WINDOW AND BACKEND INITIALIZATION ---
    let (window, mut backend): (Arc<winit::window::Window>, Box<dyn fanta_rust::backend::GraphicsBackend>) = match backend_name.as_str() {
        #[cfg(feature = "opengl")]
        "opengl" => {
            let template = ConfigTemplateBuilder::new().with_alpha_size(8);
            let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));
            let (window, gl_config) = display_builder.build(&event_loop, template, |configs| {
                configs.reduce(|accum, config| if config.num_samples() > accum.num_samples() { config } else { accum }).unwrap()
            })?;
            let window = Arc::new(window.ok_or("No window created")?);
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
            
            // Leak these to keep them alive (it's a demo)
            Box::leak(Box::new(gl_context));
            Box::leak(Box::new(surface));

            (window, Box::new(unsafe { OpenGLBackend::new(gl)? }))
        }
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            let window = Arc::new(window_builder.build(&event_loop)?);
            let size = window.inner_size();
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            });
            let surface = unsafe { instance.create_surface(window.clone()) }?;
            let b = Box::new(pollster::block_on(WgpuBackend::new_async(
                &instance,
                surface,
                size.width,
                size.height,
            )).map_err(|e| format!("WGPU creation failed: {}", e))?);
            (window, b)
        }
        #[cfg(feature = "vulkan")]
        "vulkan" => {
            let window = Arc::new(window_builder.build(&event_loop)?);
            let size = window.inner_size();
            #[cfg(target_os = "windows")]
            {
                use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
                use std::ffi::c_void;
                let (hwnd, hinstance) = match window.raw_window_handle() {
                    RawWindowHandle::Win32(handle) => {
                        (handle.hwnd as *mut c_void, handle.hinstance as *mut c_void)
                    }
                    _ => panic!("Not running on Windows/Win32"),
                };
                (window, Box::new(unsafe { VulkanBackend::new(hwnd, hinstance, size.width, size.height)? }))
            }
            #[cfg(not(target_os = "windows"))]
            panic!("Vulkan backend in this demo currently requires Windows.")
        }
        #[cfg(feature = "dx12")]
        "dx12" => {
            let window = Arc::new(window_builder.build(&event_loop)?);
            let size = window.inner_size();
            #[cfg(target_os = "windows")]
            {
                use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
                use windows::Win32::Foundation::HWND;
                let hwnd = match window.raw_window_handle() {
                    RawWindowHandle::Win32(handle) => HWND(handle.hwnd as isize),
                    _ => panic!("Not running on Windows/Win32"),
                };
                (window, Box::new(unsafe { Dx12Backend::new(hwnd, size.width, size.height)? }))
            }
            #[cfg(not(target_os = "windows"))]
            panic!("DX12 backend requires Windows.")
        }
        _ => panic!("Unsupported backend or feature not enabled: {}", backend_name),
    };
    let size = window.inner_size();

    // Initialize fonts
    fanta_rust::text::FONT_MANAGER.with(|fm| {
        fm.borrow_mut().init_fonts();
    });

    let mut current_width = size.width;
    let mut current_height = size.height;
    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;
    let start_time = std::time::Instant::now();
    let mut frame_count = 0;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width > 0 && size.height > 0 {
                    current_width = size.width;
                    current_height = size.height;
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
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let elapsed = start_time.elapsed().as_secs_f32();
                
                fanta_rust::view::interaction::update_input(cursor_x, cursor_y, mouse_pressed, false, false);
                fanta_rust::view::interaction::begin_interaction_pass();

                let arena = FrameArena::new();
                let mut ui = UIContext::new(&arena);

                // --- UI DEFINITION ---
                let root = ui.column()
                    .size(current_width as f32, current_height as f32)
                    .aurora() // Background Aurora
                    .build();

                ui.begin(root);
                
                // Centered Glassmorphism Panel
                let panel = ui.column()
                    .size(800.0, 500.0)
                    .align(Align::Center)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .squircle(32.0)
                    .backdrop_blur(15.0)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.2))
                    .elevation(30.0)
                    .padding(40.0)
                    .spacing(20.0)
                    .build();
                
                ui.begin(panel);
                
                ui.text("Fantasmagorie Ultimate Demo")
                    .font_size(48.0)
                    .fg(ColorF::white())
                    .build();
                
                let time_text = arena.alloc_str(&format!("Backend: {} | Time: {:.2}s", backend_name.to_uppercase(), elapsed));
                ui.text(time_text)
                    .font_size(18.0)
                    .fg(ColorF::new(0.7, 0.8, 1.0, 1.0))
                    .build();

                let content_row = ui.row().spacing(20.0).build();
                ui.begin(content_row);
                
                // Animated Rotating Box (SDF)
                let orbit = elapsed * 2.0;
                let offset_x = orbit.cos() * 20.0;
                let offset_y = orbit.sin() * 20.0;
                
                ui.r#box()
                    .size(150.0, 150.0)
                    .bg(ColorF::new(0.2, 0.2, 0.8, 1.0))
                    .radius(20.0 + (elapsed.sin() * 20.0))
                    .glow(1.0 + elapsed.sin().abs() * 2.0, ColorF::new(0.4, 0.4, 1.0, 1.0))
                    .margin(offset_x) 
                    .build();

                let button_col = ui.column().spacing(10.0).build();
                ui.begin(button_col);
                
                if ui.button("Standard Button").size(200.0, 50.0).radius(10.0).clicked() {
                    println!("Standard Click!");
                }

                if ui.button("Glowing Squircle")
                    .size(200.0, 50.0)
                    .squircle(15.0)
                    .bg(ColorF::new(0.8, 0.2, 0.2, 1.0))
                    .hover(ColorF::new(1.0, 0.3, 0.3, 1.0))
                    .glow(2.0, ColorF::new(0.8, 0.2, 0.2, 0.8))
                    .clicked() {
                    println!("Squircle Click!");
                }

                if ui.button("Transparent Border")
                    .size(200.0, 50.0)
                    .bg(ColorF::transparent())
                    .border(2.0, ColorF::white())
                    .radius(25.0)
                    .clicked() {
                    println!("Border Click!");
                }
                
                ui.end(); // end button_col
                ui.end(); // end content_row

                ui.text("Verification Targets:")
                    .font_size(20.0)
                    .fg(ColorF::new(0.6, 1.0, 0.6, 1.0))
                    .build();
                
                ui.text("- [x] Aurora Pixel Shader Parity")
                    .font_size(14.0).fg(ColorF::white()).build();
                ui.text("- [x] Backdrop Blur (LOD Mipmap) Sampling")
                    .font_size(14.0).fg(ColorF::white()).build();
                ui.text("- [x] Squircle Continuity (SDF)")
                    .font_size(14.0).fg(ColorF::white()).build();
                ui.text("- [x] Real-time Constant Updates")
                    .font_size(14.0).fg(ColorF::white()).build();

                ui.end(); // end panel
                ui.end(); // end root

                if let Some(root_node) = ui.root() {
                    let mut dl = fanta_rust::draw::DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::renderer::render_ui(root_node, current_width as f32, current_height as f32, &mut dl, &mut fm);
                        
                        if fm.texture_dirty {
                            backend.update_font_texture(fm.atlas.width, fm.atlas.height, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                    });
                    
                    backend.render(&dl, current_width, current_height);
                }

                frame_count += 1;
                if frame_count == 100 {
                    let filename = format!("ultimate_screenshot_{}.png", backend_name);
                    backend.capture_screenshot(&filename);
                    println!("✨ Screenshot captured: {}", filename);
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
