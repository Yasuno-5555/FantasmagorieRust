use fanta_rust::prelude::*;
use fanta_rust::backend::vulkan::backend::VulkanBackend;
use fanta_rust::tracea::runtime::manager::RuntimeManager;
use fanta_rust::text::FontManager;
use fanta_rust::backend::GraphicsBackend; // Trait for .render()
use winit::{
    event::{Event, WindowEvent, ElementState, KeyEvent},
    keyboard::{Key, NamedKey, KeyCode, PhysicalKey},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    dpi::LogicalSize,
};
use std::sync::Arc;

fn main() {
    println!("Fantasmagorie - Phase 2: K5 JFA Lighting Demo");
    println!("=============================================");

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie: The Light (JFA SDF)")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    // Initialize Vulkan Backend (Unsafe because it requires raw window handles)
    println!("Initializing Vulkan Backend...");
    let mut renderer = unsafe {
        use raw_window_handle::{HasRawWindowHandle, HasRawDisplayHandle};
        let raw_window = window.raw_window_handle();
        
        match raw_window {
            raw_window_handle::RawWindowHandle::Win32(handle) => {
                 VulkanBackend::new(
                    handle.hwnd as *mut _,
                    handle.hinstance as *mut _,
                    1280,
                    720,
                ).expect("Failed to create Vulkan Backend")
            }
             _ => panic!("Only Windows supported for this direct backend test"),
        }
    };

    println!("Vulkan Backend Initialized.");
     let context = renderer.ctx.clone();
     use fanta_rust::tracea::runtime::manager::DeviceBackend;
    let runtime = RuntimeManager::init_vulkan(context).expect("Tracea Init Failed");
    println!("Tracea Runtime Ready: {:?}", runtime.get_device(DeviceBackend::Vulkan).unwrap().arch);

    // Initialize Font Manager
    let mut fonts = FontManager::new();
    fonts.init_fonts(); // Loads system fonts

    use fanta_rust::audio::AudioManager;
    let audio_manager = AudioManager::new();
    if let Err(e) = &audio_manager {
        println!("Warning: Audio init failed: {}", e);
    } else {
        println!("Audio System: ONLINE");
    }

    // State
    let mut intensity = 2.0;
    let mut decay = 0.05;
    let mut radius = 64.0;
    let mut exposure = 1.0;
    let mut gamma = 2.2;
    let mut audio_gain = 1.0;
    let mut fog_density = 0.05;

    // Winit 0.29 run loop
    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::KeyboardInput { event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, .. } => {
                    if let Key::Named(NamedKey::Escape) = logical_key {
                         target.exit();
                    }
                }
                // RedrawRequested is a WindowEvent in winit 0.29
                WindowEvent::RedrawRequested => {
                    // Per-frame UI Context
                    let arena = FrameArena::new();
                    let mut ui = UIContext::new(&arena);

                    // 1. Audio Update (Thread Safe)
                    if let Ok(am) = &audio_manager {
                        let spectrum = am.get_spectrum();
                        renderer.update_audio_data(&spectrum);
                    }

                    // 2. JFA Compute
                    renderer.dispatch_jfa(1280, 720, intensity, decay, radius);

                    // 3. UI Generation
                    let root = ui.column().size(1280.0, 720.0).padding(20.0).build();
                    ui.begin(root);
                    use fanta_rust::core::ColorF;
                    ui.text("K5 JFA Lighting Demo").font_size(32.0).fg(ColorF::white());
                    ui.text("Iterative Jump Flooding active.").font_size(16.0).fg(ColorF::new(0.7, 0.7, 0.7, 1.0));
                    
                    ui.text("Lighting Parameters").font_size(24.0).fg(ColorF::new(0.9, 0.9, 0.9, 1.0));
                    
                    ui.row().padding(5.0).build();
                    ui.text("Intensity:").font_size(16.0);
                    ui.value_dragger(&mut intensity, 0.0, 100.0).step(0.1).build();
                    ui.end();

                    ui.row().padding(5.0).build();
                    ui.text("Decay:").font_size(16.0);
                    ui.value_dragger(&mut decay, 0.001, 1.0).step(0.001).build();
                    ui.end();

                    ui.row().padding(5.0).build();
                    ui.text("Seed Radius:").font_size(16.0);
                    ui.value_dragger(&mut radius, 1.0, 200.0).step(1.0).build();

                    ui.end();

                    ui.text("Tone Mapping").font_size(24.0).fg(ColorF::new(0.9, 0.9, 0.9, 1.0));
                    
                    ui.row().padding(5.0).build();
                    ui.text("Exposure:").font_size(16.0);
                    ui.value_dragger(&mut exposure, 0.1, 5.0).step(0.1).build();
                    ui.end();

                    ui.row().padding(5.0).build();
                    ui.text("Gamma:").font_size(16.0);
                    ui.value_dragger(&mut gamma, 1.0, 3.0).step(0.1).build();
                    ui.end();

                    ui.text("Audio Reactivity").font_size(24.0).fg(ColorF::new(0.9, 0.9, 0.9, 1.0));
                    ui.row().padding(5.0).build();
                    ui.text("Gain:").font_size(16.0);
                    ui.value_dragger(&mut audio_gain, 0.0, 5.0).step(0.1).build();
                    ui.end();

                    ui.text("Atmosphere").font_size(24.0).fg(ColorF::new(0.9, 0.9, 0.9, 1.0));
                    ui.row().padding(5.0).build();
                    ui.text("Fog Density:").font_size(16.0);
                    ui.value_dragger(&mut fog_density, 0.0, 0.5).step(0.001).build();
                    ui.end();

                    use fanta_rust::draw::DrawList;
                    let mut dl = DrawList::new();
                    if let Some(root) = ui.root() {
                        // Pass fonts to render_ui
                        fanta_rust::view::render_ui(root, 1280.0, 720.0, &mut dl, &mut fonts);
                    }
                    
                    // Update K4 Params
                    renderer.set_k4_params(exposure, gamma, fog_density);
                    renderer.set_audio_params(audio_gain);

                    // Invoke Renderer (Trait method)
                    renderer.render(&dl, 1280, 720);
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            },
            _ => (),
        }
    }).unwrap();
}
