
use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::view::header::Align;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit::raw_window_handle::{HasWindowHandle, HasDisplayHandle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - WGPU Cinematic Rich Demo");
    println!("Showcasing: Time-Driven Architecture");
    println!("Controls: [SPACE] Pause/Play, [LEFT/RIGHT] Seek +/- 1s");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie: Time-Driven Cinematic Demo")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let window = Arc::new(window);
    // WGPU Setup (Backend handles instance/surface creation)
    let size = window.inner_size();
    let mut backend = WgpuBackend::new_async(
        &*window,
        size.width,
        size.height,
    )
    .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut current_width = size.width;
    let mut current_height = size.height;
    let mut frame_count = 0;

    let mut cursor_x = 0.0;
    let mut cursor_y = 0.0;
    let mut mouse_pressed = false;
    
    // --- The Constitution: MasterClock ---
    let mut clock = fanta_rust::core::time::MasterClock::new();
    let mut last_frame = std::time::Instant::now();

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
                    backend.surface_config.width = size.width;
                    backend.surface_config.height = size.height;
                    backend.surface.configure(&backend.device, &backend.surface_config);
                    current_width = size.width;
                    current_height = size.height;
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
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event, .. },
                ..
            } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key {
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                            clock.toggle_pause();
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowLeft) => {
                            clock.seek(clock.time().seconds - 1.0);
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowRight) => {
                            clock.seek(clock.time().seconds + 1.0);
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_frame).as_secs_f64();
                last_frame = now;
                
                clock.tick(dt);
                
                let time = clock.time().seconds as f32;
                
                // --- Simulate Audio Reactivity (Driven by Absolute Time) ---
                let bass = (time * 2.0).sin() * 0.5 + 0.5;
                let mid = (time * 3.5).cos() * 0.3 + 0.3;
                let _high = (time * 7.0).sin() * 0.2 + 0.2;
                let spectrum = vec![bass, mid, _high, 0.0];
                backend.update_audio_data(&spectrum);

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

                // --- Cinematic Visual Layout ---
                let root = ui
                    .column()
                    .size(current_width as f32, current_height as f32)
                    .aurora() 
                    .align(Align::Center) 
                    .build();

                ui.begin(root);

                // Floating "Glass" Header
                let header = ui.column()
                    .size(800.0, 80.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .backdrop_blur(30.0)
                    .squircle(40.0)
                    .border(1.0, ColorF::new(1.0, 1.0, 1.0, 0.2))
                    .align(Align::Center)
                    .margin(20.0)
                    .build();
                
                ui.begin(header);
                ui.text("TIME-DRIVEN CINEMATIC STACK")
                    .font_size(36.0)
                    .fg(ColorF::new(1.0, 1.0, 1.0, 0.9))
                    .build();
                ui.end();

                // Main Content Row
                let content_row = ui.row()
                    .size(1000.0, 400.0)
                    .align(Align::Center)
                    .build();

                ui.begin(content_row);
                        
                // Left Panel: Stats
                let left_panel = ui.column()
                    .size(300.0, 350.0)
                    .bg(ColorF::new(0.0, 0.0, 0.0, 0.4))
                    .backdrop_blur(20.0)
                    .radius(20.0)
                    .padding(20.0)
                    .margin(10.0)
                    .build();
                
                ui.begin(left_panel);
                ui.text("SYSTEM STATUS")
                    .font_size(20.0)
                    .fg(ColorF::new(0.0, 1.0, 0.8, 1.0))
                    .build();
                
                let time_str = format!("TIME: {:.2}s", time);
                ui.text(&time_str)
                    .font_size(14.0)
                    .fg(ColorF::white())
                    .build();
                
                let state_str = if clock.is_paused() { "PAUSED" } else { "PLAYING" };
                let status_display = format!("STATE: {}", state_str);
                 ui.text(&status_display)
                    .font_size(14.0)
                    .fg(if clock.is_paused() { ColorF::new(1.0, 0.5, 0.0, 1.0) } else { ColorF::new(0.0, 1.0, 0.0, 1.0) })
                    .build();

                ui.text("BACKEND: WGPU")
                    .font_size(14.0)
                    .fg(ColorF::white())
                    .build();

                // Pulsing Audio Visualizer
                let vis_row = ui.row().size(260.0, 100.0).margin(20.0).build();
                ui.begin(vis_row);
                for i in 0..5 {
                    let h = 20.0 + (time * (2.0 + i as f32)).sin().abs() * 60.0;
                    ui.column()
                        .size(30.0, h)
                        .bg(ColorF::new(0.5, 0.2, 1.0, 0.8))
                        .glow(0.5, ColorF::new(0.5, 0.2, 1.0, 1.0))
                        .radius(5.0)
                        .margin(5.0)
                        .build();
                }
                ui.end();
                ui.end(); // left_panel

                // Center: Interactive Orb
                let center_col = ui.column()
                    .size(350.0, 350.0)
                    .align(Align::Center)
                    .build();
                
                ui.begin(center_col);
                let orb_size = 200.0 + bass * 50.0;
                ui.column()
                    .size(orb_size, orb_size)
                    .bg(ColorF::new(1.0, 0.2, 0.5, 0.3))
                    .squircle(orb_size / 2.0)
                    .glow(1.2 + mid, ColorF::new(1.0, 0.1, 0.4, 1.0))
                    .build();
                ui.end();

                // Right Panel: Feature Checklist
                let right_panel = ui.column()
                    .size(300.0, 350.0)
                    .bg(ColorF::new(1.0, 1.0, 1.0, 0.05))
                    .backdrop_blur(40.0)
                    .radius(20.0)
                    .padding(20.0)
                    .margin(10.0)
                    .build();
                
                ui.begin(right_panel);
                ui.text("FEATURES")
                    .font_size(20.0)
                    .fg(ColorF::new(1.0, 0.8, 0.0, 1.0))
                    .build();
                
                let features = ["Master Clock", "Seek/Pause", "f(t) Animation", "JFA Blur", "Audio Reactive"];
                let formatted_features: Vec<String> = features.iter().map(|f| format!("[+] {}", f)).collect();
                for f_text in &formatted_features {
                    let f_row = ui.row().size(260.0, 30.0).build();
                    ui.begin(f_row);
                    ui.text(f_text)
                        .font_size(16.0)
                        .fg(ColorF::white())
                        .build();
                    ui.end();
                }
                ui.end(); // right_panel

                ui.end(); // content_row

                // Bottom Footer
                ui.text("Press SPACE to Pause, Arrows to Seek - Fantasmagorie Engine v5.1")
                    .font_size(14.0)
                    .fg(ColorF::new(0.6, 0.6, 0.6, 1.0))
                    .margin(20.0)
                    .build();

                ui.end(); // root

                if let Some(root) = ui.root() {
                    let mut dl = fanta_rust::DrawList::new();
                    fanta_rust::text::FONT_MANAGER.with(|fm| {
                        let mut fm = fm.borrow_mut();
                        fanta_rust::view::render_ui(
                            root,
                            current_width as f32,
                            current_height as f32,
                            &mut dl,
                            &mut fm,
                        );
                        if fm.texture_dirty {
                            backend.update_font_texture(fm.atlas.width as u32, fm.atlas.height as u32, &fm.atlas.texture_data);
                            fm.texture_dirty = false;
                        }
                    });

                    // --- NEW: THE COMPOSER (Phase 2) ---
                    use fanta_rust::renderer::layer::{Composition, Layer, BlendMode, TimeProperty};
                    let mut composition = Composition::new();

                    // Background Layer (Just the Aurora)
                    let mut bg_layer = Layer::new("Background");
                    if let fanta_rust::renderer::layer::LayerSource::DrawList(dl) = &mut bg_layer.source {
                        dl.add_aurora(Vec2::ZERO, Vec2::new(current_width as f32, current_height as f32));
                    }
                    composition.add_layer(bg_layer);

                    // UI Layer (Everything else)
                    let mut ui_layer = Layer::new("UI");
                    ui_layer.source = fanta_rust::renderer::layer::LayerSource::DrawList(dl);
                    ui_layer.blend_mode = BlendMode::Alpha;
                    // Phase 3: Dynamic Property!
                    ui_layer.opacity = TimeProperty::Dynamic(Arc::new(|t| {
                        0.8 + (t * 0.5).sin() as f32 * 0.1
                    }));
                    composition.add_layer(ui_layer);

                    // Phase 3: Glitch Layer (Shader Slot)
                    let glitch_shader = r#"
                        @group(0) @binding(0) var t_input: texture_2d<f32>;
                        @group(0) @binding(1) var s_input: sampler;

                        struct PostProcessUniforms {
                            threshold: f32,
                            direction: vec2<f32>, 
                            intensity: f32,
                            _pad: array<vec4<f32>, 62>, 
                        };

                        @group(0) @binding(2) var<uniform> uniforms: PostProcessUniforms;

                        struct VertexOutput {
                            @builtin(position) position: vec4<f32>,
                            @location(0) uv: vec2<f32>,
                        };

                        @fragment
                        fn fs_glitch(in: VertexOutput) -> @location(0) vec4<f32> {
                            let t = uniforms.threshold; // time passed as first float
                            var uv = in.uv;
                            let intensity = uniforms.intensity; // first param passed at offset 16
                            
                            // Random horizontal offset based on time
                            let scanline = step(0.95, sin(uv.y * 80.0 + t * 40.0));
                            uv.x = uv.x + scanline * 0.04 * sin(t * 100.0) * intensity;
                            
                            let color = textureSample(t_input, s_input, uv);
                            
                            // Chromatic aberration
                            let r = textureSample(t_input, s_input, uv + vec2<f32>(0.005 * sin(t * 13.0) * intensity, 0.0)).r;
                            let g = color.g;
                            let b = textureSample(t_input, s_input, uv - vec2<f32>(0.005 * cos(t * 11.0) * intensity, 0.0)).b;
                            
                            return vec4<f32>(r, g, b, color.a);
                        }
                    "#;
                    let mut glitch_layer = Layer::with_shader("Glitch", glitch_shader);
                     let mut params = std::collections::HashMap::new();
                     params.insert("intensity".to_string(), 1.0f32);
                     glitch_layer.source = fanta_rust::renderer::layer::LayerSource::Shader(fanta_rust::renderer::layer::ShaderSlot {
                         source: glitch_shader.to_string(),
                         entry_point: "fs_glitch".to_string(),
                         parameters: params,
                     });
                    // Only apply glitch occasionally
                    glitch_layer.opacity = TimeProperty::Dynamic(Arc::new(|t| {
                        if (t * 2.0).sin() > 0.8 { 0.5 } else { 0.0 }
                    }));
                    composition.add_layer(glitch_layer);

                    // Render with the new Cinematic Compositing Pipeline
                    backend.render_composition(&composition, current_width, current_height, time as f64);
                    
                    frame_count += 1;
                    if frame_count == 120 {
                        backend.capture_screenshot("cinematic_glitch.png");
                    }
                    if frame_count > 130 {
                        elwt.exit();
                    }
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
