use fanta_rust::prelude::*;
use fanta_rust::audio::{AudioEngine, SoundType};
use fanta_rust::backend::WgpuBackend;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie - Ghost Inspector Demo")
        .with_inner_size(LogicalSize::new(1280, 800));
    
    let window = Arc::new(event_loop.create_window(window)?);
    let backend = WgpuBackend::new_async(window.clone(), 1280, 800, 1.0)?;
    let mut renderer = Renderer::new(Box::new(backend), EngineConfig::cinematic());
    
    let mut audio = AudioEngine::new();
    let mut inspector = fanta_rust::devtools::Inspector::new();
    inspector.show();

    let mut arena = FrameArena::new();
    let _start_time = std::time::Instant::now();

    // Simulation state
    let mut sine_freq = 440.0;
    let mut volume = 0.5;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                arena.reset();
                let mut ctx = UIContext::new(&arena);
                
                // 1. Update Audio Simulation
                audio.play_sound(1, SoundType::Sine(sine_freq), Vec2::ZERO, volume, 1.0, true);
                
                // 2. Poll Audio Telemetry
                while let Some(telemetry) = audio.pop_telemetry() {
                    inspector.update_audio_telemetry(telemetry);
                }

                // 3. Build UI
                ctx.column().width(1280.0).height(800.0).padding(20.0).build();
                {
                    ctx.row().height(40.0).build();
                    ctx.text("Ghost Inspector - Real-time Audio Telemetry").font_size(24.0).build();
                    ctx.end();

                    ctx.row().flex_grow(1.0).spacing(20.0).build();
                    {
                        // Left Column: Controls
                        ctx.column().width(300.0).spacing(10.0).build();
                        ctx.text("Audio Controls").font_size(18.0).build();
                        
                        ctx.row().height(40.0).build();
                        ctx.text("Frequency").width(80.0).build();
                        ctx.value_dragger(&mut sine_freq, 20.0, 2000.0).flex_grow(1.0).build();
                        ctx.end();

                        ctx.row().height(40.0).build();
                        ctx.text("Volume").width(80.0).build();
                        ctx.value_dragger(&mut volume, 0.0, 1.0).flex_grow(1.0).build();
                        ctx.end();
                        
                        ctx.end();

                        // Right Column: Telemetry
                        ctx.column().flex_grow(1.0).spacing(10.0).build();
                        ctx.text("Performance Telemetry").font_size(18.0).build();
                        
                        if let Some(latest) = inspector.audio_history.back() {
                            ctx.row().height(30.0).build();
                            let cpu_str = ctx.arena.alloc_str(&format!("CPU Usage: {:.2}%", latest.cpu_usage * 100.0));
                            ctx.text(cpu_str).build();
                            ctx.end();
                            
                            ctx.row().height(30.0).build();
                            let voice_str = ctx.arena.alloc_str(&format!("Active Voices: {}", latest.active_voices));
                            ctx.text(voice_str).build();
                            ctx.end();
                        }

                        // Sparkline (using Plot widget)
                        let cpu_data: Vec<f32> = inspector.audio_history.iter().map(|t| t.cpu_usage * 100.0).collect();
                        if !cpu_data.is_empty() {
                            ctx.text("Audio Thread CPU Load (%)").build();
                            // Note: PlotBuilder needs a reference to the data. 
                            // Since we are in immediate mode, we must ensure data lives long enough.
                            // Here we use the arena to allocate the data.
                            let cpu_slice = ctx.arena.alloc_slice(&cpu_data);
                            
                            fanta_rust::widgets::plot::PlotBuilder::new(&mut ctx)
                                .height(100.0)
                                .y_range(0.0, 1.0) // 0-1% for demo, adjust as needed
                                .fast_line(cpu_slice, 0.0, 1.0, ColorF::new(0.0, 1.0, 0.5, 1.0))
                                .build();
                        }

                        let peak_data: Vec<f32> = inspector.audio_history.iter().map(|t| t.peak_l).collect();
                        if !peak_data.is_empty() {
                            ctx.text("Audio Output Peak (L)").build();
                            let peak_slice = ctx.arena.alloc_slice(&peak_data);
                            
                            fanta_rust::widgets::plot::PlotBuilder::new(&mut ctx)
                                .height(100.0)
                                .y_range(0.0, 1.0)
                                .fast_line(peak_slice, 0.0, 1.0, ColorF::new(1.0, 0.5, 0.0, 1.0))
                                .build();
                        }

                        ctx.end();
                    }
                    ctx.end();
                }
                ctx.end();

                // 4. Render
                let root = ctx.root().expect("No root view");
                let is_dirty = ctx.dirty.get() || fanta_rust::view::needs_repaint();
                renderer.render_ui(root, 1280.0, 800.0, is_dirty);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
