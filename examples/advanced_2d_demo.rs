use fantasmagorie::renderer::{Renderer, api::FrameContext};
use fantasmagorie::core::{Vec2, ColorF};
use fantasmagorie::renderer::types::{Rect, Transform2D};
use fantasmagorie::draw::BlendMode;
use fantasmagorie::backend::WgpuBackend;
use fantasmagorie::config::EngineConfig;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[pollster::main]
async fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Fantasmagorie - Advanced 2D Demo")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let backend = WgpuBackend::new(&window).await;
    let mut renderer = Renderer::new(Box::new(backend), EngineConfig::lite());

    let mut frame_count = 0;

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => target.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                // Handle resize
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let mut frame = renderer.begin_frame();
                let time = frame_count as f32 / 60.0;
                
                // 1. Transformation Demo: Rotating Squares
                frame.set_transform(Transform2D::translation(200.0, 200.0));
                for i in 0..5 {
                    frame.set_transform(Transform2D::rotation(time + i as f32 * 0.5));
                    frame.draw(Rect::new(-50.0, -50.0, 100.0, 100.0), ColorF::new(1.0, 0.5, 0.2, 0.8))
                        .rounded(10.0)
                        .submit();
                    frame.pop_transform();
                }
                frame.pop_transform();

                // 2. Blend Mode Demo: Overlapping Circles
                let center = Vec2::new(700.0, 400.0);
                let modes = [BlendMode::Alpha, BlendMode::Add, BlendMode::Multiply, BlendMode::Screen];
                let colors = [
                    ColorF::new(1.0, 0.0, 0.0, 0.5),
                    ColorF::new(0.0, 1.0, 0.0, 0.5),
                    ColorF::new(0.0, 0.1, 1.0, 0.5),
                ];

                for (idx, mode) in modes.iter().enumerate() {
                    let offset_x = (idx % 2) as f32 * 300.0;
                    let offset_y = (idx / 2) as f32 * 300.0;
                    let pos = center + Vec2::new(offset_x, offset_y);
                    
                    frame.set_blend_mode(*mode);
                    // Draw 3 overlapping circles
                    for i in 0..3 {
                        let angle = i as f32 * 2.0 * std::f32::consts::PI / 3.0 + time;
                        let circle_pos = pos + Vec2::new(angle.cos() * 40.0, angle.sin() * 40.0);
                        frame.draw(Rect::new(circle_pos.x - 60.0, circle_pos.y - 60.0, 120.0, 120.0), colors[i])
                            .rounded(60.0)
                            .submit();
                    }
                    frame.pop_blend_mode();
                }

                renderer.end_frame(frame, 1280, 720);
                frame_count += 1;
            }
            _ => (),
        }
    }).unwrap();
}
