use fanta_rust::prelude::*;
use fanta_rust::animation::ik::IKSolver;
use winit::event::{Event, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowAttributes;
use std::sync::Arc;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window_attrs = WindowAttributes::default().with_title("Fantasmagorie - IK Demo");
    let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

    let backend = Box::new(fanta_rust::backend::wgpu::WgpuBackend::new_async(window.clone(), 1280, 800, 1.0).unwrap());
    let mut renderer = Renderer::new_lite(backend);

    // Two Bone Arm State
    let root_pos = Vec2::new(300.0, 400.0);
    let len1 = 150.0;
    let len2 = 120.0;
    
    // CCD Chain State
    let mut chain = vec![
        Vec2::new(800.0, 400.0),
        Vec2::new(850.0, 400.0),
        Vec2::new(900.0, 400.0),
        Vec2::new(950.0, 400.0),
        Vec2::new(1000.0, 400.0),
        Vec2::new(1050.0, 400.0),
        Vec2::new(1100.0, 400.0),
    ];
    // Length constraints (assuming fixed length segments for CCD is desired, but my simple CCD solver doesn't enforce length purely, it just rotates. 
    // Wait, my CCD implementation modifies positions:
    // "joints[j] = pivot + Vec2::new(x_new, y_new);" 
    // This rotates vector (joint -> pivot), preserving length!
    // So yes, it preserves segment lengths.

    let mut target = Vec2::new(500.0, 400.0);

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::CursorMoved { position, .. } => {
                    target = Vec2::new(position.x as f32, position.y as f32);
                },
                WindowEvent::RedrawRequested => {
                    let mut dl = fanta_rust::draw::DrawList::new();
                    
                    // --- Two Bone IK ---
                    dl.add_text_simple(Vec2::new(200.0, 50.0), "Two Bone IK", 20.0, ColorF::white());
                    
                    // Solve
                    // We need mid joint position and end position to draw.
                    // Solver returns angles.
                    // Root is fixed.
                    // Angle1 is absolute. Angle2 is relative?
                    // Let's check my implementation.
                    // angle2 was "PI - internal".
                    // Let's assume Angle2 is relative to Bone 1.
                    
                    // Call Solver
                    // Let's use Mouse as target.
                    if let Some((a1, a2)) = IKSolver::solve_two_bone(root_pos, len1, len2, target, true) {
                        let joint_pos = root_pos + Vec2::new(a1.cos() * len1, a1.sin() * len1);
                        let total_angle = a1 + a2; 
                        // Wait, if a2 is relative to a1:
                        // No, in my code:
                        // let angle2 = if bend_right { -angle2 } else { angle2 };
                        // This implies deviation from straight line.
                        // So absolute angle of bone 2 is a1 + a2.
                        
                        let end_pos = joint_pos + Vec2::new(total_angle.cos() * len2, total_angle.sin() * len2);
                        
                        // Draw Bones
                        dl.add_line(root_pos, joint_pos, 5.0, ColorF::green());
                        dl.add_line(joint_pos, end_pos, 5.0, ColorF::green());
                        
                        // Draw Joints
                        dl.add_circle(root_pos, 8.0, ColorF::white(), true);
                        dl.add_circle(joint_pos, 6.0, ColorF::white(), true);
                        dl.add_circle(end_pos, 4.0, ColorF::red(), true);
                    } else {
                        // Target unreachable??
                        // My solver returns Some((angle, 0.0)) if unreachable (fully extended).
                        // It returns None if too close (folded).
                        dl.add_text_simple(Vec2::new(200.0, 100.0), "Unreachable / Too Close", 16.0, ColorF::red());
                    }

                    // --- CCD IK ---
                    dl.add_text_simple(Vec2::new(800.0, 50.0), "CCD IK Chain", 20.0, ColorF::white());
                    
                    // Solve
                    // For CCD, we pass the chain slice.
                    // Target is Mouse.
                    // We need to clone chain? Or just update it every frame?
                    // If we update it every frame, it stays at last pose (good for temporal coherence).
                    
                    IKSolver::solve_ccd(&mut chain, target, 1.0, 10);
                    
                    // Draw Chain
                    for i in 0..chain.len() - 1 {
                        dl.add_line(chain[i], chain[i+1], 3.0, ColorF::blue());
                        dl.add_circle(chain[i], 4.0, ColorF::white(), true);
                    }
                    dl.add_circle(*chain.last().unwrap(), 4.0, ColorF::red(), true);
                    
                    // Target Marker
                    dl.add_circle_stroke(target, 10.0, ColorF::new(1.0, 1.0, 0.0, 0.5), 2.0);

                    renderer.render_list(&dl, 1280, 800);
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    }).unwrap();
}
