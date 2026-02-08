use fanta_rust::prelude::*;
use fanta_rust::game::world::{Transform, World};
use fanta_rust::animation::skeleton::{Bone, Skeleton};
use fanta_rust::backend::wgpu::SkinnedVertex;
use winit::event::{Event, WindowEvent, KeyEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowAttributes;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window_attrs = WindowAttributes::default().with_title("Fantasmagorie - Animation Demo");
    let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

    // Initialize Engine
    let config = EngineConfig::lite();
    let backend = Box::new(fanta_rust::backend::wgpu::WgpuBackend::new_async(window.clone(), 1280, 800, 1.0).unwrap());
    let mut renderer = Renderer::new_lite(backend);

    // Create Skeleton
    // Bone 0: Root (Base)
    // Bone 1: Child (Arm)
    let mut bones = vec![
        Bone::new("Root", 0, None, Transform::default()),
        Bone::new("Arm", 1, Some(0), Transform {
            local_position: Vec2::new(0.0, -50.0), // Attached above root
            ..Default::default()
        }),
    ];
    
    // Set Inverse Bind Pose (Assume identity for now or simple T-pose)
    // Actually, if we start at identity, inverse is identity.
    // Let's assume initial pose is bind pose.
    bones[0].inverse_bind_pose = Mat3::IDENTITY;
    bones[1].inverse_bind_pose = Mat3::IDENTITY;

    let mut skeleton = Skeleton::from_bones(bones);

    // Create Mesh Data (Simple vertical strip)
    // 2 Triangles for root part, 2 for arm part.
    // Or just one long strip for bending.
    // Let's make a 50x100 rect.
    // Vertices:
    // 0: (-25, 0)   Weight: Bone 0=1.0
    // 1: (25, 0)    Weight: Bone 0=1.0
    // 2: (-25, -50) Weight: Bone 0=0.5, Bone 1=0.5 (Joint)
    // 3: (25, -50)  Weight: Bone 0=0.5, Bone 1=0.5 (Joint)
    // 4: (-25, -100) Weight: Bone 1=1.0
    // 5: (25, -100)  Weight: Bone 1=1.0
    
    let vertices = vec![
        SkinnedVertex { pos: [-25.0, 0.0],    uv: [0.0, 1.0], color: [1.0, 0.0, 0.0, 1.0], bone_indices: [0, 0, 0, 0], bone_weights: [1.0, 0.0, 0.0, 0.0] },
        SkinnedVertex { pos: [25.0, 0.0],     uv: [1.0, 1.0], color: [0.0, 1.0, 0.0, 1.0], bone_indices: [0, 0, 0, 0], bone_weights: [1.0, 0.0, 0.0, 0.0] },
        SkinnedVertex { pos: [-25.0, -50.0],  uv: [0.0, 0.5], color: [0.0, 0.0, 1.0, 1.0], bone_indices: [0, 1, 0, 0], bone_weights: [0.5, 0.5, 0.0, 0.0] },
        SkinnedVertex { pos: [25.0, -50.0],   uv: [1.0, 0.5], color: [1.0, 1.0, 0.0, 1.0], bone_indices: [0, 1, 0, 0], bone_weights: [0.5, 0.5, 0.0, 0.0] },
        SkinnedVertex { pos: [-25.0, -100.0], uv: [0.0, 0.0], color: [0.0, 1.0, 1.0, 1.0], bone_indices: [1, 0, 0, 0], bone_weights: [1.0, 0.0, 0.0, 0.0] },
        SkinnedVertex { pos: [25.0, -100.0],  uv: [1.0, 0.0], color: [1.0, 0.0, 1.0, 1.0], bone_indices: [1, 0, 0, 0], bone_weights: [1.0, 0.0, 0.0, 0.0] },
    ];
    
    // Indices (2 quads)
    let indices = vec![
        0, 1, 2, 1, 3, 2, // Quad 1 (Bottom)
        2, 3, 4, 3, 5, 4, // Quad 2 (Top)
    ];

    let start_time = Instant::now();

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::RedrawRequested => {
                    let time = start_time.elapsed().as_secs_f32();
                    
                    // Animate Skeleton
                    // Rotate root slightly
                    // skeleton.bones[0].local_transform.local_rotation = (time * 0.5).sin() * 0.5;
                    // Rotate arm more
                    skeleton.bones[1].local_transform.local_rotation = time * 2.0;

                    // Update Global Transformations
                    skeleton.compute_global_pose();
                    
                    // Get Skinning Matrices
                    let bone_matrices = skeleton.compute_skinning_matrices();

                    let mut dl = fanta_rust::draw::DrawList::new();
                    
                    // Center transform (DrawList uses push_transform)
                    dl.push_transform(fanta_rust::core::Vec2::new(400.0, 300.0), 1.0);

                    // Add Skinned Mesh Command
                    dl.add_skinned_mesh(
                        vertices.clone(), 
                        indices.clone(), 
                        0, // Texture ID
                        bone_matrices
                    );

                    renderer.render_list(&dl, 1280, 800);
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
