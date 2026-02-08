use fanta_rust::prelude::*;
use fanta_rust::game::world::{Transform, World};
use fanta_rust::animation::skeleton::{Bone, Skeleton};
use fanta_rust::game::animation::{SkeletalAnimationComponent, SkeletalAnimationClip, BoneTrack, BoneKeyframe};
use winit::event::{Event, WindowEvent, KeyEvent, ElementState};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowAttributes;
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window_attrs = WindowAttributes::default().with_title("Fantasmagorie - Skeletal Blending Demo");
    let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

    // Initialize Engine
    let config = EngineConfig::lite();
    let backend = Box::new(fanta_rust::backend::wgpu::WgpuBackend::new_async(window.clone(), 1280, 800, 1.0).unwrap());
    let mut renderer = Renderer::new_lite(backend);

    // Create Skeleton
    // Bone 0: Root (Base)
    // Bone 1: Child (Arm)
    let mut bones = vec![
        Bone::new("Root", 0, None, Transform { local_position: Vec2::new(400.0, 600.0), ..Default::default() }),
        Bone::new("Arm", 1, Some(0), Transform {
            local_position: Vec2::new(0.0, -100.0), // Attached above root
            ..Default::default()
        }),
    ];
    let mut skeleton = Skeleton::from_bones(bones);

    // Create Clips
    let mut clips = HashMap::new();

    // Idle Clip: Arm rotates -0.2 to 0.2
    let idle_track = BoneTrack {
        bone_name: "Arm".to_string(),
        keyframes: vec![
            BoneKeyframe { time: 0.0, position: None, rotation: Some(-0.2), scale: None },
            BoneKeyframe { time: 1.0, position: None, rotation: Some(0.2), scale: None },
            BoneKeyframe { time: 2.0, position: None, rotation: Some(-0.2), scale: None },
        ],
    };
    clips.insert("Idle".to_string(), SkeletalAnimationClip {
        name: "Idle".to_string(),
        duration: 2.0,
        bone_tracks: vec![idle_track],
        loop_clip: true,
    });

    // Wave Clip: Arm rotates -1.0 to 1.0 rapidly
    let wave_track = BoneTrack {
        bone_name: "Arm".to_string(),
        keyframes: vec![
            BoneKeyframe { time: 0.0, position: None, rotation: Some(-1.0), scale: None },
            BoneKeyframe { time: 0.25, position: None, rotation: Some(1.0), scale: None },
            BoneKeyframe { time: 0.5, position: None, rotation: Some(-1.0), scale: None },
        ],
    };
    clips.insert("Wave".to_string(), SkeletalAnimationClip {
        name: "Wave".to_string(),
        duration: 0.5,
        bone_tracks: vec![wave_track],
        loop_clip: true,
    });

    // Setup Animation Component
    let mut anim = SkeletalAnimationComponent::default();
    anim.play("Idle", 0.0);

    let start_time = Instant::now();
    let mut last_frame = start_time;

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(KeyCode::Space), state: ElementState::Pressed, .. }, .. } => {
                    // Toggle animation with blend
                    if anim.current_clip == "Idle" {
                        anim.play("Wave", 0.5); // Blend over 0.5s
                        println!("Switching to Wave (Blend 0.5s)");
                    } else {
                        anim.play("Idle", 0.5);
                        println!("Switching to Idle (Blend 0.5s)");
                    }
                },
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(last_frame).as_secs_f32();
                    last_frame = now;
                    
                    // Update Animation
                    anim.update(dt, &clips);
                    anim.apply(&mut skeleton, &clips);
                    
                    // Update Global Transformations
                    skeleton.compute_global_pose();

                    let mut dl = fanta_rust::DrawList::new();
                    
                    // Draw Skeleton Debug
                    skeleton.draw_debug(&mut dl, ColorF::new(1.0, 1.0, 0.0, 1.0));
                    
                    // Draw Text
                    use fanta_rust::draw::DrawList; // Ensure extension trait is visible? No, method is on DrawList
                    // dl.add_text_simple is what I added recently
                    dl.add_text_simple(Vec2::new(10.0, 30.0), "Press Space to Blend Animations", 20.0, ColorF::white());
                    
                    let blend_status = format!("Clip: {} | Blend: {:.2}", anim.current_clip, anim.blend_factor);
                    dl.add_text_simple(Vec2::new(10.0, 60.0), &blend_status, 20.0, ColorF::white());

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
