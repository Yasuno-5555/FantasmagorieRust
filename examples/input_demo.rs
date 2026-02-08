use fanta_rust::input::{InputManager, InputBinding};
use winit::event::{Event, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};

use winit::keyboard::KeyCode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Input Demo");
    println!("Press SPACE to Jump");
    println!("Hold D to Move Right");

    let event_loop = EventLoop::new()?;
    let window_attrs = winit::window::Window::default_attributes()
        .with_title("Input Demo");
    let window = event_loop.create_window(window_attrs)?;

    let mut input_manager = InputManager::new();

    // Bind actions
    input_manager.get_action_map_mut().bind("Jump", InputBinding::Keyboard(KeyCode::Space));
    input_manager.get_action_map_mut().bind("MoveRight", InputBinding::Keyboard(KeyCode::KeyD));

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        // Process input
        input_manager.process_event(&event);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::AboutToWait => {
                input_manager.update();

                if input_manager.is_action_active("Jump") {
                     // In a real game, we'd check for "JustPressed" to avoid spamming, 
                     // but our current generic is_action_active is simple.
                     // Let's just print it.
                     // println!("Jump Action Active!"); 
                }
                
                if input_manager.is_action_active("MoveRight") {
                    println!("Moving Right...");
                }
                
                // For "Jump", let's manually check if it was just pressed in this frame 
                // (simulating what a "JustPressed" method would do if we had history execution)
                // For this demo, simple active check is fine.
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. } => {
                 if event.state == ElementState::Pressed && event.physical_key == winit::keyboard::PhysicalKey::Code(KeyCode::Space) {
                     println!("Jump! (Raw Event)");
                 }
            }
            _ => {}
        }
    })?;

    Ok(())
}
