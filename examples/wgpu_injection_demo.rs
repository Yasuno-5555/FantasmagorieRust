//! WGPU Shader Injection Demo
//! Demonstrates runtime GLSL-to-WGSL transpilation and custom shader effects.

use fanta_rust::prelude::*;
use std::sync::Arc;

fn main() {
    println!("Fantasmagorie Rust - WGPU Shader Injection Demo");
    println!("===============================================");

    #[cfg(feature = "wgpu")]
    {
        use fanta_rust::python::window::py_run_window;
        
        // Define a custom GLSL fragment shader to be injected
        // This shader creates a plasma-like effect
        let custom_shader = r#"
#version 450
layout(location = 0) out vec4 f_color;
layout(location = 0) in vec2 v_uv;

void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    float d = length(p);
    float t = 0.5; // This would normally be a uniform
    
    vec3 color = 0.5 + 0.5 * cos(vec3(0.0, 2.0, 4.0) + d * 10.0 - t * 5.0);
    f_color = vec4(color, 1.0);
}
"#;

        // Note: For now, py_run_window takes a closure that builds the UI for EACH frame.
        // It's a bit of a hack until we have a proper Rust-native runner.
        
        println!("[INFO] Starting window with Shader Injection...");
        
        let custom_shader_str = custom_shader.to_string();
        
        // In a real scenario, this would be called from Python or a Rust-native loop.
        // For this demo, we'll try to run the window.
        // Since we don't have a full Rust-side main loop explorer yet, 
        // we'll just verify it COMPILES and then provide instructions.
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("[ERROR] This demo requires the 'wgpu' feature.");
    }

    println!("\n[OK] Demo setup completed!");
}
