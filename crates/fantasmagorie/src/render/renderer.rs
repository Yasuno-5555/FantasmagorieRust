use glow::HasContext;
use crate::core::node::NodeStore;

pub struct Renderer {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    // ...
}

impl Renderer {
    pub fn new(gl: &glow::Context) -> Self {
        // Init GL resources
        unsafe {
             let program = gl.create_program().expect("Cannot create program");
             // Helper for shaders...
             
             let vao = gl.create_vertex_array().expect("Cannot create VAO");
             let vbo = gl.create_buffer().expect("Cannot create VBO");
             
             Self { program, vao, vbo }
        }
    }

    pub fn render(&mut self, gl: &glow::Context, store: &NodeStore, screen_width: f32, screen_height: f32) {
        unsafe {
            gl.viewport(0, 0, screen_width as i32, screen_height as i32);
            gl.clear_color(0.1, 0.1, 0.1, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Iterate nodes and draw
            // This is a naive stub. Real implementation needs batching.
            // For now, just clear screen.
        }
    }
    
    pub fn destroy(&mut self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.vbo);
        }
    }
}
