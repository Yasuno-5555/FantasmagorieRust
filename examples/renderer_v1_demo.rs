//! Renderer V1 Demo - verifies the Renderer → GraphicsBackend flow
//! 
//! This demo creates a minimal GraphicsBackend stub to test the pipeline.

use fanta_rust::renderer::{Renderer, Rect, Color};
use fanta_rust::backend::GraphicsBackend;
use fanta_rust::draw::DrawList;
use fanta_rust::renderer::packet::DrawPacket;

/// Minimal stub GraphicsBackend for testing the renderer pipeline
struct StubBackend;

impl GraphicsBackend for StubBackend {
    fn name(&self) -> &str {
        "StubBackend"
    }

    fn render(&mut self, _dl: &DrawList, _width: u32, _height: u32) {
        println!("StubBackend::render (high-level path)");
    }

    fn submit(&mut self, packets: &[DrawPacket]) {
        println!("StubBackend::submit - Executing {} packets (low-level path)", packets.len());
        for (i, p) in packets.iter().enumerate() {
            println!("  Packet {}: Pipeline {:?}, Range {:?}", i, p.pipeline, p.draw_range);
        }
    }

    fn present(&mut self) {
        println!("StubBackend::present");
    }

    fn update_font_texture(&mut self, _width: u32, _height: u32, _data: &[u8]) {
        println!("StubBackend::update_font_texture");
    }
}

fn main() {
    println!("--- Fantasmagorie Renderer v1.0 Demo ---");

    // 1. Initialize Backend (The Muscle)
    let backend = Box::new(StubBackend);

    // 2. Initialize Renderer (The Boundary)
    let mut renderer = Renderer::new_lite(backend);

    // 3. Begin Frame (Human API)
    println!("> Begin Frame");
    let mut ctx = renderer.begin_frame();

    // 4. Submit Commands (Immediate-ish)
    println!("> Draw Quad 1");
    ctx.draw_quad(
        Rect::new(10.0, 10.0, 100.0, 100.0), 
        Color::new(1.0, 0.0, 0.0, 1.0)
    );

    println!("> Draw Quad 2");
    ctx.draw_quad(
        Rect::new(150.0, 10.0, 100.0, 100.0), 
        Color::new(0.0, 1.0, 0.0, 1.0)
    );

    // 5. End Frame (Submission to Brain -> Muscle)
    println!("> End Frame (Submit)");
    renderer.end_frame(ctx, 1024, 768);

    println!("--- Demo Complete ---");
}

