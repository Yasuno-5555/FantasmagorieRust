use fanta_rust::backend::{GraphicsBackend, WgpuBackend};
use fanta_rust::prelude::*;
use fanta_rust::game::{World, tilemap::{TileMap, TileSet}};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fantasmagorie Rust - Tilemap Demo");

    let event_loop = EventLoop::new()?;
    let window = winit::window::WindowAttributes::default()
        .with_title("Fantasmagorie Tilemap Demo")
        .with_inner_size(LogicalSize::new(1280, 800));
    
    let window = event_loop.create_window(window)?;
    let window = Arc::new(window);
    
    let mut backend = WgpuBackend::new_async(window.clone(), 1280, 800, 1.0)
        .map_err(|e| format!("WGPU creation failed: {}", e))?;

    let mut world = World::new();

    // 1. Create a TileSet
    // We'll use 0 as the texture ID (fallback to font atlas for demo purposes if no other texture is loaded)
    let tileset = TileSet {
        texture: 0,
        tile_width: 32,
        tile_height: 32,
        columns: 8,
        spacing: 0,
        margin: 0,
    };

    // 2. Create a TileMap
    let mut tilemap = TileMap::new(tileset, 40, 25); // 1280/32 = 40, 800/32 = 25
    for y in 0..25 {
        for x in 0..40 {
            // Create a pattern
            if (x + y) % 2 == 0 {
                tilemap.set_tile(x, y, 1);
            } else {
                tilemap.set_tile(x, y, 2);
            }
            
            // Border
            if x == 0 || y == 0 || x == 39 || y == 24 {
                tilemap.set_tile(x, y, 3);
            }
        }
    }

    let map_entity = world.spawn();
    let map_idx = *world.id_to_index.get(&map_entity).unwrap();
    world.tilemaps[map_idx] = Some(tilemap);

    let mut last_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32();
                last_time = now;

                // Simple scrolling
                let map_idx = *world.id_to_index.get(&map_entity).unwrap();
                world.transforms[map_idx].local_position.x -= dt * 50.0;
                if world.transforms[map_idx].local_position.x < -32.0 {
                    world.transforms[map_idx].local_position.x = 0.0;
                }

                let mut dl = fanta_rust::DrawList::new();
                dl.add_rect(Vec2::ZERO, Vec2::new(1280.0, 800.0), ColorF::new(0.05, 0.05, 0.1, 1.0));

                // Submit tilemap to DrawList
                for i in 0..world.ids.len() {
                    if let (Some(tm), transform) = (&world.tilemaps[i], &world.transforms[i]) {
                        let pos = transform.local_position;
                        dl.add_tilemap(
                            pos,
                            Vec2::new(tm.width as f32 * tm.tileset.tile_width as f32, tm.height as f32 * tm.tileset.tile_height as f32),
                            tm.tileset.texture,
                            [tm.width, tm.height],
                            [tm.tileset.tile_width as f32, tm.tileset.tile_height as f32],
                            tm.data.clone(),
                            tm.color,
                            tm.tileset.columns,
                            [1.0 / tm.tileset.columns as f32, 1.0 / tm.tileset.columns as f32], // Simplified UV size
                        );
                    }
                }

                backend.render(&dl, 1280, 800);
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
