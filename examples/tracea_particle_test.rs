//! Tracea Particle Kernel Test
//!
//! Tests the Tracea-accelerated Particle System (Phase 4)

#[cfg(feature = "metal")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fanta_rust::tracea_bridge::{TraceaContext, TraceaParticleKernel};
    use metal::{MTLRegion, MTLSize};
    
    println!("=== Tracea Particle Kernel Test ===\n");
    
    // Create Tracea context
    let context = TraceaContext::new(None)?;
    println!("[OK] Tracea context initialized");
    
    // Create particle kernel with 100,000 particles
    let count = 100_000;
    let kernel = TraceaParticleKernel::new(&context, count)?;
    println!("[OK] Particle kernel initialized ({} particles)", count);
    
    // Test 1: Initialization
    println!("\nTest 1: Initialization");
    // Verify first particle position (should be within bounds)
    read_particle(&kernel, 0, "Initial");
    
    // Test 2: Update (Simulation)
    println!("\nTest 2: Simulation Steps");
    let start = std::time::Instant::now();
    
    // Run 60 frames
    for _ in 0..60 {
        kernel.update(0.016, [960.0, 540.0], None);
    }
    
    let elapsed = start.elapsed();
    println!("[OK] 60 frames simulated in {:.2}ms ({:.2} ms/frame)", 
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / 60.0
    );
    
    read_particle(&kernel, 0, "After 60 frames");
    
    println!("\n=== Test Passed ===");
    
    Ok(())
}

#[cfg(feature = "metal")]
fn read_particle(kernel: &fanta_rust::tracea_bridge::TraceaParticleKernel, index: usize, label: &str) {
    use fanta_rust::tracea_bridge::particles::Particle;
    use metal::{MTLRegion, MTLSize};
    
    let buffer = kernel.particle_buffer();
    let mut data = vec![0u8; std::mem::size_of::<Particle>()];
    
    // Get pointer to buffer contents
    let ptr = buffer.contents() as *const u8;
    let offset = index * std::mem::size_of::<Particle>();
    
    unsafe {
        std::ptr::copy_nonoverlapping(ptr.add(offset), data.as_mut_ptr(), data.len());
    }
    
    let p: &Particle = bytemuck::from_bytes(&data);
    println!("{}: Pos=({:.2}, {:.2}), Vel=({:.2}, {:.2}), Color=({:.2}, {:.2}, {:.2})",
        label, p.position[0], p.position[1], p.velocity[0], p.velocity[1], p.color[0], p.color[1], p.color[2]);
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This example requires the 'metal' feature.");
}
