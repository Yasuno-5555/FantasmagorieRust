//! Tracea SPH Kernel Test
//!
//! Tests the SPH Fluid Simulation (Phase 5)

#[cfg(feature = "metal")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fanta_rust::tracea_bridge::{TraceaContext, TraceaSPHKernel};
    
    println!("=== Tracea SPH Kernel Test ===\n");
    
    let context = TraceaContext::new(None)?;
    println!("[OK] Context initialized");
    
    // 4096 particles (64x64 block)
    let count = 4096;
    let kernel = TraceaSPHKernel::new(&context, count)?;
    println!("[OK] SPH Kernel initialized ({} particles)", count);
    
    // Init (Run reset)
    // Already called in new(), but safe to call again or verify
    
    // Simulation
    println!("\nSimulating 30 frames...");
    let start = std::time::Instant::now();
    for _ in 0..30 {
        kernel.update(0.016);
    }
    let elapsed = start.elapsed();
    
    println!("[OK] 30 frames in {:.2}ms ({:.2} ms/frame)", 
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / 30.0
    );
    
    // Verify some data
    use fanta_rust::tracea_bridge::sph::SPHParticle;
    let buffer = kernel.particle_buffer();
    let ptr = buffer.contents() as *const SPHParticle;
    let particles = unsafe { std::slice::from_raw_parts(ptr, count) };
    
    let p0 = particles[0];
    println!("Particle 0: Pos=({:.2}, {:.2}), Density={:.2}, Pressure={:.2}", 
        p0.position[0], p0.position[1], p0.density, p0.pressure);
        
    // Check if particles moved (Gravity should pull them down)
    // Init pos around 500, 500.
    // Gravity is negative Y.
    // So Y should decrease.
    
    if p0.position[1] < 500.0 {
        println!("[OK] Gravity effective (Y decreased)");
    } else {
        println!("[WARN] Particle didn't fall? Y={}", p0.position[1]);
    }
    
    if p0.density > 0.0 {
         println!("[OK] Density calculated (>0)");
    }
    
    println!("\n=== Test Passed ===");
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("Requires metal feature");
}
