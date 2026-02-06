//! Tracea Blur Kernel Test
//!
//! Tests the Tracea-accelerated Gaussian blur implementation

#[cfg(feature = "metal")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fanta_rust::tracea_bridge::{TraceaContext, TraceaBlurKernel};
    use metal::{TextureDescriptor, MTLPixelFormat, MTLTextureUsage};
    
    println!("=== Tracea Blur Kernel Test ===\n");
    
    // Create Tracea context
    let context = TraceaContext::new(None)?;
    println!("[OK] Tracea context initialized");
    
    // Create blur kernel with radius 8
    let blur_kernel = TraceaBlurKernel::new(&context, 8)?;
    println!("[OK] Blur kernel created (radius=8, sigma=2.67)");
    
    // Create test textures
    let device = context.device();
    
    let tex_desc = TextureDescriptor::new();
    tex_desc.set_width(256);
    tex_desc.set_height(256);
    tex_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
    tex_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    
    let input_texture = device.new_texture(&tex_desc);
    let output_texture = device.new_texture(&tex_desc);
    
    println!("[OK] Created test textures (256x256 RGBA16Float)");
    
    // Execute blur
    println!("\nExecuting blur (3 passes)...");
    let start = std::time::Instant::now();
    
    blur_kernel.execute(&input_texture, &output_texture, 3)?;
    
    let elapsed = start.elapsed();
    println!("[OK] Blur completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    
    println!("\n=== Test Passed ===");
    
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This example requires the 'metal' feature.");
    eprintln!("Run with: cargo run --example tracea_blur_test --features metal");
}
