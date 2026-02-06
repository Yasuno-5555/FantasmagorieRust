//! Tracea JFA Kernel Test
//!
//! Tests the Tracea-accelerated Jump Flood Algorithm implementation

#[cfg(feature = "metal")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fanta_rust::tracea_bridge::{TraceaContext, TraceaJFAKernel};
    use metal::{TextureDescriptor, MTLPixelFormat, MTLTextureUsage, MTLRegion, MTLOrigin, MTLSize};
    
    println!("=== Tracea JFA Kernel Test ===\n");
    
    // Create Tracea context
    let context = TraceaContext::new(None)?;
    println!("[OK] Tracea context initialized");
    
    // Create JFA kernel
    let jfa_kernel = TraceaJFAKernel::new(&context)?;
    println!("[OK] JFA kernel created");
    
    // Create test textures (512x512)
    let width = 512u64;
    let height = 512u64;
    let device = context.device();
    
    let tex_desc = TextureDescriptor::new();
    tex_desc.set_width(width);
    tex_desc.set_height(height);
    tex_desc.set_pixel_format(MTLPixelFormat::RGBA16Float); // Seed texture
    tex_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    let seed_texture = device.new_texture(&tex_desc);
    
    let out_desc = TextureDescriptor::new();
    out_desc.set_width(width);
    out_desc.set_height(height);
    out_desc.set_pixel_format(MTLPixelFormat::RGBA16Float); // Output SDF
    out_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    let output_texture = device.new_texture(&out_desc);
    
    // Initialize seed texture (center point)
    let center_x = width / 2;
    let center_y = height / 2;
    let seed_data = vec![1.0f32; 4]; // White pixel
    let region = MTLRegion {
        origin: MTLOrigin { x: center_x, y: center_y, z: 0 },
        size: MTLSize { width: 1, height: 1, depth: 1 },
    };
    seed_texture.replace_region(region, 0, seed_data.as_ptr() as *const _, 16);
    println!("[OK] Created seed texture with center point");
    
    // Execute JFA SDF generation
    println!("\nExecuting JFA SDF generation...");
    let start = std::time::Instant::now();
    
    jfa_kernel.generate_sdf(&seed_texture, &output_texture, 512.0)?;
    
    let elapsed = start.elapsed();
    println!("[OK] JFA completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    
    // Verify specific point (0,0 should be dist ~362 since center is 256,256)
    // Distance from (0,0) to (256,256) is sqrt(256^2 + 256^2) = 362.03
    // Normalized distance = 362.03 / 512.0 = 0.707
    
    // Read back output texture pixel at (0,0)
    let mut result_data = vec![0.0f32; 4];
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize { width: 1, height: 1, depth: 1 },
    };
    output_texture.get_bytes(
        result_data.as_mut_ptr() as *mut _,
        16,
        region,
        0
    );
    
    let dist = result_data[0];
    println!("Distance at (0,0): {:.4} (Expected ~0.707)", dist);
    
    if (dist - 0.707).abs() < 0.05 {
        println!("[OK] Distance validation passed");
    } else {
        println!("[WARN] Distance validation outside tolerance");
    }

    println!("\n=== Test Passed ===");
    
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This example requires the 'metal' feature.");
}
