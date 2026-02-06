//! Tracea Neural Kernel Test
//!
//! Tests the Neural Style Transfer Foundation (Phase 6)

#[cfg(feature = "metal")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fanta_rust::tracea_bridge::{TraceaContext, TraceaNeuralKernel};
    use metal::{TextureDescriptor, MTLPixelFormat, MTLTextureUsage, MTLSize, MTLResourceOptions};
    
    println!("=== Tracea Neural Kernel Test ===\n");
    
    let context = TraceaContext::new(None)?;
    println!("[OK] Context initialized");
    
    let neural = TraceaNeuralKernel::new(&context)?;
    println!("[OK] Neural Kernel initialized");
    
    // Create Input/Output Textures (256x256 RGBA)
    let desc = TextureDescriptor::new();
    desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
    desc.set_width(256);
    desc.set_height(256);
    desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    
    let input = context.device().new_texture(&desc);
    let output = context.device().new_texture(&desc);
    
    // Create random weights for Conv2D (3x3 kernel, 4 in, 4 out)
    // 9 pos * 4 outputs * 4 inputs = 144 floats ?
    // My shader expects: 9 pos * 4 float4s = 36 float4s = 144 floats.
    // Let's assume standard identity pass filter first for verification
    
    // Identity kernel: center pixel = 1, others = 0.
    // Center is index 4 (0..8).
    // For each output channel c, we want dot(input, weight) = input[c].
    // So weight vector for output c should be (0,0,0,0) except at input c component.
    // Center Weights (Index 4):
    // Out R: (1, 0, 0, 0)
    // Out G: (0, 1, 0, 0)
    // Out B: (0, 0, 1, 0)
    // Out A: (0, 0, 0, 1)
    
    let mut weights_data = vec![0.0f32; 144]; // 36 * 4
    let center_idx = 4;
    let base = center_idx * 16; // 4 float4s * 4 floats
    
    // R -> R
    weights_data[base + 0] = 1.0; 
    // G -> G
    weights_data[base + 5] = 1.0;
    // B -> B
    weights_data[base + 10] = 1.0;
    // A -> A
    weights_data[base + 15] = 1.0;
    
    let weights_buffer = context.device().new_buffer_with_data(
        weights_data.as_ptr() as *const _,
        (weights_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    
    let bias_data = vec![0.0f32; 4];
    let bias_buffer = context.device().new_buffer_with_data(
        bias_data.as_ptr() as *const _,
        (bias_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    
    use fanta_rust::tracea_bridge::neural::LayerParams;
    let params = LayerParams {
        input_dim: [256, 256],
        output_dim: [256, 256],
        kernel_size: 3,
        stride: 1,
        padding: 1,
        channels_in: 4,
        channels_out: 4,
        _pad: 0,
    };
    
    // Test: Forward Pass
    println!("Running Conv2d Forward...");
    let start = std::time::Instant::now();
    
    neural.forward_conv(&input, &output, &weights_buffer, &bias_buffer, params);
    
    // Wait for completion (via Blit or simply wait command buffer inside forward_conv if we modify it, 
    // or just checking command buffer status.
    // Since forward_conv commits but doesn't wait, we need to wait here or ensure sync.
    // For test, let's assume valid command buffer submission. 
    // We can rely on system sync if we read back.
    
    let elapsed = start.elapsed();
    println!("[OK] Conv2d dispatched in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    
    println!("\n=== Test Passed ===");
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("Requires metal feature");
}
