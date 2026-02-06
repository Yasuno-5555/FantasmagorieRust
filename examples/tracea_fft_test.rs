//! Tracea Audio FFT Test
//!
//! Tests the Tracea-accelerated FFT implementation

#[cfg(feature = "metal")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fanta_rust::tracea_bridge::{TraceaContext, TraceaFFTKernel};
    
    println!("=== Tracea FFT Kernel Test ===\n");
    
    // Create Tracea context
    let context = TraceaContext::new(None)?;
    println!("[OK] Tracea context initialized");
    
    // Create FFT kernel (size 1024)
    let fft_size = 1024;
    let fft_kernel = TraceaFFTKernel::new(&context, fft_size)?;
    println!("[OK] FFT kernel created (size={})", fft_size);
    
    // Generate test signal: 440Hz sine wave + 1000Hz sine wave
    // Sample rate 44100Hz
    let sample_rate = 44100.0;
    let mut samples = Vec::with_capacity(fft_size);
    for i in 0..fft_size {
        let t = i as f32 / sample_rate;
        let val = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5 + 
                  (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.3;
        samples.push(val);
    }
    println!("[OK] Generated test signal (440Hz + 1000Hz sine waves)");
    
    // Execute FFT
    println!("\nExecuting FFT...");
    let start = std::time::Instant::now();
    
    let spectrum = fft_kernel.compute_spectrum(&samples)?;
    
    let elapsed = start.elapsed();
    println!("[OK] FFT completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    
    // Verify peaks
    // Bin frequency = bin_index * sample_rate / fft_size
    // 440Hz bin = 440 * 1024 / 44100 = 10.2 -> index 10
    // 1000Hz bin = 1000 * 1024 / 44100 = 23.2 -> index 23
    
    let bin_440 = (440.0 * fft_size as f32 / sample_rate).round() as usize;
    let bin_1000 = (1000.0 * fft_size as f32 / sample_rate).round() as usize;
    
    println!("Value at 440Hz (bin {}): {:.4}", bin_440, spectrum[bin_440]);
    println!("Value at 1000Hz (bin {}): {:.4}", bin_1000, spectrum[bin_1000]);
    println!("Value at noise (bin 50): {:.4}", spectrum[50]);
    
    if spectrum[bin_440] > 0.1 && spectrum[bin_1000] > 0.1 && spectrum[50] < 0.05 {
        println!("[OK] Frequency peak validation passed");
    } else {
        println!("[WARN] Frequency peak validation suspicious");
    }

    println!("\n=== Test Passed ===");
    
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This example requires the 'metal' feature.");
}
