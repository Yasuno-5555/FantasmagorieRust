use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use realfft::RealFftPlanner;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

pub struct AudioManager {
    _stream: cpal::Stream, // Keep stream alive
    spectrum_data: Arc<Mutex<Vec<f32>>>,
    samples_data: Arc<Mutex<Vec<f32>>>,
}

impl AudioManager {
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        
        let device = host.default_input_device()
            .ok_or("No input device available")?;

        let config: cpal::StreamConfig = device.default_input_config()
            .map_err(|e| format!("Failed to get default input config: {}", e))?
            .into();

        log::info!("Audio Input: Using device '{}', Sample Rate: {}", 
            device.name().unwrap_or("Unknown".into()), 
            config.sample_rate.0);

        let spectrum_data = Arc::new(Mutex::new(vec![0.0; 512]));
        let spectrum_clone = spectrum_data.clone();
        
        let fft_size = 1024;

        // PCM samples storage
        let samples_data = Arc::new(Mutex::new(vec![0.0; fft_size]));
        let samples_clone = samples_data.clone();

        // FFT Setup
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let mut spectrum = r2c.make_output_vec();
        let mut input_buffer = r2c.make_input_vec();
        
        // Ring buffer for smooth windows
        let mut sample_ring = VecDeque::with_capacity(fft_size);
        for _ in 0..fft_size {
            sample_ring.push_back(0.0);
        }

        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| {
                // Add new samples to ring buffer
                for &sample in data {
                    sample_ring.pop_front();
                    sample_ring.push_back(sample);
                }

                // Copy to FFT input with windowing (Hann Window)
                for (i, &sample) in sample_ring.iter().enumerate() {
                    let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size as f32 - 1.0)).cos());
                    input_buffer[i] = sample * window;
                }
                
                // Update PCM samples for GPU (Tracea)
                if let Ok(mut lock) = samples_clone.try_lock() {
                    // We copy input_buffer (windowed)
                    // Or should we copy raw sample_ring? 
                    // Windowing is useful for FFT. TraceaFFT expects raw or windowed?
                    // TraceaFFT is generic. Passing windowed is better for spectral analysis.
                    if lock.len() != input_buffer.len() {
                        *lock = vec![0.0; input_buffer.len()];
                    }
                    lock.copy_from_slice(&input_buffer);
                }

                // Execute FFT (CPU Fallback/Legacy)
                if let Ok(()) = r2c.process(&mut input_buffer, &mut spectrum) {
                     let mut out_lock = spectrum_clone.lock().unwrap();
                     for (i, val) in spectrum.iter().enumerate().take(512) {
                        // Magnitude
                        let mag = val.norm();
                        // Logarithmic scaling for better visualization
                        let log_mag = (mag + 1.0).log10(); 
                        
                        // Time smoothing (Simple Lerp)
                        out_lock[i] = out_lock[i] * 0.8 + log_mag * 0.2; 
                     }
                }
            },
            err_fn,
            None // None for default timeout
        ).map_err(|e| format!("Failed to build input stream: {}", e))?;

        stream.play().map_err(|e| format!("Failed to play stream: {}", e))?;

        Ok(Self {
            _stream: stream,
            spectrum_data,
            samples_data,
        })
    }

    pub fn get_spectrum(&self) -> Vec<f32> {
        let lock = self.spectrum_data.lock().unwrap();
        lock.clone()
    }

    pub fn get_samples(&self) -> Vec<f32> {
        let lock = self.samples_data.lock().unwrap();
        lock.clone()
    }
}
