
// K12: Audio Spectrum Aggregator
// Input: Spectrum Buffer (Raw FFT Data)
// Output: Global Uniforms (Bass, Mid, High)

struct GlobalUniforms {
    projection: mat4x4<f32>,
    viewport: vec2<f32>,
    time: f32,
    audio_gain: f32, 
}

struct SpectrumData {
    samples: array<f32>,
}

@group(0) @binding(0) var<uniform> u_global: GlobalUniforms;

struct AudioParams {
    bass: f32,
    mid: f32,
    high: f32,
    _pad: f32,
}

@group(0) @binding(1) var<storage, read> t_spectrum: SpectrumData;
@group(0) @binding(2) var<storage, read_write> t_audio_params: AudioParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var bass_sum = 0.0;
    var mid_sum = 0.0;
    var high_sum = 0.0;
    
    let sample_count = arrayLength(&t_spectrum.samples);
    
    // Bass: 0 - 150 Hz -> Bins 0-3
    for (var i = 0u; i < 4u; i++) {
        bass_sum += t_spectrum.samples[i];
    }
    
    // Mid: 150 - 2000 Hz -> Bins 4 - 46
    for (var i = 4u; i < 46u; i++) {
        mid_sum += t_spectrum.samples[i];
    }
    
    // High: 2000+ Hz -> Bins 47+
    for (var i = 46u; i < sample_count; i++) {
        high_sum += t_spectrum.samples[i];
    }
    
    // 1. Normalize
    let raw_bass = bass_sum / 4.0;
    let raw_mid = mid_sum / 42.0;
    let raw_high = high_sum / f32(sample_count - 46u);

    // 2. Apply Gain + Non-Linear Curve (Log Soft Clip)
    // log(1.0 + x * gain) gives a nice "ear-like" response.
    let gain = u_global.audio_gain;
    
    let target_bass = log(1.0 + raw_bass * gain * 5.0); // Boost bass slightly more
    let target_mid = log(1.0 + raw_mid * gain);
    let target_high = log(1.0 + raw_high * gain * 2.0);

    // 3. Temporal Smoothing (Hysteresis)
    // Read previous frame values
    let prev_bass = t_audio_params.bass;
    let prev_mid = t_audio_params.mid;
    let prev_high = t_audio_params.high;

    // Smooth factor (Low = Slow/Smooth, High = Fast/Twitchy)
    // Bass and Mid can be smoother, High should be twitchy.
    let bass_smooth = 0.2; 
    let mid_smooth = 0.2;
    let high_smooth = 0.4; 

    t_audio_params.bass = mix(prev_bass, target_bass, bass_smooth);
    t_audio_params.mid = mix(prev_mid, target_mid, mid_smooth);
    t_audio_params.high = mix(prev_high, target_high, high_smooth);
}
