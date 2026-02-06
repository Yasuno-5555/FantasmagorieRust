#include <metal_stdlib>
using namespace metal;

// FFT parameters
struct FFTParams {
    uint fft_size;
    uint stage;
    int direction;  // 1 = forward, -1 = inverse
    uint _pad;
};

// Complex number operations
struct Complex {
    float real;
    float imag;
};

Complex complex_add(Complex a, Complex b) {
    return { a.real + b.real, a.imag + b.imag };
}

Complex complex_sub(Complex a, Complex b) {
    return { a.real - b.real, a.imag - b.imag };
}

Complex complex_mul(Complex a, Complex b) {
    return {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

// Butterfly operation for Cooley-Tukey FFT
kernel void fft_butterfly(
    device float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant FFTParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint N = params.fft_size;
    uint stage = params.stage;
    float direction = float(params.direction);
    
    // Butterfly parameters
    uint butterfly_size = 1u << (stage + 1);
    uint half_size = butterfly_size / 2;
    
    // Which butterfly group and position within group
    uint group = gid / half_size;
    uint pos = gid % half_size;
    
    // Input indices
    uint idx_top = group * butterfly_size + pos;
    uint idx_bottom = idx_top + half_size;
    
    if (idx_bottom >= N) return;
    
    // Twiddle factor: W_N^k = e^(-2πik/N) for forward, e^(2πik/N) for inverse
    float angle = -direction * 2.0 * M_PI_F * float(pos) / float(butterfly_size);
    float2 twiddle = float2(cos(angle), sin(angle));
    
    // Load values
    float2 top = input[idx_top];
    float2 bottom = input[idx_bottom];
    
    // Complex multiplication: bottom * twiddle
    float2 t = float2(
        bottom.x * twiddle.x - bottom.y * twiddle.y,
        bottom.x * twiddle.y + bottom.y * twiddle.x
    );
    
    // Butterfly
    output[idx_top] = top + t;
    output[idx_bottom] = top - t;
}

// Compute magnitude spectrum from complex FFT output
kernel void fft_magnitude(
    device float2 *complex_data [[buffer(0)]],
    device float *magnitudes [[buffer(1)]],
    constant FFTParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint N = params.fft_size;
    uint half_N = N / 2;
    
    if (gid >= half_N) return;
    
    float2 c = complex_data[gid];
    float magnitude = sqrt(c.x * c.x + c.y * c.y);
    
    // Normalize by FFT size
    magnitude /= float(N);
    
    // Apply log scale for better visualization
    magnitude = log10(1.0 + magnitude * 100.0) / 2.0;
    
    magnitudes[gid] = clamp(magnitude, 0.0, 1.0);
}

// Power spectrum (magnitude squared, useful for audio analysis)
kernel void fft_power_spectrum(
    device float2 *complex_data [[buffer(0)]],
    device float *power [[buffer(1)]],
    constant FFTParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint N = params.fft_size;
    uint half_N = N / 2;
    
    if (gid >= half_N) return;
    
    float2 c = complex_data[gid];
    float p = (c.x * c.x + c.y * c.y) / float(N * N);
    
    // Convert to dB scale
    float db = 10.0 * log10(max(p, 1e-10));
    
    // Normalize to 0-1 range (assuming -60dB to 0dB range)
    power[gid] = clamp((db + 60.0) / 60.0, 0.0, 1.0);
}

// Apply Hann window before FFT
kernel void apply_hann_window(
    device float *samples [[buffer(0)]],
    device float *windowed [[buffer(1)]],
    constant FFTParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint N = params.fft_size;
    
    if (gid >= N) return;
    
    float window = 0.5 * (1.0 - cos(2.0 * M_PI_F * float(gid) / float(N - 1)));
    windowed[gid] = samples[gid] * window;
}

// Mel-scale frequency mapping for audio visualization
kernel void mel_scale_bins(
    device float *linear_mags [[buffer(0)]],
    device float *mel_mags [[buffer(1)]],
    constant uint &num_mel_bins [[buffer(2)]],
    constant uint &fft_size [[buffer(3)]],
    constant float &sample_rate [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_mel_bins) return;
    
    // Mel scale conversion
    float mel_low = 0.0;
    float mel_high = 2595.0 * log10(1.0 + (sample_rate / 2.0) / 700.0);
    
    float mel = mel_low + float(gid) * (mel_high - mel_low) / float(num_mel_bins);
    float freq = 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
    
    // Map frequency to FFT bin
    float bin_f = freq * float(fft_size) / sample_rate;
    uint bin = uint(clamp(bin_f, 0.0, float(fft_size / 2 - 1)));
    
    mel_mags[gid] = linear_mags[bin];
}

// Bit reversal permutation for Cooley-Tukey FFT
kernel void bit_reversal(
    device float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant FFTParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.fft_size) return;
    
    // Reverse bits
    // params.stage holds log2_n
    uint bits = params.stage; 
    uint r = reverse_bits(gid) >> (32 - bits);
    
    output[r] = input[gid];
}
