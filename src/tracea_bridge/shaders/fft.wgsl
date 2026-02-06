// WGSL FFT Implementation (Cooley-Tukey)

struct Complex {
    r: f32,
    i: f32,
}

struct Params {
    size: u32,       // FFT size
    direction: i32,  // 1 for forward, -1 for inverse
    stage: u32,      // Current butterfly stage (0..log2(size)-1)
    stride: u32,     // Distance between butterfly wings
}

@group(0) @binding(0) var<storage, read_write> input_buffer: array<vec2<f32>>; // Real/Imag interleaved
@group(0) @binding(1) var<storage, read_write> output_buffer: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265;

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.r + b.r, a.i + b.i);
}

fn complex_sub(a: Complex, b: Complex) -> Complex {
    return Complex(a.r - b.r, a.i - b.i);
}

// Bit reversal kernel (for Pre-processing)
@compute @workgroup_size(64)
fn bit_reversal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.size) { return; }
    
    var val = i;
    var rev = 0u;
    let bits = u32(log2(f32(params.size))); // Note: This might be inaccurate in float, better precalc
    // Simple bit reversal loop
    // Assuming size is valid power of two.
    // Ideally bits should be passed in params or derived robustly.
    // For now, let's look at host code passing log2_size.
    // But generic WGSL is tricky.
    
    // Instead of computing bits here, assume params.stride is used as temporary storage for log2_size in bit_reversal pass?
    // Or just iterate 32 bits and shift.
    
    // Simple 32-bit reversal check
    // But we only want to swap within 'bits'.
    // Faster to do host-side bit reversal if possible?
    // Metal version likely does it on GPU. 
    
    // Let's implement iterative bit reversal
    // bit_reverse(i, N)
    var n = params.size;
    var j = 0u;
    var k = i;
    while (n > 1u) {
        j = (j << 1u) | (k & 1u);
        k = k >> 1u;
        n = n >> 1u;
    }
    
    // Only swap once
    if (j > i) {
        let temp = input_buffer[j];
        input_buffer[j] = input_buffer[i];
        input_buffer[i] = temp;
    }
}

// Radix-2 Butterfly
@compute @workgroup_size(64)
fn butterfly(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Each thread handles one pair of butterfly
    // Total pairs = size / 2.
    // So if size=1024, we need 512 threads.
    
    let half_size = params.size / 2u;
    if (idx >= half_size) { return; }
    
    // Butterfly Logic
    // group = idx / (stride)
    // offset_in_group = idx % (stride)
    
    // Pair indices:
    // a = group * (2 * stride) + offset_in_group
    // b = a + stride
    
    let stride = params.stride;
    let group = idx / stride;
    let offset = idx % stride;
    
    let a_idx = group * (2u * stride) + offset;
    let b_idx = a_idx + stride;
    
    let a = Complex(input_buffer[a_idx].x, input_buffer[a_idx].y);
    let b = Complex(input_buffer[b_idx].x, input_buffer[b_idx].y);
    
    // Twiddle Factor
    // W_N^k = exp(-i * 2 * PI * k / N_stage)
    // N_stage = 2 * stride
    // k = offset * (params.size / (2 * stride)) ? No. 
    // In Cooley-Tukey, for decimated-in-time:
    // W = exp(-i * 2 * PI * offset / (2 * stride))
    
    let angle = -2.0 * PI * f32(offset) / f32(2u * stride);
    let w = Complex(cos(angle), sin(angle));
    
    let wb = complex_mul(w, b);
    
    let new_a = complex_add(a, wb);
    let new_b = complex_sub(a, wb);
    
    // Ping-pong? Or write to separate output?
    // If we use separate input/output buffers, we need to toggle them.
    // For now assuming in-place (read_write storage). 
    // BUT race conditions! 
    // In each stage, a_idx and b_idx are unique per thread.
    // So reading a/b and writing new_a/new_b is safe?
    // Yes, because butterfly dependencies are strictly previous stage.
    // Wait, if it's in-place, we must ensure all reads for a stage finish before writes?
    // BARRIER needed. WGSL supports workgroupBarrier().
    // But we need GLOBAL barrier.
    // WGSL doesn't support global barrier across workgroups.
    // So we MUST dispatch each stage as a separate compute pass.
    
    input_buffer[a_idx] = vec2<f32>(new_a.r, new_a.i);
    input_buffer[b_idx] = vec2<f32>(new_b.r, new_b.i);
}

// Magnitude Kernel
@compute @workgroup_size(64)
fn magnitude(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.size) { return; }
    
    let c = input_buffer[i];
    let mag = length(c);
    
    // Log scaling
    let log_mag = log(mag + 1.0);
    
    // Store in x component of output (reuse vec2 buffer or separate float buffer?)
    // Let's assume output_buffer is f32 (vec2 for alignment/reuse or just .x)
    // Actually we bind output for spectrum as f32 array usually?
    // The binding 1 is array<vec2>.
    
    output_buffer[i] = vec2<f32>(log_mag, 0.0);
}
