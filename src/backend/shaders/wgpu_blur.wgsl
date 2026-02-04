// WGPU Compute Shader for Gaussian Blur
// Separable 1D Gaussian Blur

struct BlurParams {
    direction: vec2<f32>,
    sigma: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba8unorm, write>;

// Gaussian function
fn gaussian(x: f32, sigma: f32) -> f32 {
    let pi = 3.14159265359;
    return (1.0 / sqrt(2.0 * pi * sigma * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    let coords = vec2<i32>(global_id.xy);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Kernel radius based on sigma (3 * sigma rule)
    // We clamp to a reasonable max to prevent performance cliff
    let radius = i32(ceil(params.sigma * 3.0));
    let clamped_radius = min(radius, 20); // Max kernel size 41

    var color_acc = vec4<f32>(0.0);
    var weight_acc = 0.0;

    let dir = vec2<i32>(params.direction);

    for (var i = -clamped_radius; i <= clamped_radius; i++) {
        let offset = dir * i;
        let sample_coords = coords + offset;

        // Clamp to edge
        let clamped_x = clamp(sample_coords.x, 0, i32(dims.x) - 1);
        let clamped_y = clamp(sample_coords.y, 0, i32(dims.y) - 1);

        let weight = gaussian(f32(i), params.sigma);
        let color = textureLoad(input_tex, vec2<i32>(clamped_x, clamped_y), 0);
        
        color_acc += color * weight;
        weight_acc += weight;
    }

    // Normalize
    let final_color = color_acc / weight_acc;

    textureStore(output_tex, coords, final_color);
}
