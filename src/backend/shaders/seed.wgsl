@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    // Just pass through for testing (assuming full screen or handled elsewhere)
    return vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(frag_pos.xy / 1280.0, 0.0, 1.0);
}
