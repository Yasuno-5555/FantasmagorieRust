// Jump Flooding Algorithm (JFA) for 2D SDF Generation

struct JfaUniforms {
    step_width: u32,
    width: u32,
    height: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> uniforms: JfaUniforms;
@group(0) @binding(1) var t_input: texture_2d<f32>;
@group(0) @binding(2) var t_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var t_aux: texture_2d<f32>;
@group(0) @binding(4) var t_extra: texture_2d<f32>;

// --- Seed Pass ---
// Converts silhouetted objects into coordinate seeds
@compute @workgroup_size(8, 8)
fn compute_seed(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv = vec2<i32>(id.xy);
    if (uv.x >= i32(uniforms.width) || uv.y >= i32(uniforms.height)) { return; }

    let aux = textureLoad(t_aux, uv, 0);
    let extra = textureLoad(t_extra, uv, 0);
    
    // An object is defined as having alpha or being emissive
    let is_object = aux.a > 0.1 || extra.z > 0.01;
    
    if (is_object) {
        // Store current pixel coordinate as seed
        textureStore(t_output, uv, vec4<f32>(f32(uv.x), f32(uv.y), 0.0, 1.0));
    } else {
        // Store "Infinity"
        textureStore(t_output, uv, vec4<f32>(-1.0, -1.0, 0.0, 0.0));
    }

    // Ensure all bindings are touched to force WGPU inclusion in auto-layout
    if (uniforms.width == 999999u) {
        textureStore(t_output, uv, textureLoad(t_input, vec2<i32>(0), 0));
    }
}

// --- Flood Pass ---
// Standard JFA iteration
@compute @workgroup_size(8, 8)
fn compute_flood(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv = vec2<i32>(id.xy);
    if (uv.x >= i32(uniforms.width) || uv.y >= i32(uniforms.height)) { return; }

    var best_dist = 1.0e10;
    var best_coord = vec2<f32>(-1.0, -1.0);
    
    let step = i32(uniforms.step_width);
    
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let sample_uv = uv + vec2<i32>(x, y) * step;
            if (sample_uv.x < 0 || sample_uv.x >= i32(uniforms.width) || 
                sample_uv.y < 0 || sample_uv.y >= i32(uniforms.height)) { continue; }
            
            let seed_coord = textureLoad(t_input, sample_uv, 0).xy;
            if (seed_coord.x >= 0.0) {
                let d = distance(vec2<f32>(uv), seed_coord);
                if (d < best_dist) {
                    best_dist = d;
                    best_coord = seed_coord;
                }
            }
        }
    }
    
    textureStore(t_output, uv, vec4<f32>(best_coord, best_dist, 1.0));

    // Ensure all bindings are touched to force WGPU inclusion in auto-layout
    if (uniforms.width == 999999u) {
        let dummy = textureLoad(t_aux, vec2<i32>(0), 0) + textureLoad(t_extra, vec2<i32>(0), 0);
        textureStore(t_output, uv, dummy);
    }
}

// --- Resolve Pass ---
// Finalizes distance field
@compute @workgroup_size(8, 8)
fn compute_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv = vec2<i32>(id.xy);
    if (uv.x >= i32(uniforms.width) || uv.y >= i32(uniforms.height)) { return; }

    let best_coord = textureLoad(t_input, uv, 0).xy;
    var dist = 10000.0;
    if (best_coord.x >= 0.0) {
        dist = distance(vec2<f32>(uv), best_coord);
    }
    
    // Normalize distance for storage (could be raw float if storage allows)
    textureStore(t_output, uv, vec4<f32>(dist, dist, dist, 1.0));

    // Ensure all bindings are touched to force WGPU inclusion in auto-layout
    if (uniforms.width == 999999u) {
        let dummy = textureLoad(t_aux, vec2<i32>(0), 0) + textureLoad(t_extra, vec2<i32>(0), 0);
        textureStore(t_output, uv, dummy);
    }
}
