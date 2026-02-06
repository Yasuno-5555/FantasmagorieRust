// Tracea JFA Kernel (from K5 JFA + K5 Resolve)
// Jump Flooding Algorithm for SDF computation and lighting

struct JFAUniforms {
    width: u32,
    height: u32,
    step_size: u32,
    pass_type: u32,  // 0 = flood, 1 = resolve
    intensity: f32,
    decay: f32,
    radius: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: JFAUniforms;
@group(0) @binding(1) var t_input: texture_2d<f32>;
@group(0) @binding(2) var t_output: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var t_sdf_output: texture_storage_2d<r32float, write>;

// Flood pass: propagate nearest seed coordinates
@compute @workgroup_size(8, 8)
fn flood(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let step_val = i32(uniforms.step_size);
    let coords = vec2<i32>(i32(x), i32(y));
    
    var best_seed = textureLoad(t_input, coords, 0).xy;
    var best_dist = 100000000.0;
    
    if (best_seed.x >= 0.0) {
        let dx = best_seed.x - f32(x);
        let dy = best_seed.y - f32(y);
        best_dist = dx * dx + dy * dy;
    }

    // Check 8 neighbors at step distance
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            if (ox == 0 && oy == 0) { continue; }
            
            let nx = i32(x) + ox * step_val;
            let ny = i32(y) + oy * step_val;
            
            if (nx >= 0 && nx < i32(uniforms.width) && ny >= 0 && ny < i32(uniforms.height)) {
                let s = textureLoad(t_input, vec2<i32>(nx, ny), 0).xy;
                if (s.x >= 0.0) {
                    let ds_x = s.x - f32(x);
                    let ds_y = s.y - f32(y);
                    let d = ds_x * ds_x + ds_y * ds_y;
                    if (d < best_dist) {
                        best_dist = d;
                        best_seed = s;
                    }
                }
            }
        }
    }

    textureStore(t_output, coords, vec4<f32>(best_seed, 0.0, 1.0));
}

// Resolve pass: convert JFA to SDF lighting
@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let p = vec2<i32>(i32(x), i32(y));
    let seed = textureLoad(t_input, p, 0).xy;
    
    var dist = 10000.0;
    if (seed.x >= 0.0) {
        let diff = vec2<f32>(f32(x), f32(y)) - seed;
        dist = length(diff);
    }
    
    // SDF Lighting: intensity / (1.0 + decay * dist^2)
    let light = uniforms.intensity / (1.0 + uniforms.decay * dist * dist);
    
    textureStore(t_sdf_output, p, vec4<f32>(light, 0.0, 0.0, 1.0));
}

// Seed pass: initialize JFA from emissive pixels
@compute @workgroup_size(8, 8)
fn seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let coords = vec2<i32>(i32(x), i32(y));
    let color = textureLoad(t_input, coords, 0);
    
    // If luminance > threshold, this pixel is a seed
    let lum = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    
    if (lum > 0.5) {
        // Store own coordinates as seed
        textureStore(t_output, coords, vec4<f32>(f32(x), f32(y), 0.0, 1.0));
    } else {
        // No seed: mark as invalid (-1)
        textureStore(t_output, coords, vec4<f32>(-1.0, -1.0, 0.0, 1.0));
    }
}
