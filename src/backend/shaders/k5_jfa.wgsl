// K5: SDF Lighting - Jump Flooding Algorithm (WGSL)

struct JFAUniforms {
    width: u32,
    height: u32,
    jfa_step: u32,
    ping_pong_idx: u32,
    intensity: f32,
    decay: f32,
    radius: f32,
    _pad: u32,
}

var<push_constant> uniforms: JFAUniforms;

@group(0) @binding(1) var t_seeds: texture_2d<f32>;
@group(0) @binding(2) var t_seeds_secondary: texture_2d<f32>; 
@group(0) @binding(3) var t_output: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let step_val = i32(uniforms.jfa_step);
    let coords = vec2<i32>(i32(x), i32(y));
    
    var best_seed = textureLoad(t_seeds, coords, 0).xy;
    var best_dist = 100000000.0;
    
    if (best_seed.x >= 0.0) {
        let dx = best_seed.x - f32(x);
        let dy = best_seed.y - f32(y);
        best_dist = dx * dx + dy * dy;
    }

    // Check 8 neighbors
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            if (ox == 0 && oy == 0) { continue; }
            
            let nx = i32(x) + ox * step_val;
            let ny = i32(y) + oy * step_val;
            
            if (nx >= 0 && nx < i32(uniforms.width) && ny >= 0 && ny < i32(uniforms.height)) {
                let s = textureLoad(t_seeds, vec2<i32>(nx, ny), 0).xy;
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
