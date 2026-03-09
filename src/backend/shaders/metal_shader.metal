#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 pos [[attribute(0)]];
    float2 uv [[attribute(1)]];
    float4 color [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
    float2 world_pos;
    uint iid [[flat]];
    float2 velocity;
};

struct GlobalUniforms {
    float4x4 projection;
    float time;
    float2 viewport_size;
    float _padding;
};

struct DrawUniforms {
    float4x4 projection;
    float4 rect;       // x, y, w, h
    float4 radii;      // tl, tr, br, bl
    float4 border_color;
    float4 glow_color;
    float2 offset;
    float scale;
    float border_width;
    float elevation;
    float glow_strength;
    float lut_intensity;
    int mode;
    int is_squircle;
    float time;
    float2 viewport_size;
};

struct ShapeInstance {
    float4 rect;
    float4 radii;
    float4 color;
    float4 border_color;
    float4 glow_color;
    float4 params1; // border_width, elevation, glow_strength, lut_intensity
    uint4 params2;   // mode, is_squircle, _r1, _r2
    float4 material; // velocity_x, velocity_y, reflectivity, roughness
    float4 pbr_params; // normal_map_id, distortion, emissive_intensity, parallax_factor
};

vertex VertexOut vs_main(Vertex v [[stage_in]],
                       constant GlobalUniforms &glob [[buffer(0)]]) {
    VertexOut out;
    out.position = glob.projection * float4(v.pos, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

// Instanced vertex shader
vertex VertexOut vs_instanced(Vertex v [[stage_in]],
                             constant GlobalUniforms &glob [[buffer(0)]],
                             constant ShapeInstance *instances [[buffer(1)]],
                             uint instance_id [[instance_id]],
                             uint vertex_id [[vertex_id]]) {
    VertexOut out;
    // Full screen triangle
    float2 grid[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    out.position = float4(grid[vertex_id % 3], 0.0, 1.0);
    out.uv = grid[vertex_id % 3] * 0.5 + 0.5;
    out.color = float4(1.0);
    out.world_pos = out.position.xy * 1000.0;
    out.iid = instance_id;
    out.velocity = float2(0.0);
    return out;
}

float sdRoundedBox(float2 p, float2 b, float4 r) {
    float radius = r.x; 
    if (p.x > 0.0) radius = r.y;
    if (p.x > 0.0 && p.y > 0.0) radius = r.z;
    if (p.x <= 0.0 && p.y > 0.0) radius = r.w;
    float2 q = abs(p) - b + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

float sdSquircle(float2 p, float2 b, float r) {
    float2 q = abs(p) - b + r;
    float2 start = max(q, 0.0);
    float n = 4.0; 
    float2 p_n = pow(start, float2(n)); 
    float len = pow(p_n.x + p_n.y, 1.0/n);
    return len + min(max(q.x, q.y), 0.0) - r;
}

float hash(float2 p) {
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

struct ShapeData {
    float4 rect;
    float4 radii;
    float4 color;
    float4 border_color;
    float4 glow_color;
    float border_width;
    float elevation;
    float glow_strength;
    float lut_intensity;
    int mode;
    int is_squircle;
    float2 viewport_size;
    float time;
};

float4 resolve_shape(VertexOut in, ShapeData u, texture2d<float> tex, texture2d<float> tex2, sampler s, float time) {
    float4 color_linear = in.color;
    float4 final_color = float4(0.0);

    if (u.mode == 0) {
        final_color = color_linear;
    }
    else if (u.mode == 1) {
        // Standard SDF Text
        float dist = tex.sample(s, in.uv).r;
        float alpha = smoothstep(0.45, 0.55, dist);
        final_color = float4(color_linear.rgb, color_linear.a * alpha);
    }
    else if (u.mode == 2) {
        float2 center = u.rect.xy + u.rect.zw * 0.5;
        float2 half_size = u.rect.zw * 0.5;
        float2 local = in.world_pos - center;
        float d;
        if (u.is_squircle == 1) d = sdSquircle(local, half_size, u.radii.x);
        else d = sdRoundedBox(local, half_size, u.radii);
        float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        float4 bg = color_linear;
        if (u.border_width > 0.0) {
            float interior_alpha = 1.0 - smoothstep(-1.0, 1.0, d + u.border_width);
            float4 border_col_lin = float4(pow(u.border_color.rgb, 2.2), u.border_color.a);
            bg = mix(border_col_lin, color_linear, interior_alpha);
        }
        float4 main_layer = float4(bg.rgb, bg.a * alpha);
        float4 glow_layer = float4(0.0);
        if (u.glow_strength > 0.0) {
            float d_glow = d - u.glow_strength * 0.5;
            float glow_alpha = 1.0 - smoothstep(-u.glow_strength, u.glow_strength, d_glow);
            float4 glow_col_lin = float4(pow(u.glow_color.rgb, 2.2), u.glow_color.a);
            glow_layer = float4(glow_col_lin.rgb, glow_col_lin.a * glow_alpha * (1.0 - alpha));
        }
        float4 shadow_layer = float4(0.0);
        if (u.elevation > 0.0) {
            float d1 = sdRoundedBox(local - float2(0.0, u.elevation * 0.25), half_size, u.radii);
            float a1 = (1.0 - smoothstep(-u.elevation*0.5, u.elevation*0.5, d1)) * 0.4;
            shadow_layer = float4(0.0, 0.0, 0.0, a1 * color_linear.a);
        }
        final_color = main_layer + glow_layer + shadow_layer;
    }
    else if (u.mode == 3) {
        float2 center = u.rect.xy + u.rect.zw * 0.5;
        float2 half_size = u.rect.zw * 0.5;
        float2 local = in.world_pos - center;
        float d = sdRoundedBox(local, half_size, u.radii);
        float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        float4 tex_col = tex.sample(s, in.uv);
        tex_col.rgb = pow(tex_col.rgb, 2.2);
        final_color = tex_col * color_linear;
        final_color.a *= alpha;
    }
    else if (u.mode == 4) {
        // Authentic Backdrop Blur (Screen-space sampling from captured background)
        float2 screen_uv = in.position.xy / u.viewport_size;
        
        float3 blurred = float3(0.0);
        float blur_radius = u.elevation * 0.005; 
        
        // 9-tap Gaussian approximation
        float weights[5] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
        
        blurred += tex2.sample(s, screen_uv).rgb * weights[0];
        for(int i = 1; i < 5; ++i) {
            float offset = float(i) * blur_radius;
            blurred += tex2.sample(s, screen_uv + float2(offset, 0.0)).rgb * weights[i] * 0.5;
            blurred += tex2.sample(s, screen_uv - float2(offset, 0.0)).rgb * weights[i] * 0.5;
            blurred += tex2.sample(s, screen_uv + float2(0.0, offset)).rgb * weights[i] * 0.5;
            blurred += tex2.sample(s, screen_uv - float2(0.0, offset)).rgb * weights[i] * 0.5;
        }

        // Apply masking (Rounded Rect / Squircle)
        float2 center = u.rect.xy + u.rect.zw * 0.5;
        float2 half_size = u.rect.zw * 0.5;
        float2 local = in.world_pos - center;
        float d;
        if (u.is_squircle == 1) d = sdSquircle(local, half_size, u.radii.x);
        else d = sdRoundedBox(local, half_size, u.radii);
        float alpha_mask = 1.0 - smoothstep(-1.0, 1.0, d);

        // Standard glass tinting / blending
        float3 glass_color = color_linear.rgb;
        float3 final_rgb = mix(blurred, glass_color, color_linear.a);
        final_color = float4(final_rgb, alpha_mask);
    }
    else if (u.mode == 9) {
        // Aurora
        float t = u.time;
        float2 p = in.uv * 2.0 - 1.0;
        float3 color = float3(0.0);
        for (float i = 1.0; i < 4.0; i += 1.0) {
            float2 shifted_uv = p + float2(sin(t * 0.2 + i * 1.5) * 0.5, cos(t * 0.3 + i * 2.1) * 0.5);
            float dist = length(shifted_uv);
            float wave = sin(dist * 5.0 - t * 2.0) * 0.5 + 0.5;
            float hue = fract(t * 0.05 + i * 0.2);
            float3 c = hsv2rgb(float3(hue, 0.7, 0.8));
            color += c * (0.15 / (dist + 0.1)) * wave;
        }
        color = clamp(color * 0.8, 0.05, 0.8);
        final_color = float4(color, 1.0);
    }
    else if (u.mode == 10) {
        // Grid
        float2 pos = in.world_pos;
        float zoom = u.elevation;
        float g1 = 40.0 * zoom;
        if (zoom < 0.5) g1 *= 2.0;
        float2 f1 = abs(fract(pos / g1 - 0.5) - 0.5) / fwidth(pos / g1);
        float line1 = 1.0 - min(min(f1.x, f1.y), 1.0);
        float g2 = g1 * 5.0;
        float2 f2 = abs(fract(pos / g2 - 0.5) - 0.5) / fwidth(pos / g2);
        float line2 = 1.0 - min(min(f2.x, f2.y), 1.0);
        float alpha = line1 * 0.15 + line2 * 0.35;
        final_color = float4(color_linear.rgb, color_linear.a * alpha * (0.5 + 0.5 * smoothstep(0.0, 0.2, zoom)));
    }
    else if (u.mode == 11) {
        // Line segment
        float2 p = in.world_pos;
        float2 a = u.rect.xy;
        float2 b = u.rect.zw;
        float2 pa = p - a;
        float2 ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        float d = length(pa - ba * h) - u.border_width * 0.5;
        float alpha = 1.0 - smoothstep(-0.5, 0.5, d);
        final_color = float4(color_linear.rgb, color_linear.a * alpha);
    }
    else if (u.mode == 12) {
        // Cyberpunk Grid Background
        float2 pos = in.world_pos;
        float t = u.time;
        float2 g = abs(fract(pos / 50.0 - 0.5) - 0.5) / (fwidth(pos / 50.0) + 0.02);
        float grid = 1.0 - min(min(g.x, g.y), 1.0);
        
        float pulse = sin(t * 2.0 - length(pos) * 0.01) * 0.5 + 0.5;
        // Slightly HDR color to trigger a soft glow via bloom
        float3 col = mix(float3(0.02, 0.02, 0.05), float3(0.0, 1.2, 1.5), grid * pulse * 0.5);
        final_color = float4(col, 1.0);
    }
    else {
        final_color = color_linear;
    }
    
    return final_color;
}

fragment float4 fs_main(VertexOut in [[stage_in]],
                         constant DrawUniforms &u [[buffer(0)]],
                         texture2d<float> tex [[texture(0)]],
                         texture2d<float> tex2 [[texture(1)]],
                         sampler s [[sampler(0)]]) {
    ShapeData d;
    d.rect = u.rect; d.radii = u.radii; d.color = in.color; // Use vertex color as fill
    d.border_color = u.border_color;
    d.glow_color = u.glow_color; d.border_width = u.border_width;
    d.elevation = u.elevation; d.glow_strength = u.glow_strength;
    d.lut_intensity = u.lut_intensity; d.mode = u.mode; d.is_squircle = u.is_squircle;
    d.viewport_size = u.viewport_size; d.time = u.time;
    return resolve_shape(in, d, tex, tex2, s, u.time);
}

fragment float4 fs_instanced(VertexOut in [[stage_in]],
                            constant GlobalUniforms &glob [[buffer(0)]],
                            constant ShapeInstance *instances [[buffer(1)]],
                            texture2d<float> tex [[texture(2)]],
                            texture2d<float> tex2 [[texture(3)]],
                            sampler s [[sampler(4)]]) {
    constant ShapeInstance &inst = instances[in.iid];
    ShapeData d;
    d.rect = inst.rect; d.radii = inst.radii; d.color = inst.color;
    d.border_color = inst.border_color;
    d.glow_color = inst.glow_color; 
    d.border_width = inst.params1.x; d.elevation = inst.params1.y;
    d.glow_strength = inst.params1.z; d.lut_intensity = inst.params1.w;
    d.mode = inst.params2.x; d.is_squircle = inst.params2.y;
    d.viewport_size = glob.viewport_size; d.time = glob.time;
    
    float4 color = resolve_shape(in, d, tex, tex2, s, glob.time);
    
    // 2D Normal Mapping Support
    float3 normal = float3(0.5, 0.5, 1.0); // Default up
    // sampler s2 as slot 2... (placeholder for now)
    
    return color;
}

struct FragmentOutput {
    float4 color [[color(0)]];
    float4 aux [[color(1)]];
    float2 velocity [[color(2)]];
};

fragment FragmentOutput fs_instanced_gbuffer(VertexOut in [[stage_in]],
                            constant GlobalUniforms &glob [[buffer(0)]],
                            constant ShapeInstance *instances [[buffer(1)]],
                            texture2d<float> tex [[texture(2)]],
                            texture2d<float> tex2 [[texture(3)]],
                            sampler s [[sampler(4)]]) {

    FragmentOutput out;
    out.color = float4(in.color.rgb, 1.0);
    out.aux = float4(0.0, 0.0, 0.0, 1.0); 
    out.velocity = float2(0.0, 0.0);
    return out;
}

struct CinematicParams {
    float exposure;
    float ca_strength;
    float vignette_intensity;
    float bloom_intensity;
    uint tonemap_mode;
    uint bloom_mode;
    float grain_strength;
    float time;
    float lut_intensity;
    float blur_radius;
    float motion_blur_strength;
    uint debug_mode;
    float2 light_pos;
    float gi_intensity;
    float volumetric_intensity;
    float4 light_color;
    float2 jitter;      // Jitter offset for TAA parity
    float2 render_size; // Native resolution for effects
};

// Helper for 3D LUT sampling
float3 sample_lut_metal(float3 color, texture3d<float> lut, sampler s) {
    // Standard 3D LUT assumes 0-1 range
    return lut.sample(s, max(color, 0.0)).rgb;
}

float3 reinhard(float3 v) {
    return v / (1.0 + v);
}

// Phase 2: Raymarching
float raymarch_shadow(float2 pixel_pos, float2 light_pos, texture2d<float> sdf, sampler s) {
    float2 dir = normalize(light_pos - pixel_pos);
    float max_dist = distance(pixel_pos, light_pos);
    float t = 2.0; 
    float res = 1.0;
    
    float2 size = float2(sdf.get_width(), sdf.get_height());
    
    for (int i = 0; i < 32; i++) {
        float2 p = pixel_pos + dir * t;
        float2 uv = p / size;
        
        float d = sdf.sample(s, uv).r;
        
        if (d < 0.1) {
            return 0.0;
        }
        
        res = min(res, 8.0 * d / t);
        
        t += max(d, 1.0);
        if (t >= max_dist) { break; }
    }
    return clamp(res, 0.0, 1.0);
}

// Phase 3: Volumetric Lighting
float3 volumetric_lighting(float2 pixel_pos, float2 light_posd, texture2d<float> sdf, sampler s, constant CinematicParams &cinema) {
    float2 dir = normalize(light_posd - pixel_pos);
    float max_dist = distance(pixel_pos, light_posd);
    float t = 0.0;
    float accumulation = 0.0;
    float step_size = 20.0;
    float density = 0.002;
    
    float dither = hash(pixel_pos * 0.01); // Simple hash
    t += dither * step_size;

    float2 size = float2(sdf.get_width(), sdf.get_height());

    for (int i = 0; i < 16; i++) {
        if (t >= max_dist) { break; }
        
        float2 p = pixel_pos + dir * t;
        float2 uv = p / size;
        
        float d = sdf.sample(s, uv).r;
        
        if (d > 0.1) {
            float dist_to_light = max_dist - t;
            float falloff = 1.0 / (1.0 + 0.00005 * dist_to_light * dist_to_light);
            accumulation += density * falloff;
        }
        
        t += step_size;
    }
    
    return cinema.light_color.rgb * accumulation * cinema.volumetric_intensity;
}

// Phase 3: Cone Tracing GI
float3 cone_trace_gi(float2 pixel_pos, float2 normal, texture2d<float> sdf, texture2d<float> hdr, sampler s, constant CinematicParams &cinema) {
    if (cinema.gi_intensity <= 0.0) { return float3(0.0); }
    
    float2 size = float2(sdf.get_width(), sdf.get_height());
    float3 indirect = float3(0.0);
    
    int step_count = 8;
    float max_dist = 400.0;
    
    float t = 10.0;
    
    for (int i = 0; i < step_count; i++) {
        float2 p = pixel_pos + normal * t;
        float2 uv = p / size;
        
        float d = sdf.sample(s, uv).r;
        
        if (d < 2.0) {
            float3 radiance = hdr.sample(s, uv, level(2.0)).rgb;
            
            float weight = (1.0 - t / max_dist);
            if (weight > 0.0) {
                indirect += radiance * weight;
            }
            break; 
        }
        
        t += max(d, 5.0);
        if (t >= max_dist) { break; }
    }
    
    return indirect * cinema.gi_intensity * 0.5;
}

// --- Resolve Pass (HDR to LDR + Tone Mapping + Post-effects) ---

struct ResolveVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex ResolveVertexOut vs_resolve(uint vid [[vertex_id]]) {
    ResolveVertexOut out;
    float x = (vid == 2) ? 3.0 : -1.0;
    float y = (vid == 1) ? 3.0 : -1.0;
    out.position = float4(x, y, 0.0, 1.0);
    out.uv = float2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

float3 aces_approx(float3 v) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0, 1.0);
}

fragment float4 fs_resolve(ResolveVertexOut in [[stage_in]],
                            constant CinematicParams &cinema [[buffer(2)]],
                            texture2d<float> hdr_tex [[texture(0)]],
                            sampler s [[sampler(0)]]) {
    float2 uv = in.uv;
    float2 dist_from_center = uv - 0.5;
    
    // 1. Chromatic Aberration
    float3 color;
    float ca = cinema.ca_strength;
    color.r = hdr_tex.sample(s, uv + dist_from_center * ca).r;
    color.g = hdr_tex.sample(s, uv).g;
    color.b = hdr_tex.sample(s, uv - dist_from_center * ca).b;
    
    // 2. Exposure
    color *= cinema.exposure;

    // 3. Tone Mapping
    if (cinema.tonemap_mode == 1) {
        color = aces_approx(color);
    } else if (cinema.tonemap_mode == 2) {
        color = reinhard(color);
    }
    
    // 4. Vignette
    float vignette = 1.0 - dot(dist_from_center, dist_from_center) * 1.5;
    color *= max(vignette, cinema.vignette_intensity);
    
    // 5. Film Grain
    float noise = hash(uv + fract(cinema.time));
    color += (noise - 0.5) * cinema.grain_strength;
    
    // 6. Dithering
    float dither = hash(uv * 10.0) / 255.0;
    color += dither;
    
    return float4(pow(color, 1.0/2.2), 1.0);
}

// --- Bloom Pass Shaders ---

fragment float4 fs_bright_pass(ResolveVertexOut in [[stage_in]],
                               texture2d<float> tex [[texture(0)]],
                               sampler s [[sampler(0)]]) {
    float4 color = tex.sample(s, in.uv);
    float brightness = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0) {
        return color;
    } else {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
}

struct BlurUniforms {
    int horizontal;
};

fragment float4 fs_blur(ResolveVertexOut in [[stage_in]],
                        constant BlurUniforms &u [[buffer(0)]],
                        texture2d<float> tex [[texture(0)]],
                        sampler s [[sampler(0)]]) {
    float weights[5] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
    float2 tex_offset = 1.0 / float2(tex.get_width(), tex.get_height());
    float3 result = tex.sample(s, in.uv).rgb * weights[0];
    
    if (u.horizontal == 1) {
        for (int i = 1; i < 5; ++i) {
            result += tex.sample(s, in.uv + float2(tex_offset.x * i, 0.0)).rgb * weights[i];
            result += tex.sample(s, in.uv - float2(tex_offset.x * i, 0.0)).rgb * weights[i];
        }
    } else {
        for (int i = 1; i < 5; ++i) {
            result += tex.sample(s, in.uv + float2(0.0, tex_offset.y * i)).rgb * weights[i];
            result += tex.sample(s, in.uv - float2(0.0, tex_offset.y * i)).rgb * weights[i];
        }
    }
    return float4(result, 1.0);
}

// Update resolve to composite bloom
fragment float4 fs_resolve_bloom(ResolveVertexOut in [[stage_in]],
                                 texture2d<float> hdr_tex [[texture(0)]],
                                 texture2d<float> bloom_tex [[texture(1)]],
                                 texture3d<float> t_lut [[texture(2)]],
                                 texture2d<float> t_velocity [[texture(3)]],
                                 texture2d<float> t_reflection [[texture(4)]],
                                 texture2d<float> t_aux [[texture(5)]],
                                 texture2d<float> t_extra [[texture(6)]],
                                 texture2d<float> t_sdf [[texture(7)]],
                                 constant CinematicParams &cinema [[buffer(2)]],
                                 sampler s [[sampler(0)]]) {
    // Standard resolve logic
    float4 bloom = bloom_tex.sample(s, in.uv);
    float4 hdr = hdr_tex.sample(s, in.uv);

    // Apply bloom
    hdr += bloom * cinema.bloom_intensity;

    // Output
    return float4(hdr.rgb, 1.0);
}
