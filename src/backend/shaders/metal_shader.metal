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
};

struct Uniforms {
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

vertex VertexOut vs_main(Vertex v [[stage_in]],
                         constant Uniforms &u [[buffer(1)]]) {
    VertexOut out;
    float2 pos = (v.pos * u.scale) + u.offset;
    out.position = u.projection * float4(pos, 0.0, 1.0);
    out.uv = v.uv;
    // Linear Workflow:
    out.color = float4(pow(v.color.rgb, 2.2), v.color.a);
    out.world_pos = pos;
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

fragment float4 fs_main(VertexOut in [[stage_in]],
                         constant Uniforms &u [[buffer(1)]],
                         texture2d<float> tex [[texture(0)]],
                         texture2d<float> tex2 [[texture(1)]],
                         sampler s [[sampler(0)]]) {
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
             float glow_factor = exp(-max(d, 0.0) * 0.1) * u.glow_strength;
             float4 glow_col_lin = float4(pow(u.glow_color.rgb, 2.2), u.glow_color.a);
             glow_layer = glow_col_lin * glow_factor;
        }
        float4 shadow_layer = float4(0.0);
        if (u.elevation > 0.0) {
            float d1 = sdRoundedBox(local - float2(0.0, u.elevation * 0.25), half_size, u.radii);
            float a1 = (1.0 - smoothstep(-u.elevation*0.5, u.elevation*0.5, d1)) * 0.4;
            shadow_layer = float4(0.0, 0.0, 0.0, a1 * color_linear.a);
        }
        float4 comp = shadow_layer + glow_layer;
        comp.rgb = main_layer.rgb * main_layer.a + comp.rgb * (1.0 - main_layer.a);
        comp.a = max(comp.a, main_layer.a);
        final_color = comp;
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
    else {
        final_color = color_linear;
    }
    
    // Gamma correction for output
    return float4(pow(final_color.rgb, 1.0/2.2), final_color.a);
}
