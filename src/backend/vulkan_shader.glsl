#version 450
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;
layout(location = 2) out vec2 fragPos;

layout(binding = 0) uniform UniformBufferObject {
    mat4 projection;
} ubo;

layout(push_constant) uniform PushConstants {
    vec4 rect;
    vec4 radii;
    vec4 border_color;
    vec4 glow_color;
    int mode;
    float border_width;
    float elevation;
    int is_squircle;
    float glow_strength;
    float _padding;
    vec2 _pad2;
} pc;

void main() {
    // Linear Workflow: sRGB -> Linear
    vec3 linearColor = pow(abs(inColor.rgb), vec3(2.2));
    fragColor = vec4(linearColor, inColor.a);
    
    fragUV = inUV;
    fragPos = inPos;
    gl_Position = ubo.projection * vec4(inPos, 0.0, 1.0);
}
//---FRAGMENT---
#version 450
layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;
layout(location = 2) in vec2 fragPos;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    vec4 rect;
    vec4 radii;
    vec4 border_color;
    vec4 glow_color;
    int mode;
    float border_width;
    float elevation;
    int is_squircle;
    float glow_strength;
    float _padding;
    vec2 _pad2;
} pc;

// SDF Helpers
float sdRoundedBox(vec2 p, vec2 b, vec4 r) {
    float radius = r.x; // tl
    if (p.x > 0.0) radius = r.y; // tr
    if (p.x > 0.0 && p.y > 0.0) radius = r.z; // br
    if (p.x <= 0.0 && p.y > 0.0) radius = r.w; // bl
    
    vec2 q = abs(p) - b + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

float sdSquircle(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    vec2 start = max(q, 0.0);
    float n = 4.0;
    vec2 p_n = pow(start, vec2(n));
    float len = pow(p_n.x + p_n.y, 1.0/n);
    return len + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    vec4 final_color = vec4(0.0);
    
    if (pc.mode == 0) {
        // Solid
        final_color = fragColor;
    } 
    else if (pc.mode == 1) {
        // Text
        float dist = texture(texSampler, fragUV).r;
        float alpha = smoothstep(0.4, 0.6, dist);
        final_color = vec4(fragColor.rgb, fragColor.a * alpha);
    }
    else if (pc.mode == 2) {
        // Shape
        vec2 center = pc.rect.xy + pc.rect.zw * 0.5;
        vec2 half_size = pc.rect.zw * 0.5;
        vec2 local = fragPos - center;
        
        float d = 0.0;
        if (pc.is_squircle == 1) {
            d = sdSquircle(local, half_size, pc.radii.x);
        } else {
            d = sdRoundedBox(local, half_size, pc.radii);
        }
        
        float aa = 1.0;
        float alpha = 1.0 - smoothstep(-aa, aa, d);
        
        vec4 bg = fragColor;
        
        // Border
        if (pc.border_width > 0.0) {
            float interior_alpha = 1.0 - smoothstep(-aa, aa, d + pc.border_width);
            vec4 border_col_lin = vec4(pow(abs(pc.border_color.rgb), vec3(2.2)), pc.border_color.a);
            bg = mix(border_col_lin, fragColor, interior_alpha);
        }
        
        // Hairline (Inner Stroke)
        if (alpha > 0.01) {
            float border_alpha = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5));
            vec4 hairline = vec4(1.0, 1.0, 1.0, 0.15);
            bg = mix(bg, hairline, border_alpha);
        }
        
        vec4 main_layer = vec4(bg.rgb, bg.a * alpha);
        
        // Glow
         vec4 glow_layer = vec4(0,0,0,0);
         if (pc.glow_strength > 0.0) {
             float glow_factor = exp(-max(d, 0.0) * 0.1) * pc.glow_strength;
             vec4 glow_col_lin = vec4(pow(abs(pc.glow_color.rgb), vec3(2.2)), pc.glow_color.a);
             glow_layer = glow_col_lin * glow_factor;
         }
         
         // Shadow (Simplified for Vulkan Port MVP)
         vec4 shadow_layer = vec4(0,0,0,0);
         if (pc.elevation > 0.0) {
             float shadow_alpha = (1.0 - smoothstep(-pc.elevation*2.0, pc.elevation*2.0, d)) * 0.3 * fragColor.a;
             shadow_layer = vec4(0,0,0, shadow_alpha);
         }
         
         // Composite
         vec4 comp = shadow_layer;
         comp = comp + glow_layer;
         comp = vec4(
             main_layer.rgb * main_layer.a + comp.rgb * (1.0 - main_layer.a),
             max(comp.a, main_layer.a)
         );
         
         final_color = comp;
    }
    else if (pc.mode == 5) {
        // Mesh Gradient
        float t = pc.elevation; // Time
        vec2 uv = fragPos / pc.rect.zw;
        
        vec2 p1 = vec2(0.5 + 0.3*sin(t*0.5), 0.5 + 0.3*cos(t*0.3));
        vec2 p2 = vec2(0.2 + 0.4*sin(t*0.7 + 1.0), 0.8 + 0.2*cos(t*0.5 + 2.0));
        vec2 p3 = vec2(0.8 + 0.2*sin(t*0.4 + 4.0), 0.2 + 0.5*cos(t*0.6 + 3.0));
        
        float aspect = pc.rect.z / pc.rect.w;
        vec2 uv_aspect = vec2(uv.x * aspect, uv.y);
        vec2 p1_aspect = vec2(p1.x * aspect, p1.y);
        vec2 p2_aspect = vec2(p2.x * aspect, p2.y);
        vec2 p3_aspect = vec2(p3.x * aspect, p3.y);
        
        float d1 = length(uv_aspect - p1_aspect);
        float d2 = length(uv_aspect - p2_aspect);
        float d3 = length(uv_aspect - p3_aspect);
        
        vec3 c1 = vec3(0.1, 0.0, 0.3);
        vec3 c2 = vec3(0.0, 0.2, 0.4);
        vec3 c3 = vec3(0.2, 0.0, 0.1);
        
        float w1 = 1.0 / (d1 * d1 + 0.01);
        float w2 = 1.0 / (d2 * d2 + 0.01);
        float w3 = 1.0 / (d3 * d3 + 0.01);
        
        vec3 aurora = (c1*w1 + c2*w2 + c3*w3) / (w1 + w2 + w3);
        final_color = vec4(aurora, 1.0);
    }
    
    // Linear -> sRGB
    outColor = vec4(pow(abs(final_color.rgb), vec3(1.0/2.2)), final_color.a);
}
