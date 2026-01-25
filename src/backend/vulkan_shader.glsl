// Fantasmagorie Unified Vulkan Shader
// Naga-compatible style

// --- Vertex ---
#ifdef VERTEX_SHADER
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec2 v_pos;

layout(binding = 0) uniform GlobalUniforms {
    mat4 u_projection;
    vec2 u_viewport;
    float u_time;
    float _pad0;
} u_global;

layout(push_constant) uniform PushConstants {
    vec4 rect;
    vec4 radii;
    vec4 border_color;
    vec4 glow_color;
    vec2 offset;
    float scale;
    float border_width;
    float elevation;
    float glow_strength;
    float lut_intensity;
    int mode;
    int is_squircle;
    int _pad;
} pc;

void main() {
    vec2 pos = (a_pos * pc.scale) + pc.offset;
    gl_Position = u_global.u_projection * vec4(pos, 0.0, 1.0);
    v_uv = a_uv;
    v_color = vec4(pow(a_color.rgb, vec3(2.2)), a_color.a);
    v_pos = pos;
}
#endif

// --- Fragment ---
#ifdef FRAGMENT_SHADER
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec2 v_pos;

layout(location = 0) out vec4 frag_color;

layout(binding = 0) uniform GlobalUniforms {
    mat4 u_projection;
    vec2 u_viewport;
    float u_time;
    float _pad0;
} u_global;

layout(binding = 1) uniform sampler2D u_texture;

layout(push_constant) uniform PushConstants {
    vec4 rect;
    vec4 radii;
    vec4 border_color;
    vec4 glow_color;
    vec2 offset;
    float scale;
    float border_width;
    float elevation;
    float glow_strength;
    float lut_intensity;
    int mode;
    int is_squircle;
    int _pad;
} pc;

float sdRoundedBox(vec2 p, vec2 b, vec4 r) {
    float radius = r.x; 
    if (p.x > 0.0) { radius = r.y; }
    if (p.x > 0.0 && p.y > 0.0) { radius = r.z; }
    if (p.x <= 0.0 && p.y > 0.0) { radius = r.w; }
    vec2 q = abs(p) - b + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

void main() {
    vec4 col_linear = vec4(pow(v_color.rgb, vec3(2.2)), v_color.a);
    vec4 final_color = col_linear;
    
    if (pc.mode == 0) { 
        final_color = col_linear; 
    }
    else if (pc.mode == 1) { // Text
        float dist = texture(u_texture, v_uv).r;
        float alpha = smoothstep(0.4, 0.6, dist);
        final_color = vec4(col_linear.rgb, col_linear.a * alpha);
    }
    else if (pc.mode == 2) { // Generic Shape
        vec2 center = pc.rect.xy + pc.rect.zw * 0.5;
        vec2 half_size = pc.rect.zw * 0.5;
        vec2 local = v_pos - center;
        float d = sdRoundedBox(local, half_size, pc.radii);
        float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        vec4 bg = col_linear;
        if (pc.border_width > 0.0) {
            float i_alpha = 1.0 - smoothstep(-1.0, 1.0, d + pc.border_width);
            vec4 b_lin = vec4(pow(pc.border_color.rgb, vec3(2.2)), pc.border_color.a);
            bg = mix(b_lin, col_linear, i_alpha);
        }
        final_color = vec4(bg.rgb, bg.a * alpha);
    }
    else if (pc.mode == 9) { // Aurora
        float t = u_global.u_time * 0.5;
        vec2 uv = gl_FragCoord.xy / u_global.u_viewport;
        vec3 bg = mix(vec3(0.05, 0.05, 0.1), vec3(0.1, 0.2, 0.4), sin(uv.x + t) * 0.5 + 0.5);
        final_color = vec4(bg, 1.0);
    }

    frag_color = vec4(pow(final_color.rgb, vec3(1.0 / 2.2)), final_color.a);
}
#endif
