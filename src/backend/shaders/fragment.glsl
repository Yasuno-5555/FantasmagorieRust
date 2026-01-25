#version 450
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
layout(push_constant) uniform DrawUniforms {
    vec4 u_rect;
    vec4 u_radii;
    vec4 u_border_color;
    vec4 u_glow_color;
    vec2 u_offset;
    float u_scale;
    float u_border_width;
    float u_elevation;
    float u_glow_strength;
    float u_lut_intensity;
    int u_mode;
    int u_is_squircle;
    int _pad1;
} pc;
float sdRoundedBox(vec2 p, vec2 b, vec4 r) {
    float radius = r.x; 
    if (p.x > 0.0) { radius = r.y; }
    if (p.x > 0.0 && p.y > 0.0) { radius = r.z; }
    if (p.x <= 0.0 && p.y > 0.0) { radius = r.w; }
    vec2 q = abs(p) - b + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}
float sdSquircle(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    vec2 start = max(q, 0.0);
    float n = 4.0;
    float px = pow(start.x, n);
    float py = pow(start.y, n);
    float len = pow(px + py, 1.0 / n);
    return len + min(max(q.x, q.y), 0.0) - r;
}
void main() {
    vec4 col_linear = vec4(pow(v_color.rgb, vec3(2.2)), v_color.a);
    vec4 final_color = vec4(0.0);
    if (pc.u_mode == 0) { final_color = col_linear; }
    else if (pc.u_mode == 1) {
        float dist = texture(u_texture, v_uv).r;
        float alpha = smoothstep(0.4, 0.6, dist);
        final_color = vec4(col_linear.rgb, col_linear.a * alpha);
    }
    else if (pc.u_mode == 2) {
        vec2 center = pc.u_rect.xy + pc.u_rect.zw * 0.5;
        vec2 half_size = pc.u_rect.zw * 0.5;
        vec2 local = v_pos - center;
        float d;
        if (pc.u_is_squircle == 1) { d = sdSquircle(local, half_size, pc.u_radii.x); }
        else { d = sdRoundedBox(local, half_size, pc.u_radii); }
        float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        vec4 bg = col_linear;
        if (pc.u_border_width > 0.0) {
            float i_alpha = 1.0 - smoothstep(-1.0, 1.0, d + pc.u_border_width);
            vec4 b_lin = vec4(pow(pc.u_border_color.rgb, vec3(2.2)), pc.u_border_color.a);
            bg = mix(b_lin, col_linear, i_alpha);
        }
        final_color = vec4(bg.rgb, bg.a * alpha);
    }
    else if (pc.u_mode == 9) {
        float t = u_global.u_time * 0.5;
        vec2 uv = gl_FragCoord.xy / u_global.u_viewport;
        vec3 bg = mix(vec3(0.1, 0.1, 0.12), vec3(0.2, 0.1, 0.3), sin(uv.x + t)*0.5+0.5);
        final_color = vec4(bg, 1.0);
    }
    else { final_color = col_linear; }
    frag_color = vec4(pow(final_color.rgb, vec3(1.0 / 2.2)), final_color.a);
}
