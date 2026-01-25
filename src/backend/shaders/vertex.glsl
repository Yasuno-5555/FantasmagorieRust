#version 450
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
void main() {
    vec2 pos = (a_pos * pc.u_scale) + pc.u_offset;
    gl_Position = u_global.u_projection * vec4(pos, 0.0, 1.0);
    v_uv = a_uv;
    v_color = vec4(pow(a_color.rgb, vec3(2.2)), a_color.a);
    v_pos = pos;
}
