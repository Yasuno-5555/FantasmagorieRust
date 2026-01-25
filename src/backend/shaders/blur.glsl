#version 450
// Kawase Blur Shader - Separate Pass
// Run twice: horizontal then vertical

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform BlurUniforms {
    vec2 u_direction;    // (1,0) for H, (0,1) for V
    vec2 u_texel_size;   // 1.0 / texture_size
    float u_radius;
    float _pad0;
    float _pad1;
    float _pad2;
};

layout(set = 0, binding = 1) uniform sampler2D u_texture;

void main() {
    vec4 sum = vec4(0.0);
    
    // Kawase blur kernel (5 samples)
    sum += texture(u_texture, v_uv) * 0.227027;
    
    vec2 offset1 = u_direction * u_texel_size * 1.3846153846;
    sum += texture(u_texture, v_uv + offset1) * 0.3162162162;
    sum += texture(u_texture, v_uv - offset1) * 0.3162162162;
    
    vec2 offset2 = u_direction * u_texel_size * 3.2307692308;
    sum += texture(u_texture, v_uv + offset2) * 0.0702702703;
    sum += texture(u_texture, v_uv - offset2) * 0.0702702703;
    
    frag_color = sum;
}
