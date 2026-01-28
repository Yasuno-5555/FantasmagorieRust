// Fantasmagorie HLSL Shader
// Visual Revolution Port (Squircle, Glow, Mesh Gradient)

struct VertexInput {
    float2 pos : POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR;
};

struct PixelInput {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
    float2 world_pos : TEXCOORD1;
};

cbuffer Constants : register(b0) {
    matrix projection;
    float4 rect;         // x, y, w, h
    float4 radii;        // tl, tr, br, bl
    float4 border_color;
    float4 glow_color;   // Added
    
    int mode;            // 0=solid, 1=text, 2=rounded, 5=mesh
    float border_width;
    float elevation;
    int is_squircle;     // Added
    
    float glow_strength; // Added
    float3 _padding;
};

Texture2D font_texture : register(t0);
SamplerState font_sampler : register(s0);

// Vertex Shader
PixelInput VSMain(VertexInput input) {
    PixelInput output;
    output.pos = mul(projection, float4(input.pos, 0.0, 1.0));
    output.uv = input.uv;
    
    // Linear Workflow: sRGB -> Linear
    output.color = float4(pow(abs(input.color.rgb), 2.2), input.color.a);
    
    output.world_pos = input.pos;
    return output;
}

// SDF Helpers
float sdRoundedBox(float2 p, float2 b, float4 r) {
    float radius = r.x; // tl
    if (p.x > 0.0) radius = r.y; // tr
    if (p.x > 0.0 && p.y > 0.0) radius = r.z; // br
    if (p.x <= 0.0 && p.y > 0.0) radius = r.w; // bl
    
    float2 q = abs(p) - b + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

float sdSquircle(float2 p, float2 b, float r) {
    float2 q = abs(p) - b + r;
    float2 start = max(q, 0.0);
    float n = 4.0;
    float2 p_n = pow(start, n);
    float len = pow(p_n.x + p_n.y, 1.0/n);
    return len + min(max(q.x, q.y), 0.0) - r;
}

// Fragment Shader
float4 PSMain(PixelInput input) : SV_TARGET {
    float4 final_color = float4(0,0,0,0);
    
    if (mode == 0) {
        // Solid
        final_color = input.color;
    }
    else if (mode == 1) {
        // Text
        float dist = font_texture.Sample(font_sampler, input.uv).r;
        float alpha = smoothstep(0.4, 0.6, dist);
        final_color = float4(input.color.rgb, input.color.a * alpha);
    }
    else if (mode == 2) {
        // Shape
        float2 center = rect.xy + rect.zw * 0.5;
        float2 half_size = rect.zw * 0.5;
        float2 local = input.world_pos - center;
        
        float d = 0.0;
        if (is_squircle == 1) {
            d = sdSquircle(local, half_size, radii.x);
        } else {
            d = sdRoundedBox(local, half_size, radii);
        }
        
        float aa = 1.0;
        float alpha = 1.0 - smoothstep(-aa, aa, d);
        
        float4 bg = input.color;
        
        // Border
        if (border_width > 0.0) {
            float interior_alpha = 1.0 - smoothstep(-aa, aa, d + border_width);
            float4 border_col_lin = float4(pow(abs(border_color.rgb), 2.2), border_color.a);
            bg = lerp(border_col_lin, input.color, interior_alpha);
        }
        
        // Hairline (Inner Stroke)
        if (alpha > 0.01) {
            float border_alpha = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5));
            float4 hairline = float4(1.0, 1.0, 1.0, 0.15);
            bg = lerp(bg, hairline, border_alpha);
        }
        
        float4 main_layer = float4(bg.rgb, bg.a * alpha);
        
        // Glow
         float4 glow_layer = float4(0,0,0,0);
         if (glow_strength > 0.0) {
             float glow_factor = exp(-max(d, 0.0) * 0.1) * glow_strength;
             float4 glow_col_lin = float4(pow(abs(glow_color.rgb), 2.2), glow_color.a);
             glow_layer = glow_col_lin * glow_factor;
         }
         
         // Shadow
         float4 shadow_layer = float4(0,0,0,0);
         if (elevation > 0.0) {
             float2 offset1 = float2(0.0, elevation * 0.25);
             float d1 = sdRoundedBox(local - offset1, half_size, radii);
             float a1 = (1.0 - smoothstep(-elevation*0.5, elevation*0.5, d1)) * 0.4;
             
             float2 offset2 = float2(0.0, elevation * 1.5);
             float d2 = sdRoundedBox(local - offset2, half_size, radii);
             float a2 = (1.0 - smoothstep(-elevation*3.0, elevation*3.0, d2)) * 0.2;
             
             float shadow_alpha = max(a1, a2) * input.color.a;
             shadow_layer = float4(0,0,0, shadow_alpha);
         }
         
         // Composite
         float4 comp = shadow_layer;
         comp = comp + glow_layer;
         comp = float4(
             main_layer.rgb * main_layer.a + comp.rgb * (1.0 - main_layer.a),
             max(comp.a, main_layer.a)
         );
         
         final_color = comp;
    }
    else if (mode == 5) {
        // Mesh Gradient (Aurora)
        float t = time; // Updated from elevation
        float2 uv = input.pos.xy / rect.zw; // Screen space to UV
        
        // Flip Y if needed for DX coords? DX is top-left usually for window, but clip space is -1..1.
        // Actually input.pos is world pos (pixel coords).
        
        float2 p1 = float2(0.5 + 0.3*sin(t*0.5), 0.5 + 0.3*cos(t*0.3));
        float2 p2 = float2(0.2 + 0.4*sin(t*0.7 + 1.0), 0.8 + 0.2*cos(t*0.5 + 2.0));
        float2 p3 = float2(0.8 + 0.2*sin(t*0.4 + 4.0), 0.2 + 0.5*cos(t*0.6 + 3.0));
        
        float aspect = rect.z / rect.w;
        float2 uv_aspect = float2(uv.x * aspect, uv.y);
        float2 p1_aspect = float2(p1.x * aspect, p1.y);
        float2 p2_aspect = float2(p2.x * aspect, p2.y);
        float2 p3_aspect = float2(p3.x * aspect, p3.y);
        
        float d1 = length(uv_aspect - p1_aspect);
        float d2 = length(uv_aspect - p2_aspect);
        float d3 = length(uv_aspect - p3_aspect);
        
        float3 c1 = float3(0.1, 0.0, 0.3);
        float3 c2 = float3(0.0, 0.2, 0.4);
        float3 c3 = float3(0.2, 0.0, 0.1);
        
        float w1 = 1.0 / (d1 * d1 + 0.01);
        float w2 = 1.0 / (d2 * d2 + 0.01);
        float w3 = 1.0 / (d3 * d3 + 0.01);
        
        float3 aurora = (c1*w1 + c2*w2 + c3*w3) / (w1 + w2 + w3);
        
        // Simple noise
        float noise = frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453) * 0.05;
        aurora += noise;
        
        final_color = float4(aurora, 1.0);
    }
    
    // Linear -> sRGB
    return float4(pow(abs(final_color.rgb), 1.0/2.2), final_color.a);
}
