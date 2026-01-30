//! OpenGL backend using glow
//! Ported from opengl_backend.cpp
//!
//! Renders DrawList commands using SDF shaders

use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use glow::HasContext;

/// SDF vertex shader source
const VERTEX_SHADER: &str = r#"
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

out vec2 v_uv;
out vec4 v_color;
out vec2 v_pos;

uniform mat4 u_projection;
uniform vec2 u_offset;
uniform float u_scale;

void main() {
    vec2 pos = (a_pos * u_scale) + u_offset;
    gl_Position = u_projection * vec4(pos, 0.0, 1.0);
    v_uv = a_uv;
    
    // Linear Workflow:
    v_color = vec4(pow(a_color.rgb, vec3(2.2)), a_color.a);
    v_pos = pos;
}
"#;

/// SDF fragment shader source
const FRAGMENT_SHADER: &str = r#"
#version 330 core
in vec2 v_uv;
in vec4 v_color;
in vec2 v_pos;

out vec4 frag_color;

uniform sampler2D u_texture;
uniform int u_mode; // 0=solid, 1=sdf_text, 2=rounded_rect, 3=image, 4=blur, 5=image_lut, 6=arc, 7=plot, 8=heatmap, 9=aurora, 10=grid

uniform vec4 u_rect;       // x, y, w, h
uniform vec4 u_radii;      // tl, tr, br, bl
uniform float u_border_width;
uniform vec4 u_border_color;
uniform float u_elevation;
uniform int u_is_squircle;
uniform float u_glow_strength;
uniform vec4 u_glow_color;

uniform sampler3D u_lut;
uniform float u_lut_intensity;

float sdRoundedBox(vec2 p, vec2 b, vec4 r) {
    float radius = r.x; 
    if (p.x > 0.0) radius = r.y;
    if (p.x > 0.0 && p.y > 0.0) radius = r.z;
    if (p.x <= 0.0 && p.y > 0.0) radius = r.w;
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
    vec4 color_linear = vec4(pow(v_color.rgb, vec3(2.2)), v_color.a);
    vec4 final_color = vec4(0.0);

    if (u_mode == 0) {
        final_color = color_linear;
    }
    else if (u_mode == 1) {
        // MSDF/SDF Text
        vec3 msd = texture(u_texture, v_uv).rgb;
        float sd = max(min(msd.r, msd.g), min(max(msd.r, msd.g), msd.b)); 
        float alpha = smoothstep(0.4, 0.6, sd);
        final_color = vec4(color_linear.rgb, color_linear.a * alpha);
    }
    else if (u_mode == 2) {
        vec2 center = u_rect.xy + u_rect.zw * 0.5;
        vec2 half_size = u_rect.zw * 0.5;
        vec2 local = v_pos - center;
        float d;
        if (u_is_squircle == 1) d = sdSquircle(local, half_size, u_radii.x);
        else d = sdRoundedBox(local, half_size, u_radii);
        float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        vec4 bg = color_linear;
        if (u_border_width > 0.0) {
            float interior_alpha = 1.0 - smoothstep(-1.0, 1.0, d + u_border_width);
            vec4 border_col_lin = vec4(pow(u_border_color.rgb, vec3(2.2)), u_border_color.a);
            bg = mix(border_col_lin, color_linear, interior_alpha);
        }
        vec4 main_layer = vec4(bg.rgb, bg.a * alpha);
        vec4 glow_layer = vec4(0.0);
        if (u_glow_strength > 0.0) {
             float glow_factor = exp(-max(d, 0.0) * 0.1) * u_glow_strength;
             vec4 glow_col_lin = vec4(pow(u_glow_color.rgb, vec3(2.2)), u_glow_color.a);
             glow_layer = glow_col_lin * glow_factor;
        }
        vec4 shadow_layer = vec4(0.0);
        if (u_elevation > 0.0) {
            float d1 = sdRoundedBox(local - vec2(0.0, u_elevation * 0.25), half_size, u_radii);
            float a1 = (1.0 - smoothstep(-u_elevation*0.5, u_elevation*0.5, d1)) * 0.4;
            shadow_layer = vec4(0.0, 0.0, 0.0, a1 * color_linear.a);
        }
        vec4 comp = shadow_layer + glow_layer;
        comp.rgb = main_layer.rgb * main_layer.a + comp.rgb * (1.0 - main_layer.a);
        comp.a = max(comp.a, main_layer.a);
        final_color = comp;
    }
    else if (u_mode == 3) {
        vec2 center = u_rect.xy + u_rect.zw * 0.5;
        vec2 half_size = u_rect.zw * 0.5;
        vec2 local = v_pos - center;
        float d = sdRoundedBox(local, half_size, u_radii);
        float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        vec4 tex_col = texture(u_texture, v_uv) * color_linear;
        final_color = vec4(pow(tex_col.rgb, vec3(2.2)), tex_col.a * alpha);
    }
    else if (u_mode == 4) {
        vec2 center = u_rect.xy + u_rect.zw * 0.5;
        vec2 half_size = u_rect.zw * 0.5;
        vec2 local = v_pos - center;
        float d = sdRoundedBox(local, half_size, u_radii);
        float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
        float lod = u_border_width;
        vec4 bg = textureLod(u_texture, v_uv, lod) * color_linear;
        final_color = vec4(bg.rgb, bg.a * alpha);
    }
    else if (u_mode == 10) {
        vec2 pos = v_pos;
        float zoom = u_elevation;
        float g1 = 40.0 * zoom;
        if (zoom < 0.5) g1 *= 2.0;
        vec2 f1 = abs(fract(pos / g1 - 0.5) - 0.5) / fwidth(pos / g1);
        float line1 = 1.0 - min(min(f1.x, f1.y), 1.0);
        float g2 = g1 * 5.0;
        vec2 f2 = abs(fract(pos / g2 - 0.5) - 0.5) / fwidth(pos / g2);
        float line2 = 1.0 - min(min(f2.x, f2.y), 1.0);
        float alpha = line1 * 0.15 + line2 * 0.35;
        final_color = vec4(color_linear.rgb, color_linear.a * alpha * (0.5 + 0.5 * smoothstep(0.0, 0.2, zoom)));
    }
    else if (u_mode == 11) {
        // Mode 11: SDF Line segment
        // u_rect.xy = p0, u_rect.zw = p1
        // u_border_width = thickness
        vec2 p = v_pos;
        vec2 a = u_rect.xy;
        vec2 b = u_rect.zw;
        vec2 pa = p - a;
        vec2 ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        float d = length(pa - ba * h) - u_border_width * 0.5;
        float alpha = 1.0 - smoothstep(-0.5, 0.5, d);
        final_color = vec4(color_linear.rgb, color_linear.a * alpha);
    }
    else {
        final_color = color_linear;
    }
    frag_color = vec4(pow(final_color.rgb, vec3(1.0/2.2)), final_color.a);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

use std::rc::Rc;
pub struct OpenGLBackend {
    gl: Rc<glow::Context>,
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    projection_loc: glow::UniformLocation,
    mode_loc: glow::UniformLocation,
    rect_loc: Option<glow::UniformLocation>,
    radii_loc: Option<glow::UniformLocation>,
    border_width_loc: Option<glow::UniformLocation>,
    border_color_loc: Option<glow::UniformLocation>,
    elevation_loc: Option<glow::UniformLocation>,
    glow_strength_loc: Option<glow::UniformLocation>,
    glow_color_loc: Option<glow::UniformLocation>,
    is_squircle_loc: Option<glow::UniformLocation>,
    offset_loc: Option<glow::UniformLocation>,
    scale_loc: Option<glow::UniformLocation>,
    lut_loc: Option<glow::UniformLocation>,
    lut_intensity_loc: Option<glow::UniformLocation>,
    font_texture: glow::Texture,
    backdrop_texture: glow::Texture,
    heatmap_texture: glow::Texture,
    ping_pong_fbo: [glow::Framebuffer; 2],
    ping_pong_texture: [glow::Texture; 2],
    current_pp_width: u32,
    current_pp_height: u32,
    start_time: std::time::Instant,
}

impl OpenGLBackend {
    pub unsafe fn new(gl: glow::Context) -> Result<Self, String> {
        let gl = Rc::new(gl);
        let vs = gl.create_shader(glow::VERTEX_SHADER)?;
        gl.shader_source(vs, VERTEX_SHADER);
        gl.compile_shader(vs);
        let fs = gl.create_shader(glow::FRAGMENT_SHADER)?;
        gl.shader_source(fs, FRAGMENT_SHADER);
        gl.compile_shader(fs);
        let program = gl.create_program()?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        let projection_loc = gl.get_uniform_location(program, "u_projection").unwrap();
        let mode_loc = gl.get_uniform_location(program, "u_mode").unwrap();
        let rect_loc = gl.get_uniform_location(program, "u_rect");
        let radii_loc = gl.get_uniform_location(program, "u_radii");
        let border_width_loc = gl.get_uniform_location(program, "u_border_width");
        let border_color_loc = gl.get_uniform_location(program, "u_border_color");
        let elevation_loc = gl.get_uniform_location(program, "u_elevation");
        let glow_strength_loc = gl.get_uniform_location(program, "u_glow_strength");
        let glow_color_loc = gl.get_uniform_location(program, "u_glow_color");
        let is_squircle_loc = gl.get_uniform_location(program, "u_is_squircle");
        let offset_loc = gl.get_uniform_location(program, "u_offset");
        let scale_loc = gl.get_uniform_location(program, "u_scale");
        let lut_loc = gl.get_uniform_location(program, "u_lut");
        let lut_intensity_loc = gl.get_uniform_location(program, "u_lut_intensity");

        let vao = gl.create_vertex_array()?;
        gl.bind_vertex_array(Some(vao));
        let vbo = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let stride = std::mem::size_of::<Vertex>() as i32;
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 8);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_f32(2, 4, glow::FLOAT, false, stride, 16);

        let font_texture = gl.create_texture()?;
        let backdrop_texture = gl.create_texture()?;
        let heatmap_texture = gl.create_texture()?;
        let ping_pong_fbo = [gl.create_framebuffer()?, gl.create_framebuffer()?];
        let ping_pong_texture = [gl.create_texture()?, gl.create_texture()?];

        Ok(Self {
            gl, program, vao, vbo, projection_loc, mode_loc, rect_loc, radii_loc,
            border_width_loc, border_color_loc, elevation_loc, glow_strength_loc,
            glow_color_loc, is_squircle_loc, offset_loc, scale_loc, lut_loc, lut_intensity_loc,
            font_texture, backdrop_texture, heatmap_texture, ping_pong_fbo, ping_pong_texture,
            current_pp_width: 0, current_pp_height: 0, start_time: std::time::Instant::now(),
        })
    }

    fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> [f32; 16] {
        let mut m = [0.0; 16];
        m[0] = 2.0 / (r - l);
        m[5] = 2.0 / (t - b);
        m[10] = -2.0 / (f - n);
        m[12] = -(r + l) / (r - l);
        m[13] = -(t + b) / (t - b);
        m[14] = -(f + n) / (f - n);
        m[15] = 1.0;
        m
    }

    fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> [Vertex; 6] {
        let x = pos.x; let y = pos.y; let w = size.x; let h = size.y;
        let c = [color.r, color.g, color.b, color.a];
        [
            Vertex { pos: [x, y], uv: [0.0, 0.0], color: c },
            Vertex { pos: [x, y + h], uv: [0.0, 1.0], color: c },
            Vertex { pos: [x + w, y + h], uv: [1.0, 1.0], color: c },
            Vertex { pos: [x, y], uv: [0.0, 0.0], color: c },
            Vertex { pos: [x + w, y + h], uv: [1.0, 1.0], color: c },
            Vertex { pos: [x + w, y], uv: [1.0, 0.0], color: c },
        ]
    }

    fn quad_vertices_uv(pos: Vec2, size: Vec2, uv: [f32; 4], color: ColorF) -> [Vertex; 6] {
        let x = pos.x; let y = pos.y; let w = size.x; let h = size.y;
        let c = [color.r, color.g, color.b, color.a];
        [
            Vertex { pos: [x, y], uv: [uv[0], uv[1]], color: c },
            Vertex { pos: [x, y + h], uv: [uv[0], uv[3]], color: c },
            Vertex { pos: [x + w, y + h], uv: [uv[2], uv[3]], color: c },
            Vertex { pos: [x, y], uv: [uv[0], uv[1]], color: c },
            Vertex { pos: [x + w, y + h], uv: [uv[2], uv[3]], color: c },
            Vertex { pos: [x + w, y], uv: [uv[2], uv[1]], color: c },
        ]
    }

    unsafe fn upload_and_draw(&self, vertices: &[Vertex]) {
        self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
        self.gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(vertices), glow::DYNAMIC_DRAW);
        self.gl.draw_arrays(glow::TRIANGLES, 0, vertices.len() as i32);
    }

    fn draw_line_primitive(&self, p0: Vec2, p1: Vec2, thickness: f32, color: ColorF) {
        let dir = (p1 - p0).normalized();
        let normal = Vec2::new(-dir.y, dir.x);
        let pad = thickness * 0.5 + 2.0;

        let v0 = p0 + normal * pad - dir * pad;
        let v1 = p0 - normal * pad - dir * pad;
        let v2 = p1 + normal * pad + dir * pad;
        let v3 = p1 - normal * pad + dir * pad;

        let vertices = [
            Vertex { pos: [v0.x, v0.y], uv: [0.0, 0.0], color: [color.r, color.g, color.b, color.a] },
            Vertex { pos: [v1.x, v1.y], uv: [0.0, 0.0], color: [color.r, color.g, color.b, color.a] },
            Vertex { pos: [v2.x, v2.y], uv: [0.0, 0.0], color: [color.r, color.g, color.b, color.a] },
            Vertex { pos: [v1.x, v1.y], uv: [0.0, 0.0], color: [color.r, color.g, color.b, color.a] },
            Vertex { pos: [v3.x, v3.y], uv: [0.0, 0.0], color: [color.r, color.g, color.b, color.a] },
            Vertex { pos: [v2.x, v2.y], uv: [0.0, 0.0], color: [color.r, color.g, color.b, color.a] },
        ];
        unsafe {
            self.gl.uniform_1_i32(Some(&self.mode_loc), 11);
            self.gl.uniform_4_f32(self.rect_loc.as_ref(), p0.x, p0.y, p1.x, p1.y);
            self.gl.uniform_1_f32(self.border_width_loc.as_ref(), thickness);
            self.upload_and_draw(&vertices);
        }
    }
}

impl super::GraphicsBackend for OpenGLBackend {
    fn name(&self) -> &str { "OpenGL" }
    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        unsafe {
            crate::text::FONT_MANAGER.with(|fm| {
                let mut fm = fm.borrow_mut();
                if fm.texture_dirty {
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(self.font_texture));
                    self.gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
                    self.gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGB8 as i32, fm.atlas.width as i32, fm.atlas.height as i32, 0, glow::RGB, glow::UNSIGNED_BYTE, Some(&fm.atlas.texture_data));
                    fm.texture_dirty = false;
                }
            });
            self.gl.viewport(0, 0, width as i32, height as i32);
            self.gl.clear_color(0.05, 0.05, 0.07, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.enable(glow::BLEND);
            self.gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            self.gl.use_program(Some(self.program));
            self.gl.bind_vertex_array(Some(self.vao));
            let projection = Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0);
            self.gl.uniform_matrix_4_f32_slice(Some(&self.projection_loc), false, &projection);
            self.gl.uniform_2_f32(self.offset_loc.as_ref(), 0.0, 0.0);
            self.gl.uniform_1_f32(self.scale_loc.as_ref(), 1.0);

            for cmd in dl.commands() {
                match cmd {
                    DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, .. } => {
                        self.gl.uniform_1_i32(Some(&self.mode_loc), 2);
                        self.gl.uniform_4_f32(self.rect_loc.as_ref(), pos.x, pos.y, size.x, size.y);
                        self.gl.uniform_4_f32(self.radii_loc.as_ref(), radii[0], radii[1], radii[2], radii[3]);
                        self.gl.uniform_1_f32(self.border_width_loc.as_ref(), *border_width);
                        self.gl.uniform_4_f32(self.border_color_loc.as_ref(), border_color.r, border_color.g, border_color.b, border_color.a);
                        self.gl.uniform_1_f32(self.elevation_loc.as_ref(), *elevation);
                        self.gl.uniform_1_i32(self.is_squircle_loc.as_ref(), if *is_squircle { 1 } else { 0 });
                        self.gl.uniform_1_f32(self.glow_strength_loc.as_ref(), *glow_strength);
                        self.gl.uniform_4_f32(self.glow_color_loc.as_ref(), glow_color.r, glow_color.g, glow_color.b, glow_color.a);
                        let vertices = Self::quad_vertices(*pos, *size, *color);
                        self.upload_and_draw(&vertices);
                    }
                    DrawCommand::Text { pos, size, uv, color } => {
                        self.gl.uniform_1_i32(Some(&self.mode_loc), 1);
                        self.gl.active_texture(glow::TEXTURE0);
                        self.gl.bind_texture(glow::TEXTURE_2D, Some(self.font_texture));
                        let vertices = Self::quad_vertices_uv(*pos, *size, *uv, *color);
                        self.upload_and_draw(&vertices);
                    }
                    DrawCommand::Line { p0, p1, thickness, color } => self.draw_line_primitive(*p0, *p1, *thickness, *color),
                    DrawCommand::Bezier { p0, p1, p2, p3, thickness, color } => {
                        let mut points = Vec::new();
                        let tess = crate::draw::path::BezierTessellator::new();
                        tess.tessellate_cubic_recursive(*p0, *p1, *p2, *p3, 0, &mut points);
                        points.push(*p3);
                        let mut prev = *p0;
                        for p in points { self.draw_line_primitive(prev, p, *thickness, *color); prev = p; }
                    }
                    DrawCommand::Grid { pos, size, color, zoom } => {
                        self.gl.uniform_1_i32(Some(&self.mode_loc), 10);
                        self.gl.uniform_1_f32(self.elevation_loc.as_ref(), *zoom);
                        self.gl.uniform_4_f32(self.rect_loc.as_ref(), pos.x, pos.y, size.x, size.y);
                        let vertices = Self::quad_vertices(*pos, *size, *color);
                        self.upload_and_draw(&vertices);
                    }
                    DrawCommand::PushTransform { offset, scale } => {
                        self.gl.uniform_2_f32(self.offset_loc.as_ref(), offset.x, offset.y);
                        self.gl.uniform_1_f32(self.scale_loc.as_ref(), *scale);
                    }
                    DrawCommand::PopTransform => {
                        self.gl.uniform_2_f32(self.offset_loc.as_ref(), 0.0, 0.0);
                        self.gl.uniform_1_f32(self.scale_loc.as_ref(), 1.0);
                    }
                    _ => {}
                }
            }
        }
    }
    fn update_font_texture(&mut self, _w: u32, _h: u32, _data: &[u8]) {}
}
