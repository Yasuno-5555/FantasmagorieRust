//! Shared shader types and uniform structures
//! 
//! Design principles:
//! - Uniform structures split to respect Push Constants 128B limit
//! - std140/std430 alignment via bytemuck
//! - Coordinate system handled via projection matrix

use bytemuck::{Pod, Zeroable};

/// Global Uniforms - bound once per frame (Uniform Buffer)
/// Must match WGSL struct layout (80 bytes)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GlobalUniforms {
    pub projection: [[f32; 4]; 4], // 64 bytes at offset 0
    pub time: f32,                  // 4 bytes at offset 64
    pub _pad0: f32,                 // 4 bytes at offset 68 (padding for vec2 alignment to 8 bytes)
    pub viewport_size: [f32; 2],    // 8 bytes at offset 72 -> total 80 bytes
}

/// Cinematic post-processing parameters
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, serde::Serialize, serde::Deserialize)]
pub struct CinematicParams {
    pub exposure: f32,
    pub ca_strength: f32,
    pub vignette_intensity: f32,
    pub bloom_intensity: f32,
    pub tonemap_mode: u32,
    pub bloom_mode: u32,
    pub grain_strength: f32,
    pub time: f32,
    pub lut_intensity: f32,
    pub blur_radius: f32,
    pub motion_blur_strength: f32,
    pub debug_mode: u32,
    pub light_pos: [f32; 2],
    pub gi_intensity: f32,
    pub volumetric_intensity: f32,
    pub light_color: [f32; 4],
    pub jitter: [f32; 2],            // 8 bytes at offset 80
    pub render_size: [f32; 2],       // 8 bytes at offset 88 -> total 96 bytes
}

impl Default for CinematicParams {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            ca_strength: 0.0015,
            vignette_intensity: 0.7,
            bloom_intensity: 0.4,
            tonemap_mode: 1, // Aces
            bloom_mode: 1,   // Soft
            grain_strength: 0.0,
            time: 0.0,
            lut_intensity: 1.0,
            blur_radius: 0.0,
            motion_blur_strength: 0.0,
            debug_mode: 0,
            light_pos: [500.0, 300.0],
            gi_intensity: 0.5,       // Default GI strength
            volumetric_intensity: 0.0, // Default off for now
            light_color: [1.0, 0.9, 0.7, 1.0],
            jitter: [0.0, 0.0],
            render_size: [1280.0, 720.0],
        }
    }
}

/// Audio reactive parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct AudioParams {
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub _pad: f32,
}

/// Blur pass parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct BlurParams {
    /// Direction (1.0, 0.0) or (0.0, 1.0)
    pub direction: [f32; 2],
    /// Blur sigma (radius/strength)
    pub sigma: f32,
    pub _pad: f32,
}

/// Shape Instance Data - for batching/instancing
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShapeInstance {
    /// x, y, w, h
    pub rect: [f32; 4],
    /// tl, tr, br, bl
    pub radii: [f32; 4],
    /// rgba fill color
    pub color: [f32; 4],
    /// rgba
    pub border_color: [f32; 4],
    /// rgba
    pub glow_color: [f32; 4],
    /// border_width, elevation, glow_strength, lut_intensity
    pub params1: [f32; 4],
    /// mode, is_squircle, _r1, _r2
    pub params2: [i32; 4],
    /// velocity_x, velocity_y, reflectivity, roughness
    pub material: [f32; 4],
    /// normal_map_id (i32), distortion, emissive_intensity, parallax_factor
    pub pbr_params: [f32; 4],
}

/// Per-draw uniforms - legacy/fallback path
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawUniforms {
    pub projection: [f32; 16],
    pub rect: [f32; 4],
    pub radii: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub offset: [f32; 2],
    pub scale: f32,
    pub border_width: f32,
    pub elevation: f32,
    pub glow_strength: f32,
    pub lut_intensity: f32,
    pub mode: i32,
    pub is_squircle: i32,
    pub time: f32,
    pub viewport_size: [f32; 2],
}
// Total: 16*4 + 8 + 4*7 + 4 = 104 bytes ✅ Under 128B limit

/// Shader rendering modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ShaderMode {
    Solid = 0,
    SdfText = 1,
    Shape = 2,
    Image = 3,
    // Note: Glass/Blur is handled as separate render pass, not a mode
    Lut = 5,
    Arc = 6,
    Plot = 7,
    Heatmap = 8,
    Aurora = 9,
    Custom = 100,
}

/// Render pass types (Blur handled separately from modes)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPassType {
    /// Main rendering pass with all modes
    Main,
    /// Horizontal blur pass
    BlurHorizontal,
    /// Vertical blur pass  
    BlurVertical,
}

/// Create projection matrix with Y-flip option
pub fn create_projection(width: f32, height: f32, flip_y: bool, z_near_far: (f32, f32)) -> [[f32; 4]; 4] {
    let (l, r, b, t) = (0.0, width, height, 0.0);
    let (n, f) = z_near_far;
    
    let mut proj = [
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 1.0 / (f - n), 0.0],
        [-(r + l) / (r - l), -(t + b) / (t - b), -n / (f - n), 1.0],
    ];
    
    // Y-flip for Vulkan/wgpu/DX12
    if flip_y {
        proj[1][1] = -proj[1][1];
        proj[3][1] = -proj[3][1];
    }
    
    proj
}

/// Backend-specific projection parameters
pub struct ProjectionParams {
    /// Flip Y axis (Vulkan/wgpu/DX12 = true, OpenGL = false)
    pub flip_y: bool,
    /// Z depth range (OpenGL = -1..1, others = 0..1)
    pub z_range: (f32, f32),
}

impl ProjectionParams {
    pub fn opengl() -> Self {
        Self { flip_y: false, z_range: (-1.0, 1.0) }
    }
    
    pub fn vulkan() -> Self {
        Self { flip_y: true, z_range: (0.0, 1.0) }
    }
    
    pub fn wgpu() -> Self {
        Self { flip_y: false, z_range: (0.0, 1.0) }
    }
    
    pub fn dx12() -> Self {
        Self { flip_y: false, z_range: (0.0, 1.0) }
    }
}
