//! Shared shader types and uniform structures
//! 
//! Design principles:
//! - Uniform structures split to respect Push Constants 128B limit
//! - std140/std430 alignment via bytemuck
//! - Coordinate system handled via projection matrix

use bytemuck::{Pod, Zeroable};

/// Global uniforms - bound once per frame (Uniform Buffer)
/// Contains stable per-frame data
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GlobalUniforms {
    pub projection: [f32; 16],    // 64 (0)
    pub viewport_size: [f32; 2], // 8 (64)
    pub time: f32,               // 4 (72)
    pub _pad: [f32; 237],        // 948 (76) -> Total 1024
}

/// Cinematic post-processing parameters
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CinematicParams {
    pub exposure: f32,
    pub gamma: f32,
    pub fog_density: f32,
    pub _pad: f32,
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

/// Per-draw uniforms - updated per draw call (Push Constants or Dynamic UBO)
/// Must be ≤ 1024 bytes for cross-backend compatibility
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawUniforms {
    pub projection: [f32; 16],    // 64 (0)
    pub rect: [f32; 4],          // 16 (64)
    pub radii: [f32; 4],         // 16 (80)
    pub border_color: [f32; 4],  // 16 (96)
    pub glow_color: [f32; 4],    // 16 (112)
    pub offset: [f32; 2],        // 8 (128)
    pub scale: f32,              // 4 (136)
    pub border_width: f32,       // 4 (140)
    pub elevation: f32,          // 4 (144)
    pub glow_strength: f32,      // 4 (148)
    pub lut_intensity: f32,      // 4 (152)
    pub mode: i32,               // 4 (156)
    pub is_squircle: i32,        // 4 (160)
    pub time: f32,               // 4 (164)
    pub _pad_inner: f32,         // 4 (168)
    pub _pad_to_vec2: f32,       // 4 (172) -> Aligns viewport_size to 8
    pub viewport_size: [f32; 2], // 8 (176)
    pub _pad_to_array: [f32; 2], // 8 (184) -> Aligns array to 16
    pub _pad: [f32; 208],        // 832 (192-1024) -> Total 1024
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PostProcessUniforms {
    pub threshold: f32,          // 4 (0)
    pub _pad_to_vec2: f32,       // 4 (4) -> Aligns direction to 8
    pub direction: [f32; 2],     // 8 (8)
    pub intensity: f32,          // 4 (16)
    pub _pad_to_array: [f32; 3], // 12 (20) -> Aligns array to 32
    pub _pad: [f32; 248],        // 992 (32-1024) -> Total 1024. 248 * 4 = 992.
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlendUniforms {
    pub opacity: f32,            // 4 (0)
    pub mode: u32,               // 4 (4)
    pub _pad: [f32; 254],        // 1016 (8-1024) -> Total 1024
}

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
