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
    /// MVP projection matrix (64 bytes)
    /// Y-flip and Z-range handled here per backend
    pub projection: [[f32; 4]; 4],
    /// Viewport size (8 bytes)
    pub viewport_size: [f32; 2],
    /// Frame time for animations (4 bytes)
    pub time: f32,
    /// Padding (4 bytes)
    pub _pad: f32,
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
/// Must be ≤ 128 bytes for Push Constants compatibility
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DrawUniforms {
    /// MVP projection matrix (64 bytes)
    pub projection: [f32; 16],
    /// Rect bounds: x, y, w, h (16 bytes)
    pub rect: [f32; 4],
    /// Corner radii: tl, tr, br, bl (16 bytes)  
    pub radii: [f32; 4],
    /// Border RGBA (16 bytes)
    pub border_color: [f32; 4],
    /// Glow RGBA (16 bytes)
    pub glow_color: [f32; 4],
    /// Offset for transform (8 bytes)
    pub offset: [f32; 2],
    /// Scale factor (4 bytes)
    pub scale: f32,
    /// Border width (4 bytes)
    pub border_width: f32,
    /// Elevation for shadow (4 bytes)
    pub elevation: f32,
    /// Glow strength (4 bytes)
    pub glow_strength: f32,
    /// LUT intensity (4 bytes)
    pub lut_intensity: f32,
    /// Rendering mode 0-9 (4 bytes)
    pub mode: i32,
    /// Is squircle flag (4 bytes)
    pub is_squircle: i32,
    /// Time in seconds (4 bytes)
    pub time: f32,
    /// Padding (8 bytes)
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
        Self { flip_y: true, z_range: (0.0, 1.0) }
    }
    
    pub fn dx12() -> Self {
        Self { flip_y: true, z_range: (0.0, 1.0) }
    }
}
