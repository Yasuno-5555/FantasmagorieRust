//! Tracea-accelerated Jump Flood Algorithm (JFA) for SDF Generation
//!
//! Implements JFA using Metal compute shaders for fast SDF generation.
//! This is used for shadow casting, Voronoi diagrams, and distance fields.

use super::context::TraceaContext;

#[cfg(feature = "metal")]
use metal::{
    Device, CommandQueue, ComputePipelineState, Texture, TextureDescriptor,
    MTLPixelFormat, MTLTextureUsage, MTLSize,
};

/// JFA (Jump Flood Algorithm) kernel for SDF generation
pub struct TraceaJFAKernel {
    #[cfg(feature = "metal")]
    device: metal::Device,
    #[cfg(feature = "metal")]
    command_queue: metal::CommandQueue,
    #[cfg(feature = "metal")]
    seed_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    flood_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    resolve_pipeline: metal::ComputePipelineState,
}

/// JFA parameters passed to compute shader
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JFAParams {
    pub step_size: u32,
    pub width: u32,
    pub height: u32,
    pub max_distance: f32,
}

#[cfg(feature = "metal")]
impl TraceaJFAKernel {
    /// Create a new JFA kernel
    pub fn new(context: &TraceaContext) -> Result<Self, String> {
        let device = context.device().clone();
        let command_queue = device.new_command_queue();
        
        // Compile the compute shader
        let shader_src = include_str!("shaders/jfa_compute.metal");
        let options = metal::CompileOptions::new();
        let library = device.new_library_with_source(shader_src, &options)
            .map_err(|e| format!("Failed to compile JFA shader: {}", e))?;
        
        let seed_fn = library.get_function("jfa_seed", None)
            .map_err(|e| format!("Failed to get seed function: {}", e))?;
        let flood_fn = library.get_function("jfa_flood", None)
            .map_err(|e| format!("Failed to get flood function: {}", e))?;
        let resolve_fn = library.get_function("jfa_resolve", None)
            .map_err(|e| format!("Failed to get resolve function: {}", e))?;
        
        let seed_pipeline = device.new_compute_pipeline_state_with_function(&seed_fn)
            .map_err(|e| format!("Failed to create seed pipeline: {}", e))?;
        let flood_pipeline = device.new_compute_pipeline_state_with_function(&flood_fn)
            .map_err(|e| format!("Failed to create flood pipeline: {}", e))?;
        let resolve_pipeline = device.new_compute_pipeline_state_with_function(&resolve_fn)
            .map_err(|e| format!("Failed to create resolve pipeline: {}", e))?;
        
        Ok(Self {
            device,
            command_queue,
            seed_pipeline,
            flood_pipeline,
            resolve_pipeline,
        })
    }
    
    /// Generate SDF from a seed texture (binary mask)
    /// 
    /// - `seed_texture`: Binary mask where non-zero pixels are seeds
    /// - `output_texture`: Output SDF texture (RG16Float recommended)
    /// - `max_distance`: Maximum distance to compute (in pixels)
    pub fn generate_sdf(
        &self,
        seed_texture: &Texture,
        output_texture: &Texture,
        max_distance: f32,
    ) -> Result<(), String> {
        let width = seed_texture.width() as u32;
        let height = seed_texture.height() as u32;
        
        // Create ping-pong textures for JFA
        let tex_desc = TextureDescriptor::new();
        tex_desc.set_width(width as u64);
        tex_desc.set_height(height as u64);
        tex_desc.set_pixel_format(MTLPixelFormat::RG32Float); // Store (closest_x, closest_y)
        tex_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        
        let temp_a = self.device.new_texture(&tex_desc);
        let temp_b = self.device.new_texture(&tex_desc);
        
        let command_buffer = self.command_queue.new_command_buffer();
        
        // Compute optimal threadgroup size
        let threadgroup_size = MTLSize { width: 8, height: 8, depth: 1 };
        let grid_size = MTLSize {
            width: ((width as u64 + 7) / 8) * 8,
            height: ((height as u64 + 7) / 8) * 8,
            depth: 1,
        };
        
        // --- SEED PASS ---
        {
            let params = JFAParams {
                step_size: 0,
                width,
                height,
                max_distance,
            };
            let params_buffer = self.device.new_buffer_with_data(
                bytemuck::bytes_of(&params).as_ptr() as *const _,
                std::mem::size_of::<JFAParams>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.seed_pipeline);
            encoder.set_texture(0, Some(seed_texture));
            encoder.set_texture(1, Some(&temp_a));
            encoder.set_buffer(0, Some(&params_buffer), 0);
            encoder.dispatch_threads(grid_size, threadgroup_size);
            encoder.end_encoding();
        }
        
        // --- FLOOD PASSES ---
        // Calculate number of passes: log2(max(width, height))
        let max_dim = width.max(height);
        let mut step = 1u32 << (31 - max_dim.leading_zeros());
        let mut ping = true;
        
        while step > 0 {
            let params = JFAParams {
                step_size: step,
                width,
                height,
                max_distance,
            };
            let params_buffer = self.device.new_buffer_with_data(
                bytemuck::bytes_of(&params).as_ptr() as *const _,
                std::mem::size_of::<JFAParams>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            
            let (src, dst) = if ping { (&temp_a, &temp_b) } else { (&temp_b, &temp_a) };
            
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.flood_pipeline);
            encoder.set_texture(0, Some(src));
            encoder.set_texture(1, Some(dst));
            encoder.set_buffer(0, Some(&params_buffer), 0);
            encoder.dispatch_threads(grid_size, threadgroup_size);
            encoder.end_encoding();
            
            step /= 2;
            ping = !ping;
        }
        
        // --- RESOLVE PASS ---
        {
            let final_src = if ping { &temp_a } else { &temp_b };
            
            let params = JFAParams {
                step_size: 0,
                width,
                height,
                max_distance,
            };
            let params_buffer = self.device.new_buffer_with_data(
                bytemuck::bytes_of(&params).as_ptr() as *const _,
                std::mem::size_of::<JFAParams>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.resolve_pipeline);
            encoder.set_texture(0, Some(final_src));
            encoder.set_texture(1, Some(output_texture));
            encoder.set_buffer(0, Some(&params_buffer), 0);
            encoder.dispatch_threads(grid_size, threadgroup_size);
            encoder.end_encoding();
        }
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
    
    /// Generate Voronoi diagram from seed points
    pub fn generate_voronoi(
        &self,
        seed_texture: &Texture,
        output_texture: &Texture,
    ) -> Result<(), String> {
        // Voronoi uses same algorithm but outputs seed ID instead of distance
        self.generate_sdf(seed_texture, output_texture, f32::MAX)
    }
}

#[cfg(not(feature = "metal"))]
impl TraceaJFAKernel {
    pub fn new(_context: &TraceaContext) -> Result<Self, String> {
        Err("TraceaJFAKernel only supported on macOS".to_string())
    }
}
