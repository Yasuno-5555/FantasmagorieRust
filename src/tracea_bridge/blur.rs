//! Tracea-accelerated Gaussian Blur Kernel
//!
//! Implements separable Gaussian blur using Metal compute shaders.
//! This replaces the fragment shader-based blur with optimized compute.

use super::context::TraceaContext;

#[cfg(feature = "metal")]
use metal::{
    Device, CommandQueue, ComputePipelineState, Texture, TextureDescriptor,
    MTLPixelFormat, MTLTextureUsage, MTLSize, MTLOrigin,
};

/// Separable Gaussian blur kernel using Metal compute
pub struct TraceaBlurKernel {
    #[cfg(feature = "metal")]
    device: metal::Device,
    #[cfg(feature = "metal")]
    command_queue: metal::CommandQueue,
    #[cfg(feature = "metal")]
    horizontal_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    vertical_pipeline: metal::ComputePipelineState,
    pub kernel_radius: u32,
    pub sigma: f32,
}

#[cfg(feature = "metal")]
impl TraceaBlurKernel {
    /// Create a new blur kernel with the given radius
    pub fn new(context: &TraceaContext, radius: u32) -> Result<Self, String> {
        let device = context.device().clone();
        let command_queue = device.new_command_queue();
        
        // Compile the compute shader
        let shader_src = include_str!("shaders/blur_compute.metal");
        let options = metal::CompileOptions::new();
        let library = device.new_library_with_source(shader_src, &options)
            .map_err(|e| format!("Failed to compile blur shader: {}", e))?;
        
        let horizontal_fn = library.get_function("blur_horizontal", None)
            .map_err(|e| format!("Failed to get horizontal blur function: {}", e))?;
        let vertical_fn = library.get_function("blur_vertical", None)
            .map_err(|e| format!("Failed to get vertical blur function: {}", e))?;
        
        let horizontal_pipeline = device.new_compute_pipeline_state_with_function(&horizontal_fn)
            .map_err(|e| format!("Failed to create horizontal pipeline: {}", e))?;
        let vertical_pipeline = device.new_compute_pipeline_state_with_function(&vertical_fn)
            .map_err(|e| format!("Failed to create vertical pipeline: {}", e))?;
        
        // Calculate sigma from radius (common approximation)
        let sigma = (radius as f32) / 3.0;
        
        Ok(Self {
            device,
            command_queue,
            horizontal_pipeline,
            vertical_pipeline,
            kernel_radius: radius,
            sigma,
        })
    }
    
    /// Execute the blur on input texture, writing to output texture
    pub fn execute(
        &self,
        input: &Texture,
        output: &Texture,
        passes: u32,
    ) -> Result<(), String> {
        let width = input.width();
        let height = input.height();
        
        // Create intermediate texture for ping-pong
        let temp_desc = TextureDescriptor::new();
        temp_desc.set_width(width);
        temp_desc.set_height(height);
        temp_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        temp_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let temp_texture = self.device.new_texture(&temp_desc);
        
        let command_buffer = self.command_queue.new_command_buffer();
        
        // Compute optimal threadgroup size
        let threadgroup_size = MTLSize { width: 16, height: 16, depth: 1 };
        let grid_size = MTLSize {
            width: (width + 15) / 16 * 16,
            height: (height + 15) / 16 * 16,
            depth: 1,
        };
        
        // Blur parameters
        let params = BlurParams {
            radius: self.kernel_radius,
            sigma: self.sigma,
            width: width as u32,
            height: height as u32,
        };
        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<BlurParams>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        for pass in 0..passes {
            let (src, dst_h, dst_v) = if pass == 0 {
                (input, &temp_texture, output)
            } else {
                (output, &temp_texture, output)
            };
            
            // Horizontal pass
            {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.horizontal_pipeline);
                encoder.set_texture(0, Some(src));
                encoder.set_texture(1, Some(dst_h));
                encoder.set_buffer(0, Some(&params_buffer), 0);
                encoder.dispatch_threads(grid_size, threadgroup_size);
                encoder.end_encoding();
            }
            
            // Vertical pass
            {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.vertical_pipeline);
                encoder.set_texture(0, Some(dst_h));
                encoder.set_texture(1, Some(dst_v));
                encoder.set_buffer(0, Some(&params_buffer), 0);
                encoder.dispatch_threads(grid_size, threadgroup_size);
                encoder.end_encoding();
            }
        }
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
    
    /// Update blur radius (recomputes sigma)
    pub fn set_radius(&mut self, radius: u32) {
        self.kernel_radius = radius;
        self.sigma = (radius as f32) / 3.0;
    }
}

/// Parameters passed to the compute shader
#[repr(C)]
#[derive(Clone, Copy)]
struct BlurParams {
    radius: u32,
    sigma: f32,
    width: u32,
    height: u32,
}

#[cfg(not(feature = "metal"))]
impl TraceaBlurKernel {
    pub fn new(_context: &TraceaContext, _radius: u32) -> Result<Self, String> {
        Err("TraceaBlurKernel only supported on macOS".to_string())
    }
}
