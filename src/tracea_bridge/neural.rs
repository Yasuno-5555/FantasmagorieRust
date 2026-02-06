//! Tracea Neural Engine for Style Transfer
//!
//! Implements a lightweight inference engine for Fast Style Transfer models using Metal Compute.
//! layers: Conv2D, InstanceNorm, ReLU, ResidualBlock, Upsample.

use super::context::TraceaContext;

#[cfg(feature = "metal")]
use metal::{
    Device, CommandQueue, ComputePipelineState, Buffer, Texture, TextureDescriptor,
    MTLResourceOptions, MTLSize, MTLPixelFormat, MTLTextureUsage,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerParams {
    pub input_dim: [u32; 2],
    pub output_dim: [u32; 2],
    pub kernel_size: u32,
    pub stride: u32,
    pub padding: u32,
    pub channels_in: u32,
    pub channels_out: u32,
    pub _pad: u32,
}

pub struct TraceaNeuralKernel {
    #[cfg(feature = "metal")]
    device: metal::Device,
    #[cfg(feature = "metal")]
    command_queue: metal::CommandQueue,
    
    #[cfg(feature = "metal")]
    conv_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    inst_norm_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    relu_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    add_pipeline: metal::ComputePipelineState,
}

#[cfg(feature = "metal")]
impl TraceaNeuralKernel {
    pub fn new(context: &TraceaContext) -> Result<Self, String> {
        let device = context.device().clone();
        let command_queue = device.new_command_queue();
        
        let shader_src = include_str!("shaders/neural_compute.metal");
        let options = metal::CompileOptions::new();
        let library = device.new_library_with_source(shader_src, &options)
            .map_err(|e| format!("Neural compile error: {}", e))?;
            
        let mk_pipeline = |name: &str| -> Result<metal::ComputePipelineState, String> {
            let func = library.get_function(name, None)
                .map_err(|e| format!("Fn {} not found: {}", name, e))?;
            device.new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Pipeline {} error: {}", name, e))
        };
        
        let conv_pipeline = mk_pipeline("conv2d_3x3")?;
        let inst_norm_pipeline = mk_pipeline("instance_norm")?;
        let relu_pipeline = mk_pipeline("relu_activ")?;
        let add_pipeline = mk_pipeline("eltwise_add")?;
        
        Ok(Self {
            device: device.clone(), // Clone for struct to avoid move issue
            command_queue,
            conv_pipeline,
            inst_norm_pipeline,
            relu_pipeline,
            add_pipeline,
        })
    }
    
    pub fn forward_conv(
        &self,
        input: &Texture,
        output: &Texture,
        weights: &Buffer,
        bias: &Buffer,
        params: LayerParams,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.conv_pipeline);
        encoder.set_texture(0, Some(input));
        encoder.set_texture(1, Some(output));
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(bias), 0);
        
        let p_buf = self.device.new_buffer_with_data(
            bytemuck::bytes_of(&params).as_ptr() as *const _,
            std::mem::size_of::<LayerParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(2, Some(&p_buf), 0);
        
        let w = output.width();
        let h = output.height();
        encoder.dispatch_threads(
            MTLSize { width: w, height: h, depth: 1 }, // Handling channels inside shader or loop?
            // Usually, distinct threads per (x,y), loop over input channels
            // For simple implementation, assuming output texture handles slicing or just RGBA inputs.
            // Let's assume standard texture processing (RGBA).
            // If channels > 4, we need texture arrays or multiple textures.
            // For this bridge, we support RGBA (4 channels) processing primarily.
            MTLSize { width: 16, height: 16, depth: 1 }
        );
        
        encoder.end_encoding();
        command_buffer.commit();
        // Check performance: in real inference, we chain commands in one buffer.
    }
}

#[cfg(not(feature = "metal"))]
impl TraceaNeuralKernel {
    pub fn new(_context: &TraceaContext) -> Result<Self, String> {
        Err("Not supported".into())
    }
}
