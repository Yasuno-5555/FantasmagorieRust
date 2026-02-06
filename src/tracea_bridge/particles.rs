//! Tracea-accelerated Particle System
//!
//! Implements high-performance particle simulation using Metal compute shaders.
//! Supports:
//! - 1M+ particles
//! - Physics update (velocity, position, damping)
//! - Attractor/Repulsor fields
//! - Collision with SDF (from JFA)

use super::context::TraceaContext;

#[cfg(feature = "metal")]
use metal::{
    Device, CommandQueue, ComputePipelineState, Buffer, Texture,
    MTLResourceOptions, MTLSize,
};

/// Particle data structure (GPU compatible)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub color: [f32; 4],
    pub life: f32,
    pub size: f32,
    pub _pad: [f32; 2],
}

/// Simulation parameters
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub dt: f32,
    pub damping: f32,
    pub gravity: [f32; 2],
    pub count: u32,
    pub width: u32,
    pub height: u32,
    pub _pad0: u32, // Pad to 32-byte alignment for attractor_pos
    pub attractor_pos: [f32; 2],
    pub attractor_strength: f32,
    pub _pad1: u32, // Rounding to 48 bytes
}

pub struct TraceaParticleKernel {
    #[cfg(feature = "metal")]
    device: Option<metal::Device>,
    #[cfg(feature = "metal")]
    command_queue: Option<metal::CommandQueue>,
    #[cfg(feature = "metal")]
    update_pipeline: Option<metal::ComputePipelineState>,
    #[cfg(feature = "metal")]
    init_pipeline: Option<metal::ComputePipelineState>,
    #[cfg(feature = "metal")]
    particle_buffer: Option<metal::Buffer>,
    
    #[cfg(feature = "wgpu")]
    wgpu_state: Option<WgpuState>,
    
    count: usize,
}

#[cfg(feature = "metal")]
impl TraceaParticleKernel {
    fn new_metal(context: &TraceaContext, count: usize) -> Result<Self, String> {
        let device = context.device().clone();
        let command_queue = device.new_command_queue();
        
        // Compile shaders
        let shader_src = include_str!("shaders/particles_compute.metal");
        let options = metal::CompileOptions::new();
        let library = device.new_library_with_source(shader_src, &options)
            .map_err(|e| format!("Failed to compile particle shader: {}", e))?;
            
        let update_fn = library.get_function("update_particles", None)
            .map_err(|e| format!("Failed to get update function: {}", e))?;
        let init_fn = library.get_function("init_particles", None)
            .map_err(|e| format!("Failed to get init function: {}", e))?;
            
        let update_pipeline = device.new_compute_pipeline_state_with_function(&update_fn)
            .map_err(|e| format!("Failed to create update pipeline: {}", e))?;
        let init_pipeline = device.new_compute_pipeline_state_with_function(&init_fn)
            .map_err(|e| format!("Failed to create init pipeline: {}", e))?;
            
        // Create particle buffer
        let buffer_size = (count * std::mem::size_of::<Particle>()) as u64;
        let particle_buffer = device.new_buffer(
            buffer_size,
            MTLResourceOptions::StorageModeShared, // Allow CPU init/debug for now
        );
        
        // Initialize particles
        let kernel = Self {
            device: Some(device),
            command_queue: Some(command_queue),
            update_pipeline: Some(update_pipeline),
            init_pipeline: Some(init_pipeline),
            particle_buffer: Some(particle_buffer),
            #[cfg(feature = "wgpu")]
            wgpu_state: None,
            count,
        };
        
        kernel.reset();
        
        Ok(kernel)
    }
    
    /// Reset particles to initial state
    pub fn reset(&self) {
        let command_queue = self.command_queue.as_ref().unwrap();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(self.init_pipeline.as_ref().unwrap());
        encoder.set_buffer(0, Some(self.particle_buffer.as_ref().unwrap()), 0);
        
        let params = SimParams {
            dt: 0.0,
            damping: 0.0,
            gravity: [0.0, 0.0],
            count: self.count as u32,
            width: 1920, // Default bounds
            height: 1080,
            attractor_pos: [0.0, 0.0],
            attractor_strength: 0.0,
            _pad0: 0,
            _pad1: 0,
        };
        let device = self.device.as_ref().unwrap();
        let params_buffer = device.new_buffer_with_data(
            bytemuck::bytes_of(&params).as_ptr() as *const _,
            std::mem::size_of::<SimParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(1, Some(&params_buffer), 0);
        
        let threadgroup_size = MTLSize { width: 256, height: 1, depth: 1 };
        let grid_size = MTLSize {
            width: self.count as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    /// Update particle simulation
    pub fn update(
        &self,
        dt: f32,
        attractor: [f32; 2],
        sdf_texture: Option<&Texture>,
    ) {
        let command_queue = self.command_queue.as_ref().unwrap();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(self.update_pipeline.as_ref().unwrap());
        encoder.set_buffer(0, Some(self.particle_buffer.as_ref().unwrap()), 0);
        
        let params = SimParams {
            dt,
            damping: 0.98,
            gravity: [0.0, -9.8],
            count: self.count as u32,
            width: 1920,
            height: 1080,
            attractor_pos: attractor,
            attractor_strength: 1000.0,
            _pad0: 0,
            _pad1: 0,
        };
        let device = self.device.as_ref().unwrap();
        let params_buffer = device.new_buffer_with_data(
            bytemuck::bytes_of(&params).as_ptr() as *const _,
            std::mem::size_of::<SimParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(1, Some(&params_buffer), 0);
        
        encoder.set_texture(0, sdf_texture.map(|v| &**v));
        
        let threadgroup_size = MTLSize { width: 256, height: 1, depth: 1 };
        let grid_size = MTLSize {
            width: self.count as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        // Don't wait here for max performance in real app, but for test we might
    }
    
    pub fn particle_buffer(&self) -> &Buffer {
        self.particle_buffer.as_ref().unwrap()
    }
}

// WGPU Implementation
#[cfg(feature = "wgpu")]
impl TraceaParticleKernel {
    fn new_wgpu(context: &TraceaContext, count: usize) -> Result<Self, String> {
        let device = context.wgpu_device();
        
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tracea Particles"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/particles.wgsl"))),
        });
        
        // Utils
        let buffer_size = (count * std::mem::size_of::<Particle>()) as u64;
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tracea Particle Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pipeline Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "update_particles",
        });
        
        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle Init Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "init_particles",
        });
        
        // Params Buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Params"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Dummy texture setup for initial bind (since we need valid bind group)
        // Ideally we update bind group every frame or use optional bindings (not supported in WGSL well yet without different pipelines)
        // We act lazy: We store the layout and create bindgroup on update.
        
        Ok(Self {
            #[cfg(feature = "metal")]
            device: None,
            #[cfg(feature = "metal")]
            command_queue: None,
            #[cfg(feature = "metal")]
            update_pipeline: None,
            #[cfg(feature = "metal")]
            init_pipeline: None,
            #[cfg(feature = "metal")]
            particle_buffer: None,
            
            wgpu_state: Some(WgpuState {
                update_pipeline,
                init_pipeline,
                particle_buffer,
                params_buffer,
                bind_group_layout,
            }),
            count,
        })
    }
    
    pub fn update_wgpu(&self, context: &TraceaContext, dt: f32, attractor: [f32; 2], sdf_view: Option<&wgpu::TextureView>, sampler: Option<&wgpu::Sampler>) -> Result<(), String> {
        if let Some(state) = &self.wgpu_state {
            let queue = context.wgpu_queue();
            
            // Update Params
             let params = SimParams {
                dt,
                damping: 0.98,
                gravity: [0.0, -9.8],
                count: self.count as u32,
                width: 1920,
                height: 1080,
                _pad0: 0,
                attractor_pos: attractor,
                attractor_strength: 1000.0,
                _pad1: 0,
            };
            queue.write_buffer(&state.params_buffer, 0, bytemuck::bytes_of(&params));
            
            // Create BindGroup (Dynamic if SDF changes)
            // If SDF is None, we need a dummy or handle it.
            // For now assume caller provides valid SDF/Sampler or we fail/skip.
            if let (Some(view), Some(samp)) = (sdf_view, sampler) {
                let device = context.wgpu_device();
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Particle Work BG"),
                    layout: &state.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: state.particle_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: state.params_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(view) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(samp) },
                    ],
                });
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cpass.set_pipeline(&state.update_pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    let group_count = (self.count as u32 + 63) / 64;
                    cpass.dispatch_workgroups(group_count, 1, 1);
                }
                queue.submit(Some(encoder.finish()));
            }
        }
        Ok(())
    }
    
    pub fn particle_buffer_wgpu(&self) -> Option<&wgpu::Buffer> {
        self.wgpu_state.as_ref().map(|s| &s.particle_buffer)
    }
}

// Helper structs
#[cfg(feature = "wgpu")]
struct WgpuState {
    update_pipeline: wgpu::ComputePipeline,
    init_pipeline: wgpu::ComputePipeline,
    particle_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
}

// Common Implementation
impl TraceaParticleKernel {
    pub fn new(context: &TraceaContext, count: usize) -> Result<Self, String> {
        #[cfg(feature = "metal")]
        {
            if context.is_ready() {
                return Self::new_metal(context, count);
            }
        }
        
        #[cfg(feature = "wgpu")]
        {
            if context.is_ready() {
                return Self::new_wgpu(context, count);
            }
        }
        
        Err("No active backend context for Tracea particles".into())
    }
}
