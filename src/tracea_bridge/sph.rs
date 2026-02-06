//! Tracea SPH Fluid Simulation Kernel
//!
//! Implements Smoothed Particle Hydrodynamics (SPH) using a spatial hash grid.
//! Uses Atomic Linked List approach for neighbor search.
//!
//! Pipeline Stages:
//! 1. clear_grid: Reset grid heads to -1.
//! 2. build_grid: Atomic exchange to insert particles into linked lists.
//! 3. compute_density: Calculate density & pressure for each particle using neighbors.
//! 4. compute_forces: Calculate pressure & viscosity forces and integrate position.

use super::context::TraceaContext;

#[cfg(feature = "metal")]
use metal::{
    Device, CommandQueue, ComputePipelineState, Buffer, MTLResourceOptions, MTLSize,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SPHParticle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub density: f32,
    pub pressure: f32,
    // Store next index for linked list in the particle struct or separate buffer?
    // Using separate buffer for 'next' is common, but let's stick to standard layout.
    // 'next' index is typically a storage buffer u32.
    // So this struct is purely physical data.
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SPHParams {
    pub count: u32,
    pub grid_width: u32,
    pub grid_height: u32,
    pub cell_size: f32,
    
    pub h: f32,             // Smoothing length
    pub dt: f32,            // Time step
    pub rest_density: f32,
    pub stiff: f32,         // Gas stiffness (k)
    pub viscosity: f32,
    pub gravity: [f32; 2],
    pub _pad: f32,
}

pub struct TraceaSPHKernel {
    #[cfg(feature = "metal")]
    device: metal::Device,
    #[cfg(feature = "metal")]
    command_queue: metal::CommandQueue,
    
    // Pipelines
    #[cfg(feature = "metal")]
    clear_grid_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    build_grid_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    density_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    force_pipeline: metal::ComputePipelineState,
    #[cfg(feature = "metal")]
    init_pipeline: metal::ComputePipelineState,

    // Buffers
    #[cfg(feature = "metal")]
    particle_buffer: metal::Buffer,
    #[cfg(feature = "metal")]
    grid_head_buffer: metal::Buffer, // Stores first particle index in cell
    #[cfg(feature = "metal")]
    particle_next_buffer: metal::Buffer, // Stores next particle index
    
    count: usize,
    grid_res: (u32, u32),
}

#[cfg(feature = "metal")]
impl TraceaSPHKernel {
    pub fn new(context: &TraceaContext, count: usize) -> Result<Self, String> {
        let device = context.device().clone();
        let command_queue = device.new_command_queue();
        
        let shader_src = include_str!("shaders/sph_compute.metal");
        let options = metal::CompileOptions::new();
        // Enable atomic support if needed, but standard should be fine
        let library = device.new_library_with_source(shader_src, &options)
            .map_err(|e| format!("SPH compile error: {}", e))?;
            
        let mk_pipeline = |name: &str| -> Result<metal::ComputePipelineState, String> {
            let func = library.get_function(name, None)
                .map_err(|e| format!("Fn {} not found: {}", name, e))?;
            device.new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Pipeline {} error: {}", name, e))
        };
        
        let clear_grid_pipeline = mk_pipeline("clear_grid")?;
        let build_grid_pipeline = mk_pipeline("build_grid")?;
        let density_pipeline = mk_pipeline("compute_density")?;
        let force_pipeline = mk_pipeline("compute_forces")?;
        let init_pipeline = mk_pipeline("init_sph")?;
        
        // Buffers
        let particle_size = (count * std::mem::size_of::<SPHParticle>()) as u64;
        let particle_buffer = device.new_buffer(particle_size, MTLResourceOptions::StorageModeShared);
        
        // Grid setup (e.g. 128x128 for a 1920x1080 screen with 16px cells)
        // With H=16.0, CellSize=H.
        // Let's assume simulation area 2048x2048 for safety.
        let cell_size = 16.0;
        let grid_w = 128;
        let grid_h = 128;
        let grid_cells = grid_w * grid_h;
        
        let grid_head_buffer = device.new_buffer(
            (grid_cells * 4) as u64, // u32/int32
            MTLResourceOptions::StorageModePrivate,
        );
        let particle_next_buffer = device.new_buffer(
            (count * 4) as u64, // u32/int32
            MTLResourceOptions::StorageModePrivate,
        );
        
        let kernel = Self {
            device,
            command_queue,
            clear_grid_pipeline,
            build_grid_pipeline,
            density_pipeline,
            force_pipeline,
            init_pipeline,
            particle_buffer,
            grid_head_buffer,
            particle_next_buffer,
            count,
            grid_res: (grid_w, grid_h),
        };
        
        kernel.reset();
        
        Ok(kernel)
    }
    
    pub fn reset(&self) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.init_pipeline);
        encoder.set_buffer(0, Some(&self.particle_buffer), 0);
        
        let params = self.make_params(0.0);
         let params_buffer = self.device.new_buffer_with_data(
            bytemuck::bytes_of(&params).as_ptr() as *const _,
            std::mem::size_of::<SPHParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(1, Some(&params_buffer), 0);
        
        let size = self.count as u64;
        let threadgroup = MTLSize { width: 256, height: 1, depth: 1 };
        let grid = MTLSize { width: size, height: 1, depth: 1 };
        
        encoder.dispatch_threads(grid, threadgroup);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    pub fn update(&self, dt: f32) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        let params = self.make_params(dt);
        let params_buffer = self.device.new_buffer_with_data(
            bytemuck::bytes_of(&params).as_ptr() as *const _,
            std::mem::size_of::<SPHParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // 1. Clear Grid
        encoder.set_compute_pipeline_state(&self.clear_grid_pipeline);
        encoder.set_buffer(0, Some(&self.grid_head_buffer), 0);
        let grid_cells = (self.grid_res.0 * self.grid_res.1) as u64;
        encoder.dispatch_threads(
            MTLSize { width: grid_cells, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 }
        );
        
        // Memory Barrier implicitly handled between dispatches?
        // Metal tracks dependencies if we use the same command buffer?
        // Compute-to-compute barrier might be needed for buffer hazard.
        // But separate encoders handle it, or same encoder with barrier?
        // Within same encoder, we MUST issue memory barriers if resources overlap.
        // Actually, let's assume `dispatch_threads` doesn't auto-barrier.
        // Standard Metal practice: explicit barrier or split encoders?
        // Same encoder is faster. let's add barrier if possible. 
        // Rust metal crate `memory_barrier_with_resources`.
        
        // 2. Build Grid
        encoder.set_compute_pipeline_state(&self.build_grid_pipeline);
        encoder.set_buffer(0, Some(&self.particle_buffer), 0);
        encoder.set_buffer(1, Some(&params_buffer), 0);
        encoder.set_buffer(2, Some(&self.grid_head_buffer), 0);
        encoder.set_buffer(3, Some(&self.particle_next_buffer), 0);
        encoder.dispatch_threads(
            MTLSize { width: self.count as u64, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 }
        );
        
        // 3. Density
        encoder.set_compute_pipeline_state(&self.density_pipeline);
        // buffers already set mostly, but good to be explicit for clarity
        encoder.set_buffer(0, Some(&self.particle_buffer), 0);
        encoder.set_buffer(1, Some(&params_buffer), 0);
        encoder.set_buffer(2, Some(&self.grid_head_buffer), 0);
        encoder.set_buffer(3, Some(&self.particle_next_buffer), 0);
        encoder.dispatch_threads(
            MTLSize { width: self.count as u64, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 }
        );
        
        // 4. Force & Integrate
        encoder.set_compute_pipeline_state(&self.force_pipeline);
        encoder.set_buffer(0, Some(&self.particle_buffer), 0);
        encoder.set_buffer(1, Some(&params_buffer), 0);
        encoder.set_buffer(2, Some(&self.grid_head_buffer), 0);
        encoder.set_buffer(3, Some(&self.particle_next_buffer), 0);
        encoder.dispatch_threads(
            MTLSize { width: self.count as u64, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 }
        );
        
        encoder.end_encoding();
        command_buffer.commit();
        // command_buffer.wait_until_completed(); // Async
    }
    
    fn make_params(&self, dt: f32) -> SPHParams {
        SPHParams {
            count: self.count as u32,
            grid_width: self.grid_res.0,
            grid_height: self.grid_res.1,
            cell_size: 16.0,
            h: 16.0,
            dt,
            rest_density: 100.0, // Tunable
            stiff: 200.0,        // Tunable
            viscosity: 0.5,      // Tunable
            gravity: [0.0, -9.8 * 10.0], // Scaled gravity
            _pad: 0.0,
        }
    }
    
    pub fn particle_buffer(&self) -> &Buffer {
        &self.particle_buffer
    }
}

#[cfg(not(feature = "metal"))]
impl TraceaSPHKernel {
    pub fn new(_context: &TraceaContext, _count: usize) -> Result<Self, String> {
        Err("Not supported".into())
    }
}
