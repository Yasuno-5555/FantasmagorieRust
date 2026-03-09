//! Tracea Indirect Dispatch Kernel
//! GPU-driven command generation for sprites, particles, and compute

use super::context::TraceaContext;

/// Indirect dispatch kernel for GPU-driven rendering
pub struct TraceaIndirectKernel {
    #[cfg(feature = "wgpu")]
    wgpu_state: Option<WgpuIndirectState>,
}

#[cfg(feature = "wgpu")]
struct WgpuIndirectState {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    draw_commands: wgpu::Buffer,
    dispatch_commands: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl TraceaIndirectKernel {
    #[cfg(feature = "wgpu")]
    pub fn new_wgpu(context: &TraceaContext) -> Result<Self, String> {
        use wgpu::util::DeviceExt;
        
        let device = context.wgpu_device();
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tracea Indirect"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/indirect.wgsl"))),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Indirect BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Indirect Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Indirect Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        // Pre-allocate command buffers
        let draw_commands = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Draw Indirect Commands"),
            contents: bytemuck::cast_slice(&[DrawIndirectCommand { vertex_count: 0, instance_count: 0, first_vertex: 0, first_instance: 0 }; 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });
        
        let dispatch_commands = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dispatch Indirect Commands"),
            contents: bytemuck::cast_slice(&[DispatchIndirectCommand { x: 0, y: 0, z: 0 }; 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });
        
        Ok(Self {
            wgpu_state: Some(WgpuIndirectState { pipeline, bind_group_layout, draw_commands, dispatch_commands }),
        })
    }
    
    #[cfg(feature = "wgpu")]
    pub fn dispatch(
        &self,
        context: &TraceaContext,
        counter_buffer: &wgpu::Buffer,
    ) -> Result<(), String> {
        let state = self.wgpu_state.as_ref().ok_or("WGPU state not initialized")?;
        let device = context.wgpu_device();
        let queue = context.wgpu_queue();
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Indirect Bind Group"),
            layout: &state.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: counter_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: state.draw_commands.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: state.dispatch_commands.as_entire_binding() },
            ],
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Indirect Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Indirect Pass"), timestamp_writes: None });
            cpass.set_pipeline(&state.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        Ok(())
    }
    
    #[cfg(feature = "wgpu")]
    pub fn draw_commands(&self) -> &wgpu::Buffer {
        &self.wgpu_state.as_ref().unwrap().draw_commands
    }
    
    #[cfg(not(feature = "wgpu"))]
    pub fn new_wgpu(_context: &TraceaContext) -> Result<Self, String> {
        Err("WGPU not enabled".into())
    }
}
