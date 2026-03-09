//! Tracea Visibility Kernel
//! HZB-based occlusion culling for 2D sprites

use super::context::TraceaContext;

/// Visibility kernel for GPU-driven culling
pub struct TraceaVisibilityKernel {
    #[cfg(feature = "wgpu")]
    wgpu_state: Option<WgpuVisibilityState>,
}

#[cfg(feature = "wgpu")]
#[cfg(feature = "wgpu")]
struct WgpuVisibilityState {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullingUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub num_instances: u32,
    pub hzb_mip_levels: u32,
    pub _pad: [u32; 2],
}

impl TraceaVisibilityKernel {
    #[cfg(feature = "wgpu")]
    pub fn new_wgpu(context: &TraceaContext) -> Result<Self, String> {
        let device = context.wgpu_device();
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tracea Visibility"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/visibility.wgsl"))),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Visibility BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Visibility Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Visibility Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        Ok(Self {
            wgpu_state: Some(WgpuVisibilityState { pipeline, bind_group_layout }),
        })
    }
    
    #[cfg(feature = "wgpu")]
    pub fn dispatch(
        &self,
        context: &TraceaContext,
        uniforms: &CullingUniforms,
        instances: &wgpu::Buffer,
        hzb: &wgpu::TextureView,
        visible_indices: &wgpu::Buffer,
        visible_counter: &wgpu::Buffer,
    ) -> Result<(), String> {
        use wgpu::util::DeviceExt;
        let state = self.wgpu_state.as_ref().ok_or("WGPU state not initialized")?;
        let device = context.wgpu_device();
        let queue = context.wgpu_queue();
        
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Culling Uniforms"),
            contents: bytemuck::cast_slice(&[*uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Visibility Bind Group"),
            layout: &state.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: instances.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(hzb) },
                wgpu::BindGroupEntry { binding: 3, resource: visible_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: visible_counter.as_entire_binding() },
            ],
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visibility Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Visibility Pass"), timestamp_writes: None });
            cpass.set_pipeline(&state.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (uniforms.num_instances + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        Ok(())
    }
    
    #[cfg(not(feature = "wgpu"))]
    pub fn new_wgpu(_context: &TraceaContext) -> Result<Self, String> {
        Err("WGPU not enabled".into())
    }
}
