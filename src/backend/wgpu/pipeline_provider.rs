use std::sync::Arc;
use crate::backend::hal::GpuPipelineProvider;

pub struct WgpuPipelineProvider {
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
}

impl WgpuPipelineProvider {
    pub fn new(device: Arc<wgpu::Device>, surface_format: wgpu::TextureFormat) -> Self {
        Self { device, surface_format }
    }
}

impl GpuPipelineProvider for WgpuPipelineProvider {
    type RenderPipeline = wgpu::RenderPipeline;
    type ComputePipeline = wgpu::ComputePipeline;
    type BindGroupLayout = wgpu::BindGroupLayout;

    fn create_render_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
        layout: Option<&Self::BindGroupLayout>,
    ) -> Result<Self::RenderPipeline, String> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let layouts = if let Some(l) = layout { vec![l] } else { vec![] };
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            push_constant_ranges: &[],
        });

        Ok(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // Vertex buffer layout should be configurable? For now keep it simple or take from somewhere
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float, // Always render to HDR internal buffer
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }))
    }

    fn create_compute_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
        layout: Option<&Self::BindGroupLayout>,
    ) -> Result<Self::ComputePipeline, String> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let layouts = if let Some(l) = layout { vec![l] } else { vec![] };
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            push_constant_ranges: &[],
        });

        Ok(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        }))
    }
}
