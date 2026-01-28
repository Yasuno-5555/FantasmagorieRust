
use wgpu::util::DeviceExt;
use std::sync::Arc;

pub struct ComputePipelines {
    pub k4_pipeline: wgpu::ComputePipeline,
    pub k4_bind_group_layout: wgpu::BindGroupLayout,
    
    pub k5_pipeline: wgpu::ComputePipeline,
    pub k5_bind_group_layout: wgpu::BindGroupLayout,
    
    pub k6_update_pipeline: wgpu::ComputePipeline,
    pub k6_spawn_pipeline: wgpu::ComputePipeline,
    pub k6_bind_group_layout: wgpu::BindGroupLayout,
    
    pub k13_pipeline: wgpu::ComputePipeline,
    pub k13_bind_group_layout: wgpu::BindGroupLayout,
    
    pub k8_pipeline: wgpu::ComputePipeline,
    pub k8_bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        // --- K4: Cinematic Resolver ---
        let k4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K4 Resolver Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/k4_resolver.wgsl").into()),
        });

        let k4_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K4 Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let k4_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K4 Pipeline Layout"),
            bind_group_layouts: &[&k4_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let k4_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K4 Resolver Pipeline"),
            layout: Some(&k4_pipeline_layout),
            module: &k4_shader,
            entry_point: "main",
        });

        // --- K5: JFA ---
        let k5_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K5 JFA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/k5_jfa.wgsl").into()),
        });

        let k5_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K5 Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let k5_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K5 Pipeline Layout"),
            bind_group_layouts: &[&k5_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..32,
            }],
        });

        let k5_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K5 JFA Pipeline"),
            layout: Some(&k5_pipeline_layout),
            module: &k5_shader,
            entry_point: "main",
        });

        // --- K6: Particles ---
        let k6_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K6 Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/k6_particle.wgsl").into()),
        });

        let k6_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K6 Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let k6_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K6 Pipeline Layout"),
            bind_group_layouts: &[&k6_bind_group_layout],
            push_constant_ranges: &[],
        });

        let k6_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K6 Update Pipeline"),
            layout: Some(&k6_pipeline_layout),
            module: &k6_shader,
            entry_point: "update",
        });

        let k6_spawn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K6 Spawn Pipeline"),
            layout: Some(&k6_pipeline_layout),
            module: &k6_shader,
            entry_point: "spawn",
        });

        // --- K13: Indirect Dispatch ---
        let k13_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K13 Indirect Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/k13_indirect.wgsl").into()),
        });

        let k13_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K13 Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let k13_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K13 Pipeline Layout"),
            bind_group_layouts: &[&k13_bind_group_layout],
            push_constant_ranges: &[],
        });

        let k13_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K13 Indirect Pipeline"),
            layout: Some(&k13_pipeline_layout),
            module: &k13_shader,
            entry_point: "main",
        });

        // --- K8: Visibility & Occlusion ---
        let k8_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K8 Visibility Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/k8_visibility.wgsl").into()),
        });

        let k8_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K8 Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let k8_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K8 Pipeline Layout"),
            bind_group_layouts: &[&k8_bind_group_layout],
            push_constant_ranges: &[],
        });

        let k8_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K8 Visibility Pipeline"),
            layout: Some(&k8_pipeline_layout),
            module: &k8_shader,
            entry_point: "main",
        });

        Self {
            k4_pipeline,
            k4_bind_group_layout,
            k5_pipeline,
            k5_bind_group_layout,
            k6_update_pipeline,
            k6_spawn_pipeline,
            k6_bind_group_layout,
            k13_pipeline,
            k13_bind_group_layout,
            k8_pipeline,
            k8_bind_group_layout,
        }
    }
}
