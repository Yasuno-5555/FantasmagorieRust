use std::sync::Arc;
use std::collections::HashMap;

pub struct PipelineManager {
    pub main_pipeline: Arc<wgpu::RenderPipeline>,
    pub instanced_pipeline: Arc<wgpu::RenderPipeline>,
    pub instanced_gbuffer_pipeline: Arc<wgpu::RenderPipeline>,
    pub culling_pipeline: Arc<wgpu::ComputePipeline>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub instanced_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub culling_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub pipeline_layout: Arc<wgpu::PipelineLayout>,
    
    pub blit_pipeline: Arc<wgpu::RenderPipeline>,
    pub blit_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub upscale_pipeline: Arc<wgpu::RenderPipeline>,
    pub upscale_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub k4_pipeline: Arc<wgpu::RenderPipeline>,
    pub k4_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub lighting_pipeline: Arc<wgpu::RenderPipeline>,
    pub lighting_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub post_pipeline: Arc<wgpu::RenderPipeline>,
    pub post_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub dof_pipeline: Arc<wgpu::RenderPipeline>,
    pub dof_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub flare_pipeline: Arc<wgpu::RenderPipeline>,
    pub flare_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub fxaa_pipeline: Arc<wgpu::RenderPipeline>,
    pub fxaa_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pub bright_pipeline: Arc<wgpu::RenderPipeline>,
    pub blur_pipeline: Arc<wgpu::RenderPipeline>,
    pub bloom_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub motion_blur_pipeline: Arc<wgpu::RenderPipeline>,
    pub motion_blur_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub ssr_pipeline: Arc<wgpu::RenderPipeline>,
    pub taa_pipeline: Arc<wgpu::RenderPipeline>,
    pub taa_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub ssr_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    
    pub tilemap_pipeline: Arc<wgpu::RenderPipeline>,
    pub tilemap_gbuffer_pipeline: Arc<wgpu::RenderPipeline>,
    pub tilemap_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pub skinned_pipeline: Arc<wgpu::RenderPipeline>,
    pub skinned_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pub cache: std::sync::Mutex<HashMap<String, Arc<wgpu::RenderPipeline>>>,
}

impl PipelineManager {
    fn load_shader(device: &wgpu::Device, label: &str, relative_path: &str, fallback: &'static str) -> wgpu::ShaderModule {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("src/backend");
        if relative_path.starts_with("../") {
             // Handle relative paths from src/backend
             path.push(&relative_path[3..]);
        } else {
             path.push(relative_path);
        }

        let source = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("WARNING: Failed to load shader from {:?}: {}. Using fallback.", path, e);
            fallback.to_string()
        });

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    pub fn new(
        device: &wgpu::Device,
        hdr_format: wgpu::TextureFormat,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self::create_internal(device, hdr_format, surface_format)
    }

    pub fn reload(&mut self, device: &wgpu::Device, hdr_format: wgpu::TextureFormat, surface_format: wgpu::TextureFormat) {
        println!("DEBUG: [PIPELINES] Reloading all pipelines...");
        let new_manager = Self::create_internal(device, hdr_format, surface_format);
        *self = new_manager;
    }

    fn create_internal(
        device: &wgpu::Device,
        hdr_format: wgpu::TextureFormat,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // --- Shader Modules ---
        let main_shader = Self::load_shader(device, "Main Shader", "wgpu_shader.wgsl", include_str!("../wgpu_shader.wgsl"));
        let bloom_shader = Self::load_shader(device, "Bloom Shader", "shaders/wgpu_bloom.wgsl", include_str!("../shaders/wgpu_bloom.wgsl"));
        let resolve_shader = Self::load_shader(device, "Resolve Shader", "shaders/wgpu_resolve.wgsl", include_str!("../shaders/wgpu_resolve.wgsl"));
        let upscale_shader = Self::load_shader(device, "Upscale Shader", "shaders/wgpu_upscale.wgsl", include_str!("../shaders/wgpu_upscale.wgsl"));
        let dof_shader = Self::load_shader(device, "DoF Shader", "shaders/dof.wgsl", include_str!("../shaders/dof.wgsl"));
        let flare_shader = Self::load_shader(device, "Flare Shader", "shaders/flare.wgsl", include_str!("../shaders/flare.wgsl"));
        let fxaa_shader = Self::load_shader(device, "FXAA Shader", "shaders/wgpu_fxaa.wgsl", include_str!("../shaders/wgpu_fxaa.wgsl"));
        let motion_blur_shader = &main_shader; // Placeholder
        let ssr_shader = Self::load_shader(device, "SSR Shader", "shaders/ssr.wgsl", include_str!("../shaders/ssr.wgsl"));
        let skinned_shader = Self::load_shader(device, "Skinned Shader", "shaders/skinned.wgsl", include_str!("../shaders/skinned.wgsl"));
        let lighting_shader = Self::load_shader(device, "Lighting Shader", "shaders/wgpu_lighting.wgsl", include_str!("../shaders/wgpu_lighting.wgsl"));
        let post_shader = Self::load_shader(device, "Post Shader", "shaders/wgpu_post.wgsl", include_str!("../shaders/wgpu_post.wgsl"));
        let blit_shader = Self::load_shader(device, "Blit Shader", "shaders/wgpu_blit.wgsl", include_str!("../shaders/wgpu_blit.wgsl"));
        let taa_shader = Self::load_shader(device, "TAA Shader", "shaders/taa.wgsl", include_str!("../shaders/taa.wgsl"));
        let stub_shader = Self::load_shader(device, "Stub Shader", "shaders/stub.wgsl", include_str!("../shaders/stub.wgsl"));

        // --- Bind Group Layouts ---
        let bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let instanced_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Instanced Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let bloom_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let k4_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K4 Resolve Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            ],
        }));

        let lighting_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            ],
        }));

        let post_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D3, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            ],
        }));

        let ssr_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Depth }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            ],
        }));

        let taa_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let motion_blur_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion Blur Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            ],
        }));

        let dof_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Depth }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let flare_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let upscale_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Upscale Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let skinned_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skinned Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let fxaa_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FXAA Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        }));

        let blit_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        }));

        let tilemap_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tilemap Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        }));

        let pipeline_layout = Arc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        }));

        // --- Pipelines ---
        let main_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_main", buffers: &[super::Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { front_face: wgpu::FrontFace::Cw, cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let instanced_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&instanced_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_instanced", buffers: &[super::Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_instanced",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let instanced_gbuffer_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced G-Buffer Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&instanced_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_instanced", buffers: &[super::Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_instanced_gbuffer",
                targets: &[
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let culling_pipeline = Arc::new(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Culling Pipeline"),
            layout: None,
            module: &main_shader,
            entry_point: "culling_main",
        }));
        let culling_bind_group_layout = Arc::new(culling_pipeline.get_bind_group_layout(0));

        let blit_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&blit_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &blit_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let upscale_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Upscale Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&upscale_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &upscale_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let k4_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("K4 Resolve Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&k4_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &resolve_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let lighting_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lighting Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&lighting_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &lighting_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &lighting_shader,
                entry_point: "fs_lighting",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let post_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Post Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&post_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &post_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: "fs_post",
                targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let dof_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&dof_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &dof_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let flare_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Flare Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&flare_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &flare_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &flare_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let fxaa_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FXAA Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&fxaa_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &fxaa_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &fxaa_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let bright_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bright Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bloom_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: "fs_bright",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let blur_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bloom_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &bloom_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: "fs_blur",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let motion_blur_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Motion Blur Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&motion_blur_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &stub_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &stub_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let ssr_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSR Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&ssr_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &stub_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &stub_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let tilemap_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tilemap Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&instanced_bind_group_layout, &tilemap_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_tilemap", buffers: &[super::Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_tilemap",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let tilemap_gbuffer_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tilemap G-Buffer Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&instanced_bind_group_layout, &tilemap_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &main_shader, entry_point: "vs_tilemap", buffers: &[super::Vertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_tilemap_gbuffer",
                targets: &[
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let skinned_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skinned Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&skinned_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &skinned_shader, entry_point: "vs_main", buffers: &[super::SkinnedVertex::desc()] },
            fragment: Some(wgpu::FragmentState {
                module: &skinned_shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        let taa_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&taa_bind_group_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &taa_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &taa_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: hdr_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        Self {
            main_pipeline,
            instanced_pipeline,
            instanced_gbuffer_pipeline,
            culling_pipeline,
            bind_group_layout,
            instanced_bind_group_layout,
            culling_bind_group_layout,
            pipeline_layout,
            blit_pipeline,
            blit_bind_group_layout,
            upscale_pipeline,
            upscale_bind_group_layout,
            k4_pipeline,
            k4_bind_group_layout,
            lighting_pipeline,
            lighting_bind_group_layout,
            post_pipeline,
            post_bind_group_layout,
            dof_pipeline,
            dof_bind_group_layout,
            flare_pipeline,
            flare_bind_group_layout,
            fxaa_pipeline,
            fxaa_bind_group_layout,
            bright_pipeline,
            blur_pipeline,
            bloom_bind_group_layout,
            motion_blur_pipeline,
            motion_blur_bind_group_layout,
            ssr_pipeline,
            ssr_bind_group_layout,
            taa_pipeline,
            taa_bind_group_layout,
            tilemap_pipeline,
            tilemap_gbuffer_pipeline,
            tilemap_bind_group_layout,
            skinned_pipeline,
            skinned_bind_group_layout,
            cache: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

