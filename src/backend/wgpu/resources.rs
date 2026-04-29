use std::sync::{Arc, Mutex};
use crate::backend::hal::{BufferUsage, TextureDescriptor, TextureUsage, TextureFormat};
use crate::renderer::graph::TransientPool;
use super::WgpuBackend;

pub struct ResourceManager {
    pub sampler: Arc<wgpu::Sampler>,
    pub font_texture: Arc<wgpu::Texture>,
    pub font_view: Arc<wgpu::TextureView>,
    pub backdrop_view: Arc<wgpu::TextureView>,
    
    pub hdr_texture: Arc<wgpu::Texture>,
    pub hdr_view: Arc<wgpu::TextureView>,
    pub aux_texture: Arc<wgpu::Texture>,
    pub aux_view: Arc<wgpu::TextureView>,
    pub extra_texture: Arc<wgpu::Texture>,
    pub extra_view: Arc<wgpu::TextureView>,
    pub reflection_view: Option<Arc<wgpu::TextureView>>,
    pub velocity_view: Option<Arc<wgpu::TextureView>>,
    pub velocity_texture: Option<Arc<wgpu::Texture>>,
    pub depth_texture: Arc<wgpu::Texture>,
    pub depth_view: Arc<wgpu::TextureView>,
    pub backdrop_texture: Arc<wgpu::Texture>,
    pub lut_texture: Option<Arc<wgpu::Texture>>,
    pub lut_view: Option<Arc<wgpu::TextureView>>,

    pub ldr_texture: Arc<wgpu::Texture>,
    pub ldr_view: Arc<wgpu::TextureView>,
    
    pub bloom_textures: Vec<wgpu::Texture>,
    pub bloom_views: Vec<Arc<wgpu::TextureView>>,
    
    pub dummy_velocity_view: Arc<wgpu::TextureView>,
    pub dummy_lut_view: Arc<wgpu::TextureView>,

    pub ssr_history_texture: Option<Arc<wgpu::Texture>>,
    pub ssr_history_view: Option<Arc<wgpu::TextureView>>,
    pub taa_history_texture: Option<Arc<wgpu::Texture>>,
    pub taa_history_view: Option<Arc<wgpu::TextureView>>,
    pub sdf_view: Option<Arc<wgpu::TextureView>>,

    pub blur_uniform_buffer: Arc<wgpu::Buffer>,
    pub cinematic_buffer: Arc<wgpu::Buffer>,
    pub dummy_storage_buffer: Arc<wgpu::Buffer>,
    
    pub transient_pool: Mutex<TransientPool<WgpuBackend>>,
}

impl ResourceManager {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        i_width: u32,
        i_height: u32,
    ) -> Self {
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let font_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fallback Font Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &font_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[255, 255, 255, 255],
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let font_view = Arc::new(font_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let hdr_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let hdr_view = Arc::new(hdr_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let extra_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Extra G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let extra_view = Arc::new(extra_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let aux_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Aux G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let aux_view = Arc::new(aux_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let velocity_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Velocity G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let velocity_view = Arc::new(velocity_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let depth_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let depth_view = Arc::new(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let backdrop_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Backdrop Texture"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let backdrop_view = Arc::new(backdrop_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let ldr_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LDR Texture"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let ldr_view = Arc::new(ldr_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let mut bloom_textures = Vec::new();
        let mut bloom_views = Vec::new();
        for i in 0..3 {
            let width = (surface_config.width / (2u32.pow(i as u32 + 1))).max(1);
            let height = (surface_config.height / (2u32.pow(i as u32 + 1))).max(1);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Texture {}", i)),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            bloom_textures.push(tex);
            bloom_views.push(view);
        }

        let dummy_velocity_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Velocity"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_velocity_view = Arc::new(dummy_velocity_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let dummy_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy LUT"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_lut_view = Arc::new(dummy_lut_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let ssr_history_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR History"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let ssr_history_view = Arc::new(ssr_history_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let blur_uniform_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let cinematic_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cinematic Params Buffer"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let dummy_storage_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Storage Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        Self {
            sampler,
            font_texture,
            font_view,
            backdrop_view,
            hdr_texture,
            hdr_view,
            aux_texture,
            aux_view,
            extra_texture,
            extra_view,
            reflection_view: None,
            velocity_view: None,
            velocity_texture: Some(velocity_texture),
            depth_texture,
            depth_view,
            backdrop_texture,
            lut_texture: None,
            lut_view: None,
            ldr_texture,
            ldr_view,
            bloom_textures,
            bloom_views,
            dummy_velocity_view,
            dummy_lut_view,
            ssr_history_texture: Some(ssr_history_texture),
            ssr_history_view: Some(ssr_history_view),
            taa_history_texture: None, // Will be initialized on resize or specifically
            taa_history_view: None,
            sdf_view: None,
            blur_uniform_buffer,
            cinematic_buffer,
            dummy_storage_buffer,
            transient_pool: Mutex::new(TransientPool::new()),
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        i_width: u32,
        i_height: u32,
    ) {
        self.hdr_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        self.hdr_view = Arc::new(self.hdr_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        self.extra_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Extra G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        self.extra_view = Arc::new(self.extra_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        self.aux_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Aux G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        self.aux_view = Arc::new(self.aux_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let velocity_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Velocity G-Buffer Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        self.velocity_view = Some(Arc::new(velocity_texture.create_view(&wgpu::TextureViewDescriptor::default())));
        self.velocity_texture = Some(velocity_texture);

        self.depth_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width: i_width, height: i_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        self.depth_view = Arc::new(self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        self.backdrop_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Backdrop Texture"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        self.backdrop_view = Arc::new(self.backdrop_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        self.ldr_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LDR Texture"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        self.ldr_view = Arc::new(self.ldr_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        self.bloom_textures.clear();
        self.bloom_views.clear();
        for i in 0..3 {
            let width = (surface_config.width / (2u32.pow(i as u32 + 1))).max(1);
            let height = (surface_config.height / (2u32.pow(i as u32 + 1))).max(1);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Texture {}", i)),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.bloom_textures.push(tex);
            self.bloom_views.push(view);
        }

        self.ssr_history_texture = Some(Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR History"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })));
        self.ssr_history_view = self.ssr_history_texture.as_ref().map(|t| Arc::new(t.create_view(&wgpu::TextureViewDescriptor::default())));

        self.taa_history_texture = Some(Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TAA History"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })));
        self.taa_history_view = self.taa_history_texture.as_ref().map(|t| Arc::new(t.create_view(&wgpu::TextureViewDescriptor::default())));
    }
}

