use std::sync::Mutex;
use crate::tracea_bridge::{
    TraceaContext, TraceaParticleKernel, TraceaFFTKernel, 
    TraceaVisibilityKernel, TraceaIndirectKernel
};

pub struct TraceaManager {
    pub context: crate::tracea_bridge::TraceaContext,
    pub particle_cache: Mutex<Option<crate::tracea_bridge::TraceaParticleKernel>>,
    pub fft_kernel: Mutex<Option<crate::tracea_bridge::TraceaFFTKernel>>,
    pub visibility_kernel: Mutex<Option<crate::tracea_bridge::TraceaVisibilityKernel>>,
    pub indirect_kernel: Mutex<Option<crate::tracea_bridge::TraceaIndirectKernel>>,
    pub audio_data: Mutex<Vec<f32>>,
}

impl TraceaManager {
    pub fn new() -> Self {
        Self {
            context: TraceaContext::default(),
            particle_cache: Mutex::new(None),
            fft_kernel: Mutex::new(None),
            visibility_kernel: Mutex::new(None),
            indirect_kernel: Mutex::new(None),
            audio_data: Mutex::new(Vec::new()),
        }
    }

    pub fn dispatch_particles(
        &self,
        device: &wgpu::Device,
        dt: f32,
        attractor: [f32; 2],
        sdf_texture: Option<&wgpu::Texture>,
    ) -> Result<bool, String> {
        let mut cache = self.particle_cache.lock().unwrap();
        if cache.is_none() {
            if let Ok(kernel) = TraceaParticleKernel::new(&self.context, 100_000) {
                *cache = Some(kernel);
            }
        }

        if let Some(kernel) = cache.as_ref() {
            let view = sdf_texture.map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            let _ = kernel.update_wgpu(&self.context, dt, attractor, view.as_ref(), Some(&sampler));
            return Ok(true);
        }
        Ok(false)
    }

    pub fn update_audio_pcm(&self, samples: &[f32]) {
        if !self.context.is_ready() { return; }

        let mut kernel_lock = self.fft_kernel.lock().unwrap();
        if kernel_lock.is_none() {
            let size = samples.len().max(1024).next_power_of_two();
            if let Ok(k) = TraceaFFTKernel::new(&self.context, size) {
                *kernel_lock = Some(k);
            }
        }

        if let Some(kernel) = kernel_lock.as_ref() {
            if kernel.fft_size() == samples.len() {
                if let Ok(spectrum) = kernel.compute_spectrum(&self.context, samples) {
                    *self.audio_data.lock().unwrap() = spectrum;
                }
            } else if !samples.is_empty() {
                let size = samples.len().next_power_of_two();
                if let Ok(k) = TraceaFFTKernel::new(&self.context, size) {
                    *kernel_lock = Some(k);
                    if let Ok(spectrum) = kernel_lock.as_ref().unwrap().compute_spectrum(&self.context, samples) {
                        *self.audio_data.lock().unwrap() = spectrum;
                    }
                }
            }
        }
    }

    pub fn dispatch_visibility(
        &self,
        projection: [[f32; 4]; 4],
        num_instances: u32,
        instances: &wgpu::Buffer,
        hzb: &wgpu::TextureView,
        visible_indices: &wgpu::Buffer,
        visible_counter: &wgpu::Buffer,
    ) -> Result<(), String> {
        let mut kernel_lock = self.visibility_kernel.lock().unwrap();
        if kernel_lock.is_none() {
            *kernel_lock = Some(TraceaVisibilityKernel::new_wgpu(&self.context)?);
        }

        if let Some(kernel) = kernel_lock.as_ref() {
            let uniforms = crate::tracea_bridge::visibility::CullingUniforms {
                view_proj: projection,
                num_instances,
                hzb_mip_levels: 1,
                _pad: [0, 0],
            };
            kernel.dispatch(&self.context, &uniforms, instances, hzb, visible_indices, visible_counter)?;
            return Ok(());
        }
        Err("Visibility kernel not available".into())
    }

    pub fn dispatch_indirect_command(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        counter_buffer: &wgpu::Buffer,
        draw_commands: &wgpu::Buffer,
    ) -> Result<(), String> {
        let mut kernel_lock = self.indirect_kernel.lock().unwrap();
        if kernel_lock.is_none() {
            *kernel_lock = Some(TraceaIndirectKernel::new_wgpu(&self.context)?);
        }

        if let Some(kernel) = kernel_lock.as_ref() {
            kernel.dispatch(&self.context, counter_buffer)?;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Indirect Copy") });
            encoder.copy_buffer_to_buffer(kernel.draw_commands(), 0, draw_commands, 0, 16);
            queue.submit(Some(encoder.finish()));
            return Ok(());
        }
        Err("Indirect kernel not available".into())
    }
}
