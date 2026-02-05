use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, SDF_HANDLE, VELOCITY_HANDLE, DEPTH_HANDLE};
use crate::backend::hal::{GpuExecutor, BufferUsage, BindGroupEntry, BindingResource};
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
    color: [f32; 4],
    life: f32,
    size: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleControl {
    count: u32,
    emit_rate: f32,
    gravity: [f32; 2],
    delta_time: f32,
    seed: f32,
    drag_coefficient: f32,
    _pad0: u32,
    jitter: [f32; 2],
    _pad1: [u32; 2],
}

pub struct ParticleSystem {
    particle_buffer: Option<Arc<dyn std::any::Any + Send + Sync>>, 
    control_buffer: Option<Arc<dyn std::any::Any + Send + Sync>>, 
    compute_pipeline: Option<Arc<dyn std::any::Any + Send + Sync>>,
    render_pipeline: Option<Arc<dyn std::any::Any + Send + Sync>>,
    initialized: bool,
    particle_count: u32,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particle_buffer: None,
            control_buffer: None,
            compute_pipeline: None,
            render_pipeline: None,
            initialized: false,
            particle_count: 100000,
        }
    }
}

pub struct ParticleNode {
    system: Arc<Mutex<ParticleSystem>>,
}

impl ParticleNode {
    pub fn new(system: Arc<Mutex<ParticleSystem>>) -> Self {
        Self { system }
    }
}

impl<E: GpuExecutor + 'static> RenderNode<E> for ParticleNode {
    fn name(&self) -> &str { "ParticleNode" }
    
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        let mut sys = self.system.lock().map_err(|_| "LocK poisoned")?;
        
        // 1. Initialize
        if !sys.initialized {
            let size = std::mem::size_of::<Particle>() as u64 * sys.particle_count as u64;
            let buffer = ctx.executor.create_buffer(
                size, 
                BufferUsage::Storage | BufferUsage::Vertex, 
                "Particle Buffer"
            )?;
            
            let control_size = std::mem::size_of::<ParticleControl>() as u64;
            let c_buffer = ctx.executor.create_buffer(
                control_size,
                BufferUsage::Uniform | BufferUsage::CopyDst,
                "Particle Control"
            )?;
            
            sys.particle_buffer = Some(Arc::new(buffer));
            sys.control_buffer = Some(Arc::new(c_buffer));
            
            let shader_source = if cfg!(feature = "metal") {
                 include_str!("../../backend/shaders/particles.metal")
            } else {
                 include_str!("../../backend/shaders/particles.wgsl")
            };
            
            let compute = ctx.executor.create_compute_pipeline("ParticleUpdate", shader_source, Some("update"))?;
            sys.compute_pipeline = Some(Arc::new(compute));
            
            // Auto-layout render pipeline
            let render = ctx.executor.create_render_pipeline("ParticleRender", shader_source, None)?;
            sys.render_pipeline = Some(Arc::new(render));

            sys.initialized = true;
        }

        let p_buffer = sys.particle_buffer.as_ref().unwrap().downcast_ref::<E::Buffer>().ok_or("buf cast")?;
        let c_buffer = sys.control_buffer.as_ref().unwrap().downcast_ref::<E::Buffer>().ok_or("buf cast")?;
        let c_pipeline = sys.compute_pipeline.as_ref().unwrap().downcast_ref::<E::ComputePipeline>().ok_or("pipe cast")?;
        let r_pipeline = sys.render_pipeline.as_ref().unwrap().downcast_ref::<E::RenderPipeline>().ok_or("pipe cast")?;

        // 2. Update Control
        let dt = 0.016; 
        let control = ParticleControl {
            count: sys.particle_count,
            emit_rate: 1000.0,
            gravity: [0.0, 98.0], 
            delta_time: dt,
            seed: (ctx.executor.get_default_sampler() as *const _ as usize) as f32,
            drag_coefficient: 2.0, // Interaction strength
            _pad0: 0,
            jitter: [ctx.jitter.0, ctx.jitter.1],
            _pad1: [0, 0],
        };
        ctx.executor.write_buffer(c_buffer, 0, bytemuck::bytes_of(&control));

        // 3. Resources
        let sdf_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&SDF_HANDLE) {
             ctx.executor.create_texture_view(tex)?
        } else { return Ok(()); };
        
        let velocity_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&VELOCITY_HANDLE) {
             ctx.executor.create_texture_view(tex)?
        } else { return Ok(()); };

        let depth_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&DEPTH_HANDLE) {
             ctx.executor.create_texture_view(tex)?
        } else { return Ok(()); };
        
        let cinematic_buffer = ctx.executor.get_cinematic_buffer();
        let sampler = ctx.executor.get_default_sampler();

        // 4. Compute Bindings
        let c_layout = ctx.executor.get_compute_pipeline_layout(c_pipeline, 0)?;
        let c_entries = [
            BindGroupEntry { binding: 0, resource: BindingResource::Buffer(p_buffer) },
            BindGroupEntry { binding: 1, resource: BindingResource::Buffer(cinematic_buffer) },
            BindGroupEntry { binding: 2, resource: BindingResource::Texture(&sdf_view) },
            BindGroupEntry { binding: 3, resource: BindingResource::Sampler(sampler) },
            BindGroupEntry { binding: 4, resource: BindingResource::Buffer(c_buffer) },
            BindGroupEntry { binding: 5, resource: BindingResource::Texture(&velocity_view) },
        ];
        let c_bind_group = ctx.executor.create_bind_group(&c_layout, &c_entries)?;

        // 5. Render Bindings
        // Note: Render shader might use different binding set. We'll reuse slots 0-4 for simplicity and correctness.
        // We assume render shader uses same core resources + Depth instead of Velocity.
        let r_layout = ctx.executor.get_render_pipeline_layout(r_pipeline, 0)?;
        let r_entries = [
            BindGroupEntry { binding: 0, resource: BindingResource::Buffer(p_buffer) },
            BindGroupEntry { binding: 1, resource: BindingResource::Buffer(cinematic_buffer) },
            BindGroupEntry { binding: 2, resource: BindingResource::Texture(&sdf_view) },
            BindGroupEntry { binding: 3, resource: BindingResource::Sampler(sampler) },
            // Binding 4 (Control) might not be used in render, but if shader declares it, we bind it.
            BindGroupEntry { binding: 4, resource: BindingResource::Buffer(c_buffer) }, 
            BindGroupEntry { binding: 6, resource: BindingResource::Texture(&depth_view) },
        ];
        let r_bind_group = ctx.executor.create_bind_group(&r_layout, &r_entries)?;

        // 6. Execute
        let groups_x = (sys.particle_count + 63) / 64;
        ctx.executor.dispatch(c_pipeline, Some(&c_bind_group), [groups_x, 1, 1], &[])?;
        ctx.executor.draw_particles(r_pipeline, &r_bind_group, sys.particle_count)?;
        
        Ok(())
    }
}
