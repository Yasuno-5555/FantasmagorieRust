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
            let tracea_buffer = if ctx.executor.supports_tracea_particles() {
                 ctx.executor.get_tracea_particle_buffer()
            } else { None };
            
            if let Some(buffer) = tracea_buffer {
                 sys.particle_buffer = Some(Arc::new(buffer));
                 println!("[ParticlesNode] Using Tracea Particle Buffer");
            } else {
                 let size = std::mem::size_of::<Particle>() as u64 * sys.particle_count as u64;
                 let buffer = ctx.executor.create_buffer(
                     size, 
                     BufferUsage::Storage | BufferUsage::Vertex, 
                     "Particle Buffer"
                 )?;
                 sys.particle_buffer = Some(Arc::new(buffer));
            }
            
            // Common Control Buffer
            let control_size = std::mem::size_of::<ParticleControl>() as u64;
            let c_buffer = ctx.executor.create_buffer(
                control_size,
                BufferUsage::Uniform | BufferUsage::CopyDst,
                "Particle Control"
            )?;
            sys.control_buffer = Some(Arc::new(c_buffer));
            
            let shader_source = if cfg!(feature = "metal") {
                 include_str!("../../backend/shaders/particles.metal")
            } else {
                 include_str!("../../backend/shaders/particles.wgsl")
            };
            
            // Create Pipelines
            // If Tracea is used (tracea_buffer is Some), we don't need COMPUTE pipeline from engine
            if ctx.executor.supports_tracea_particles() {
                sys.compute_pipeline = None; 
            } else {
                match ctx.executor.create_compute_pipeline("ParticleUpdate", shader_source, Some("update")) {
                    Ok(compute) => sys.compute_pipeline = Some(Arc::new(compute)),
                    Err(e) => {
                        eprintln!("[ParticlesNode] Failed to create compute pipeline: {}", e);
                        sys.compute_pipeline = None;
                    }
                }
            }
            
            // Auto-layout render pipeline (Always needed)
            match ctx.executor.create_render_pipeline("ParticleRender", shader_source, None) {
                Ok(render) => sys.render_pipeline = Some(Arc::new(render)),
                Err(e) => {
                    eprintln!("[ParticlesNode] Failed to create render pipeline: {}", e);
                    sys.render_pipeline = None;
                }
            }

            sys.initialized = true;
        }

        let p_buffer = sys.particle_buffer.as_ref().ok_or("no p_buf")?.downcast_ref::<E::Buffer>().ok_or("buf cast")?;
        let c_buffer = sys.control_buffer.as_ref().ok_or("no c_buf")?.downcast_ref::<E::Buffer>().ok_or("buf cast")?;
        
        // If render pipeline is missing, we can't do anything useful
        let r_pipeline = if let Some(p) = sys.render_pipeline.as_ref() {
            p.downcast_ref::<E::RenderPipeline>().ok_or("pipe cast")?
        } else {
            return Ok(());
        };

        // 2. Update Control
        let dt = 0.016; 
        let control = ParticleControl {
            count: sys.particle_count,
            emit_rate: 1000.0,
            gravity: [0.0, 98.0], 
            delta_time: dt,
            seed: (ctx.executor.get_default_sampler() as *const _ as usize) as f32,
            drag_coefficient: 2.0, 
            _pad0: 0,
            jitter: [ctx.jitter.0, ctx.jitter.1],
            _pad1: [0, 0],
        };
        ctx.executor.write_buffer(c_buffer, 0, bytemuck::bytes_of(&control));

        // 3. Resources
        // We need 'tex' (Texture) for Tracea and 'view' (TextureView) for Render
        let (sdf_tex, sdf_view) = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&SDF_HANDLE) {
             (Some(tex), Some(ctx.executor.create_texture_view(tex)?))
        } else { (None, None) };
        
        let velocity_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&VELOCITY_HANDLE) {
             Some(ctx.executor.create_texture_view(tex)?)
        } else { None };

        let depth_view = if let Some(GraphResource::Texture(_, tex)) = ctx.resources.get(&DEPTH_HANDLE) {
             Some(ctx.executor.create_texture_view(tex)?)
        } else { None };
        
        // 4. Update (Compute or Tracea)
        if sys.compute_pipeline.is_none() {
            // Use Tracea
            // We need to pass sdf texture as Texture, not View
            let attractor = [0.0, 0.0]; // Should come from control or input
            ctx.executor.dispatch_tracea_particles(dt, attractor, sdf_tex)?;
        } else {
            let c_pipeline = sys.compute_pipeline.as_ref().unwrap().downcast_ref::<E::ComputePipeline>().ok_or("pipe cast")?;
            
            let cinematic_buffer = ctx.executor.get_cinematic_buffer();
            let sampler = ctx.executor.get_default_sampler();
            
            if let Some(sdf) = &sdf_view {
                if let Some(vel) = &velocity_view {
                    let c_layout = ctx.executor.get_compute_pipeline_layout(c_pipeline, 0)?;
                    let c_entries = [
                        BindGroupEntry { binding: 0, resource: BindingResource::Buffer(p_buffer) },
                        BindGroupEntry { binding: 1, resource: BindingResource::Buffer(cinematic_buffer) },
                        BindGroupEntry { binding: 2, resource: BindingResource::Texture(sdf) },
                        BindGroupEntry { binding: 3, resource: BindingResource::Sampler(sampler) },
                        BindGroupEntry { binding: 4, resource: BindingResource::Buffer(c_buffer) },
                        BindGroupEntry { binding: 5, resource: BindingResource::Texture(vel) },
                    ];
                    let c_bind_group = ctx.executor.create_bind_group(&c_layout, &c_entries)?;
                    
                    let groups_x = (sys.particle_count + 63) / 64;
                    ctx.executor.dispatch(c_pipeline, Some(&c_bind_group), [groups_x, 1, 1], &[])?;
                }
            }
        }

        // 5. Render Bindings & Draw
        if let Some(depth) = depth_view {
            // Need sdf_view too?
             if let Some(sdf) = &sdf_view {
                let cinematic_buffer = ctx.executor.get_cinematic_buffer();
                let sampler = ctx.executor.get_default_sampler();
                
                let r_layout = ctx.executor.get_render_pipeline_layout(r_pipeline, 0)?;
                let r_entries = [
                    BindGroupEntry { binding: 0, resource: BindingResource::Buffer(p_buffer) },
                    BindGroupEntry { binding: 2, resource: BindingResource::Texture(sdf) },
                    BindGroupEntry { binding: 3, resource: BindingResource::Sampler(sampler) },
                    BindGroupEntry { binding: 4, resource: BindingResource::Buffer(c_buffer) }, 
                    BindGroupEntry { binding: 6, resource: BindingResource::Texture(&depth) },
                ];
                let r_bind_group = ctx.executor.create_bind_group(&r_layout, &r_entries)?;
                
                ctx.executor.draw_particles(r_pipeline, &r_bind_group, sys.particle_count)?;
             }
        }
        
        Ok(())
    }
}
