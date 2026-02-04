use crate::backend::hal::{GpuExecutor, TextureDescriptor, TextureFormat, TextureUsage, BufferUsage};
use crate::backend::shaders::types::BlurParams;
use crate::renderer::graph::{RenderNode, RenderContext, GraphResource, BACKDROP_HANDLE};
use bytemuck::bytes_of;

pub struct BlurNode {
    pub sigma: f32,
}

impl<E: GpuExecutor> RenderNode<E> for BlurNode {
    fn name(&self) -> &str { "BlurPass" }
    fn execute(&mut self, ctx: &mut RenderContext<'_, E>) -> Result<(), String> {
        // 1. Get Backdrop Texture
        let backdrop_res = ctx.resources.get(&BACKDROP_HANDLE).ok_or("BlurNode: No backdrop found")?;
        let (backdrop_desc, backdrop_tex) = match backdrop_res {
             GraphResource::Texture(d, t) => (d.clone(), t), // Clone desc to use its format/size
             _ => return Err("BlurNode: Backdrop is not a texture".to_string()),
        };

        if self.sigma <= 0.0 {
            return Ok(());
        }

        // 2. Acquire Intermediate Texture
        let intermediate_desc = TextureDescriptor {
            label: Some("Blur Intermediate"),
            width: ctx.width,
            height: ctx.height,
            format: backdrop_desc.format,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::STORAGE_BINDING | TextureUsage::COPY_DST,
        };
        let intermediate_tex = ctx.executor.acquire_transient_texture(&intermediate_desc)?;

        // 3. Create/Get Pipeline
        let shader_source = if cfg!(feature = "metal") { 
            include_str!("../../backend/shaders/metal_blur.metal") 
        } else { 
            include_str!("../../backend/shaders/wgpu_blur.wgsl") 
        };
        
        let pipeline = ctx.executor.create_compute_pipeline(
            "blur_compute",
            shader_source,
            Some(if cfg!(feature = "metal") { "blur_compute" } else { "main" })
        )?;

        // 4. Ping-Pong Passes
        let dim_x = (ctx.width + 15) / 16;
        let dim_y = (ctx.height + 15) / 16;
        let groups = [dim_x, dim_y, 1];

        // Pass 1: Horizontal (Backdrop -> Intermediate)
        {
            let params_h = BlurParams {
                direction: [1.0, 0.0],
                sigma: self.sigma,
                _pad: 0.0,
            };
            let buf_h = ctx.executor.create_buffer(std::mem::size_of::<BlurParams>() as u64, BufferUsage::Uniform, "Blur Param H")?;
            ctx.executor.write_buffer(&buf_h, 0, bytes_of(&params_h));

            let layout = ctx.executor.get_compute_pipeline_layout(&pipeline, 0)?;
            
            // Views
            let input_view = ctx.executor.create_texture_view(backdrop_tex)?;
            let output_view = ctx.executor.create_texture_view(&intermediate_tex)?;

            // Bind Group
            let bind_group = ctx.executor.create_bind_group(
                &layout,
                &[&buf_h], // Buffer 0 (Param)
                &[&input_view, &output_view], // Tex 1 (Input), Tex 2 (Output) (Assuming Bindings 0,1,2 map to Buffer0, Tex0, Tex1 by HAL? 
                // Wait, WGPU create_bind_group iterates buffers then textures.
                // So Buffer -> Binding 0 (matches shader @binding(0))
                // Texture1 -> Binding 1 (matches shader @binding(1))
                // Texture2 -> Binding 2 (matches shader @binding(2))
                // Correct.
                &[]
            )?;

            ctx.executor.dispatch(&pipeline, Some(&bind_group), groups, &[])?;
            ctx.executor.destroy_bind_group(bind_group);
            ctx.executor.destroy_buffer(buf_h);
        }

        // Pass 2: Vertical (Intermediate -> Backdrop)
        {
            let params_v = BlurParams {
                direction: [0.0, 1.0],
                sigma: self.sigma,
                _pad: 0.0,
            };
            let buf_v = ctx.executor.create_buffer(std::mem::size_of::<BlurParams>() as u64, BufferUsage::Uniform, "Blur Param V")?;
            ctx.executor.write_buffer(&buf_v, 0, bytes_of(&params_v));

            let layout = ctx.executor.get_compute_pipeline_layout(&pipeline, 0)?;
            
            // Views
            let input_view = ctx.executor.create_texture_view(&intermediate_tex)?; // Input is intermediate
            let output_view = ctx.executor.create_texture_view(backdrop_tex)?; // Output is original backdrop

            // Bind Group
            let bind_group = ctx.executor.create_bind_group(
                &layout,
                &[&buf_v],
                &[&input_view, &output_view],
                &[]
            )?;

            ctx.executor.dispatch(&pipeline, Some(&bind_group), groups, &[])?;
            ctx.executor.destroy_bind_group(bind_group);
            ctx.executor.destroy_buffer(buf_v);
        }
        
        // Cleanup intermediates
        ctx.executor.release_transient_texture(intermediate_tex, &intermediate_desc);

        Ok(())
    }
}
