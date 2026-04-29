use super::WgpuBackend;
use crate::backend::GraphicsBackend;
use wgpu::util::DeviceExt;

pub fn draw_bloom_pass(backend: &WgpuBackend, input_view: &wgpu::TextureView) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    let pipelines = backend.pipelines.lock().unwrap();
    // 1. Bright Pass
    let bright_bg = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bright Bind Group"),
        layout: &pipelines.bloom_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: backend.resources.blur_uniform_buffer.as_entire_binding() },
        ],
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Bright Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &backend.resources.bloom_views[0],
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.bright_pipeline);
        rpass.set_bind_group(0, &bright_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    // 2. Horizontal Blur
    backend.queue.write_buffer(&backend.resources.blur_uniform_buffer, 0, bytemuck::cast_slice(&[1.0f32, 0.0f32, 0.0f32, 0.0f32]));
    let blur_h_bg = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blur H Bind Group"),
        layout: &pipelines.bloom_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&backend.resources.bloom_views[0]) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: backend.resources.blur_uniform_buffer.as_entire_binding() },
        ],
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Blur H Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &backend.resources.bloom_views[1],
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.blur_pipeline);
        rpass.set_bind_group(0, &blur_h_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    // 3. Vertical Blur
    backend.queue.write_buffer(&backend.resources.blur_uniform_buffer, 0, bytemuck::cast_slice(&[0.0f32, 1.0f32, 0.0f32, 0.0f32]));
    let blur_v_bg = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Blur V Bind Group"),
        layout: &pipelines.bloom_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&backend.resources.bloom_views[1]) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: backend.resources.blur_uniform_buffer.as_entire_binding() },
        ],
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Blur V Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &backend.resources.bloom_views[2],
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.blur_pipeline);
        rpass.set_bind_group(0, &blur_v_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    Ok(())
}

pub fn draw_ssr(
    backend: &WgpuBackend,
    hdr_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    aux_view: &wgpu::TextureView,
    velocity_view: &wgpu::TextureView,
    output_texture: &wgpu::Texture,
) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let pipelines = backend.pipelines.lock().unwrap();
    let ssr_bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SSR Bind Group"),
        layout: &pipelines.ssr_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: backend.resources.cinematic_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(hdr_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(aux_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(
                backend.resources.ssr_history_view.as_ref().unwrap_or(&backend.resources.dummy_velocity_view)
            )},
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(velocity_view) },
        ],
    });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.ssr_pipeline);
        rpass.set_bind_group(0, &ssr_bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    if let Some(history_tex) = &backend.resources.ssr_history_texture {
         encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: history_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            output_texture.size(),
        );
    }

    Ok(())
}

pub fn upscale(
    backend: &WgpuBackend,
    input: &wgpu::TextureView,
    output: &wgpu::TextureView,
    _params: crate::backend::hal::UpscaleParams,
) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    let pipelines = backend.pipelines.lock().unwrap();
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pipelines.upscale_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: backend.resources.cinematic_buffer.as_entire_binding() },
        ],
        label: Some("Upscale Bind Group"),
    });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Upscale Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.upscale_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    Ok(())
}

pub fn draw_taa_pass(
    backend: &WgpuBackend,
    current_view: &wgpu::TextureView,
    history_view: &wgpu::TextureView,
    velocity_view: &wgpu::TextureView,
    output_view: &wgpu::TextureView,
) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    let pipelines = backend.pipelines.lock().unwrap();
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TAA Bind Group"),
        layout: &pipelines.taa_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(current_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(history_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(velocity_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: backend.resources.cinematic_buffer.as_entire_binding() },
        ],
    });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TAA Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.taa_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    Ok(())
}

pub fn draw_fxaa_pass(backend: &WgpuBackend, input_view: &wgpu::TextureView) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    if let Some(qs) = &backend.profiler.query_set {
         encoder.write_timestamp(qs, 6);
    }

    let pipelines = backend.pipelines.lock().unwrap();
    let fxaa_bg = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FXAA Bind Group"),
        layout: &pipelines.fxaa_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
        ],
    });

    let view_guard = backend.current_view.lock().unwrap();
    let view = view_guard.as_ref().ok_or("No active swapchain view")?;

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.fxaa_pipeline);
        rpass.set_bind_group(0, &fxaa_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    if let Some(qs) = &backend.profiler.query_set {
         encoder.write_timestamp(qs, 7);
    }

    Ok(())
}

pub fn draw_motion_blur(
    backend: &WgpuBackend,
    dst_view: &wgpu::TextureView,
    src_view: &wgpu::TextureView,
    vel_view: &wgpu::TextureView,
) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    let pipelines = backend.pipelines.lock().unwrap();
    let mb_bg = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Motion Blur Bind Group"),
        layout: &pipelines.motion_blur_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(vel_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: backend.resources.cinematic_buffer.as_entire_binding() },
        ],
    });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Motion Blur Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.motion_blur_pipeline);
        rpass.set_bind_group(0, &mb_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    Ok(())
}

pub fn draw_dof_pass(backend: &WgpuBackend, input_view: &wgpu::TextureView, depth_view: &wgpu::TextureView, output_view: &wgpu::TextureView) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    let pipelines = backend.pipelines.lock().unwrap();
    let dof_bg = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DoF Bind Group"),
        layout: &pipelines.dof_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: backend.resources.cinematic_buffer.as_entire_binding() },
        ],
    });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DoF Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.dof_pipeline);
        rpass.set_bind_group(0, &dof_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    Ok(())
}

pub fn draw_flare_pass(backend: &WgpuBackend, input_view: &wgpu::TextureView, output_view: &wgpu::TextureView) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    let pipelines = backend.pipelines.lock().unwrap();
    let flare_bg = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Flare Bind Group"),
        layout: &pipelines.flare_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: backend.resources.cinematic_buffer.as_entire_binding() },
        ],
    });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Flare Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.flare_pipeline);
        rpass.set_bind_group(0, &flare_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    Ok(())
}

pub fn draw_lighting_pass(backend: &WgpuBackend, output_view: &wgpu::TextureView) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;
    
    // Prepare resources
    let dummy_vel = &backend.resources.dummy_velocity_view;
    let hdr = &backend.resources.hdr_view;
    
    let velocity = if let Some(v) = &backend.resources.velocity_view { v.as_ref() } else { dummy_vel.as_ref() };
    let reflection = if let Some(v) = &backend.resources.ssr_history_view { v.as_ref() } else { hdr.as_ref() };
    let aux = backend.resources.aux_view.as_ref();
    let extra = backend.resources.extra_view.as_ref();
    let sdf = if let Some(v) = &backend.resources.sdf_view { v.as_ref() } else { dummy_vel.as_ref() };

    let pipelines = backend.pipelines.lock().unwrap();
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pipelines.lighting_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&*hdr) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*backend.resources.bloom_views[2]) },
            wgpu::BindGroupEntry { binding: 3, resource: backend.resources.cinematic_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&*velocity) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&*reflection) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&*aux) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&*extra) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&*sdf) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(backend.resources.lut_view.as_ref().map(|v| &**v).unwrap_or(&backend.resources.dummy_lut_view)) },
        ],
        label: Some("Lighting Bind Group"),
    });
    
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Lighting Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.lighting_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
    
    Ok(())
}

pub fn draw_post_process_pass(backend: &WgpuBackend, input_view: &wgpu::TextureView, output_view: Option<&wgpu::TextureView>) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    if let Some(qs) = &backend.profiler.query_set {
         encoder.write_timestamp(qs, 4);
    }
    
    let pipelines = backend.pipelines.lock().unwrap();
    if output_view.is_none() {
         let view_guard = backend.current_view.lock().unwrap();
         let target_view = view_guard.as_ref().ok_or("No current view for post-process output")?;
         
         let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipelines.post_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*backend.resources.bloom_views[2]) },
                wgpu::BindGroupEntry { binding: 3, resource: backend.resources.cinematic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(backend.resources.lut_view.as_ref().map(|v| &**v).unwrap_or(&backend.resources.dummy_lut_view)) },
            ],
            label: Some("Post Bind Group"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Post Process Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&*pipelines.post_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        return Ok(());
    }

    let target_view = output_view.unwrap();
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pipelines.post_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&*backend.resources.bloom_views[2]) },
            wgpu::BindGroupEntry { binding: 3, resource: backend.resources.cinematic_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(backend.resources.lut_view.as_ref().map(|v| &**v).unwrap_or(&backend.resources.dummy_lut_view)) },
        ],
        label: Some("Post Bind Group"),
    });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Post Process Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&*pipelines.post_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    Ok(())
}

pub fn draw_tilemap(
    backend: &WgpuBackend,
    params: crate::backend::shaders::types::TilemapParams,
    data: &[u32],
    texture_view: &wgpu::TextureView,
    global_buffer: &wgpu::Buffer,
    aux_view: Option<&wgpu::TextureView>,
    velocity_view: Option<&wgpu::TextureView>,
    depth_view: Option<&wgpu::TextureView>,
) -> Result<(), String> {
    let mut encoder_guard = backend.current_encoder.lock().unwrap();
    let encoder = encoder_guard.as_mut().ok_or("No active encoder")?;

    // 1. Create Buffers
    let params_buffer = backend.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tilemap Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let data_buffer = backend.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tilemap Data"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Vertex Buffer (Unit Quad)
    let quad_verts = crate::renderer::nodes::geometry::unit_quad();
    let vertex_buffer = backend.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tilemap Quad Vertex"),
        contents: &quad_verts,
        usage: wgpu::BufferUsages::VERTEX,
    });

    let pipelines = backend.pipelines.lock().unwrap();
    // 2. Create Bind Group 1 (Tilemap Specific)
    let bg1 = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tile BG1"),
        layout: &pipelines.tilemap_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
        ],
    });

    // 3. Create Bind Group 0 (Shared)
    let bg0 = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tile BG0"),
        layout: &pipelines.instanced_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: global_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: backend.resources.dummy_storage_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(texture_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&backend.resources.backdrop_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&backend.resources.sampler) },
        ],
    });

    // 4. Determine Pipeline & Attachments
    let use_gbuffer = aux_view.is_some() && velocity_view.is_some() && depth_view.is_some();
    let pipeline = if use_gbuffer { &*pipelines.tilemap_gbuffer_pipeline } else { &*pipelines.tilemap_pipeline };

    // 5. Begin Render Pass
    {
        let color_attachments = if use_gbuffer {
             vec![
                Some(wgpu::RenderPassColorAttachment { view: &backend.resources.hdr_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                Some(wgpu::RenderPassColorAttachment { view: aux_view.unwrap(), resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                Some(wgpu::RenderPassColorAttachment { view: velocity_view.unwrap(), resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
                Some(wgpu::RenderPassColorAttachment { view: &backend.resources.extra_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
             ]
        } else {
             vec![
                Some(wgpu::RenderPassColorAttachment { view: &backend.resources.hdr_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
             ]
        };

        let depth_att = if use_gbuffer {
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view.unwrap(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            })
        } else {
            None
        };
        
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Tilemap Draw"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_att,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, &bg0, &[]);
        rpass.set_bind_group(1, &bg1, &[]);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..6, 0..params.map_size[0] * params.map_size[1]);
    }

    Ok(())
}
