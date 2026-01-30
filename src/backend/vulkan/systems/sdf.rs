use ash::vk;
use std::sync::Arc;
use std::ffi::CStr;
use crate::backend::vulkan::VulkanContext;
use crate::backend::vulkan::managed::ManagedImage;
use crate::backend::vulkan::pipelines;
use crate::backend::vulkan::resources;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct JFAUniforms {
    pub width: u32,
    pub height: u32,
    pub jfa_step: u32,
    pub ping_pong_idx: u32,
    pub intensity: f32,
    pub decay: f32,
    pub radius: f32,
    pub _pad: u32,
}

pub struct SdfSystem {
    pub ctx: Arc<VulkanContext>,
    
    // Resources
    pub jfa_images: [ManagedImage; 2],
    pub sdf_image: ManagedImage,
    pub jfa_framebuffers: [vk::Framebuffer; 2],
    
    // Pipelines
    pub seed_render_pass: vk::RenderPass,
    pub seed_pipeline: vk::Pipeline,
    pub seed_layout: vk::PipelineLayout,
    
    pub k5_pipeline: vk::Pipeline,
    pub k5_layout: vk::PipelineLayout,
    pub k5_descriptor_sets: [vk::DescriptorSet; 2],
    pub k5_set_layout: vk::DescriptorSetLayout,
    
    pub k5_resolve_pipeline: vk::Pipeline,
    pub k5_resolve_layout: vk::PipelineLayout,
    pub k5_resolve_descriptor_sets: [vk::DescriptorSet; 2],
    pub k5_resolve_set_layout: vk::DescriptorSetLayout,
}

impl SdfSystem {
    pub fn new(
        ctx: Arc<VulkanContext>,
        width: u32,
        height: u32,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffer: vk::Buffer,
        uniform_size: vk::DeviceSize,
        main_descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, String> {
        let device = &ctx.device;

        // 1. Create Images (Managed)
        let jfa_images = [
            Self::create_jfa_image(ctx.clone(), width, height)?,
            Self::create_jfa_image(ctx.clone(), width, height)?,
        ];
        
        let sdf_image = Self::create_sdf_image(ctx.clone(), width, height)?;

        // 2. Create Render Pass and Framebuffers for Seed
        let seed_render_pass = Self::create_seed_render_pass(device)?;
        let jfa_framebuffers = [
            Self::create_framebuffer(device, seed_render_pass, jfa_images[0].view, width, height)?,
            Self::create_framebuffer(device, seed_render_pass, jfa_images[1].view, width, height)?,
        ];

        // 3. K5 JFA Pipeline
        let k5_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k5_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k5_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k5_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<JFAUniforms>() as u32)];
        let k5_layout = unsafe { device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&k5_set_layout))
                .push_constant_ranges(&k5_ranges), 
            None) 
        }.map_err(|e| format!("{:?}", e))?;
        let k5_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k5_layout, include_str!("../../shaders/k5_jfa.wgsl"), "main") }?;

        // 4. K5 Resolve Pipeline
        let k5_resolve_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k5_resolve_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k5_resolve_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let resolve_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<JFAUniforms>() as u32)];
        let k5_resolve_layout = unsafe { device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&k5_resolve_set_layout))
                .push_constant_ranges(&resolve_ranges), 
            None) 
        }.map_err(|e| format!("{:?}", e))?;
        let k5_resolve_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k5_resolve_layout, include_str!("../../shaders/k5_resolve.wgsl"), "main") }?;

        // 5. Seed Pipeline
        let seed_wgsl = include_str!("../../shaders/seed.wgsl");
        let seed_spv = pipelines::compile_wgsl(seed_wgsl)?;
        let seed_module_info = vk::ShaderModuleCreateInfo::default().code(&seed_spv);
        let seed_module = unsafe { device.create_shader_module(&seed_module_info, None) }.map_err(|e| format!("{:?}", e))?;

        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(128)];
        let seed_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&main_descriptor_set_layout))
            .push_constant_ranges(&ranges); // Standard push constants size
        let seed_layout = unsafe { device.create_pipeline_layout(&seed_layout_info, None) }.map_err(|e| format!("{:?}", e))?;

        let seed_pipeline = Self::create_seed_pipeline(device, seed_render_pass, seed_layout, seed_module)?;
        unsafe { device.destroy_shader_module(seed_module, None); }

        // 6. Allocate and Update Descriptor Sets
        let k5_descriptor_sets = Self::allocate_and_update_k5_sets(device, descriptor_pool, k5_set_layout, &jfa_images)?;
        let k5_resolve_descriptor_sets = Self::allocate_and_update_resolve_sets(device, descriptor_pool, k5_resolve_set_layout, &jfa_images, &sdf_image)?;

        Ok(Self {
            ctx,
            jfa_images,
            sdf_image,
            jfa_framebuffers,
            seed_render_pass,
            seed_pipeline,
            seed_layout,
            k5_pipeline,
            k5_layout,
            k5_descriptor_sets,
            k5_set_layout,
            k5_resolve_pipeline,
            k5_resolve_layout,
            k5_resolve_descriptor_sets,
            k5_resolve_set_layout,
        })
    }

    fn create_jfa_image(ctx: Arc<VulkanContext>, width: u32, height: u32) -> Result<ManagedImage, String> {
        let (image, memory, view) = resources::create_texture(
            &ctx.device, &ctx.instance, ctx.physical_device, width, height, 1, None,
            vk::Format::R32G32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR
        )?;
        Ok(ManagedImage::new(ctx, image, memory, view, vk::Format::R32G32_SFLOAT, vk::Extent3D { width, height, depth: 1 }, vk::ImageUsageFlags::STORAGE, vk::ImageAspectFlags::COLOR))
    }

    fn create_sdf_image(ctx: Arc<VulkanContext>, width: u32, height: u32) -> Result<ManagedImage, String> {
        let (image, memory, view) = resources::create_texture(
            &ctx.device, &ctx.instance, ctx.physical_device, width, height, 1, None,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR
        )?;
        Ok(ManagedImage::new(ctx, image, memory, view, vk::Format::R32_SFLOAT, vk::Extent3D { width, height, depth: 1 }, vk::ImageUsageFlags::STORAGE, vk::ImageAspectFlags::COLOR))
    }

    fn create_seed_render_pass(device: &ash::Device) -> Result<vk::RenderPass, String> {
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R32G32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL);

        let attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&attachment_ref));

        let info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass));

        unsafe { device.create_render_pass(&info, None).map_err(|e| format!("{:?}", e)) }
    }

    fn create_framebuffer(device: &ash::Device, rp: vk::RenderPass, view: vk::ImageView, width: u32, height: u32) -> Result<vk::Framebuffer, String> {
        let attachments = [view];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(rp)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        unsafe { device.create_framebuffer(&info, None).map_err(|e| format!("{:?}", e)) }
    }

    fn create_seed_pipeline(device: &ash::Device, rp: vk::RenderPass, layout: vk::PipelineLayout, module: vk::ShaderModule) -> Result<vk::Pipeline, String> {
        let entry = unsafe { CStr::from_bytes_with_nul_unchecked(b"vs_main\0") };
        let frag_entry = unsafe { CStr::from_bytes_with_nul_unchecked(b"fs_main\0") };
        
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(module).name(entry),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(module).name(frag_entry),
        ];

        let vi_info = vk::PipelineVertexInputStateCreateInfo::default(); 
        let ia_info = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vp_info = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let rs_info = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let ms_info = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds_info = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(false);
        let cb_attach = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);
        let cb_info = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&cb_attach));

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi_info)
            .input_assembly_state(&ia_info)
            .viewport_state(&vp_info)
            .dynamic_state(&dyn_info)
            .rasterization_state(&rs_info)
            .multisample_state(&ms_info)
            .depth_stencil_state(&ds_info)
            .color_blend_state(&cb_info)
            .layout(layout)
            .render_pass(rp)
            .subpass(0);

        unsafe { 
            device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None)
                .map_err(|e| format!("{:?}", e))
                .map(|p| p[0])
        }
    }

    fn allocate_and_update_k5_sets(device: &ash::Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout, jfa_images: &[ManagedImage; 2]) -> Result<[vk::DescriptorSet; 2], String> {
        let layouts = [layout, layout];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).map_err(|e| format!("{:?}", e))? };
        
        for i in 0..2 {
            let src_idx = i;
            let dst_idx = (i + 1) % 2;
            let img_info_src_0 = vk::DescriptorImageInfo::default().image_view(jfa_images[src_idx].view).image_layout(vk::ImageLayout::GENERAL);
            let img_info_src_1 = vk::DescriptorImageInfo::default().image_view(jfa_images[dst_idx].view).image_layout(vk::ImageLayout::GENERAL);
            let img_info_dst = vk::DescriptorImageInfo::default().image_view(jfa_images[dst_idx].view).image_layout(vk::ImageLayout::GENERAL);

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&img_info_src_0)),
                vk::WriteDescriptorSet::default().dst_set(sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&img_info_src_1)),
                vk::WriteDescriptorSet::default().dst_set(sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(std::slice::from_ref(&img_info_dst)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]); }
        }
        Ok([sets[0], sets[1]])
    }

    fn allocate_and_update_resolve_sets(device: &ash::Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout, jfa_images: &[ManagedImage; 2], sdf_image: &ManagedImage) -> Result<[vk::DescriptorSet; 2], String> {
        let layouts = [layout, layout];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).map_err(|e| format!("{:?}", e))? };
        
        for i in 0..2 {
            let info_src = vk::DescriptorImageInfo::default().image_view(jfa_images[i].view).image_layout(vk::ImageLayout::GENERAL);
            let info_dst = vk::DescriptorImageInfo::default().image_view(sdf_image.view).image_layout(vk::ImageLayout::GENERAL);

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&info_src)),
                vk::WriteDescriptorSet::default().dst_set(sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(std::slice::from_ref(&info_dst)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]); }
        }
        Ok([sets[0], sets[1]])
    }

    pub fn execute(
        &mut self,
        cmd: vk::CommandBuffer,
        width: u32,
        height: u32,
        intensity: f32,
        decay: f32,
        radius: f32,
        descriptor_set: vk::DescriptorSet, // Main descriptor set for seed pass
    ) {
        unsafe {
            let device = &self.ctx.device;

            // 1. Seed Pass (Graphics)
            // Transition JFA image 0 to COLOR_ATTACHMENT
            self.jfa_images[0].transition(
                cmd,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            );

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [-1.0, -1.0, 0.0, 0.0],
                },
            }];

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.seed_render_pass)
                .framebuffer(self.jfa_framebuffers[0])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.seed_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.seed_layout, 0, &[descriptor_set], &[]);
            // The actual draw call would happen here if we used geometry for seeding.
            // Currently it relies on clear + potential future logic.
            device.cmd_end_render_pass(cmd);

            // 2. JFA Compute Loop
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_pipeline);

            let max_dim = width.max(height);
            let steps = (max_dim as f32).log2().ceil() as u32; 
            
            for i in 0..steps {
                let step_width = 1 << (steps - 1 - i);
                let (read_idx, write_idx) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
                
                // Transition Read Image to GENERAL (Sampled)
                self.jfa_images[read_idx as usize].transition(
                    cmd,
                    vk::ImageLayout::GENERAL,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                );

                // Transition Write Image to GENERAL (Storage)
                self.jfa_images[write_idx as usize].transition(
                    cmd,
                    vk::ImageLayout::GENERAL,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                );

                let uniforms = JFAUniforms {
                    width,
                    height,
                    jfa_step: step_width,
                    ping_pong_idx: read_idx,
                    intensity,
                    decay,
                    radius,
                    _pad: 0,
                };
                let pc_bytes = std::slice::from_raw_parts(&uniforms as *const _ as *const u8, std::mem::size_of::<JFAUniforms>());
                device.cmd_push_constants(cmd, self.k5_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_layout, 0, &[self.k5_descriptor_sets[read_idx as usize]], &[]);

                let group_x = (width + 7) / 8;
                let group_y = (height + 7) / 8;
                device.cmd_dispatch(cmd, group_x, group_y, 1);
            }
            
            // 3. Resolve Pass
            let final_jfa_idx = if steps % 2 == 0 { 0 } else { 1 };
            
            self.jfa_images[final_jfa_idx as usize].transition(
                cmd,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );

            self.sdf_image.transition(
                cmd,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_resolve_pipeline);
            
            let uniforms = JFAUniforms {
                width,
                height,
                jfa_step: 0,
                ping_pong_idx: final_jfa_idx,
                intensity,
                decay,
                radius,
                _pad: 0,
            };
            let pc_bytes = std::slice::from_raw_parts(&uniforms as *const _ as *const u8, std::mem::size_of::<JFAUniforms>());
            device.cmd_push_constants(cmd, self.k5_resolve_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_resolve_layout, 0, &[self.k5_resolve_descriptor_sets[final_jfa_idx as usize]], &[]);

            let group_x = (width + 7) / 8;
            let group_y = (height + 7) / 8;
            device.cmd_dispatch(cmd, group_x, group_y, 1);
            
            // Final transition of SDF for shader use
            self.sdf_image.transition(
                cmd,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );
        }
    }
}

impl Drop for SdfSystem {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            for &fb in &self.jfa_framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            device.destroy_render_pass(self.seed_render_pass, None);
            device.destroy_pipeline(self.seed_pipeline, None);
            device.destroy_pipeline_layout(self.seed_layout, None);
            device.destroy_pipeline(self.k5_pipeline, None);
            device.destroy_pipeline_layout(self.k5_layout, None);
            device.destroy_descriptor_set_layout(self.k5_set_layout, None);
            device.destroy_pipeline(self.k5_resolve_pipeline, None);
            device.destroy_pipeline_layout(self.k5_resolve_layout, None);
            device.destroy_descriptor_set_layout(self.k5_resolve_set_layout, None);
        }
    }
}
