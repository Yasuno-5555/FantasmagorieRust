use ash::vk;
use std::sync::Arc;
use crate::backend::vulkan::VulkanContext;
use crate::backend::vulkan::pipelines;
use crate::backend::vulkan::resources;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct K4PushConstants {
    pub exposure: f32,
    pub gamma: f32,
    pub fog_density: f32,
    pub _pad: f32,
}

pub struct PostProcessSystem {
    pub ctx: Arc<VulkanContext>,
    
    // Pipelines
    pub k4_pipeline: vk::Pipeline,
    pub k4_layout: vk::PipelineLayout,
    pub k4_descriptor_set: vk::DescriptorSet,
    pub k4_set_layout: vk::DescriptorSetLayout,
    
    // Render Pass for HDR
    pub hdr_render_pass: vk::RenderPass,
    pub hdr_framebuffer: vk::Framebuffer,
}

impl PostProcessSystem {
    pub fn new(
        ctx: Arc<VulkanContext>,
        width: u32,
        height: u32,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffer: vk::Buffer,
        uniform_size: vk::DeviceSize,
        backdrop_view: vk::ImageView,
        sdf_view: vk::ImageView,
        font_view: vk::ImageView,
        sampler: vk::Sampler,
        audio_params_buffer: vk::Buffer,
    ) -> Result<Self, String> {
        let device = &ctx.device;

        // 1. Create HDR Render Pass
        let hdr_render_pass = Self::create_hdr_render_pass(device)?;
        let hdr_framebuffer = Self::create_hdr_framebuffer(device, hdr_render_pass, backdrop_view, width, height)?;

        // 2. K4: Cinematic Resolver Pipeline
        let k4_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(4).descriptor_type(vk::DescriptorType::SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(5).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(6).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let k4_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k4_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k4_pc_range = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<K4PushConstants>() as u32);
        let k4_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k4_set_layout)).push_constant_ranges(std::slice::from_ref(&k4_pc_range)), None) }.map_err(|e| format!("{:?}", e))?;
        let k4_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k4_layout, include_str!("../../shaders/k4_resolver.wgsl"), "main") }?;
        
        let k4_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k4_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];

        // Update K4 Descriptor Set
        unsafe {
            let u_info = [vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size)];
            let img_backdrop = [vk::DescriptorImageInfo::default().image_view(backdrop_view).image_layout(vk::ImageLayout::GENERAL)];
            let img_sdf = [vk::DescriptorImageInfo::default().image_view(sdf_view).image_layout(vk::ImageLayout::GENERAL)];
            let img_font = [vk::DescriptorImageInfo::default().image_view(font_view).image_layout(vk::ImageLayout::GENERAL)];
            let s_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let a_info = [vk::DescriptorBufferInfo::default().buffer(audio_params_buffer).offset(0).range(vk::WHOLE_SIZE)];

            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&u_info),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&img_backdrop),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&img_sdf),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&img_backdrop), // Backdrop used twice? (In old code)
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(4).descriptor_type(vk::DescriptorType::SAMPLER).image_info(&s_info),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(5).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&img_backdrop),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(6).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&a_info),
            ], &[]);
        }

        Ok(Self {
            ctx,
            k4_pipeline,
            k4_layout,
            k4_descriptor_set,
            k4_set_layout,
            hdr_render_pass,
            hdr_framebuffer,
        })
    }

    pub fn execute(&mut self, cmd: vk::CommandBuffer, width: u32, height: u32, params: K4PushConstants) {
        unsafe {
            let device = &self.ctx.device;

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k4_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k4_layout, 0, &[self.k4_descriptor_set], &[]);
            
            let pc_bytes = std::slice::from_raw_parts(&params as *const _ as *const u8, std::mem::size_of::<K4PushConstants>());
            device.cmd_push_constants(cmd, self.k4_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

            let group_x = (width + 7) / 8;
            let group_y = (height + 7) / 8;
            device.cmd_dispatch(cmd, group_x, group_y, 1);
        }
    }

    fn create_hdr_render_pass(device: &ash::Device) -> Result<vk::RenderPass, String> {
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

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

    fn create_hdr_framebuffer(device: &ash::Device, rp: vk::RenderPass, view: vk::ImageView, width: u32, height: u32) -> Result<vk::Framebuffer, String> {
        let attachments = [view];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(rp)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        unsafe { device.create_framebuffer(&info, None).map_err(|e| format!("{:?}", e)) }
    }
}

impl Drop for PostProcessSystem {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            device.destroy_framebuffer(self.hdr_framebuffer, None);
            device.destroy_render_pass(self.hdr_render_pass, None);
            device.destroy_pipeline(self.k4_pipeline, None);
            device.destroy_pipeline_layout(self.k4_layout, None);
            device.destroy_descriptor_set_layout(self.k4_set_layout, None);
        }
    }
}
