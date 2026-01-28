//! Vulkan backend - Native Vulkan rendering via ash
//!
//! Full implementation using ash crate for Vulkan 1.2+ support.
//! Targets: Windows (Vulkan), Linux (Vulkan), Android (Vulkan)

use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;
use super::VulkanContext;
use super::swapchain::VulkanSurfaceContext;
use super::resources;
use super::pipelines;

/// Vulkan-based rendering backend


pub struct VulkanBackend {
    pub ctx: Arc<VulkanContext>,
    pub surface_ctx: VulkanSurfaceContext,
    
    command_buffer: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    render_pass_load: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,

    // Buffers
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    uniform_buffer: vk::Buffer,
    uniform_memory: vk::DeviceMemory,

    // Texture
    font_texture: vk::Image,
    font_texture_memory: vk::DeviceMemory,
    font_texture_view: vk::ImageView,
    sampler: vk::Sampler,
    descriptor_set: vk::DescriptorSet,

    // Sync
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,

    // Compute Pipelines (Aggressive Plan)
    k13_pipeline: vk::Pipeline,
    k13_layout: vk::PipelineLayout,
    k8_pipeline: vk::Pipeline,
    k8_layout: vk::PipelineLayout,
    k6_update_pipeline: vk::Pipeline,
    k6_spawn_pipeline: vk::Pipeline,
    k6_layout: vk::PipelineLayout,
    k5_pipeline: vk::Pipeline,
    k5_layout: vk::PipelineLayout,
    k4_pipeline: vk::Pipeline,
    k4_layout: vk::PipelineLayout,

    k13_descriptor_set: vk::DescriptorSet,
    k8_descriptor_set: vk::DescriptorSet,
    k6_descriptor_set: vk::DescriptorSet,
    k5_descriptor_sets: [vk::DescriptorSet; 2],
    k4_descriptor_set: vk::DescriptorSet,

    // Compute Storage Buffers
    indirect_dispatch_buffer: vk::Buffer,
    indirect_dispatch_memory: vk::DeviceMemory,
    indirect_draw_buffer: vk::Buffer,
    indirect_draw_memory: vk::DeviceMemory,
    counter_buffer: vk::Buffer,
    counter_memory: vk::DeviceMemory,
    instance_buffer: vk::Buffer,
    instance_memory: vk::DeviceMemory,
    particle_buffer: vk::Buffer,
    particle_memory: vk::DeviceMemory,
    counter_readback_buffer: vk::Buffer,
    counter_readback_memory: vk::DeviceMemory,

    start_time: std::time::Instant,
    last_image_index: u32,

    // Backdrop for blur
    backdrop_image: vk::Image,
    backdrop_memory: vk::DeviceMemory,
    backdrop_view: vk::ImageView,

    // JFA SDF Lighting (K5)
    jfa_images: [(vk::Image, vk::DeviceMemory, vk::ImageView); 2],
    sdf_image: (vk::Image, vk::DeviceMemory, vk::ImageView),
    jfa_framebuffers: [vk::Framebuffer; 2],
    seed_render_pass: vk::RenderPass,
    seed_pipeline: vk::Pipeline,
    seed_layout: vk::PipelineLayout,
    k5_resolve_pipeline: vk::Pipeline,
    k5_resolve_layout: vk::PipelineLayout,
    k5_resolve_descriptor_sets: [vk::DescriptorSet; 2],

    pub exposure: f32,
    pub gamma: f32,
    pub fog_density: f32,
    pub audio_params: crate::backend::shaders::types::AudioParams,
    pub audio_gain: f32,

    // Profiling
    query_pools: [vk::QueryPool; 2],
    timestamp_period: f32,
    frame_index: usize,

    // HDR Pipeline
    hdr_render_pass: vk::RenderPass,
    hdr_framebuffer: vk::Framebuffer,

    // K12 Audio
    k12_pipeline: vk::Pipeline,
    k12_layout: vk::PipelineLayout,
    k12_set_layout: vk::DescriptorSetLayout, 
    k12_descriptor_set: vk::DescriptorSet,
    k4_set_layout: vk::DescriptorSetLayout,
    spectrum_buffer: vk::Buffer,
    spectrum_memory: vk::DeviceMemory,
    audio_params_buffer: vk::Buffer,
    audio_params_memory: vk::DeviceMemory,
}

/// Vertex format for Vulkan
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

/// Push constants for per-draw data (Matches DrawUniforms in shaders)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PushConstants {
    rect: [f32; 4],
    radii: [f32; 4],
    border_color: [f32; 4],
    glow_color: [f32; 4],
    offset: [f32; 2],
    scale: f32,
    border_width: f32,
    elevation: f32,
    glow_strength: f32,
    lut_intensity: f32,
    mode: i32,
    is_squircle: i32,
    time: f32,
    _pad: f32,
}
// Note: 104 bytes, fits in 128B Push Constant limit. 
// Standardized to match crate::backend::shaders::types::DrawUniforms

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct JFAUniforms {
    width: u32,
    height: u32,
    pub jfa_step: u32,
    ping_pong_idx: u32,
    intensity: f32,
    decay: f32,
    radius: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct K4PushConstants {
    exposure: f32,
    gamma: f32,
    fog_density: f32,
    _pad: f32,
}

// Using crate::backend::shaders::types::AudioParams for unified audio reactive parameters

/// Uniform buffer for global data (Matches GlobalUniforms in shaders)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    projection: [[f32; 4]; 4],
    viewport: [f32; 2],
    time: f32,
    audio_gain: f32, // Replaced _pad
}

impl VulkanBackend {
    /// Create a new Vulkan backend
    ///
    /// # Safety
    /// Requires valid window handle for surface creation
    #[cfg(target_os = "windows")]
    pub unsafe fn new(
        hwnd: *mut std::ffi::c_void,
        hinstance: *mut std::ffi::c_void,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let ctx = VulkanContext::new(hinstance, hwnd)?;
        Self::new_with_context(ctx, hinstance, hwnd, width, height)
    }

    /// Create a new Vulkan backend with an existing context
    ///
    /// # Safety
    /// Requires valid context and window handle
    #[cfg(target_os = "windows")]
    pub unsafe fn new_with_context(
        ctx: Arc<VulkanContext>,
        hinstance: *mut std::ffi::c_void,
        hwnd: *mut std::ffi::c_void,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        { use std::io::Write; let mut f = std::fs::OpenOptions::new().append(true).create(true).open("debug_log.txt").unwrap(); writeln!(f, "[VulkanBackend] new_with_context starting...").unwrap(); f.sync_all().unwrap(); }
        // Create render pass (needed for swapchain framebuffers)
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::B8G8R8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&attachment_ref));

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass));

        let render_pass = ctx.device.create_render_pass(&render_pass_info, None).map_err(|e| format!("{:?}", e))?;

        // Create Surface Context (Swapchain)
        let surface_ctx = VulkanSurfaceContext::new(&ctx, hinstance, hwnd, width, height, render_pass)?;

        let render_pass_load = {
            let attachment_load = vk::AttachmentDescription::default()
                .format(surface_ctx.format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

            let render_pass_info_load = vk::RenderPassCreateInfo::default()
                .attachments(std::slice::from_ref(&attachment_load))
                .subpasses(std::slice::from_ref(&subpass));

            ctx.device.create_render_pass(&render_pass_info_load, None).map_err(|e| format!("{:?}", e))?
        };

        let device = &ctx.device;
        let instance = &ctx.instance;
        let physical_device = ctx.physical_device;
        let graphics_family = ctx.graphics_family;
        let descriptor_pool = ctx.descriptor_pool;
        let command_pool = ctx.command_pool;

        // Create descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE) // Font Texture
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER) // Universal Sampler
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE) // LUT Texture
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE) // Backdrop Texture
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER) // Instance Data
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| format!("Failed to create descriptor set layout: {:?}", e))?;

        // Pipeline Layout with Push Constants
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))?;

        // Sync Objects
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(|e| format!("{:?}", e))?;
        let render_finished_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(|e| format!("{:?}", e))?;
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = unsafe { device.create_fence(&fence_info, None) }.map_err(|e| format!("{:?}", e))?;

        // Command Buffer
        let command_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&command_alloc_info) }.map_err(|e| format!("{:?}", e))?[0];

        // 1. Create Counter Buffer (STORAGE + VERTEX + TRANSFER)
        let (counter_buffer, counter_memory) = resources::create_buffer(
            device, instance, physical_device, 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 2. Create Vertex Buffer
        let vertex_buffer_size = 65536 * std::mem::size_of::<Vertex>() as vk::DeviceSize;
        let (vertex_buffer, vertex_memory) = resources::create_buffer(
            device, instance, physical_device, vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 3. Create Uniform Buffer
        let uniform_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let (uniform_buffer, uniform_memory) = resources::create_buffer(
            device, instance, physical_device, uniform_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 4. Create Compute Storage Buffers
        let (indirect_dispatch_buffer, indirect_dispatch_memory) = resources::create_buffer(
            device, instance, physical_device, 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (indirect_draw_buffer, indirect_draw_memory) = resources::create_buffer(
            device, instance, physical_device, 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (instance_buffer, instance_memory) = resources::create_buffer(
            device, instance, physical_device, 1024 * 1024 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let (particle_buffer, particle_memory) = resources::create_buffer(
            device, instance, physical_device, 1024 * 1024 * 32,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let (counter_readback_buffer, counter_readback_memory) = resources::create_buffer(
            device, instance, physical_device, 1024,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 5. Create Textures
        let (font_texture, font_texture_memory, font_texture_view) = resources::create_texture(
            device, instance, physical_device, 1024, 1024, None, vk::Format::R8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED, vk::ImageAspectFlags::COLOR,
        )?;

        let (backdrop_image, backdrop_memory, backdrop_view) = resources::create_texture(
            device, instance, physical_device, width, height, None, vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE, vk::ImageAspectFlags::COLOR,
        )?;

        let jfa_images = [
            resources::create_texture(device, instance, physical_device, width, height, None, vk::Format::R32G32_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageAspectFlags::COLOR)?,
            resources::create_texture(device, instance, physical_device, width, height, None, vk::Format::R32G32_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageAspectFlags::COLOR)?,
        ];

        let (sdf_image_raw, sdf_memory_raw, sdf_view_raw) = resources::create_texture(
            device, instance, physical_device, width, height, None, vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC, vk::ImageAspectFlags::COLOR,
        )?;
        let sdf_image = (sdf_image_raw, sdf_memory_raw, sdf_view_raw);

        // 6. Create Sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }.map_err(|e| format!("{:?}", e))?;

        // 7. Descriptor Updates
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&descriptor_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];
        
        let db_info = vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size);
        let font_info = vk::DescriptorImageInfo::default().image_view(font_texture_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);
        let backdrop_info = vk::DescriptorImageInfo::default().image_view(backdrop_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let instance_info = vk::DescriptorBufferInfo::default().buffer(instance_buffer).offset(0).range(vk::WHOLE_SIZE);

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(std::slice::from_ref(&db_info)),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&font_info)),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLER).image_info(std::slice::from_ref(&sampler_info)),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&font_info)),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(4).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&backdrop_info)),
            vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(5).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&instance_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]); }

        // HDR Render Pass
        let hdr_attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let hdr_ref = vk::AttachmentReference::default().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let hdr_subpass = vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS).color_attachments(std::slice::from_ref(&hdr_ref));
        let hdr_rp_info = vk::RenderPassCreateInfo::default().attachments(std::slice::from_ref(&hdr_attachment)).subpasses(std::slice::from_ref(&hdr_subpass));
        let hdr_render_pass = unsafe { device.create_render_pass(&hdr_rp_info, None).map_err(|e| format!("{:?}", e))? };

        let hdr_framebuffer = unsafe { device.create_framebuffer(&vk::FramebufferCreateInfo::default().render_pass(hdr_render_pass).attachments(&[backdrop_view]).width(width).height(height).layers(1), None).map_err(|e| format!("{:?}", e))? };

        let wgsl_src = include_str!("../vulkan.wgsl");
        let spv_binary = pipelines::compile_wgsl(wgsl_src)?;

        // Create Modules
        let shader_code = vk::ShaderModuleCreateInfo::default().code(&spv_binary);
        let shader_module = unsafe {
            device
                .create_shader_module(&shader_code, None)
                .map_err(|e| format!("{:?}", e))?
        };

        let v_main = unsafe { CStr::from_bytes_with_nul_unchecked(b"vs_main\0") };
        let f_main = unsafe { CStr::from_bytes_with_nul_unchecked(b"fs_main\0") };

        let vert_stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(shader_module).name(v_main);
        let frag_stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(shader_module).name(f_main);
        let shader_stages = [vert_stage, frag_stage];

        // Pipeline States
        let v_binding = vk::VertexInputBindingDescription { binding: 0, stride: std::mem::size_of::<Vertex>() as u32, input_rate: vk::VertexInputRate::VERTEX };
        let v_attrs = [
            vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 0 },
            vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 8 },
            vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R32G32B32A32_SFLOAT, offset: 16 },
        ];
        // Use explicit slices to ensure pointers remain valid
        let v_bindings = [v_binding];
        // Consolidate states to avoid builder pointer issues
        let v_bindings = [v_binding];
        let v_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&v_bindings)
            .vertex_attribute_descriptions(&v_attrs);
        let ia_state_info = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST).primitive_restart_enable(false);
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dyn_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let rs_state_info = vk::PipelineRasterizationStateCreateInfo::default().depth_clamp_enable(false).rasterizer_discard_enable(false).polygon_mode(vk::PolygonMode::FILL).line_width(1.0).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::CLOCKWISE).depth_bias_enable(false);
        let ms_state_info = vk::PipelineMultisampleStateCreateInfo::default().sample_shading_enable(false).rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let cb_attach = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA).blend_enable(true).src_color_blend_factor(vk::BlendFactor::SRC_ALPHA).dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA).color_blend_op(vk::BlendOp::ADD).src_alpha_blend_factor(vk::BlendFactor::ONE).dst_alpha_blend_factor(vk::BlendFactor::ZERO).alpha_blend_op(vk::BlendOp::ADD);
        let cb_attachments = [cb_attach];
        let cb_state_info = vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false).attachments(&cb_attachments);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&v_input_state_info)
            .input_assembly_state(&ia_state_info)
            .viewport_state(&viewport_state_info)
            .dynamic_state(&dyn_state_info)
            .rasterization_state(&rs_state_info)
            .multisample_state(&ms_state_info)
            .color_blend_state(&cb_state_info)
            .layout(pipeline_layout)
            .render_pass(hdr_render_pass)
            .subpass(0);
        let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None).map_err(|e| format!("{:?}", e))? }[0];
        device.destroy_shader_module(shader_module, None);

        // K13: Indirect Dispatch

        let k13_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k13_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k13_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k13_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k13_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        let k13_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k13_layout, include_str!("../shaders/k13_indirect.wgsl"), "main") }?;
        let k13_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k13_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];

        // Update K13 Writes
        let k13_bis = [
            vk::DescriptorBufferInfo::default().buffer(counter_buffer).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(indirect_draw_buffer).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(indirect_dispatch_buffer).offset(0).range(vk::WHOLE_SIZE),
        ];
        unsafe {
            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k13_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&k13_bis[0..1]),
                vk::WriteDescriptorSet::default().dst_set(k13_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&k13_bis[1..2]),
                vk::WriteDescriptorSet::default().dst_set(k13_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&k13_bis[2..3]),
            ], &[]);
        }

        // Query Pools
        let query_pool_info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(128);
        let qp0 = unsafe { device.create_query_pool(&query_pool_info, None) }.map_err(|e| format!("{:?}", e))?;
        let qp1 = unsafe { device.create_query_pool(&query_pool_info, None) }.map_err(|e| format!("{:?}", e))?;
        let timestamp_period = ctx.physical_device_properties.limits.timestamp_period;

        // K8: Visibility Culling

        let k8_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k8_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k8_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k8_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k8_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        let k8_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k8_layout, include_str!("../shaders/k8_visibility.wgsl"), "main") }?;
        let k8_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k8_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];
        unsafe { device.update_descriptor_sets(&[
            vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&[vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size)]),
            vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[vk::DescriptorBufferInfo::default().buffer(instance_buffer).offset(0).range(vk::WHOLE_SIZE)]),
            vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&[vk::DescriptorImageInfo::default().image_view(backdrop_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
            vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[vk::DescriptorBufferInfo::default().buffer(indirect_draw_buffer).offset(0).range(vk::WHOLE_SIZE)]),
            vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[vk::DescriptorBufferInfo::default().buffer(counter_buffer).offset(0).range(vk::WHOLE_SIZE)]),
        ], &[]); }

        // K6: Particles

        let k6_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE), // Combined counters
        ];
        let k6_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k6_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k6_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k6_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        let k6_update_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k6_layout, include_str!("../shaders/k6_particle.wgsl"), "update") }?;
        let k6_spawn_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k6_layout, include_str!("../shaders/k6_particle.wgsl"), "spawn") }?;
        let k6_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k6_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];
        unsafe { device.update_descriptor_sets(&[
            vk::WriteDescriptorSet::default().dst_set(k6_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&[vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size)]),
            vk::WriteDescriptorSet::default().dst_set(k6_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[vk::DescriptorBufferInfo::default().buffer(particle_buffer).offset(0).range(vk::WHOLE_SIZE)]),
            vk::WriteDescriptorSet::default().dst_set(k6_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[vk::DescriptorBufferInfo::default().buffer(counter_buffer).offset(0).range(vk::WHOLE_SIZE)]),
        ], &[]); }

        // K5: JFA SDF
        let k5_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k5_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k5_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k5_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k5_set_layout)).push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<JFAUniforms>() as u32)]), None) }.map_err(|e| format!("{:?}", e))?;
        let k5_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k5_layout, include_str!("../shaders/k5_jfa.wgsl"), "main") }?;

        // K5 Resolve
        let k5_resolve_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k5_resolve_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k5_resolve_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k5_resolve_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k5_resolve_set_layout)).push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<JFAUniforms>() as u32)]), None) }.map_err(|e| format!("{:?}", e))?;
        let k5_resolve_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k5_resolve_layout, include_str!("../shaders/k5_resolve.wgsl"), "main") }?;
        let k5_resolve_descriptor_sets_vec = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&[k5_resolve_set_layout, k5_resolve_set_layout])) }.map_err(|e| format!("{:?}", e))?;
        let k5_resolve_descriptor_sets = [k5_resolve_descriptor_sets_vec[0], k5_resolve_descriptor_sets_vec[1]];

        let k5_descriptor_sets_vec = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&[k5_set_layout, k5_set_layout])) }.map_err(|e| format!("{:?}", e))?;
        let k5_descriptor_sets = [k5_descriptor_sets_vec[0], k5_descriptor_sets_vec[1]];
        for i in 0..2 {
            let resolve_img_info_src = vk::DescriptorImageInfo::default().image_view(jfa_images[i].2).image_layout(vk::ImageLayout::GENERAL);
            let resolve_img_info_dst = vk::DescriptorImageInfo::default().image_view(sdf_image.2).image_layout(vk::ImageLayout::GENERAL);
            unsafe { device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k5_resolve_descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&resolve_img_info_src)),
                vk::WriteDescriptorSet::default().dst_set(k5_resolve_descriptor_sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(std::slice::from_ref(&resolve_img_info_dst)),
            ], &[]); }
        }
        let seed_wgsl = include_str!("../shaders/seed.wgsl");

        let seed_spv = pipelines::compile_wgsl(seed_wgsl)?;
        let seed_module_info = vk::ShaderModuleCreateInfo::default().code(&seed_spv);
        let seed_module = unsafe { device.create_shader_module(&seed_module_info, None) }
            .map_err(|e| format!("Failed to create seed shader module: {:?}", e))?;

        let seed_render_pass_attachments = [vk::AttachmentDescription::default()
            .format(vk::Format::R32G32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL)];
        
        let seed_color_attachment_ref = vk::AttachmentReference::default().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let seed_subpass = vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS).color_attachments(std::slice::from_ref(&seed_color_attachment_ref));
        let seed_render_pass_info = vk::RenderPassCreateInfo::default().attachments(&seed_render_pass_attachments).subpasses(std::slice::from_ref(&seed_subpass));
        let seed_render_pass = unsafe { device.create_render_pass(&seed_render_pass_info, None) }.map_err(|e| format!("Seed RP: {:?}", e))?;

        let jfa_framebuffers = [
            unsafe { device.create_framebuffer(&vk::FramebufferCreateInfo::default().render_pass(seed_render_pass).attachments(&[jfa_images[0].2]).width(width).height(height).layers(1), None) }.map_err(|e| format!("JFA FB 0: {:?}", e))?,
            unsafe { device.create_framebuffer(&vk::FramebufferCreateInfo::default().render_pass(seed_render_pass).attachments(&[jfa_images[1].2]).width(width).height(height).layers(1), None) }.map_err(|e| format!("JFA FB 1: {:?}", e))?,
        ];

        let layouts = [descriptor_set_layout];
            let push_ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).offset(0).size(std::mem::size_of::<PushConstants>() as u32)];
            let seed_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_ranges);
        let seed_layout = unsafe { device.create_pipeline_layout(&seed_layout_info, None) }.map_err(|e| format!("Seed Layout: {:?}", e))?;

        // Pipeline state... (I'll keep it minimal)
        let seed_stage_infos = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(seed_module).name(unsafe { CStr::from_bytes_with_nul_unchecked(b"vs_main\0") }),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(seed_module).name(unsafe { CStr::from_bytes_with_nul_unchecked(b"fs_main\0") }),
        ];
        // Reuse vertex input from main pipeline but for RG32_SFLOAT
        let vi_info = vk::PipelineVertexInputStateCreateInfo::default(); // Using unit quad in VS, no buffers needed for position if hardcoded, but we use Vertex for simplicity
        let ia_info = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        
        // Fix: Enable Dynamic State for Seed Pipeline too
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dyn_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let vp_info = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        
        let rs_info = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let ms_info = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds_info = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(false);
        let cb_attach = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);
        let cb_attachments = [cb_attach];
        let cb_info = vk::PipelineColorBlendStateCreateInfo::default().attachments(&cb_attachments);

        let seed_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&seed_stage_infos)
            .vertex_input_state(&vi_info)
            .input_assembly_state(&ia_info)
            .viewport_state(&vp_info)
            .dynamic_state(&dyn_state_info) // Added dynamic state
            .rasterization_state(&rs_info)
            .multisample_state(&ms_info)
            .depth_stencil_state(&ds_info)
            .color_blend_state(&cb_info)
            .layout(seed_layout)
            .render_pass(seed_render_pass)
            .subpass(0);

        let seed_pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[seed_pipeline_info], None) }.map_err(|e| format!("Seed Pipe: {:?}", e))?[0];
        unsafe { device.destroy_shader_module(seed_module, None); }

        let layouts = [k5_set_layout, k5_set_layout];
        let k5_alloc_info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
        let k5_descriptor_sets_vec = unsafe { device.allocate_descriptor_sets(&k5_alloc_info) }
            .map_err(|e| format!("Failed to allocate K5 descriptor sets: {:?}", e))?;
        let k5_descriptor_sets = [k5_descriptor_sets_vec[0], k5_descriptor_sets_vec[1]];

        for i in 0..2 {
            let src_idx = i;
            let dst_idx = (i + 1) % 2;
            
            let img_info_src_0 = vk::DescriptorImageInfo::default().image_view(jfa_images[src_idx].2).image_layout(vk::ImageLayout::GENERAL);
            let img_info_src_1 = vk::DescriptorImageInfo::default().image_view(jfa_images[dst_idx].2).image_layout(vk::ImageLayout::GENERAL);
            let img_info_dst = vk::DescriptorImageInfo::default().image_view(jfa_images[dst_idx].2).image_layout(vk::ImageLayout::GENERAL);
            let db_info = vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size);

            let k5_writes = [
                vk::WriteDescriptorSet::default().dst_set(k5_descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&img_info_src_0)),
                vk::WriteDescriptorSet::default().dst_set(k5_descriptor_sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&img_info_src_1)),
                vk::WriteDescriptorSet::default().dst_set(k5_descriptor_sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(std::slice::from_ref(&img_info_dst)),
            ];
            unsafe { device.update_descriptor_sets(&k5_writes, &[]); }
        }


        // K4: Cinematic Resolver Pipeline
        let k4_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(4).descriptor_type(vk::DescriptorType::SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(5).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(6).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE), // Audio Params
        ];
        let k4_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k4_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k4_pc_range = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<K4PushConstants>() as u32);
        let k4_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k4_set_layout)).push_constant_ranges(std::slice::from_ref(&k4_pc_range)), None) }.map_err(|e| format!("{:?}", e))?;
        let k4_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k4_layout, include_str!("../shaders/k4_resolver.wgsl"), "main") }?;

        // K12 Descriptor Set Layout (Audio)
        let k12_bindings = [
             vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
             vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
             vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k12_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k12_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k12_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k12_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        let k12_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k12_layout, include_str!("../shaders/k12_audio.wgsl"), "main") }?;

        // Audio Buffers
        let (spectrum_buffer, spectrum_memory) = resources::create_buffer(device, instance, physical_device, (512 * 4) as vk::DeviceSize, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let (audio_params_buffer, audio_params_memory) = resources::create_buffer(device, instance, physical_device, std::mem::size_of::<crate::backend::shaders::types::AudioParams>() as vk::DeviceSize, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        // Allocate and Update Descriptor Sets
        let k4_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k4_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];
        let k12_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k12_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];

        let k4_db_info = vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size);
        let audio_params_info = vk::DescriptorBufferInfo::default().buffer(audio_params_buffer).offset(0).range(vk::WHOLE_SIZE);
        let spectrum_buffer_info = vk::DescriptorBufferInfo::default().buffer(spectrum_buffer).offset(0).range(vk::WHOLE_SIZE);
        
        unsafe {
            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&[k4_db_info]),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&[vk::DescriptorImageInfo::default().image_view(backdrop_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&[vk::DescriptorImageInfo::default().image_view(sdf_image.2).image_layout(vk::ImageLayout::GENERAL)]),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&[vk::DescriptorImageInfo::default().image_view(backdrop_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(4).descriptor_type(vk::DescriptorType::SAMPLER).image_info(&[vk::DescriptorImageInfo::default().sampler(sampler)]),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(5).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&[vk::DescriptorImageInfo::default().image_view(backdrop_view).image_layout(vk::ImageLayout::GENERAL)]),
                vk::WriteDescriptorSet::default().dst_set(k4_descriptor_set).dst_binding(6).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[audio_params_info]),
            ], &[]);

            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k12_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&[k4_db_info]),
                vk::WriteDescriptorSet::default().dst_set(k12_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[spectrum_buffer_info]),
                vk::WriteDescriptorSet::default().dst_set(k12_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&[audio_params_info]),
            ], &[]);
        }

        Ok(Self {
            ctx, surface_ctx, command_buffer, render_pass, render_pass_load,
            pipeline_layout, pipeline, descriptor_set_layout,
            vertex_buffer, vertex_memory, uniform_buffer, uniform_memory,
            indirect_dispatch_buffer, indirect_dispatch_memory, indirect_draw_buffer, indirect_draw_memory,
            counter_buffer, counter_memory, instance_buffer, instance_memory,
            particle_buffer, particle_memory, counter_readback_buffer, counter_readback_memory,
            start_time: std::time::Instant::now(), last_image_index: 0,
            image_available_semaphore, render_finished_semaphore, in_flight_fence,
            descriptor_set,
            k5_descriptor_sets,
            k12_set_layout,
            k12_descriptor_set,
            k4_set_layout,
            backdrop_image, backdrop_memory, backdrop_view,
            jfa_images, sdf_image,
            jfa_framebuffers,
            seed_render_pass,
            seed_pipeline,
            seed_layout,
            k5_resolve_pipeline,
            k5_resolve_layout,
            k5_resolve_descriptor_sets,
            query_pools: [qp0, qp1], timestamp_period, frame_index: 0,
            hdr_render_pass, hdr_framebuffer, 
            exposure: 1.0, gamma: 2.2, audio_gain: 1.0, fog_density: 0.05,
            audio_params: crate::backend::shaders::types::AudioParams { bass: 0.0, mid: 0.0, high: 0.0, _pad: 0.0 },
            k12_pipeline, k12_layout,
            spectrum_buffer, spectrum_memory,
            audio_params_buffer, audio_params_memory,
            font_texture, font_texture_memory, font_texture_view, sampler,
            k13_pipeline, k13_layout, k13_descriptor_set,
            k8_pipeline, k8_layout, k8_descriptor_set,
            k6_update_pipeline, k6_spawn_pipeline, k6_layout, k6_descriptor_set,
            k5_pipeline, k5_layout,
            k4_pipeline, k4_layout, k4_descriptor_set,
        })
    }

    pub fn update_audio_data(&mut self, spectrum: &[f32]) {
        // Map Spectrum Buffer and write
        unsafe {
            let size = (spectrum.len() * 4) as vk::DeviceSize;
            let ptr = self.ctx.device.map_memory(self.spectrum_memory, 0, size, vk::MemoryMapFlags::empty()).map_err(|e| format!("{:?}", e)).unwrap();
            let slice = std::slice::from_raw_parts_mut(ptr as *mut f32, spectrum.len());
            slice.copy_from_slice(spectrum);
            self.ctx.device.unmap_memory(self.spectrum_memory);
        }
    }

    pub fn set_k4_params(&mut self, exposure: f32, gamma: f32, fog_density: f32) {
        self.exposure = exposure;
        self.gamma = gamma;
        self.fog_density = fog_density;
    }

    pub fn set_audio_params(&mut self, gain: f32) {
        self.audio_gain = gain;
    }

    #[cfg(not(target_os = "windows"))]
    pub unsafe fn new(
        _hwnd: *mut std::ffi::c_void,
        _hinstance: *mut std::ffi::c_void,
        _width: u32,
        _height: u32,
    ) -> Result<Self, String> {
        Err("Vulkan backend Windows surface not available on this platform".to_string())
    }









    fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        [
            [2.0 / (right - left), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (top - bottom), 0.0, 0.0],
            [0.0, 0.0, -2.0 / (far - near), 0.0],
            [tx, ty, tz, 1.0],
        ]
    }

    #[allow(dead_code)]
    fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> [Vertex; 6] {
        Self::quad_vertices_uv(pos, size, [0.0, 0.0, 1.0, 1.0], color)
    }

    fn quad_vertices_uv(pos: Vec2, size: Vec2, uv: [f32; 4], color: ColorF) -> [Vertex; 6] {
        let (x0, y0) = (pos.x, pos.y);
        let (x1, y1) = (pos.x + size.x, pos.y + size.y);
        let c = [color.r, color.g, color.b, color.a];
        let (u0, v0, u1, v1) = (uv[0], uv[1], uv[2], uv[3]);

        [
            Vertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color: c,
            },
            Vertex {
                pos: [x0, y1],
                uv: [u0, v1],
                color: c,
            },
            Vertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color: c,
            },
            Vertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color: c,
            },
            Vertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color: c,
            },
            Vertex {
                pos: [x1, y0],
                uv: [u1, v0],
                color: c,
            },
        ]
    }

    unsafe fn cmd_transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            _ => (
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
            ),
        };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        self.ctx.device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    unsafe fn cmd_generate_mipmaps(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) {
        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            // Transition level i-1 to TRANSFER_SRC
            let barrier_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_src],
            );

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.ctx.device.cmd_blit_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );

            // Transition level i-1 to SHADER_READ
            let barrier_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_read],
            );

            if mip_width > 1 { mip_width /= 2; }
            if mip_height > 1 { mip_height /= 2; }
        }

        // Final level transition
        let barrier_final = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip_levels - 1,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        self.ctx.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_final],
        );
    }

    unsafe fn cmd_capture_backdrop(
        &self,
        command_buffer: vk::CommandBuffer,
        swapchain_image: vk::Image,
    ) {
        // 1. Transition swapchain to TRANSFER_SRC
        self.cmd_transition_image_layout(
            command_buffer,
            swapchain_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            1,
        );

        // 2. Transition backdrop to TRANSFER_DST
        self.cmd_transition_image_layout(
            command_buffer,
            self.backdrop_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            6, // 6 mip levels
        );

        // 3. Blit image (Full Screen to Backdrop)
        let blit = vk::ImageBlit::default()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.surface_ctx.extent.width as i32,
                    y: self.surface_ctx.extent.height as i32,
                    z: 1,
                },
            ])
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.surface_ctx.extent.width as i32,
                    y: self.surface_ctx.extent.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            });

        self.ctx.device.cmd_blit_image(
            command_buffer,
            swapchain_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.backdrop_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        // 4. Generate Mipmaps (transitions to SHADER_READ_ONLY)
        self.cmd_generate_mipmaps(
            command_buffer,
            self.backdrop_image,
            self.surface_ctx.extent.width,
            self.surface_ctx.extent.height,
            6,
        );

        // 5. Transition swapchain back to COLOR_ATTACHMENT
        self.cmd_transition_image_layout(
            command_buffer,
            swapchain_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1,
        );
    }

    pub unsafe fn reset_profiler(&self, cb: vk::CommandBuffer) {
        let pool_idx = self.frame_index % 2;
        self.ctx.device.cmd_reset_query_pool(cb, self.query_pools[pool_idx], 0, 64);
    }

    pub unsafe fn write_timestamp(&self, cb: vk::CommandBuffer, index: u32) {
        let pool_idx = self.frame_index % 2;
        self.ctx.device.cmd_write_timestamp(
            cb,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.query_pools[pool_idx],
            index,
        );
    }

    pub fn get_vulkan_stats(&self) -> (Vec<u64>, f32) {
        // Return results from the PREVIOUS frame (N-1) to avoid stalling
        if self.frame_index == 0 {
            return (vec![0u64; 64], self.timestamp_period);
        }

        let read_idx = (self.frame_index + 1) % 2; 
        
        let mut results = vec![0u64; 64];
        let pool = self.query_pools[read_idx];
        
        unsafe {
            let res = self.ctx.device.get_query_pool_results(
                pool,
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64,
            );
            
            if res.is_err() {
                 return (vec![0u64; 64], self.timestamp_period);
            }
        }
        (results, self.timestamp_period)
    }

    fn check_gpu_sanity(&self) {
        unsafe {
            let ptr = self.ctx.device.map_memory(self.counter_readback_memory, 0, 1024, vk::MemoryMapFlags::empty()).unwrap();
            let counters = std::slice::from_raw_parts(ptr as *const u32, 256);
            
            // Layout:
            // 0: visible instances
            // 1: indirect dispatch X
            // 8: start of error counters
            // 16: start of sanity flags (bitmask)
            
            let visible_count = counters[0];
            let error_count = counters[8];
            let sanity_flags = counters[16];

            if error_count > 0 || sanity_flags != 0 {
                eprintln!("⚠️ GPU SANITY ALERT [Frame {}]: Errors={}, Flags=0x{:08X}, Visible={}", self.frame_index, error_count, sanity_flags, visible_count);
                if error_count > 0 {
                    eprintln!("  - Total GPU detected errors: {}", error_count);
                }
                if sanity_flags & 0x1 != 0 { eprintln!("  - FLAG: K8 Visibility output empty while Input was not."); }
                if sanity_flags & 0x2 != 0 { eprintln!("  - FLAG: K13 Indirect generation overflow."); }
            }

            self.ctx.device.unmap_memory(self.counter_readback_memory);
        }
    }
}

impl VulkanBackend {
    /// Execute the K5 Jump Flooding Algorithm (JFA) to generate an SDF from seeds
    pub fn dispatch_jfa(&mut self, width: u32, height: u32, intensity: f32, decay: f32, radius: f32) {
        unsafe {
            let cmd = self.command_buffer;
            
            // 1. Seed Pass (Graphics)
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [-1.0, -1.0, 0.0, 0.0], // Negative ID indicates invalid
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

            self.ctx.device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
            self.ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.seed_pipeline);
            // Placeholder: no draw call for now, relying on clear to verify infrastructure
            self.ctx.device.cmd_end_render_pass(cmd);

            // Barrier: Graphics -> Compute
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL) // Was created as GENERAL/ColorAttach compatible
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(self.jfa_images[0].0)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.ctx.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);

            // 2. JFA Compute Loop
            self.ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_pipeline);

            let max_dim = width.max(height);
            let steps = (max_dim as f32).log2().ceil() as u32; 
            
            for i in 0..steps {
                let step_width = 1 << (steps - 1 - i);
                let (read_idx, write_idx) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
                
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
                self.ctx.device.cmd_push_constants(cmd, self.k5_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
                self.ctx.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_layout, 0, &[self.k5_descriptor_sets[read_idx as usize]], &[]);

                let group_x = (width + 7) / 8;
                let group_y = (height + 7) / 8;
                self.ctx.device.cmd_dispatch(cmd, group_x, group_y, 1);

                let img_barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .image(self.jfa_images[write_idx as usize].0)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                self.ctx.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[img_barrier]);
            }
            
            // 3. Resolve Pass
            let final_jfa_idx = if steps % 2 == 0 { 0 } else { 1 };
            self.ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_resolve_pipeline);
            
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
            self.ctx.device.cmd_push_constants(cmd, self.k5_resolve_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
            self.ctx.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k5_resolve_layout, 0, &[self.k5_resolve_descriptor_sets[final_jfa_idx as usize]], &[]);

            let group_x = (width + 7) / 8;
            let group_y = (height + 7) / 8;
            self.ctx.device.cmd_dispatch(cmd, group_x, group_y, 1);
            
            let sdf_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(self.sdf_image.0)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.ctx.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[sdf_barrier]);
        }
    }
}

impl crate::backend::GraphicsBackend for VulkanBackend {
    fn name(&self) -> &str {
        "Vulkan"
    }

    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        unsafe {
            // Create staging buffer
            let size = (width * height) as vk::DeviceSize;
            let (staging_buffer, staging_memory) = resources::create_buffer(
                &self.ctx.device,
                &self.ctx.instance,
                self.ctx.physical_device,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).unwrap();

            // Map and copy
            let ptr = self.ctx.device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, size as usize);
            self.ctx.device.unmap_memory(staging_memory);

            // Recreate image if size changed
            // For now, assume size is compatible with existing image (1024x1024 usually)
            // But let's check size
            
            // Transition to TRANSFER_DST
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.ctx.device.begin_command_buffer(self.command_buffer, &begin_info).unwrap();

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(self.font_texture)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.ctx.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width, height, depth: 1 });

            self.ctx.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging_buffer,
                self.font_texture,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition to SHADER_READ_ONLY
            let barrier = barrier
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(self.font_texture);

            self.ctx.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            self.ctx.device.end_command_buffer(self.command_buffer).unwrap();

            // Submit and wait
            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            self.ctx.device.queue_submit(self.ctx.graphics_queue, &[submit_info], vk::Fence::null()).unwrap();
            self.ctx.device.queue_wait_idle(self.ctx.graphics_queue).unwrap();

            // Clean up staging resources
            self.ctx.device.destroy_buffer(staging_buffer, None);
            self.ctx.device.free_memory(staging_memory, None);
        }
    }
    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        unsafe {
            // Wait for previous frame
            self.ctx.device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                .unwrap();
            
            self.check_gpu_sanity();

            self.ctx.device.reset_fences(&[self.in_flight_fence]).unwrap();

            // Acquire swapchain image
            let (image_index, _) = self
                .surface_ctx.swapchain_loader
                .acquire_next_image(
                    self.surface_ctx.swapchain,
                    u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();

            let instance_count = dl.commands().len() as u32;

            // Update uniform buffer
            let ubo = UniformBufferObject {
                projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
                viewport: [width as f32, height as f32],
                time: self.start_time.elapsed().as_secs_f32(),
                audio_gain: self.audio_gain,
            };
            let ptr = self
                .ctx
                .device
                .map_memory(
                    self.uniform_memory,
                    0,
                    std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            std::ptr::copy_nonoverlapping(
                &ubo as *const _ as *const u8,
                ptr as *mut u8,
                std::mem::size_of::<UniformBufferObject>(),
            );
            self.ctx.device.unmap_memory(self.uniform_memory);

            // Reset and begin command buffer
            self.ctx.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.ctx.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .unwrap();

            // K11: Reset Profiler & Mark Start
            self.reset_profiler(self.command_buffer);
            self.write_timestamp(self.command_buffer, 0); // Frame Start
            
            // --- Aggressive GPU logic Phase 1 (K8, K13) ---

            if instance_count > 0 {
                // 0. Transfer DrawCommands to Instance Buffer
                let instance_size = (instance_count as usize * std::mem::size_of::<PushConstants>()) as vk::DeviceSize;
                let inst_ptr = self.ctx.device.map_memory(self.instance_memory, 0, instance_size, vk::MemoryMapFlags::empty()).unwrap();
                let inst_slice = std::slice::from_raw_parts_mut(inst_ptr as *mut PushConstants, instance_count as usize);

                for (i, cmd) in dl.commands().iter().enumerate() {
                    let mut pc = PushConstants {
                        rect: [0.0; 4],
                        radii: [0.0; 4],
                        border_color: [0.0; 4],
                        glow_color: [0.0; 4],
                        offset: [0.0; 2],
                        scale: 1.0,
                        border_width: 0.0,
                        elevation: 0.0,
                        glow_strength: 0.0,
                        lut_intensity: 0.0,
                        mode: 0,
                        is_squircle: 0,
                        time: 0.0,
                        _pad: 0.0,
                    };
                    // Fill PC based on command (Duplicated logic from main loop for now, we should unify this)
                    match cmd {
                        DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, .. } => {
                            pc.rect = [pos.x, pos.y, size.x, size.y]; pc.radii = *radii;
                            pc.border_color = [border_color.r, border_color.g, border_color.b, border_color.a];
                            pc.glow_color = [glow_color.r, glow_color.g, glow_color.b, glow_color.a];
                            pc.mode = 2; pc.border_width = *border_width; pc.elevation = *elevation;
                            pc.is_squircle = if *is_squircle { 1 } else { 0 }; pc.glow_strength = *glow_strength;
                        }
                        DrawCommand::Text { pos, size, uv, color } => {
                            pc.rect = [pos.x, pos.y, size.x, size.y]; pc.border_color = [color.r, color.g, color.b, color.a]; pc.mode = 1; pc.radii = *uv;
                        }
                        DrawCommand::Image { pos, size, uv, color, radii, .. } => {
                            pc.rect = [pos.x, pos.y, size.x, size.y]; pc.radii = *radii; pc.border_color = [color.r, color.g, color.b, color.a]; pc.glow_color = *uv; pc.mode = 3;
                        }
                        _ => {}
                    }
                    inst_slice[i] = pc;
                }
                self.ctx.device.unmap_memory(self.instance_memory);

                // 1. Reset counters
                self.ctx.device.cmd_fill_buffer(self.command_buffer, self.counter_buffer, 0, 1024, 0);

                // 2. Memory Barrier for instance buffer and counters
                let barriers = [
                    vk::BufferMemoryBarrier::default().src_access_mask(vk::AccessFlags::HOST_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).buffer(self.instance_buffer).size(vk::WHOLE_SIZE),
                    vk::BufferMemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).buffer(self.counter_buffer).size(vk::WHOLE_SIZE),
                ];
                self.ctx.device.cmd_pipeline_barrier(self.command_buffer, vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &barriers, &[]);

                // 3. Dispatch K8: Visibility Culling
                self.ctx.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k8_pipeline);
                self.ctx.device.cmd_bind_descriptor_sets(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k8_layout, 0, &[self.k8_descriptor_set], &[]);
                self.ctx.device.cmd_dispatch(self.command_buffer, (instance_count + 63) / 64, 1, 1);

                // 4. Memory Barrier (K8 -> K13)
                let k8_to_k13_barrier = vk::BufferMemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).buffer(self.counter_buffer).size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier(self.command_buffer, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[k8_to_k13_barrier], &[]);

                // 5. Dispatch K13: Indirect Dispatch Generator
                self.ctx.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k13_pipeline);
                self.ctx.device.cmd_bind_descriptor_sets(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k13_layout, 0, &[self.k13_descriptor_set], &[]);
                self.ctx.device.cmd_dispatch(self.command_buffer, 1, 1, 1);

                // 6. Memory Barrier (K13 -> Draw Indirect)
                let barriers = [
                    vk::BufferMemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ).buffer(self.indirect_draw_buffer).size(vk::WHOLE_SIZE),
                    vk::BufferMemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ).buffer(self.indirect_dispatch_buffer).size(vk::WHOLE_SIZE),
                ];
                self.ctx.device.cmd_pipeline_barrier(self.command_buffer, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::DRAW_INDIRECT, vk::DependencyFlags::empty(), &[], &barriers, &[]);
            }

            // --- End Aggressive GPU logic ---

            println!("Vulkan: Compute dispatch added, command buffer begun");

            // Begin render pass
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.12, 1.0],
                },
            }];

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.hdr_render_pass)
                .framebuffer(self.hdr_framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: self.surface_ctx.extent.width,
                        height: self.surface_ctx.extent.height,
                    },
                })
                .clear_values(&clear_values);

            self.ctx.device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            // Bind pipeline and descriptors
            if self.pipeline != vk::Pipeline::null() {
                self.ctx.device.cmd_bind_pipeline(
                    self.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    self.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );

                // Set viewport and scissor
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.surface_ctx.extent.width as f32,
                    height: self.surface_ctx.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                self.ctx.device
                    .cmd_set_viewport(self.command_buffer, 0, &[viewport]);

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: self.surface_ctx.extent.width,
                        height: self.surface_ctx.extent.height,
                    },
                };
                self.ctx.device
                    .cmd_set_scissor(self.command_buffer, 0, &[scissor]);

                // --- GPU-driven Graphics Pass ---
                let unit_quad = [
                    Vertex { pos: [0.0, 0.0], uv: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
                    Vertex { pos: [0.0, 1.0], uv: [0.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
                    Vertex { pos: [1.0, 1.0], uv: [1.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
                    Vertex { pos: [0.0, 0.0], uv: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
                    Vertex { pos: [1.0, 1.0], uv: [1.0, 1.0], color: [1.0, 1.0, 1.0, 1.0] },
                    Vertex { pos: [1.0, 0.0], uv: [1.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
                ];
                let vb_size = (unit_quad.len() * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
                let vb_ptr = self.ctx.device.map_memory(self.vertex_memory, 0, vb_size, vk::MemoryMapFlags::empty()).unwrap();
                std::ptr::copy_nonoverlapping(unit_quad.as_ptr(), vb_ptr as *mut Vertex, unit_quad.len());
                self.ctx.device.unmap_memory(self.vertex_memory);

                self.ctx.device.cmd_bind_vertex_buffers(self.command_buffer, 0, &[self.vertex_buffer], &[0]);

                let pc_mode_instanced = PushConstants {
                    rect: [0.0; 4], radii: [0.0; 4], border_color: [1.0; 4], glow_color: [0.0; 4],
                    offset: [0.0; 2], scale: 1.0, border_width: 0.0, elevation: 0.0, glow_strength: 0.0,
                    lut_intensity: 0.0, mode: -1, is_squircle: 0, time: 0.0, _pad: 0.0,
                };
                self.ctx.device.cmd_push_constants(
                    self.command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(&pc_mode_instanced as *const _ as *const u8, std::mem::size_of::<PushConstants>()),
                );

                if instance_count > 0 {
                    self.ctx.device.cmd_draw_indirect(self.command_buffer, self.indirect_draw_buffer, 0, 1, std::mem::size_of::<vk::DrawIndirectCommand>() as u32);
                }
                /* Old graphics loop - deactivated
                if false {
                    let mut vertices: [Vertex; 6] = [Vertex {
                        pos: [0.0; 2],
                        uv: [0.0; 2],
                        color: [0.0; 4],
                    }; 6];
                    let mut has_draw = false;

                    let mut pc = PushConstants {
                        rect: [0.0; 4],
                        radii: [0.0; 4],
                        border_color: [0.0; 4],
                        glow_color: [0.0; 4],
                        offset: [0.0; 2],
                        scale: 1.0,
                        border_width: 0.0,
                        elevation: 0.0,
                        glow_strength: 0.0,
                        lut_intensity: 0.0,
                        mode: 0,
                        is_squircle: 0,
                        time: 0.0,
                        _pad: 0.0,
                    };

                    match cmd {
                        DrawCommand::RoundedRect {
                            pos,
                            size,
                            radii,
                            color,
                            elevation,
                            is_squircle,
                            border_width,
                            border_color,
                            glow_strength,
                            glow_color,
                            ..
                        } => {
                            let pad = if *elevation > 0.0 || *glow_strength > 0.0 {
                                100.0
                            } else {
                                0.0
                            };
                            vertices = Self::quad_vertices(
                                Vec2::new(pos.x - pad, pos.y - pad),
                                Vec2::new(size.x + pad * 2.0, size.y + pad * 2.0),
                                *color,
                            );
                            has_draw = true;
                            pc.rect = [pos.x, pos.y, size.x, size.y];
                            pc.radii = *radii;
                            pc.border_color =
                                [border_color.r, border_color.g, border_color.b, border_color.a];
                            pc.glow_color =
                                [glow_color.r, glow_color.g, glow_color.b, glow_color.a];
                            pc.mode = 2; // Shape
                            pc.border_width = *border_width;
                            pc.elevation = *elevation;
                            pc.is_squircle = if *is_squircle { 1 } else { 0 };
                            pc.glow_strength = *glow_strength;
                        }
                        DrawCommand::Text {
                            pos,
                            size,
                            uv,
                            color,
                        } => {
                            vertices = Self::quad_vertices_uv(*pos, *size, *uv, *color);
                            has_draw = true;
                            pc.mode = 1; // Text (SDF)
                        }
                        DrawCommand::Image {
                            pos,
                            size,
                            uv,
                            color,
                            radii,
                            ..
                        } => {
                            vertices = Self::quad_vertices_uv(*pos, *size, *uv, *color);
                            has_draw = true;
                            pc.rect = [pos.x, pos.y, size.x, size.y];
                            pc.radii = *radii;
                            pc.mode = 3; // Image
                        }
                        DrawCommand::Arc {
                             center,
                             radius,
                             start_angle,
                             end_angle,
                             thickness,
                             color,
                        } => {
                             let s = *radius * 2.0 + *thickness * 2.0;
                             let pos = Vec2::new(center.x - s * 0.5, center.y - s * 0.5);
                             vertices = Self::quad_vertices(pos, Vec2::new(s, s), *color);
                             has_draw = true;
                             pc.rect = [pos.x, pos.y, s, s];
                             pc.radii = [*radius, *thickness, 0.0, 0.0];
                             pc.elevation = *start_angle;
                             pc.glow_strength = *end_angle;
                             pc.mode = 6; // Arc
                        }
                         DrawCommand::BlurRect {
                            pos,
                            size,
                            radii,
                            color: _,
                            sigma,
                            is_squircle,
                         } => {
                            // High-Quality Blur requires backdrop capture
                            unsafe {
                                self.ctx.device.cmd_end_render_pass(self.command_buffer);
                                
                                self.cmd_capture_backdrop(
                                    self.command_buffer,
                                    self.surface_ctx.swapchain_images[image_index as usize],
                                );
                                
                                // RE-START RENDER PASS (LOAD)
                                let render_pass_info = vk::RenderPassBeginInfo::default()
                                    .render_pass(self.render_pass_load)
                                    .framebuffer(self.surface_ctx.framebuffers[image_index as usize])
                                    .render_area(vk::Rect2D {
                                        offset: vk::Offset2D { x: 0, y: 0 },
                                        extent: vk::Extent2D { width: self.surface_ctx.extent.width, height: self.surface_ctx.extent.height },
                                    });
                                self.ctx.device.cmd_begin_render_pass(self.command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
                                
                                // RE-BIND PIPELINE AND DESCRIPTORS
                                self.ctx.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                                self.ctx.device.cmd_bind_descriptor_sets(self.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.descriptor_set], &[]);
                                
                                // Re-set dynamic state
                                let viewport = vk::Viewport { x: 0.0, y: 0.0, width: self.surface_ctx.extent.width as f32, height: self.surface_ctx.extent.height as f32, min_depth: 0.0, max_depth: 1.0 };
                                self.ctx.device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
                                let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.surface_ctx.extent.width, height: self.surface_ctx.extent.height } };
                                self.ctx.device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
                                self.ctx.device.cmd_bind_vertex_buffers(self.command_buffer, 0, &[self.vertex_buffer], &[0]);
                            }

                            vertices = Self::quad_vertices(*pos, *size, ColorF::white());
                            has_draw = true;
                            pc.rect = [pos.x, pos.y, size.x, size.y];
                            pc.radii = *radii;
                            pc.mode = 4; // Blur
                            pc.border_width = (*sigma / 2.0).clamp(0.0, 5.0); // Abuse border_width for LOD
                            pc.is_squircle = if *is_squircle { 1 } else { 0 };
                         }
                        DrawCommand::Plot { .. } => {
                            pc.mode = 7; // Plot
                        }
                        DrawCommand::Heatmap {
                            pos,
                            size,
                            min,
                            max,
                            ..
                        } => {
                            vertices = Self::quad_vertices(*pos, *size, ColorF::white());
                            has_draw = true;
                            pc.elevation = *min;
                            pc.glow_strength = *max;
                            pc.mode = 8; // Heatmap
                        }
                        DrawCommand::Aurora { pos, size } => {
                            vertices = Self::quad_vertices(*pos, *size, ColorF::white());
                            has_draw = true;
                            pc.mode = 9; // Aurora
                        }
                        _ => {}
                    }

                    if has_draw {
                        // Copy vertices
                        unsafe {
                            let dest = vb_ptr.add(vertex_offset);
                            std::ptr::copy_nonoverlapping(vertices.as_ptr(), dest, 6);
                        }

                        // Push Constants
                        let constants_bytes = std::slice::from_raw_parts(
                            &pc as *const _ as *const u8,
                            std::mem::size_of::<PushConstants>(),
                        );

                        self.ctx.device.cmd_push_constants(
                            self.command_buffer,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            constants_bytes,
                        );

                        // Draw
                        self.ctx.device
                            .cmd_draw(self.command_buffer, 6, 1, vertex_offset as u32, 0);

                        vertex_offset += 6;
                    }
                }
                */
            }

            self.ctx.device.cmd_end_render_pass(self.command_buffer);

            // --- K4: Cinematic Resolver (Composition) ---
            unsafe {
                // 1. Prepare for K4 Dispatch
                // Transition Backdrop: COLOR_ATTACHMENT -> SHADER_READ_ONLY (For K4 Read)
                let back_read_barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .image(self.backdrop_image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });

                // Transition Swapchain: UNDEFINED (or anything) -> GENERAL (For K4 Write)
                // Since we didn't render to swapchain yet, it's whatever acquisition gave us (usually UNDEFINED or PRESENT_SRC).
                // We use UNDEFINED as old_layout to discard contents (we overwrite anyway).
                let swap_write_barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .image(self.surface_ctx.swapchain_images[image_index as usize])
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });

                self.ctx.device.cmd_pipeline_barrier(self.command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[back_read_barrier, swap_write_barrier]);

                // 2. Update K4 Output Descriptor (Binding 5) to point to valid Target
                let out_info = vk::DescriptorImageInfo::default()
                    .image_view(self.surface_ctx.swapchain_image_views[image_index as usize])
                    .image_layout(vk::ImageLayout::GENERAL);
                
                let write_desc = vk::WriteDescriptorSet::default()
                    .dst_set(self.k4_descriptor_set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&out_info));
                self.ctx.device.update_descriptor_sets(&[write_desc], &[]);

                // --- K12: Audio Aggregation ---
                self.ctx.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k12_pipeline);
                self.ctx.device.cmd_bind_descriptor_sets(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k12_layout, 0, &[self.k12_descriptor_set], &[]);
                self.ctx.device.cmd_dispatch(self.command_buffer, 1, 1, 1);

                // Barrier: K12 (Write Param) -> K4 (Read Param)
                let audio_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(self.audio_params_buffer)
                    .size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier(self.command_buffer, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[audio_barrier], &[]);

                // 3. Dispatch K4
                self.ctx.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k4_pipeline);
                self.ctx.device.cmd_bind_descriptor_sets(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.k4_layout, 0, &[self.k4_descriptor_set], &[]);
                
                let k4_pc = K4PushConstants { exposure: self.exposure, gamma: self.gamma, fog_density: self.fog_density, _pad: 0.0 };
                let k4_pc_bytes = std::slice::from_raw_parts(&k4_pc as *const _ as *const u8, std::mem::size_of::<K4PushConstants>());
                self.ctx.device.cmd_push_constants(self.command_buffer, self.k4_layout, vk::ShaderStageFlags::COMPUTE, 0, k4_pc_bytes);

                let k4_group_x = (self.surface_ctx.extent.width + 7) / 8;
                let k4_group_y = (self.surface_ctx.extent.height + 7) / 8;
                self.ctx.device.cmd_dispatch(self.command_buffer, k4_group_x, k4_group_y, 1);

                // 4. Barrier: Swapchain GENERAL -> PRESENT_SRC
                let final_barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ) // Scanout read?
                    .image(self.surface_ctx.swapchain_images[image_index as usize])
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
                
                // Also transition Backdrop back to SHADER_READ_ONLY (It is already, but for correctness if we loop? Actually we loop at Framebuffer clear)
                // Framebuffer uses LoadOp::CLEAR which doesn't care about previous content if we treat it as undefined.
                // But for safety, keep it compatible or transition.
                // The next pass will Transition UNDEFINED -> COLOR_ATTACHMENT via render pass implicit subpass dependency. 
                // So we don't need to transition backdrop back manually.

                self.ctx.device.cmd_pipeline_barrier(self.command_buffer, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &[final_barrier]);
            }

            // Sanity Readback Copy
            let copy_region = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(1024);
            self.ctx.device.cmd_copy_buffer(self.command_buffer, self.counter_buffer, self.counter_readback_buffer, &[copy_region]);

            // K11: Mark End
            self.write_timestamp(self.command_buffer, 1); // Frame End

            self.ctx.device.end_command_buffer(self.command_buffer).unwrap();

            println!("Vulkan: Command buffer ended, submitting...");

            // Submit
            let wait_semaphores = [self.image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::COMPUTE_SHADER];
            let signal_semaphores = [self.render_finished_semaphore];
            let command_buffers = [self.command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.ctx.device
                .queue_submit(self.ctx.graphics_queue, &[submit_info], self.in_flight_fence)
                .unwrap();

            println!("Vulkan: Frame submitted");

            // Present
            let swapchains = [self.surface_ctx.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            self.surface_ctx.swapchain_loader
                .queue_present(self.ctx.graphics_queue, &present_info)
                .unwrap();

            self.last_image_index = image_index;

            // Increment frame index for double-buffered query pools (K11)
            self.frame_index = self.frame_index.wrapping_add(1);
        }
    }
    
    fn capture_screenshot(&mut self, path: &str) {
        unsafe {
            // Wait for GPU to finish rendering
            self.ctx.device.queue_wait_idle(self.ctx.graphics_queue).unwrap();
            
            let width = self.surface_ctx.extent.width;
            let height = self.surface_ctx.extent.height;
            let image = self.surface_ctx.swapchain_images[self.last_image_index as usize];
            
            // Create staging buffer
            let size = (width * height * 4) as vk::DeviceSize;
            let (staging_buffer, staging_memory) = resources::create_buffer(
                &self.ctx.device,
                &self.ctx.instance,
                self.ctx.physical_device,
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).unwrap();
            
            // Transition image to TRANSFER_SRC_OPTIMAL
            self.ctx.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.ctx.device.begin_command_buffer(self.command_buffer, &begin_info).unwrap();
            
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::MEMORY_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
                
            self.ctx.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier]
            );
            
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(width)
                .buffer_image_height(height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width, height, depth: 1 });
                
            self.ctx.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buffer,
                &[region]
            );
            
            // Transition back to PRESENT_SRC_KHR
            let barrier = barrier
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ);
                
             self.ctx.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier]
            );
            
            self.ctx.device.end_command_buffer(self.command_buffer).unwrap();
            
            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            self.ctx.device.queue_submit(self.ctx.graphics_queue, &[submit_info], vk::Fence::null()).unwrap();
            self.ctx.device.queue_wait_idle(self.ctx.graphics_queue).unwrap();
            
            // Copy buffer to image crate and save
            let ptr = self.ctx.device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            let slice = std::slice::from_raw_parts(ptr as *const u8, size as usize);
            
            // Swap BGRA to RGBA if needed
            let mut rgba = vec![0u8; size as usize];
            for i in (0..size as usize).step_by(4) {
                rgba[i] = slice[i + 2];     // R
                rgba[i + 1] = slice[i + 1]; // G
                rgba[i + 2] = slice[i];     // B
                rgba[i + 3] = slice[i + 3]; // A
            }
            
            image::save_buffer(
                path,
                &rgba,
                width,
                height,
                image::ColorType::Rgba8,
            ).unwrap();
            
            self.ctx.device.unmap_memory(staging_memory);
            self.ctx.device.destroy_buffer(staging_buffer, None);
            self.ctx.device.free_memory(staging_memory, None);
            
            log::info!("📸 Screenshot saved to: {}", path);
        }
    }

    fn get_profiling_results(&self) -> Option<(Vec<u64>, f32)> {
        Some(self.get_vulkan_stats())
    }

    fn update_audio_data(&mut self, spectrum: &[f32]) {
        self.update_audio_data(spectrum);
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            self.surface_ctx.destroy(&self.ctx.device);
            self.ctx.device.destroy_render_pass(self.hdr_render_pass, None);
            self.ctx.device.destroy_framebuffer(self.hdr_framebuffer, None);
            self.ctx.device.destroy_render_pass(self.render_pass, None);
            self.ctx.device.destroy_render_pass(self.render_pass_load, None);

            // Destroy other internal resources
            self.ctx.device.destroy_pipeline_layout(self.pipeline_layout, None);
            
            // K12 Cleanup
            self.ctx.device.destroy_pipeline(self.k12_pipeline, None);
            self.ctx.device.destroy_pipeline_layout(self.k12_layout, None);
            self.ctx.device.destroy_buffer(self.spectrum_buffer, None);
            self.ctx.device.free_memory(self.spectrum_memory, None);
            self.ctx.device.destroy_buffer(self.audio_params_buffer, None);
            self.ctx.device.free_memory(self.audio_params_memory, None);
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.ctx.device.destroy_descriptor_set_layout(self.k4_set_layout, None);
            self.ctx.device.destroy_descriptor_set_layout(self.k12_set_layout, None);

            self.ctx.device.destroy_buffer(self.vertex_buffer, None);
            self.ctx.device.free_memory(self.vertex_memory, None);
            self.ctx.device.destroy_buffer(self.uniform_buffer, None);
            self.ctx.device.free_memory(self.uniform_memory, None);
            
            // Note: self.ctx will handle device/instance destruction via its own Drop
        }
    }
}
