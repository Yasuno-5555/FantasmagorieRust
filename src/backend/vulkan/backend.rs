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
use super::managed::image::ManagedImage;
use super::systems::sdf::SdfSystem;
use super::systems::visibility::VisibilitySystem;
use super::systems::particles::ParticleSystem;
use super::systems::post_process::PostProcessSystem;
use super::systems::audio::AudioSystem;
use super::wrappers::{VulkanBuffer, VulkanTexture};

/// Vulkan-based rendering backend


pub struct VulkanBackend {
    pub ctx: Arc<VulkanContext>,
    pub surface_ctx: VulkanSurfaceContext,
    
    // Systems
    sdf: SdfSystem,
    visibility: VisibilitySystem,
    particles: ParticleSystem,
    post_process: PostProcessSystem,
    audio: AudioSystem,

    // Core Managed Resources
    font_texture: ManagedImage,
    backdrop_image: ManagedImage,
    sampler: vk::Sampler,

    // Core Sync and State
    command_buffer: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    render_pass_load: vk::RenderPass,
    
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,

    // Shared State
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,

    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    uniform_buffer: vk::Buffer,
    uniform_memory: vk::DeviceMemory,
    instance_buffer: vk::Buffer,
    instance_memory: vk::DeviceMemory,

    // Profiling/Counters
    counter_buffer: vk::Buffer,
    counter_memory: vk::DeviceMemory,
    counter_readback_buffer: vk::Buffer,
    counter_readback_memory: vk::DeviceMemory,

    last_image_index: std::sync::atomic::AtomicU32,
    frame_index: std::sync::atomic::AtomicUsize,
    start_time: std::time::Instant,

    query_pools: [vk::QueryPool; 2],
    timestamp_period: f32,

    pub exposure: f32,
    pub gamma: f32,
    pub audio_gain: f32,
    pub fog_density: f32,
    pub sdf_intensity: f32,
    pub sdf_decay: f32,
    pub sdf_radius: f32,
    pub audio_params: crate::backend::shaders::types::AudioParams,

    pending_cleanup_buffers: std::sync::Mutex<Vec<VulkanBuffer>>,
    pending_cleanup_sets: std::sync::Mutex<Vec<vk::DescriptorSet>>,
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
pub struct PushConstants {
    pub rect: [f32; 4],
    pub radii: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub offset: [f32; 2],
    pub scale: f32,
    pub border_width: f32,
    pub elevation: f32,
    pub glow_strength: f32,
    pub lut_intensity: f32,
    pub mode: i32,
    pub is_squircle: i32,
    pub time: f32,
    pub _pad: f32,
    pub _pad2: f32,
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
pub struct UniformBufferObject {
    pub projection: [[f32; 4]; 4],
    pub viewport: [f32; 2],
    pub time: f32,
    pub audio_gain: f32,
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
        let device = &ctx.device;
        let descriptor_pool = ctx.descriptor_pool;

        // 1. Core Render Passes
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::B8G8R8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let attachment_ref = vk::AttachmentReference::default().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let subpass = vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS).color_attachments(std::slice::from_ref(&attachment_ref));
        let render_pass_info = vk::RenderPassCreateInfo::default().attachments(std::slice::from_ref(&attachment)).subpasses(std::slice::from_ref(&subpass));
        let render_pass = device.create_render_pass(&render_pass_info, None).map_err(|e| format!("{:?}", e))?;

        let surface_ctx = VulkanSurfaceContext::new(&ctx, hinstance, hwnd, width, height, render_pass)?;

        let render_pass_load = {
            let attachment_load = attachment.load_op(vk::AttachmentLoadOp::LOAD).initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let info = vk::RenderPassCreateInfo::default().attachments(std::slice::from_ref(&attachment_load)).subpasses(std::slice::from_ref(&subpass));
            device.create_render_pass(&info, None).map_err(|e| format!("{:?}", e))?
        };

        // 2. Command Buffers and Sync
        let command_alloc_info = vk::CommandBufferAllocateInfo::default().command_pool(ctx.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&command_alloc_info).map_err(|e| format!("{:?}", e))?[0];
        
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available_semaphore = device.create_semaphore(&semaphore_info, None).map_err(|e| format!("{:?}", e))?;
        let render_finished_semaphore = device.create_semaphore(&semaphore_info, None).map_err(|e| format!("{:?}", e))?;
        let in_flight_fence = device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).map_err(|e| format!("{:?}", e))?;

        // 3. Main Descriptor Set Layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(4).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(5).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        ];
        let descriptor_set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("{:?}", e))?;
        let pipeline_layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_set_layout)).push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).size(std::mem::size_of::<PushConstants>() as u32)]), None).map_err(|e| format!("{:?}", e))?;

        // 4. Core Buffers
        let (vertex_buffer, vertex_memory) = resources::create_buffer(device, &ctx.instance, ctx.physical_device, 65536 * std::mem::size_of::<Vertex>() as vk::DeviceSize, vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let uniform_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let (uniform_buffer, uniform_memory) = resources::create_buffer(device, &ctx.instance, ctx.physical_device, uniform_size, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let (instance_buffer, instance_memory) = resources::create_buffer(device, &ctx.instance, ctx.physical_device, 1024 * 1024, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        // 4b. Counter Buffers
        let (counter_buffer, counter_memory) = resources::create_buffer(device, &ctx.instance, ctx.physical_device, 1024, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let (counter_readback_buffer, counter_readback_memory) = resources::create_buffer(device, &ctx.instance, ctx.physical_device, 1024, vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;

        // 5. Core Textures (Managed)
        let font_texture = ManagedImage::create(ctx.clone(), 1024, 1024, vk::Format::R8_UNORM, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED, vk::ImageAspectFlags::COLOR)?;
        let backdrop_image = ManagedImage::create(ctx.clone(), width, height, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE, vk::ImageAspectFlags::COLOR)?;
        let sampler = device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR).address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE), None).map_err(|e| format!("{:?}", e))?;

        // 6. Initialize Systems
        let audio = AudioSystem::new(ctx.clone(), descriptor_pool, uniform_buffer, uniform_size)?;
        let particles = ParticleSystem::new(ctx.clone(), descriptor_pool, uniform_buffer, uniform_size, counter_buffer)?;
        let visibility = VisibilitySystem::new(ctx.clone(), descriptor_pool, uniform_buffer, uniform_size, backdrop_image.view)?;
        let sdf = SdfSystem::new(ctx.clone(), width, height, descriptor_pool, uniform_buffer, uniform_size, descriptor_set_layout)?;
        let post_process = PostProcessSystem::new(ctx.clone(), width, height, descriptor_pool, uniform_buffer, uniform_size, backdrop_image.view, sdf.sdf_image.view, font_texture.view, sampler, audio.audio_params_buffer.buffer)?;

        // 7. Main Pipeline
        let vertex_source = crate::backend::shaders::sources::VERTEX;
        let fragment_source = crate::backend::shaders::sources::FRAGMENT;
        let pipeline = pipelines::create_render_pipeline(device, render_pass, pipeline_layout, vertex_source, fragment_source, false)?;

        // 8. Descriptor Updates
        let descriptor_set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&descriptor_set_layout))).map_err(|e| format!("{:?}", e))?[0];
        
        // 9. Profiling
        let qp_info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(128);
        let query_pools = [device.create_query_pool(&qp_info, None).map_err(|e| format!("{:?}", e))?, device.create_query_pool(&qp_info, None).map_err(|e| format!("{:?}", e))?];
        let timestamp_period = ctx.physical_device_properties.limits.timestamp_period;

        Ok(Self {
            ctx, surface_ctx, sdf, visibility, particles, post_process, audio,
            font_texture, backdrop_image, sampler,
            command_buffer, render_pass, render_pass_load,
            image_available_semaphore, render_finished_semaphore, in_flight_fence,
            pipeline_layout, pipeline, descriptor_set_layout, descriptor_set,
            vertex_buffer, vertex_memory, uniform_buffer, uniform_memory,
            instance_buffer, instance_memory,
            counter_buffer, counter_memory, counter_readback_buffer, counter_readback_memory,
            last_image_index: std::sync::atomic::AtomicU32::new(0), frame_index: std::sync::atomic::AtomicUsize::new(0),
            start_time: std::time::Instant::now(),
            query_pools, timestamp_period,
            exposure: 1.0, gamma: 2.2, audio_gain: 1.0, fog_density: 0.05,
            sdf_intensity: 1.0, sdf_decay: 0.9, sdf_radius: 10.0,
            audio_params: crate::backend::shaders::types::AudioParams::default(),
            pending_cleanup_buffers: std::sync::Mutex::new(Vec::new()),
            pending_cleanup_sets: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn update_audio_data(&mut self, spectrum: &[f32]) {
        self.audio.update_spectrum(spectrum);
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
            self.backdrop_image.image,
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
            self.backdrop_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        // 4. Generate Mipmaps (transitions to SHADER_READ_ONLY)
        self.cmd_generate_mipmaps(
            command_buffer,
            self.backdrop_image.image,
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
        let pool_idx = self.frame_index.load(std::sync::atomic::Ordering::Relaxed) % 2;
        self.ctx.device.cmd_reset_query_pool(cb, self.query_pools[pool_idx], 0, 64);
    }

    pub unsafe fn write_timestamp(&self, cb: vk::CommandBuffer, index: u32) {
        let pool_idx = self.frame_index.load(std::sync::atomic::Ordering::Relaxed) % 2;
        self.ctx.device.cmd_write_timestamp(
            cb,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.query_pools[pool_idx],
            index,
        );
    }

    pub fn get_vulkan_stats(&self) -> (Vec<u64>, f32) {
        // Return results from the PREVIOUS frame (N-1) to avoid stalling
        let frame_idx = self.frame_index.load(std::sync::atomic::Ordering::Relaxed);
        if frame_idx == 0 {
            return (vec![0u64; 64], self.timestamp_period);
        }

        let read_idx = (frame_idx + 1) % 2; 
        
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
                eprintln!("⚠️ GPU SANITY ALERT [Frame {}]: Errors={}, Flags=0x{:08X}, Visible={}", self.frame_index.load(std::sync::atomic::Ordering::Relaxed), error_count, sanity_flags, visible_count);
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
                .image(self.font_texture.image)
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
                self.font_texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition to SHADER_READ_ONLY
            let barrier = barrier
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(self.font_texture.image);

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
        let orchestrator = crate::renderer::orchestrator::RenderOrchestrator::new();
        let tasks = orchestrator.plan(dl);
        let time = self.start_time.elapsed().as_secs_f32();
        
        // Use self as the executor (casted to avoid borrow checker issues if necessary, but here mutable self is fine depending on trait signature)
        // GpuExecutor takes &self. execute takes &E.
        // So &*self calls GpuExecutor methods.
        // However, begin_execute needs mutability hack I added.
        if let Err(e) = orchestrator.execute(self, &tasks, time, width, height) {
            eprintln!("Vulkan Render Error: {}", e);
        }
        self.frame_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn capture_screenshot(&mut self, path: &str) {
        unsafe {
            // Wait for GPU to finish rendering
            self.ctx.device.queue_wait_idle(self.ctx.graphics_queue).unwrap();
            
            let width = self.surface_ctx.extent.width;
            let height = self.surface_ctx.extent.height;
            let image = self.surface_ctx.swapchain_images[self.last_image_index.load(std::sync::atomic::Ordering::Relaxed) as usize];
            
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

use crate::backend::hal::{GpuResourceProvider, GpuPipelineProvider, GpuExecutor, BufferUsage, TextureDescriptor, TextureUsage};

impl GpuResourceProvider for VulkanBackend {
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;
    type TextureView = vk::ImageView;
    type Sampler = vk::Sampler;

    fn create_buffer(&self, size: u64, usage: BufferUsage, label: &str) -> Result<Self::Buffer, String> {
        let vk_usage = match usage {
            BufferUsage::Vertex => vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsage::Index => vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsage::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferUsage::Storage => vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUsage::CopySrc => vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUsage::CopyDst => vk::BufferUsageFlags::TRANSFER_DST,
        };
        
        // Host visible for writing from CPU (Vertex/Index/Uniform), device local for others?
        let properties = if matches!(usage, BufferUsage::Uniform | BufferUsage::CopySrc | BufferUsage::Vertex | BufferUsage::Index) {
             vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
             vk::MemoryPropertyFlags::DEVICE_LOCAL
        };

        let (buffer, memory) = resources::create_buffer(
            &self.ctx.device, 
            &self.ctx.instance, 
            self.ctx.physical_device, 
            size, 
            vk_usage, 
            properties
        )?;

        Ok(VulkanBuffer { buffer, memory, size })
    }

    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Self::Texture, String> {
        let mut usage = vk::ImageUsageFlags::empty();
        if desc.usage.contains(TextureUsage::COPY_SRC) { usage |= vk::ImageUsageFlags::TRANSFER_SRC; }
        if desc.usage.contains(TextureUsage::COPY_DST) { usage |= vk::ImageUsageFlags::TRANSFER_DST; }
        if desc.usage.contains(TextureUsage::TEXTURE_BINDING) { usage |= vk::ImageUsageFlags::SAMPLED; }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) { usage |= vk::ImageUsageFlags::STORAGE; }
        if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) { 
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT; 
             if desc.format == crate::backend::hal::TextureFormat::Depth32Float {
                usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
             }
        }

        let format = match desc.format {
            crate::backend::hal::TextureFormat::R8Unorm => vk::Format::R8_UNORM,
            crate::backend::hal::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            crate::backend::hal::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            crate::backend::hal::TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        };

        let (image, memory, _) = resources::create_texture(
             &self.ctx.device,
             &self.ctx.instance,
             self.ctx.physical_device,
             desc.width,
             desc.height,
             1, // mip_levels
             None,
             format,
             usage,
             vk::ImageAspectFlags::COLOR, 
        )?;

        Ok(VulkanTexture { image, memory, width: desc.width, height: desc.height, mip_levels: 1, format })
    }

    fn create_texture_view(&self, texture: &Self::Texture) -> Result<Self::TextureView, String> {
        unsafe {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(texture.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(texture.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR, // TODO: Depth
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            
            self.ctx.device.create_image_view(&view_info, None).map_err(|e| e.to_string())
        }
    }

    fn create_sampler(&self, label: &str) -> Result<Self::Sampler, String> {
         unsafe {
             let info = vk::SamplerCreateInfo::default()
                 .mag_filter(vk::Filter::LINEAR)
                 .min_filter(vk::Filter::LINEAR)
                 .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                 .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                 .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
             
             self.ctx.device.create_sampler(&info, None).map_err(|e| e.to_string())
         }
    }
    
    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]) {
        unsafe {
            if let Ok(ptr) = self.ctx.device.map_memory(buffer.memory, offset, data.len() as u64, vk::MemoryMapFlags::empty()) {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, data.len());
                slice.copy_from_slice(data);
                self.ctx.device.unmap_memory(buffer.memory);
            } else {
                eprintln!("Failed to map memory for write_buffer");
            }
        }
    }

    fn write_texture(&self, texture: &Self::Texture, data: &[u8], width: u32, height: u32) {
         eprintln!("write_texture not fully implemented for VulkanBackend in HAL");
    }

    fn destroy_buffer(&self, buffer: Self::Buffer) {
        self.pending_cleanup_buffers.lock().unwrap().push(buffer);
    }

    fn destroy_texture(&self, texture: Self::Texture) {
        // Textures are usually not per-frame in Orchestrator for now, 
        // but if they were, we'd defer them too.
        // For currently known use cases, immediate destruction is only for shared persistent ones in Drop.
        // But to be safe, let's just make it immediate for now since Orchestrator doesn't create temporary textures yet.
        unsafe {
            self.ctx.device.destroy_image(texture.image, None);
            self.ctx.device.free_memory(texture.memory, None);
        }
    }
}

impl GpuPipelineProvider for VulkanBackend {
    type RenderPipeline = vk::Pipeline;
    type ComputePipeline = vk::Pipeline;
    type BindGroupLayout = vk::DescriptorSetLayout;
    type BindGroup = vk::DescriptorSet;

    fn create_render_pipeline(&self, label: &str, wgsl_source: &str, layout: Option<&Self::BindGroupLayout>) -> Result<Self::RenderPipeline, String> {
        Err("Dynamic pipeline creation from WGSL not yet implemented in Vulkan HAL adaptation".into())
    }

    fn create_compute_pipeline(&self, label: &str, wgsl_source: &str, layout: Option<&Self::BindGroupLayout>) -> Result<Self::ComputePipeline, String> {
         Err("Dynamic compute pipeline creation from WGSL not yet implemented".into())
    }

    fn destroy_bind_group(&self, bind_group: Self::BindGroup) {
        self.pending_cleanup_sets.lock().unwrap().push(bind_group);
    }
}



impl GpuExecutor for VulkanBackend {
    fn begin_execute(&self) -> Result<(), String> {
        let device = &self.ctx.device;
        unsafe {
            // 1. Wait for previous frame to finish before we touch its resources
            device.wait_for_fences(&[self.in_flight_fence], true, std::u64::MAX).map_err(|e| format!("{:?}", e))?;
            
            // 2. Perform deferred cleanup (safe now that GPU is idle)
            {
                let mut buffers = self.pending_cleanup_buffers.lock().unwrap();
                for buf in buffers.drain(..) {
                    device.destroy_buffer(buf.buffer, None);
                    device.free_memory(buf.memory, None);
                }
                let mut sets = self.pending_cleanup_sets.lock().unwrap();
                if !sets.is_empty() {
                    let _ = device.free_descriptor_sets(self.ctx.descriptor_pool, &sets);
                    sets.clear();
                }
            }

            // 3. Reset fence and acquire next image
            device.reset_fences(&[self.in_flight_fence]).unwrap();
            
            let (image_index, _is_suboptimal) = self.surface_ctx.swapchain_loader.acquire_next_image(
                self.surface_ctx.swapchain,
                std::u64::MAX,
                self.image_available_semaphore,
                vk::Fence::null(),
            ).map_err(|e| format!("Failed to acquire next image: {:?}", e))?;
            
            // 4. Update last_image_index
            self.last_image_index.store(image_index, std::sync::atomic::Ordering::Relaxed);

            // 5. Begin Command Buffer
            device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(self.command_buffer, &begin_info).unwrap();
        }
        Ok(())
    }

    fn end_execute(&self) -> Result<(), String> {
        unsafe {
            self.ctx.device.end_command_buffer(self.command_buffer)
                .map_err(|e| format!("Failed to end command buffer: {:?}", e))?;

            // Submit
            let wait_semaphores = [self.image_available_semaphore];
            let signal_semaphores = [self.render_finished_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffer];
            
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            
            self.ctx.device.reset_fences(&[self.in_flight_fence]).unwrap();
            
            self.ctx.device.queue_submit(
                self.ctx.graphics_queue,
                &[submit_info],
                self.in_flight_fence,
            ).map_err(|e| format!("Failed to submit queue: {:?}", e))?;

            // Wait for fence (Synchronous execution for simplicity in verification)
            self.ctx.device.wait_for_fences(&[self.in_flight_fence], true, std::u64::MAX).unwrap();
        }
        Ok(())
    }

    fn draw(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        vertex_count: u32,
        uniform_data: &[u8],
    ) -> Result<(), String> {
        unsafe {
            let image_index = self.last_image_index.load(std::sync::atomic::Ordering::Relaxed);
            let clear_values = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }];
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.surface_ctx.framebuffers[image_index as usize])
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: self.surface_ctx.extent })
                .clear_values(&clear_values);

            self.ctx.device.cmd_begin_render_pass(self.command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
            
            // Set dynamic viewport and scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.surface_ctx.extent.width as f32,
                height: self.surface_ctx.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.ctx.device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
            
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.surface_ctx.extent,
            };
            self.ctx.device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);

            self.ctx.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline);
            self.ctx.device.cmd_bind_vertex_buffers(self.command_buffer, 0, &[vertex_buffer.buffer], &[0]);
            
            if let Some(bg) = bind_group {
                self.ctx.device.cmd_bind_descriptor_sets(self.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[*bg], &[]);
            }
            
            if !uniform_data.is_empty() {
                self.ctx.device.cmd_push_constants(self.command_buffer, self.pipeline_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, uniform_data);
            }

            self.ctx.device.cmd_draw(self.command_buffer, vertex_count, 1, 0, 0);
            self.ctx.device.cmd_end_render_pass(self.command_buffer);
        }
        Ok(())
    }

    fn dispatch(&self, pipeline: &Self::ComputePipeline, _layout: Option<&Self::BindGroupLayout>, groups: [u32; 3], _pc: &[u8]) -> Result<(), String> {
        unsafe {
            self.ctx.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, *pipeline);
            self.ctx.device.cmd_dispatch(self.command_buffer, groups[0], groups[1], groups[2]);
        }
        Ok(())
    }

    fn copy_texture(&self, src: &Self::Texture, dst: &Self::Texture) -> Result<(), String> {
        unsafe {
            let copy_region = vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width: src.width,
                    height: src.height,
                    depth: 1,
                });

            self.cmd_transition_image_layout(self.command_buffer, src.image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src.mip_levels);
            self.cmd_transition_image_layout(self.command_buffer, dst.image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL, dst.mip_levels);

            self.ctx.device.cmd_copy_image(
                self.command_buffer,
                src.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );

            self.cmd_transition_image_layout(self.command_buffer, src.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, src.mip_levels);
            self.cmd_transition_image_layout(self.command_buffer, dst.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, dst.mip_levels);
        }
        Ok(())
    }

    fn generate_mipmaps(&self, texture: &Self::Texture) -> Result<(), String> {
        unsafe {
            self.cmd_generate_mipmaps(self.command_buffer, texture.image, texture.width, texture.height, texture.mip_levels);
        }
        Ok(())
    }

    fn create_bind_group(&self, layout: &Self::BindGroupLayout, buffers: &[&Self::Buffer], textures: &[&Self::TextureView], samplers: &[&Self::Sampler]) -> Result<Self::BindGroup, String> {
         unsafe {
             let layouts = [layout.clone()];
             let alloc_info = vk::DescriptorSetAllocateInfo::default()
                 .descriptor_pool(self.ctx.descriptor_pool)
                 .set_layouts(&layouts);
             
             let set = self.ctx.device.allocate_descriptor_sets(&alloc_info)
                 .map_err(|e| format!("Failed to allocate descriptor set: {:?}", e))?[0];
             
             // 1. Prepare Descriptor Infos (on stack, indexed by binding)
             let mut b_infos = [vk::DescriptorBufferInfo::default(); 6];
             let mut i_infos = [vk::DescriptorImageInfo::default(); 6];
             let mut b_active = [false; 6];
             let mut i_active = [false; 6];

             // Binding 0: Uniform Buffer
             if let Some(buf) = buffers.first() {
                 b_infos[0] = vk::DescriptorBufferInfo::default().buffer(buf.buffer).range(vk::WHOLE_SIZE);
                 b_active[0] = true;
             }

             // Binding 1: Combined Image Sampler
             if let (Some(view), Some(samp)) = (textures.first(), samplers.first()) {
                 i_infos[1] = vk::DescriptorImageInfo::default()
                     .image_view(**view)
                     .sampler(**samp)
                     .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                 i_active[1] = true;
             }

             // Binding 2: Sampled Image (Placeholder)
             if let Some(view) = textures.first() {
                 i_infos[2] = vk::DescriptorImageInfo::default()
                     .image_view(**view)
                     .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                 i_active[2] = true;
             }

             // Binding 3: Sampled Image (Dummy)
             if let Some(view) = textures.first() {
                 i_infos[3] = vk::DescriptorImageInfo::default()
                     .image_view(**view)
                     .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                 i_active[3] = true;
             }

             // Binding 4: Backdrop
             let backdrop = textures.get(1).map(|&t| *t).unwrap_or(self.backdrop_image.view);
             i_infos[4] = vk::DescriptorImageInfo::default()
                 .image_view(backdrop)
                 .image_layout(vk::ImageLayout::GENERAL);
             i_active[4] = true;

             // Binding 5: Instance Buffer
             b_infos[5] = vk::DescriptorBufferInfo::default()
                 .buffer(self.instance_buffer)
                 .range(vk::WHOLE_SIZE);
             b_active[5] = true;

             // 2. Create WriteDescriptorSet structures from the prepared data
             let mut writes = Vec::new();
             
             if b_active[0] {
                 writes.push(vk::WriteDescriptorSet::default()
                     .dst_set(set)
                     .dst_binding(0)
                     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                     .buffer_info(std::slice::from_ref(&b_infos[0])));
             }

             if i_active[1] {
                 writes.push(vk::WriteDescriptorSet::default()
                     .dst_set(set)
                     .dst_binding(1)
                     .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                     .image_info(std::slice::from_ref(&i_infos[1])));
             }

             if i_active[2] {
                 writes.push(vk::WriteDescriptorSet::default()
                     .dst_set(set)
                     .dst_binding(2)
                     .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                     .image_info(std::slice::from_ref(&i_infos[2])));
             }

             if i_active[3] {
                 writes.push(vk::WriteDescriptorSet::default()
                     .dst_set(set)
                     .dst_binding(3)
                     .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                     .image_info(std::slice::from_ref(&i_infos[3])));
             }

             if i_active[4] {
                 writes.push(vk::WriteDescriptorSet::default()
                     .dst_set(set)
                     .dst_binding(4)
                     .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                     .image_info(std::slice::from_ref(&i_infos[4])));
             }

             if b_active[5] {
                 writes.push(vk::WriteDescriptorSet::default()
                     .dst_set(set)
                     .dst_binding(5)
                     .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                     .buffer_info(std::slice::from_ref(&b_infos[5])));
             }

             self.ctx.device.update_descriptor_sets(&writes, &[]);
             
             Ok(set)
         }
    }

    fn get_font_view(&self) -> &Self::TextureView {
        &self.font_texture.view
    }

    fn get_backdrop_view(&self) -> &Self::TextureView {
        &self.backdrop_image.view
    }

    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout {
        &self.descriptor_set_layout
    }

    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline {
        &self.pipeline
    }

    fn get_default_sampler(&self) -> &Self::Sampler {
        &self.sampler
    }

    fn resolve(&mut self) -> Result<(), String> {
        let width = self.surface_ctx.extent.width;
        let height = self.surface_ctx.extent.height;
        
        // 1. Run Systems
        self.audio.execute(self.command_buffer);
        self.particles.execute(self.command_buffer, 1000, 10); // Stub particle counts
        self.visibility.execute(self.command_buffer, 1); // Stub fog
        self.sdf.execute(self.command_buffer, width, height, self.sdf_intensity, self.sdf_decay, self.sdf_radius, self.descriptor_set);
        
        let post_params = crate::backend::vulkan::systems::post_process::K4PushConstants {
            exposure: self.exposure,
            gamma: self.gamma,
            fog_density: self.fog_density,
            _pad: 0.0,
        };
        self.post_process.execute(self.command_buffer, width, height, post_params);
        
        Ok(())
    }

    fn present(&self) -> Result<(), String> {
        unsafe {
             let wait_semaphores = [self.render_finished_semaphore];
             let swapchains = [self.surface_ctx.swapchain];
             let image_indices = [self.last_image_index.load(std::sync::atomic::Ordering::Relaxed)];
 
             let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
 
            self.surface_ctx.swapchain_loader.queue_present(self.ctx.graphics_queue, &present_info)
                .map_err(|e| format!("Failed to present: {:?}", e))?;
        }
        Ok(())
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            device.device_wait_idle().ok();

            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.in_flight_fence, None);

            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_memory, None);
            device.destroy_buffer(self.uniform_buffer, None);
            device.free_memory(self.uniform_memory, None);
            device.destroy_buffer(self.instance_buffer, None);
            device.free_memory(self.instance_memory, None);
            device.destroy_buffer(self.counter_buffer, None);
            device.free_memory(self.counter_memory, None);
            device.destroy_buffer(self.counter_readback_buffer, None);
            device.free_memory(self.counter_readback_memory, None);

            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_sampler(self.sampler, None);

            for &pool in &self.query_pools {
                device.destroy_query_pool(pool, None);
            }

            device.destroy_render_pass(self.render_pass, None);
            device.destroy_render_pass(self.render_pass_load, None);

            // Managed images and systems will drop themselves
        }
    }
}

