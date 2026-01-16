//! Vulkan backend - Native Vulkan rendering via ash
//!
//! Full implementation using ash crate for Vulkan 1.2+ support.
//! Targets: Windows (Vulkan), Linux (Vulkan), Android (Vulkan)

use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use ash::vk;
use std::ffi::CStr;


/// Vulkan-based rendering backend
pub struct VulkanBackend {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    
    // Swapchain
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    
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
    
    width: u32,
    height: u32,
}

/// Vertex format for Vulkan
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

/// Push constants for per-draw data
/// Push constants for per-draw data (Matches GLSL)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PushConstants {
    rect: [f32; 4],
    radii: [f32; 4],
    border_color: [f32; 4],
    glow_color: [f32; 4],
    mode: i32,
    border_width: f32,
    elevation: f32,
    is_squircle: i32,
    glow_strength: f32,
    _padding: [f32; 1],
    _pad2: [f32; 2],
}

/// Uniform buffer for projection matrix
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    projection: [[f32; 4]; 4],
}

impl VulkanBackend {
    /// Create a new Vulkan backend
    /// 
    /// # Safety
    /// Requires valid window handle for surface creation
    #[cfg(target_os = "windows")]
    pub unsafe fn new(hwnd: *mut std::ffi::c_void, hinstance: *mut std::ffi::c_void, width: u32, height: u32) -> Result<Self, String> {
        // Load Vulkan
        let entry = ash::Entry::load()
            .map_err(|e| format!("Failed to load Vulkan: {:?}", e))?;

        // Create instance
        let app_name = c"Fantasmagorie";
        let engine_name = c"Fantasmagorie Engine";
        
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let extension_names = [
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        let instance = entry.create_instance(&create_info, None)
            .map_err(|e| format!("Failed to create Vulkan instance: {:?}", e))?;

        println!("✅ Vulkan instance created");

        // Create Win32 surface
        let win32_surface_loader = ash::khr::win32_surface::Instance::new(&entry, &instance);
        let surface_create_info = vk::Win32SurfaceCreateInfoKHR::default()
            .hinstance(hinstance as vk::HINSTANCE)
            .hwnd(hwnd as vk::HWND);
        
        let surface = win32_surface_loader.create_win32_surface(&surface_create_info, None)
            .map_err(|e| format!("Failed to create Win32 surface: {:?}", e))?;

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // Select physical device
        let physical_devices = instance.enumerate_physical_devices()
            .map_err(|e| format!("Failed to enumerate physical devices: {:?}", e))?;

        let physical_device = physical_devices.into_iter()
            .find(|&pd| {
                let props = instance.get_physical_device_properties(pd);
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU ||
                props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            })
            .ok_or("No suitable GPU found")?;

        let props = instance.get_physical_device_properties(physical_device);
        let device_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy();
        println!("🎮 Vulkan device: {}", device_name);

        // Find graphics queue family
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let graphics_family = queue_families.iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("No graphics queue family found")? as u32;

        // Create logical device
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family)
            .queue_priorities(&queue_priorities);

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extensions);

        let device = instance.create_device(physical_device, &device_create_info, None)
            .map_err(|e| format!("Failed to create logical device: {:?}", e))?;

        let graphics_queue = device.get_device_queue(graphics_family, 0);

        // Create swapchain
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        
        let surface_caps = surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
            .map_err(|e| format!("Failed to get surface capabilities: {:?}", e))?;

        let surface_format = vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(vk::Extent2D { width, height })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None)
            .map_err(|e| format!("Failed to create swapchain: {:?}", e))?;

        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain)
            .map_err(|e| format!("Failed to get swapchain images: {:?}", e))?;

        // Create image views
        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images.iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                device.create_image_view(&create_info, None).unwrap()
            })
            .collect();

        // Create render pass
        let attachment = vk::AttachmentDescription::default()
            .format(surface_format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
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

        let render_pass = device.create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create render pass: {:?}", e))?;

        // Create framebuffers
        let framebuffers: Vec<vk::Framebuffer> = swapchain_image_views.iter()
            .map(|&view| {
                let attachments = [view];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1);
                device.create_framebuffer(&create_info, None).unwrap()
            })
            .collect();

        // Create descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| format!("Failed to create descriptor set layout: {:?}", e))?;

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))?;

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = device.create_command_pool(&pool_info, None)
            .map_err(|e| format!("Failed to create command pool: {:?}", e))?;

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = device.allocate_command_buffers(&alloc_info)
            .map_err(|e| format!("Failed to allocate command buffers: {:?}", e))?;
        let command_buffer = command_buffers[0];

        // Create sync objects
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let image_available_semaphore = device.create_semaphore(&semaphore_info, None)
            .map_err(|e| format!("Failed to create semaphore: {:?}", e))?;
        let render_finished_semaphore = device.create_semaphore(&semaphore_info, None)
            .map_err(|e| format!("Failed to create semaphore: {:?}", e))?;
        let in_flight_fence = device.create_fence(&fence_info, None)
            .map_err(|e| format!("Failed to create fence: {:?}", e))?;

        // Create vertex buffer (dynamic, host visible)
        let buffer_size = 65536 * std::mem::size_of::<Vertex>() as vk::DeviceSize;
        let (vertex_buffer, vertex_memory) = Self::create_buffer(
            &device,
            &instance,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Create uniform buffer
        let uniform_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let (uniform_buffer, uniform_memory) = Self::create_buffer(
            &device,
            &instance,
            physical_device,
            uniform_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Create font texture (placeholder 1x1 white)
        let (font_texture, font_texture_memory, font_texture_view) = Self::create_texture(
            &device,
            &instance,
            physical_device,
            1, 1,
            &[255u8],
        )?;

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

        let sampler = device.create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create sampler: {:?}", e))?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let descriptor_pool = device.create_descriptor_pool(&pool_info, None)
            .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?;

        // Allocate descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("Failed to allocate descriptor sets: {:?}", e))?;
        let descriptor_set = descriptor_sets[0];

        // Update descriptor set
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer)
            .offset(0)
            .range(uniform_size);

        let image_info = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(font_texture_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info)),
        ];

        device.update_descriptor_sets(&writes, &[]);

        // Helper to compile GLSL to SPIR-V using Naga
        fn compile_glsl(src: &str, stage: naga::ShaderStage) -> Result<Vec<u32>, String> {
             let mut parser = naga::front::glsl::Frontend::default();
             let options = naga::front::glsl::Options {
                 stage,
                 defines: Default::default(),
             };
             
             let module = parser.parse(&options, src)
                 .map_err(|e| format!("GLSL Parse Error: {:?}", e))?;
                 
             let info = naga::valid::Validator::new(
                 naga::valid::ValidationFlags::all(),
                 naga::valid::Capabilities::all(),
             ).validate(&module)
                 .map_err(|e| format!("Validation Error: {:?}", e))?;
                 
             let mut options = naga::back::spv::Options::default();
             options.lang_version = (1, 0); // Vulkan 1.0 compat
             
             let binary = naga::back::spv::write_vec(
                 &module, 
                 &info, 
                 &options, 
                 None
             ).map_err(|e| format!("SPIR-V Write Error: {:?}", e))?;
             
             Ok(binary)
        }

        let shader_src = include_str!("vulkan_shader.glsl");
        let parts: Vec<&str> = shader_src.split("//---FRAGMENT---").collect();
        let vert_src = parts[0];
        let frag_src = parts[1];

        // Compile Vertex Shader
        let vert_binary = compile_glsl(vert_src, naga::ShaderStage::Vertex)?;
        
        // Compile Fragment Shader
        let frag_binary = compile_glsl(frag_src, naga::ShaderStage::Fragment)?;

        // Create Modules
        // Note: ash expects &[u32] for code, but allows u8 alignment via helper if needed.
        // Direct u32 slice is preferred.
        let vert_code = vk::ShaderModuleCreateInfo::default().code(&vert_binary);
        let frag_code = vk::ShaderModuleCreateInfo::default().code(&frag_binary);

        let vert_module = device.create_shader_module(&vert_code, None).unwrap();
        let frag_module = device.create_shader_module(&frag_code, None).unwrap();

        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") });
        
        let frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") });

        let shader_stages = [vert_stage, frag_stage];

        // Pipeline States
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }))
            .vertex_attribute_descriptions(&[
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 8,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 16,
                }
            ]);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| format!("Failed to create graphics pipeline: {:?}", e))?;
        let pipeline = pipelines[0];

        // Clean up shader modules
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        println!("   ✅ Vulkan backend initialized");

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            command_pool,
            command_buffer,
            render_pass,
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            framebuffers,
            vertex_buffer,
            vertex_memory,
            uniform_buffer,
            uniform_memory,
            font_texture,
            font_texture_memory,
            font_texture_view,
            sampler,
            descriptor_set,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            width,
            height,
        })
    }

    #[cfg(not(target_os = "windows"))]
    pub unsafe fn new(_hwnd: *mut std::ffi::c_void, _hinstance: *mut std::ffi::c_void, _width: u32, _height: u32) -> Result<Self, String> {
        Err("Vulkan backend Windows surface not available on this platform".to_string())
    }

    unsafe fn create_buffer(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create buffer: {:?}", e))?;

        let mem_requirements = device.get_buffer_memory_requirements(buffer);
        let mem_properties = instance.get_physical_device_memory_properties(physical_device);

        let memory_type = Self::find_memory_type(
            mem_properties,
            mem_requirements.memory_type_bits,
            properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let memory = device.allocate_memory(&alloc_info, None)
            .map_err(|e| format!("Failed to allocate buffer memory: {:?}", e))?;

        device.bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("Failed to bind buffer memory: {:?}", e))?;

        Ok((buffer, memory))
    }

    unsafe fn create_texture(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), String> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::PREINITIALIZED);

        let image = device.create_image(&image_info, None)
            .map_err(|e| format!("Failed to create image: {:?}", e))?;

        let mem_requirements = device.get_image_memory_requirements(image);
        let mem_properties = instance.get_physical_device_memory_properties(physical_device);

        let memory_type = Self::find_memory_type(
            mem_properties,
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let memory = device.allocate_memory(&alloc_info, None)
            .map_err(|e| format!("Failed to allocate image memory: {:?}", e))?;

        device.bind_image_memory(image, memory, 0)
            .map_err(|e| format!("Failed to bind image memory: {:?}", e))?;

        // Upload texture data
        let ptr = device.map_memory(memory, 0, data.len() as vk::DeviceSize, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("Failed to map image memory: {:?}", e))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        device.unmap_memory(memory);

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8_UNORM)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = device.create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create image view: {:?}", e))?;

        Ok((image, memory, view))
    }

    fn find_memory_type(
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0 &&
               mem_properties.memory_types[i as usize].property_flags.contains(properties) {
                return Ok(i);
            }
        }
        Err("Failed to find suitable memory type".to_string())
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
            Vertex { pos: [x0, y0], uv: [u0, v0], color: c },
            Vertex { pos: [x0, y1], uv: [u0, v1], color: c },
            Vertex { pos: [x1, y1], uv: [u1, v1], color: c },
            Vertex { pos: [x0, y0], uv: [u0, v0], color: c },
            Vertex { pos: [x1, y1], uv: [u1, v1], color: c },
            Vertex { pos: [x1, y0], uv: [u1, v0], color: c },
        ]
    }
}

impl super::Backend for VulkanBackend {
    fn name(&self) -> &str {
        "Vulkan"
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        unsafe {
            // Wait for previous frame
            self.device.wait_for_fences(&[self.in_flight_fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.in_flight_fence]).unwrap();

            // Acquire swapchain image
            let (image_index, _) = self.swapchain_loader
                .acquire_next_image(self.swapchain, u64::MAX, self.image_available_semaphore, vk::Fence::null())
                .unwrap();

            // Update uniform buffer
            let ubo = UniformBufferObject {
                projection: Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0),
            };
            let ptr = self.device.map_memory(
                self.uniform_memory, 
                0, 
                std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
                vk::MemoryMapFlags::empty()
            ).unwrap();
            std::ptr::copy_nonoverlapping(&ubo as *const _ as *const u8, ptr as *mut u8, std::mem::size_of::<UniformBufferObject>());
            self.device.unmap_memory(self.uniform_memory);

            // Reset and begin command buffer
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
            
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.command_buffer, &begin_info).unwrap();

            // Begin render pass
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.08, 0.08, 0.1, 1.0],
                },
            }];

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(&clear_values);

            self.device.cmd_begin_render_pass(self.command_buffer, &render_pass_info, vk::SubpassContents::INLINE);

            // Bind pipeline and descriptors
            if self.pipeline != vk::Pipeline::null() {
                self.device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                self.device.cmd_bind_descriptor_sets(
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
                    width: self.width as f32,
                    height: self.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                self.device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                };
                self.device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);

            // Process draw commands
                 let mut vb_ptr: *mut Vertex = std::ptr::null_mut();
                 let vb_total_size = 65536 * std::mem::size_of::<Vertex>() as vk::DeviceSize;
                 
                 // Map buffer once
                 let raw_ptr = self.device.map_memory(
                     self.vertex_memory,
                     0,
                     vb_total_size,
                     vk::MemoryMapFlags::empty()
                 ).unwrap();
                 vb_ptr = raw_ptr as *mut Vertex;
                 
                 let mut vertex_offset = 0;
                 
                 self.device.cmd_bind_vertex_buffers(self.command_buffer, 0, &[self.vertex_buffer], &[0]);
 
                 for cmd in dl.commands() {
                     let mut vertices: [Vertex; 6] = [Vertex{pos:[0.0;2],uv:[0.0;2],color:[0.0;4]}; 6];
                     let mut has_draw = false;
                     
                     let mut pc = PushConstants {
                        rect: [0.0; 4],
                        radii: [0.0; 4],
                        border_color: [0.0; 4],
                        glow_color: [0.0; 4],
                        mode: 0,
                        border_width: 0.0,
                        elevation: 0.0,
                        is_squircle: 0,
                        glow_strength: 0.0,
                        _padding: [0.0; 1],
                        _pad2: [0.0; 2],
                     };
 
                     match cmd {
                         DrawCommand::RoundedRect { pos, size, radii, color, elevation, is_squircle, border_width, border_color, glow_strength, glow_color, .. } => {
                              vertices = Self::quad_vertices(*pos, *size, *color);
                              has_draw = true;
                              pc.rect = [pos.x, pos.y, size.x, size.y];
                              pc.radii = *radii;
                              pc.border_color = [border_color.r, border_color.g, border_color.b, border_color.a];
                              pc.glow_color = [glow_color.r, glow_color.g, glow_color.b, glow_color.a];
                              pc.mode = 2; // Shape
                              pc.border_width = *border_width;
                              pc.elevation = *elevation;
                              pc.is_squircle = if *is_squircle { 1 } else { 0 };
                              pc.glow_strength = *glow_strength;
                         }
                         DrawCommand::Text { pos, size, uv, color } => {
                              vertices = Self::quad_vertices_uv(*pos, *size, *uv, *color);
                              has_draw = true;
                              pc.mode = 1; // Text (SDF)
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
                             std::mem::size_of::<PushConstants>()
                         );
                         
                         self.device.cmd_push_constants(
                             self.command_buffer,
                             self.pipeline_layout,
                             vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                             0,
                             constants_bytes
                         );
 
                         // Draw
                         self.device.cmd_draw(self.command_buffer, 6, 1, vertex_offset as u32, 0);
                         
                         vertex_offset += 6;
                     }
                 }
                 
                 self.device.unmap_memory(self.vertex_memory);
            }

            self.device.cmd_end_render_pass(self.command_buffer);
            self.device.end_command_buffer(self.command_buffer).unwrap();

            // Submit
            let wait_semaphores = [self.image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished_semaphore];
            let command_buffers = [self.command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.device.queue_submit(self.graphics_queue, &[submit_info], self.in_flight_fence).unwrap();

            // Present
            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let _ = self.swapchain_loader.queue_present(self.graphics_queue, &present_info);
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_semaphore(self.image_available_semaphore, None);
            
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_image_view(self.font_texture_view, None);
            self.device.destroy_image(self.font_texture, None);
            self.device.free_memory(self.font_texture_memory, None);
            
            self.device.destroy_buffer(self.uniform_buffer, None);
            self.device.free_memory(self.uniform_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_memory, None);
            
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            
            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
