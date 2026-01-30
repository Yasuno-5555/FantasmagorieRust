use ash::vk;
use std::sync::Arc;
use super::VulkanContext;

/// Transient surface-related resources that depend on window size/state
pub struct VulkanSurfaceContext {
    pub surface_loader: ash::khr::surface::Instance,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl VulkanSurfaceContext {
    pub unsafe fn new(
        ctx: &VulkanContext,
        hinstance: *mut std::ffi::c_void,
        hwnd: *mut std::ffi::c_void,
        width: u32,
        height: u32,
        render_pass: vk::RenderPass,
    ) -> Result<Self, String> {
        { use std::io::Write; let mut f = std::fs::OpenOptions::new().append(true).create(true).open("debug_log.txt").unwrap(); writeln!(f, "[VulkanSurfaceContext] new() entered!").unwrap(); f.sync_all().unwrap(); }
        eprintln!("[VulkanSurfaceContext] Creating loaders...");
        { use std::io::Write; let _ = std::io::stderr().flush(); }
        let surface_loader = ash::khr::surface::Instance::new(&ctx.entry, &ctx.instance);
        
        // Win32 Surface creation (Note: hwnd is *mut void)
        eprintln!("[VulkanSurfaceContext] Creating Win32 surface...");
        { use std::io::Write; let _ = std::io::stderr().flush(); }
        let create_info = vk::Win32SurfaceCreateInfoKHR::default()
             .hinstance(hinstance as isize)
             .hwnd(hwnd as isize);
        let win32_loader = ash::khr::win32_surface::Instance::new(&ctx.entry, &ctx.instance);
        let surface = win32_loader.create_win32_surface(&create_info, None)
            .map_err(|e| format!("Surface: {:?}", e))?;
        eprintln!("[VulkanSurfaceContext] Surface created");
        { use std::io::Write; let _ = std::io::stderr().flush(); }

        let swapchain_loader = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);

        // Capabilities & Formats
        eprintln!("[VulkanSurfaceContext] Querying surface capabilities...");
        { use std::io::Write; let _ = std::io::stderr().flush(); }
        let caps = surface_loader.get_physical_device_surface_capabilities(ctx.physical_device, surface).map_err(|e| format!("{:?}", e))?;
        let formats = surface_loader.get_physical_device_surface_formats(ctx.physical_device, surface).map_err(|e| format!("{:?}", e))?;
        let format = formats.iter().find(|f| f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .unwrap_or(&formats[0]);

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D { width, height }
        };

        eprintln!("[VulkanSurfaceContext] Creating swapchain ({}x{})...", extent.width, extent.height);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(caps.min_image_count + 1)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None).map_err(|e| format!("{:?}", e))?;
        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).map_err(|e| format!("{:?}", e))?;
        eprintln!("[VulkanSurfaceContext] Swapchain created ({} images)", swapchain_images.len());

        eprintln!("[VulkanSurfaceContext] Creating image views...");
        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images.iter().map(|&image| {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1));
            ctx.device.create_image_view(&create_info, None).expect("Image View creation failed")
        }).collect();
        eprintln!("[VulkanSurfaceContext] Image views created");

        // Framebuffers
        eprintln!("[VulkanSurfaceContext] Creating framebuffers...");
        let framebuffers: Vec<vk::Framebuffer> = swapchain_image_views.iter().map(|&view| {
            let attachments = [view];
            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            ctx.device.create_framebuffer(&create_info, None).expect("Framebuffer creation failed")
        }).collect();
        eprintln!("[VulkanSurfaceContext] Framebuffers created");

        Ok(Self {
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            framebuffers,
            format: format.format,
            extent,
        })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for &fb in &self.framebuffers {
            device.destroy_framebuffer(fb, None);
        }
        for &view in &self.swapchain_image_views {
            device.destroy_image_view(view, None);
        }
        self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        self.surface_loader.destroy_surface(self.surface, None);
    }
}
