use ash::vk;
use std::sync::Arc;
use crate::backend::vulkan::VulkanContext;
use crate::backend::vulkan::resources;

/// A managed Vulkan Image that tracks its own layout and access state
pub struct ManagedImage {
    pub ctx: Arc<VulkanContext>,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    
    // Internal state tracking for automatic transitions
    pub current_layout: vk::ImageLayout,
    pub current_access: vk::AccessFlags,
    pub current_stage: vk::PipelineStageFlags,
}

impl ManagedImage {
    pub fn new(
        ctx: Arc<VulkanContext>,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        format: vk::Format,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Self {
        Self {
            ctx,
            image,
            memory,
            view,
            format,
            extent,
            usage,
            aspect,
            current_layout: vk::ImageLayout::UNDEFINED,
            current_access: vk::AccessFlags::empty(),
            current_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
        }
    }

    pub fn create(
        ctx: Arc<VulkanContext>,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, String> {
        let device = &ctx.device;
        let extent = vk::Extent3D { width, height, depth: 1 };
        
        let (image, memory) = unsafe {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            
            let image = device.create_image(&image_info, None).map_err(|e| format!("{:?}", e))?;
            let mem_reqs = device.get_image_memory_requirements(image);
            let mem_type = resources::find_memory_type(
                &ctx.instance,
                ctx.physical_device,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).map_err(|_| "Failed to find memory type for image")?;
            
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type);
            
            let memory = device.allocate_memory(&alloc_info, None).map_err(|e| format!("{:?}", e))?;
            device.bind_image_memory(image, memory, 0).map_err(|e| format!("{:?}", e))?;
            
            (image, memory)
        };

        let view = unsafe {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.create_image_view(&view_info, None).map_err(|e| format!("{:?}", e))?
        };

        Ok(Self::new(ctx, image, memory, view, format, extent, usage, aspect))
    }

    /// Transition to a new layout and access state automatically
    pub unsafe fn transition(
        &mut self,
        cb: vk::CommandBuffer,
        new_layout: vk::ImageLayout,
        new_access: vk::AccessFlags,
        new_stage: vk::PipelineStageFlags,
    ) {
        if self.current_layout == new_layout && self.current_access == new_access {
            return; // Already in desired state
        }

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(self.current_layout)
            .new_layout(new_layout)
            .src_access_mask(self.current_access)
            .dst_access_mask(new_access)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        self.ctx.device.cmd_pipeline_barrier(
            cb,
            self.current_stage,
            new_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );

        self.current_layout = new_layout;
        self.current_access = new_access;
        self.current_stage = new_stage;
    }
}

impl Drop for ManagedImage {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_image_view(self.view, None);
            self.ctx.device.destroy_image(self.image, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}
