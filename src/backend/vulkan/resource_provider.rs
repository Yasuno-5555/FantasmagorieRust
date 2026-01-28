use ash::vk;
use std::sync::Arc;
use crate::backend::vulkan::VulkanContext;

pub struct ResourceProvider {
    ctx: Arc<VulkanContext>,
}

impl ResourceProvider {
    pub fn new(ctx: Arc<VulkanContext>) -> Self {
        Self { ctx }
    }

    pub unsafe fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = self.ctx.device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create buffer: {:?}", e))?;

        let mem_requirements = self.ctx.device.get_buffer_memory_requirements(buffer);
        let mem_properties = self.ctx.instance.get_physical_device_memory_properties(self.ctx.physical_device);

        let memory_type = self.find_memory_type(
            mem_properties,
            mem_requirements.memory_type_bits,
            properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let memory = self.ctx.device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("Failed to allocate buffer memory: {:?}", e))?;

        self.ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("Failed to bind buffer memory: {:?}", e))?;

        Ok((buffer, memory))
    }

    pub unsafe fn create_texture(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), String> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = self.ctx.device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create image: {:?}", e))?;

        let mem_requirements = self.ctx.device.get_image_memory_requirements(image);
        let mem_properties = self.ctx.instance.get_physical_device_memory_properties(self.ctx.physical_device);

        let memory_type = self.find_memory_type(
            mem_properties,
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let memory = self.ctx.device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("Failed to allocate image memory: {:?}", e))?;

        self.ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| format!("Failed to bind image memory: {:?}", e))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = self.ctx.device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create image view: {:?}", e))?;

        Ok((image, memory, view))
    }

    pub fn find_memory_type(
        &self,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }
        Err("Failed to find suitable memory type".to_string())
    }

    pub unsafe fn map_memory_safe(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> Result<*mut std::ffi::c_void, String> {
        // In a more advanced implementation, we would check if the memory is actually HOST_VISIBLE here.
        // For now, this is a wrapper that we can expand for better diagnostics.
        self.ctx.device.map_memory(memory, offset, size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("Failed to map memory: {:?}", e))
    }
}
