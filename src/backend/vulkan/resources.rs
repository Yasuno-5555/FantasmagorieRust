use ash::vk;

pub fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    unsafe {
        let mem_properties = instance.get_physical_device_memory_properties(physical_device);
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize].property_flags & properties)
                    == properties
            {
                return Ok(i);
            }
        }
        Err("Failed to find suitable memory type".to_string())
    }
}

pub fn create_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    unsafe {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create buffer: {:?}", e))?;

        let mem_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type = find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let buffer_memory = device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("Failed to allocate buffer memory: {:?}", e))?;

        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .map_err(|e| format!("Failed to bind buffer memory: {:?}", e))?;

        Ok((buffer, buffer_memory))
    }
}

pub fn create_texture(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    mip_levels: u32,
    _data: Option<&[u8]>,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect_mask: vk::ImageAspectFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), String> {
    unsafe {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create image: {:?}", e))?;

        let mem_requirements = device.get_image_memory_requirements(image);
        let memory_type = find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);

        let image_memory = device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("Failed to allocate image memory: {:?}", e))?;

        device
            .bind_image_memory(image, image_memory, 0)
            .map_err(|e| format!("Failed to bind image memory: {:?}", e))?;
        
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });
            
        let bind = device.create_image_view(&view_info, None).map_err(|e| format!("Failed to create image view: {:?}", e))?;

        Ok((image, image_memory, bind))
    }
}
