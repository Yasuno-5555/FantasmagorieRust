use ash::vk;

#[derive(Debug, Clone, Copy)]
pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

#[derive(Debug, Clone, Copy)]
pub struct VulkanTexture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub format: vk::Format,
}
