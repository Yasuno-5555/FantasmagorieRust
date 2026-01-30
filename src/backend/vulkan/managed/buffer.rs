use ash::vk;
use std::sync::Arc;
use crate::backend::vulkan::VulkanContext;
use crate::backend::vulkan::resources;

/// A managed Vulkan Buffer that tracks its own access state
pub struct ManagedBuffer {
    pub ctx: Arc<VulkanContext>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    
    pub current_access: vk::AccessFlags,
    pub current_stage: vk::PipelineStageFlags,
}

impl ManagedBuffer {
    pub fn new(
        ctx: Arc<VulkanContext>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self {
            ctx,
            buffer,
            memory,
            size,
            usage,
            current_access: vk::AccessFlags::empty(),
            current_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
        }
    }

    pub fn create(
        ctx: Arc<VulkanContext>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self, String> {
        let (buffer, memory) = resources::create_buffer(
            &ctx.device,
            &ctx.instance,
            ctx.physical_device,
            size,
            usage,
            properties,
        )?;
        Ok(Self::new(ctx, buffer, memory, size, usage))
    }

    pub unsafe fn barrier(
        &mut self,
        cb: vk::CommandBuffer,
        new_access: vk::AccessFlags,
        new_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(self.current_access)
            .dst_access_mask(new_access)
            .buffer(self.buffer)
            .offset(0)
            .size(self.size);

        self.ctx.device.cmd_pipeline_barrier(
            cb,
            self.current_stage,
            new_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );

        self.current_access = new_access;
        self.current_stage = new_stage;
    }
}

impl Drop for ManagedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_buffer(self.buffer, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}
