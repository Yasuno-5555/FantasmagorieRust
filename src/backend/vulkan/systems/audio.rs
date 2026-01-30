use ash::vk;
use std::sync::Arc;
use crate::backend::vulkan::VulkanContext;
use crate::backend::vulkan::managed::ManagedBuffer;
use crate::backend::vulkan::pipelines;
use crate::backend::vulkan::resources;

pub struct AudioSystem {
    pub ctx: Arc<VulkanContext>,
    
    // Buffers (Managed)
    pub spectrum_buffer: ManagedBuffer,
    pub audio_params_buffer: ManagedBuffer,
    
    // Pipelines
    pub k12_pipeline: vk::Pipeline,
    pub k12_layout: vk::PipelineLayout,
    pub k12_descriptor_set: vk::DescriptorSet,
    pub k12_set_layout: vk::DescriptorSetLayout,
}

impl AudioSystem {
    pub fn new(
        ctx: Arc<VulkanContext>,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffer: vk::Buffer,
        uniform_size: vk::DeviceSize,
    ) -> Result<Self, String> {
        let device = &ctx.device;

        // 1. Create Buffers (Managed)
        let spectrum_buffer = Self::create_managed_buffer(
            ctx.clone(), (512 * 4) as vk::DeviceSize, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        )?;

        let audio_params_buffer = Self::create_managed_buffer(
            ctx.clone(), std::mem::size_of::<crate::backend::shaders::types::AudioParams>() as vk::DeviceSize, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        // 2. K12: Audio Reactive Pipeline
        let k12_bindings = [
             vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
             vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
             vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let k12_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k12_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k12_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k12_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        let k12_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k12_layout, include_str!("../../shaders/k12_audio.wgsl"), "main") }?;
        
        let k12_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k12_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];

        // Update K12 Descriptor Set
        unsafe {
            let u_info = [vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size)];
            let s_info = [vk::DescriptorBufferInfo::default().buffer(spectrum_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
            let a_info = [vk::DescriptorBufferInfo::default().buffer(audio_params_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];

            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k12_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&u_info),
                vk::WriteDescriptorSet::default().dst_set(k12_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&s_info),
                vk::WriteDescriptorSet::default().dst_set(k12_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&a_info),
            ], &[]);
        }

        Ok(Self {
            ctx,
            spectrum_buffer,
            audio_params_buffer,
            k12_pipeline,
            k12_layout,
            k12_descriptor_set,
            k12_set_layout,
        })
    }

    pub fn execute(&mut self, cmd: vk::CommandBuffer) {
        unsafe {
            let device = &self.ctx.device;

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k12_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k12_layout, 0, &[self.k12_descriptor_set], &[]);
            device.cmd_dispatch(cmd, 1, 1, 1);

            // Barrier after update
            self.audio_params_buffer.barrier(
                cmd,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );
        }
    }

    pub fn update_spectrum(&mut self, data: &[f32]) {
        unsafe {
            let size = (data.len() * 4) as vk::DeviceSize;
            if let Ok(ptr) = self.ctx.device.map_memory(self.spectrum_buffer.memory, 0, size, vk::MemoryMapFlags::empty()) {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut f32, data.len());
                slice.copy_from_slice(data);
                self.ctx.device.unmap_memory(self.spectrum_buffer.memory);
            }
        }
    }

    fn create_managed_buffer(ctx: Arc<VulkanContext>, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> Result<ManagedBuffer, String> {
        let (buffer, memory) = resources::create_buffer(
            &ctx.device, &ctx.instance, ctx.physical_device, size, usage, properties
        )?;
        Ok(ManagedBuffer::new(ctx, buffer, memory, size, usage))
    }
}

impl Drop for AudioSystem {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            device.destroy_pipeline(self.k12_pipeline, None);
            device.destroy_pipeline_layout(self.k12_layout, None);
            device.destroy_descriptor_set_layout(self.k12_set_layout, None);
        }
    }
}
