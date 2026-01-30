use ash::vk;
use std::sync::Arc;
use crate::backend::vulkan::VulkanContext;
use crate::backend::vulkan::managed::ManagedBuffer;
use crate::backend::vulkan::pipelines;
use crate::backend::vulkan::resources;

pub struct ParticleSystem {
    pub ctx: Arc<VulkanContext>,
    
    // Buffers (Managed)
    pub particle_buffer: ManagedBuffer,
    
    // Pipelines
    pub k6_update_pipeline: vk::Pipeline,
    pub k6_spawn_pipeline: vk::Pipeline,
    pub k6_layout: vk::PipelineLayout,
    pub k6_descriptor_set: vk::DescriptorSet,
    pub k6_set_layout: vk::DescriptorSetLayout,
}

impl ParticleSystem {
    pub fn new(
        ctx: Arc<VulkanContext>,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffer: vk::Buffer,
        uniform_size: vk::DeviceSize,
        counter_buffer: vk::Buffer,
    ) -> Result<Self, String> {
        let device = &ctx.device;

        // 1. Create Buffers (Managed)
        let particle_buffer = Self::create_managed_buffer(
            ctx.clone(), 1024 * 1024 * 4, // 4MB for particles
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        // 2. K6: Particles Pipelines
        let k6_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k6_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k6_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k6_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k6_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        
        let k6_source = include_str!("../../shaders/k6_particle.wgsl");
        let k6_update_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k6_layout, k6_source, "update") }?;
        let k6_spawn_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k6_layout, k6_source, "spawn") }?;
        
        let k6_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k6_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];

        // Update K6 Descriptor Set
        unsafe {
            let u_info = [vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size)];
            let p_info = [vk::DescriptorBufferInfo::default().buffer(particle_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
            let c_info = [vk::DescriptorBufferInfo::default().buffer(counter_buffer).offset(0).range(vk::WHOLE_SIZE)];

            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k6_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&u_info),
                vk::WriteDescriptorSet::default().dst_set(k6_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&p_info),
                vk::WriteDescriptorSet::default().dst_set(k6_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&c_info),
            ], &[]);
        }

        Ok(Self {
            ctx,
            particle_buffer,
            k6_update_pipeline,
            k6_spawn_pipeline,
            k6_layout,
            k6_descriptor_set,
            k6_set_layout,
        })
    }

    pub fn execute(&mut self, cmd: vk::CommandBuffer, particle_count: u32, spawn_count: u32) {
        unsafe {
            let device = &self.ctx.device;

            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k6_layout, 0, &[self.k6_descriptor_set], &[]);

            // 1. Update Particles
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k6_update_pipeline);
            let group_x = (particle_count + 63) / 64;
            device.cmd_dispatch(cmd, group_x, 1, 1);

            // Barrier
            self.particle_buffer.barrier(
                cmd,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );

            // 2. Spawn Particles
            if spawn_count > 0 {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k6_spawn_pipeline);
                let group_spawn = (spawn_count + 63) / 64;
                device.cmd_dispatch(cmd, group_spawn, 1, 1);
                
                self.particle_buffer.barrier(
                    cmd,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                );
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

impl Drop for ParticleSystem {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            device.destroy_pipeline(self.k6_update_pipeline, None);
            device.destroy_pipeline(self.k6_spawn_pipeline, None);
            device.destroy_pipeline_layout(self.k6_layout, None);
            device.destroy_descriptor_set_layout(self.k6_set_layout, None);
        }
    }
}
