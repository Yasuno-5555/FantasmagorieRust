use ash::vk;
use std::sync::Arc;
use crate::backend::vulkan::VulkanContext;
use crate::backend::vulkan::managed::ManagedBuffer;
use crate::backend::vulkan::pipelines;
use crate::backend::vulkan::resources;

pub struct VisibilitySystem {
    pub ctx: Arc<VulkanContext>,
    
    // Buffers (Managed)
    pub indirect_dispatch_buffer: ManagedBuffer,
    pub indirect_draw_buffer: ManagedBuffer,
    pub counter_buffer: ManagedBuffer,
    pub instance_buffer: ManagedBuffer,
    pub counter_readback_buffer: ManagedBuffer,
    
    // Pipelines
    pub k8_pipeline: vk::Pipeline,
    pub k8_layout: vk::PipelineLayout,
    pub k8_descriptor_set: vk::DescriptorSet,
    pub k8_set_layout: vk::DescriptorSetLayout,
    
    pub k13_pipeline: vk::Pipeline,
    pub k13_layout: vk::PipelineLayout,
    pub k13_descriptor_set: vk::DescriptorSet,
    pub k13_set_layout: vk::DescriptorSetLayout,
}

impl VisibilitySystem {
    pub fn new(
        ctx: Arc<VulkanContext>,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffer: vk::Buffer,
        uniform_size: vk::DeviceSize,
        backdrop_view: vk::ImageView,
    ) -> Result<Self, String> {
        let device = &ctx.device;

        // 1. Create Buffers (Managed)
        let indirect_dispatch_buffer = Self::create_managed_buffer(
            ctx.clone(), 1024, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;
        
        let indirect_draw_buffer = Self::create_managed_buffer(
            ctx.clone(), 1024, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        let counter_buffer = Self::create_managed_buffer(
            ctx.clone(), 1024, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        let instance_buffer = Self::create_managed_buffer(
            ctx.clone(), 1024 * 1024, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        let counter_readback_buffer = Self::create_managed_buffer(
            ctx.clone(), 1024, 
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        )?;

        // 2. K13: Indirect Dispatch Pipeline
        let k13_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k13_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k13_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k13_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k13_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        let k13_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k13_layout, include_str!("../../shaders/k13_indirect.wgsl"), "main") }?;
        let k13_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k13_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];

        // Update K13 Descriptor Set
        let k13_bis = [
            vk::DescriptorBufferInfo::default().buffer(counter_buffer.buffer).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(indirect_draw_buffer.buffer).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(indirect_dispatch_buffer.buffer).offset(0).range(vk::WHOLE_SIZE),
        ];
        unsafe {
            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k13_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&k13_bis[0..1]),
                vk::WriteDescriptorSet::default().dst_set(k13_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&k13_bis[1..2]),
                vk::WriteDescriptorSet::default().dst_set(k13_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&k13_bis[2..3]),
            ], &[]);
        }

        // 3. K8: Visibility Culling Pipeline
        let k8_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let k8_set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&k8_bindings), None) }.map_err(|e| format!("{:?}", e))?;
        let k8_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&k8_set_layout)), None) }.map_err(|e| format!("{:?}", e))?;
        let k8_pipeline = unsafe { pipelines::create_compute_pipeline(&device, k8_layout, include_str!("../../shaders/k8_visibility.wgsl"), "main") }?;
        let k8_descriptor_set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&k8_set_layout))) }.map_err(|e| format!("{:?}", e))?[0];
        
        unsafe {
            let u_info = [vk::DescriptorBufferInfo::default().buffer(uniform_buffer).offset(0).range(uniform_size)];
            let i_info = [vk::DescriptorBufferInfo::default().buffer(instance_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
            let img_info = [vk::DescriptorImageInfo::default().image_view(backdrop_view).image_layout(vk::ImageLayout::GENERAL)];
            let id_info = [vk::DescriptorBufferInfo::default().buffer(indirect_draw_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];
            let c_info = [vk::DescriptorBufferInfo::default().buffer(counter_buffer.buffer).offset(0).range(vk::WHOLE_SIZE)];

            device.update_descriptor_sets(&[
                vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&u_info),
                vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&i_info),
                vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&img_info),
                vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&id_info),
                vk::WriteDescriptorSet::default().dst_set(k8_descriptor_set).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&c_info),
            ], &[]);
        }

        Ok(Self {
            ctx,
            indirect_dispatch_buffer,
            indirect_draw_buffer,
            counter_buffer,
            instance_buffer,
            counter_readback_buffer,
            k8_pipeline,
            k8_layout,
            k8_descriptor_set,
            k8_set_layout,
            k13_pipeline,
            k13_layout,
            k13_descriptor_set,
            k13_set_layout,
        })
    }

    pub fn execute(&mut self, cmd: vk::CommandBuffer, instance_count: u32) {
        unsafe {
            let device = &self.ctx.device;

            // 1. Reset Counters
            device.cmd_fill_buffer(cmd, self.counter_buffer.buffer, 0, 1024, 0);
            
            self.counter_buffer.barrier(
                cmd, 
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
            );

            // 2. K8 Visibility Culling
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k8_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k8_layout, 0, &[self.k8_descriptor_set], &[]);
            let group_x = (instance_count + 63) / 64;
            device.cmd_dispatch(cmd, group_x, 1, 1);

            self.counter_buffer.barrier(
                cmd,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );

            // 3. K13 Indirect Dispatch Generation
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.k13_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.k13_layout, 0, &[self.k13_descriptor_set], &[]);
            device.cmd_dispatch(cmd, 1, 1, 1);

            // Multi-barrier for indirect buffers
            self.indirect_draw_buffer.barrier(
                cmd,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );
            self.indirect_dispatch_buffer.barrier(
                cmd,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );
        }
    }

    fn create_managed_buffer(ctx: Arc<VulkanContext>, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> Result<ManagedBuffer, String> {
        let (buffer, memory) = resources::create_buffer(
            &ctx.device, &ctx.instance, ctx.physical_device, size, usage, properties
        )?;
        Ok(ManagedBuffer::new(ctx, buffer, memory, size, usage))
    }
}

impl Drop for VisibilitySystem {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            device.destroy_pipeline(self.k8_pipeline, None);
            device.destroy_pipeline_layout(self.k8_layout, None);
            device.destroy_descriptor_set_layout(self.k8_set_layout, None);
            device.destroy_pipeline(self.k13_pipeline, None);
            device.destroy_pipeline_layout(self.k13_layout, None);
            device.destroy_descriptor_set_layout(self.k13_set_layout, None);
        }
    }
}
