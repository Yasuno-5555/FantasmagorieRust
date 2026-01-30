use ash::vk;
use std::sync::Arc;
use std::ffi::{CStr};
use crate::backend::vulkan::VulkanContext;

pub struct PipelineProvider {
    ctx: Arc<VulkanContext>,
}

impl PipelineProvider {
    pub fn new(ctx: Arc<VulkanContext>) -> Self {
        Self { ctx }
    }

    pub fn compile_wgsl(&self, src: &str) -> Result<Vec<u32>, String> {
        let module = naga::front::wgsl::parse_str(src)
            .map_err(|e| format!("WGSL Parse Error: {:?}", e))?;
        
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        
        let info = validator.validate(&module)
            .map_err(|e| format!("WGSL Validation Error: {:?}", e))?;
        
        let spirv = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options {
                lang_version: (1, 3),
                flags: naga::back::spv::WriterFlags::empty(),
                ..naga::back::spv::Options::default()
            },
            None,
        ).map_err(|e| format!("SPIR-V Export Error: {:?}", e))?;
        
        Ok(spirv)
    }

    pub unsafe fn create_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<vk::DescriptorSetLayout, String> {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        self.ctx.device.create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| format!("Failed to create descriptor set layout: {:?}", e))
    }

    pub unsafe fn create_pipeline_layout(
        &self,
        set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<vk::PipelineLayout, String> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);
        self.ctx.device.create_pipeline_layout(&layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))
    }

    pub unsafe fn create_graphics_pipeline(
        &self,
        create_info: &vk::GraphicsPipelineCreateInfo,
    ) -> Result<vk::Pipeline, String> {
        let pipelines = self.ctx.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(create_info), None)
            .map_err(|e| format!("Failed to create graphics pipeline: {:?}", e))?;
        Ok(pipelines[0])
    }

    pub unsafe fn create_compute_pipeline(
        &self,
        create_info: &vk::ComputePipelineCreateInfo,
    ) -> Result<vk::Pipeline, String> {
        let pipelines = self.ctx.device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(create_info), None)
            .map_err(|e| format!("Failed to create compute pipeline: {:?}", e))?;
        Ok(pipelines[0])
    }
    
    pub fn get_shader_stage(&self, module: vk::ShaderModule, stage: vk::ShaderStageFlags, entry_point: &str) -> vk::PipelineShaderStageCreateInfo {
        // Warning: This requires the entry_point string to live as long as the stage info if used in a collection.
        // For inline creation it's fine.
        let name = std::ffi::CString::new(entry_point).unwrap();
        vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .module(module)
            .name(unsafe { CStr::from_ptr(name.as_ptr()) })
    }
}
