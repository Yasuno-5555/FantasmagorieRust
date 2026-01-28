use ash::vk;
use std::ffi::CStr;

/// Compile WGSL source code to SPIR-V
pub fn compile_wgsl(wgsl: &str) -> Result<Vec<u32>, String> {
    #[cfg(feature = "vulkan")]
    {
        let module = naga::front::wgsl::parse_str(wgsl)
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
                lang_version: (1, 2),
                flags: naga::back::spv::WriterFlags::empty(),
                ..naga::back::spv::Options::default()
            },
            None,
        ).map_err(|e| format!("SPIR-V Export Error: {:?}", e))?;
        
        Ok(spirv)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = wgsl;
        Err("Vulkan feature not enabled".to_string())
    }
}

/// Create a compute pipeline from WGSL source
///
/// Handles shader compilation, module creation, pipeline creation, and module cleanup.
pub unsafe fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    wgsl_code: &str,
    entry_point: &str,
) -> Result<vk::Pipeline, String> {
    let spv = compile_wgsl(wgsl_code)?;
    
    let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&spv);
    let shader_module = device.create_shader_module(&shader_module_info, None)
        .map_err(|e| format!("Failed to create shader module: {:?}", e))?;
        
    let entry_name = std::ffi::CString::new(entry_point).unwrap();
    
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(entry_name.as_c_str());
        
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(layout);
        
    let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .map_err(|e| format!("Failed to create compute pipeline: {:?}", e))?[0];
        
    // Destroy shader module immediately as it's no longer needed after pipeline creation
    device.destroy_shader_module(shader_module, None);
    
    Ok(pipeline)
}
