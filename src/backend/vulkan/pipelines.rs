#![cfg(feature = "vulkan")]

use ash::vk;
use std::ffi::CStr;

/// Compile GLSL source code to SPIR-V using shaderc
pub fn compile_glsl(glsl: &str, stage: vk::ShaderStageFlags) -> Result<Vec<u32>, String> {
    let compiler = shaderc::Compiler::new().ok_or("Failed to initialize shaderc compiler")?;
    let options = shaderc::CompileOptions::new().ok_or("Failed to create shaderc options")?;
    
    let shader_kind = match stage {
        vk::ShaderStageFlags::VERTEX => shaderc::ShaderKind::Vertex,
        vk::ShaderStageFlags::FRAGMENT => shaderc::ShaderKind::Fragment,
        vk::ShaderStageFlags::COMPUTE => shaderc::ShaderKind::Compute,
        _ => return Err(format!("Unsupported shader stage: {:?}", stage)),
    };
    
    let artifact = compiler.compile_into_spirv(glsl, shader_kind, "shader.glsl", "main", Some(&options))
        .map_err(|e| format!("Shaderc Compilation Error: {:?}", e))?;
    
    Ok(artifact.as_binary().to_vec())
}

/// Compile WGSL source code to SPIR-V
pub fn compile_wgsl(wgsl: &str) -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(wgsl)
        .map_err(|e| format!("WGSL Parse Error: {:?}", e))?;
    
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    
    let info = validator.validate(&module)
        .map_err(|e| format!("WGSL Validation Error: {:?}", e))?;
    
    // SPIR-V 1.0 is often more compatible with Vulkan validation layers regarding atomics
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

/// Create a render pipeline from GLSL source (Vertex + Fragment)
pub unsafe fn create_render_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs_code: &str,
    fs_code: &str,
    is_wgsl: bool,
) -> Result<vk::Pipeline, String> {
    let vs_spv = if is_wgsl { compile_wgsl(vs_code)? } else { compile_glsl(vs_code, vk::ShaderStageFlags::VERTEX)? };
    let fs_spv = if is_wgsl { compile_wgsl(fs_code)? } else { compile_glsl(fs_code, vk::ShaderStageFlags::FRAGMENT)? };
    
    let vs_module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs_spv), None).map_err(|e| format!("VS Creation failed: {:?}", e))?;
    let fs_module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fs_spv), None).map_err(|e| format!("FS Creation failed: {:?}", e))?;
    
    let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let entry_wgsl_vs = CStr::from_bytes_with_nul(b"vs_main\0").unwrap();
    let entry_wgsl_fs = CStr::from_bytes_with_nul(b"fs_main\0").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs_module).name(if is_wgsl { entry_wgsl_vs } else { entry }),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs_module).name(if is_wgsl { entry_wgsl_fs } else { entry }),
    ];
    
    let vertex_binding = [vk::VertexInputBindingDescription::default().binding(0).stride(32).input_rate(vk::VertexInputRate::VERTEX)];
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::builder()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(16)
            .build(),
    ];
    
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding)
        .vertex_attribute_descriptions(&vertex_attrs);
    
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)];
    
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachment);
    
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_info)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
        
    let pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .map_err(|e| format!("Graphics Pipeline failed: {:?}", e))?[0];
    
    device.destroy_shader_module(vs_module, None);
    device.destroy_shader_module(fs_module, None);
    
    Ok(pipeline)
}

/// Create a compute pipeline from WGSL or GLSL source
pub unsafe fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    source: &str,
    entry_point: &str,
) -> Result<vk::Pipeline, String> {
    let is_wgsl = source.contains("@computer") || source.contains("@compute") || source.contains("fn ");
    let spv = if is_wgsl { compile_wgsl(source)? } else { compile_glsl(source, vk::ShaderStageFlags::COMPUTE)? };
    
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
        
    device.destroy_shader_module(shader_module, None);
    
    Ok(pipeline)
}
