//! DX12 Pipeline and Shader Management
//!
//! Handles WGSL -> HLSL transpilation using Naga and pipeline creation.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D::Fxc::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::*;
use std::ffi::{c_void, CString};
use naga::back::hlsl;
use std::result::Result as StdResult;

/// Transpile WGSL source to HLSL
pub fn transpile_wgsl_to_hlsl(wgsl_source: &str, stage: naga::ShaderStage) -> StdResult<String, String> {
    let module = naga::front::wgsl::Frontend::new().parse(wgsl_source)
        .map_err(|e| format!("Failed to parse WGSL: {:?}", e))?;
    
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let info = validator.validate(&module)
        .map_err(|e| format!("WGSL validation failed: {:?}", e))?;
    
    let hlsl_options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V5_0,
        binding_map: hlsl::BindingMap::default(),
        fake_missing_bindings: true,
        special_constants_binding: None,
        ..Default::default()
    };
    
    let mut output = String::new();
    let mut writer = hlsl::Writer::new(&mut output, &hlsl_options);
    writer.write(&module, &info)
        .map_err(|e| format!("HLSL transpilation failed: {:?}", e))?;
    
    Ok(output)
}

/// Compile HLSL to bytecode
pub unsafe fn compile_shader(
    source: &str,
    entry_point: &str,
    target: &str,
) -> StdResult<ID3DBlob, String> {
    let mut blob: Option<ID3DBlob> = None;
    let mut error_blob: Option<ID3DBlob> = None;
    
    let s_source = source.as_bytes();
    let entry_point_c = CString::new(entry_point).unwrap();
    let target_c = CString::new(target).unwrap();
    let s_entry = PCSTR(entry_point_c.as_ptr() as *const u8);
    let s_target = PCSTR(target_c.as_ptr() as *const u8);
    
    let hr = D3DCompile(
        s_source.as_ptr() as *const c_void,
        s_source.len(),
        None,
        None,
        None,
        s_entry,
        s_target,
        D3DCOMPILE_ENABLE_STRICTNESS,
        0,
        &mut blob,
        Some(&mut error_blob),
    );
    
    if hr.is_err() {
        if let Some(err) = error_blob {
            let msg = std::slice::from_raw_parts(
                err.GetBufferPointer() as *const u8,
                err.GetBufferSize(),
            );
            return Err(String::from_utf8_lossy(msg).into_owned());
        }
        return Err(format!("Shader compilation failed with HRESULT: {:?}", hr));
    }
    
    blob.ok_or_else(|| "Shader compilation failed: no output blob".to_string())
}

/// Create a basic root signature for standard draw calls
pub unsafe fn create_root_signature(device: &ID3D12Device) -> StdResult<ID3D12RootSignature, String> {
    let root_parameters = [
        // Constant Buffer (Register 0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // Descriptor Table (SRVs) (Register 0, Space 0) - For Textures
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &D3D12_DESCRIPTOR_RANGE {
                        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                        NumDescriptors: 8, // Support up to 8 textures
                        BaseShaderRegister: 0,
                        RegisterSpace: 0,
                        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                    },
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];

    let root_sig_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: root_parameters.len() as u32,
        pParameters: root_parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };

    let mut signature_blob: Option<ID3DBlob> = None;
    let mut error_blob: Option<ID3DBlob> = None;

    D3D12SerializeRootSignature(
        &root_sig_desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &mut signature_blob,
        Some(&mut error_blob),
    ).map_err(|e| format!("SerializeRootSignature failed: {:?}", e))?;

    let signature_blob = signature_blob.unwrap();

    let root_signature: ID3D12RootSignature = device.CreateRootSignature(
        0,
        std::slice::from_raw_parts(
            signature_blob.GetBufferPointer() as *const u8,
            signature_blob.GetBufferSize(),
        ),
    ).map_err(|e| format!("CreateRootSignature failed: {:?}", e))?;

    Ok(root_signature)
}
