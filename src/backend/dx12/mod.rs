//! DirectX 12 Backend - Modular Implementation
//!
//! This module implements the GpuExecutor trait for DirectX 12.

use windows::{
    core::*, Win32::Foundation::*, Win32::Graphics::Direct3D12::*, Win32::Graphics::Dxgi::Common::*,
    Win32::Graphics::Dxgi::*, Win32::System::Threading::*,
    Win32::Graphics::Direct3D::*,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::mem::ManuallyDrop;
pub type StdResult<T, E> = std::result::Result<T, E>;
use crate::backend::hal::*;
use crate::draw::DrawList;

pub mod resources;
pub mod pipelines;

const FRAME_COUNT: usize = 2;

// --- HAL Types ---
#[repr(transparent)]
pub struct Dx12Buffer(pub ID3D12Resource);
#[repr(transparent)]
pub struct Dx12Texture(pub ID3D12Resource);
#[repr(transparent)]
pub struct Dx12TextureView(pub D3D12_CPU_DESCRIPTOR_HANDLE);
#[repr(transparent)]
pub struct Dx12Sampler(pub D3D12_CPU_DESCRIPTOR_HANDLE);

pub struct Dx12RenderPipeline {
    pub pso: ID3D12PipelineState,
    pub root_signature: ID3D12RootSignature,
}
pub struct Dx12ComputePipeline {
    pub pso: ID3D12PipelineState,
    pub root_signature: ID3D12RootSignature,
}
#[repr(transparent)]
pub struct Dx12BindGroupLayout(pub ID3D12RootSignature);
pub struct Dx12BindGroup {
    pub heap: ID3D12DescriptorHeap,
    pub start_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
}

pub struct Dx12Backend {
    device: ID3D12Device,
    command_queue: ID3D12CommandQueue,
    swap_chain: IDXGISwapChain3,
    
    // Command Infrastructure
    command_allocators: [ID3D12CommandAllocator; FRAME_COUNT],
    command_list: ID3D12GraphicsCommandList,
    
    // Heaps
    rtv_heap: ID3D12DescriptorHeap,
    rtv_descriptor_size: u32,
    cbv_srv_uav_heap: ID3D12DescriptorHeap,
    cbv_srv_uav_descriptor_size: u32,
    
    // Render Targets
    render_targets: [ID3D12Resource; FRAME_COUNT],
    
    // Synchronization
    fence: ID3D12Fence,
    fence_values: [AtomicU64; FRAME_COUNT],
    fence_event: HANDLE,
    frame_index: AtomicUsize,

    // Resources (Managed)
    vertex_buffer: ID3D12Resource,
    vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    
    // Default samplers and views for HAL consistency
    font_texture: Option<ID3D12Resource>,
    font_view: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    backdrop_texture: Option<ID3D12Resource>,
    backdrop_view: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    default_sampler: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    
    // Default pipelines
    default_pipeline: Option<Dx12RenderPipeline>,
    default_layout: Option<Dx12BindGroupLayout>,

    // HAL state
    width: u32,
    height: u32,
    
    // Simple descriptor allocator
    cbv_srv_uav_ptr: AtomicUsize,
}

impl GpuResourceProvider for Dx12Backend {
    type Buffer = Dx12Buffer;
    type Texture = Dx12Texture;
    type TextureView = Dx12TextureView;
    type Sampler = Dx12Sampler;

    fn create_buffer(&self, size: u64, usage: BufferUsage, _label: &str) -> StdResult<Self::Buffer, String> {
        let (heap_type, state) = match usage {
            BufferUsage::Uniform => (D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ),
            _ => (D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON),
        };
        unsafe {
            resources::create_buffer(&self.device, size, heap_type, state)
                .map(Dx12Buffer)
                .map_err(|e| format!("{:?}", e))
        }
    }

    fn create_texture(&self, desc: &TextureDescriptor) -> StdResult<Self::Texture, String> {
        let dx_format = match desc.format {
            TextureFormat::R8Unorm => DXGI_FORMAT_R8_UNORM,
            TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
            TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
            TextureFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        };
        
        let mut flags = D3D12_RESOURCE_FLAG_NONE;
        if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        unsafe {
            resources::create_texture(&self.device, desc.width, desc.height, dx_format, flags, D3D12_RESOURCE_STATE_COMMON)
                .map(Dx12Texture)
                .map_err(|e| format!("{:?}", e))
        }
    }

    fn create_texture_view(&self, texture: &Self::Texture) -> StdResult<Self::TextureView, String> {
        unsafe {
            let desc = texture.0.GetDesc();
            let mut srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: desc.Format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            srv_desc.Anonymous.Texture2D.MipLevels = desc.MipLevels as u32;

            let idx = self.cbv_srv_uav_ptr.load(Ordering::Relaxed);
            let mut handle = self.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += idx * self.cbv_srv_uav_descriptor_size as usize;
            
            self.device.CreateShaderResourceView(&texture.0, Some(&srv_desc), handle);
            self.cbv_srv_uav_ptr.store(idx + 1, Ordering::Relaxed);
            
            Ok(Dx12TextureView(handle))
        }
    }

    fn create_sampler(&self, _label: &str) -> StdResult<Self::Sampler, String> {
        Err("Sampler creation not yet implemented for DX12".to_string())
    }

    fn write_buffer(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]) {
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            if buffer.0.Map(0, None, Some(&mut ptr)).is_ok() {
                std::ptr::copy_nonoverlapping(data.as_ptr(), (ptr as *mut u8).add(offset as usize), data.len());
                buffer.0.Unmap(0, None);
            }
        }
    }

    fn write_texture(&self, _texture: &Self::Texture, _data: &[u8], _width: u32, _height: u32) {}

    fn destroy_buffer(&self, _buffer: Self::Buffer) {}
    fn destroy_texture(&self, _texture: Self::Texture) {}
}

impl GpuPipelineProvider for Dx12Backend {
    type RenderPipeline = Dx12RenderPipeline;
    type ComputePipeline = Dx12ComputePipeline;
    type BindGroupLayout = Dx12BindGroupLayout;
    type BindGroup = Dx12BindGroup;

    fn create_render_pipeline(&self, _label: &str, wgsl_source: &str, layout: Option<&Self::BindGroupLayout>) -> StdResult<Self::RenderPipeline, String> {
        unsafe {
            let hlsl_vs = pipelines::transpile_wgsl_to_hlsl(wgsl_source, naga::ShaderStage::Vertex)?;
            let vs = pipelines::compile_shader(&hlsl_vs, "main", "vs_5_0")?;
            
            let hlsl_fs = pipelines::transpile_wgsl_to_hlsl(wgsl_source, naga::ShaderStage::Fragment)?;
            let ps = pipelines::compile_shader(&hlsl_fs, "main", "ps_5_0")?;

            let root_signature = if let Some(l) = layout {
                l.0.clone()
            } else {
                pipelines::create_root_signature(&self.device)?
            };

            let input_element_descs = [
                D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"POSITION\0".as_ptr()), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32_FLOAT,
                    InputSlot: 0, AlignedByteOffset: 0, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0,
                },
                D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"TEXCOORD\0".as_ptr()), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32_FLOAT,
                    InputSlot: 0, AlignedByteOffset: 8, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0,
                },
                D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"COLOR\0".as_ptr()), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    InputSlot: 0, AlignedByteOffset: 16, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0,
                },
            ];

            let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.GetBufferPointer(), BytecodeLength: vs.GetBufferSize() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.GetBufferPointer(), BytecodeLength: ps.GetBufferSize() },
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, ..Default::default() },
                BlendState: D3D12_BLEND_DESC {
                    RenderTarget: [
                        D3D12_RENDER_TARGET_BLEND_DESC {
                            BlendEnable: true.into(),
                            SrcBlend: D3D12_BLEND_SRC_ALPHA, DestBlend: D3D12_BLEND_INV_SRC_ALPHA, BlendOp: D3D12_BLEND_OP_ADD,
                            SrcBlendAlpha: D3D12_BLEND_ONE, DestBlendAlpha: D3D12_BLEND_ZERO, BlendOpAlpha: D3D12_BLEND_OP_ADD,
                            RenderTargetWriteMask: 0x0F, // D3D12_COLOR_WRITE_ENABLE_ALL
                            ..Default::default()
                        },
                        Default::default(), Default::default(), Default::default(),
                        Default::default(), Default::default(), Default::default(), Default::default(),
                    ],
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC { DepthEnable: false.into(), StencilEnable: false.into(), ..Default::default() },
                InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: input_element_descs.as_ptr(), NumElements: input_element_descs.len() as u32 },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: [DXGI_FORMAT_R8G8B8A8_UNORM, Default::default(), Default::default(), Default::default(), Default::default(), Default::default(), Default::default(), Default::default()],
                SampleMask: u32::MAX,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };

            let pso = self.device.CreateGraphicsPipelineState(&pso_desc).map_err(|e| format!("{:?}", e))?;
            Ok(Dx12RenderPipeline { pso, root_signature })
        }
    }

    fn create_compute_pipeline(&self, _label: &str, wgsl_source: &str, layout: Option<&Self::BindGroupLayout>) -> StdResult<Self::ComputePipeline, String> {
        unsafe {
            let hlsl = pipelines::transpile_wgsl_to_hlsl(wgsl_source, naga::ShaderStage::Compute)?;
            let cs = pipelines::compile_shader(&hlsl, "main", "cs_5_0")?;

            let root_signature = if let Some(l) = layout {
                l.0.clone()
            } else {
                pipelines::create_root_signature(&self.device)?
            };

            let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.GetBufferPointer(), BytecodeLength: cs.GetBufferSize() },
                ..Default::default()
            };

            let pso = self.device.CreateComputePipelineState(&pso_desc).map_err(|e| format!("{:?}", e))?;
            Ok(Dx12ComputePipeline { pso, root_signature })
        }
    }

    fn destroy_bind_group(&self, _bind_group: Self::BindGroup) {}
}

impl GpuExecutor for Dx12Backend {
    fn begin_execute(&self) -> StdResult<(), String> {
        unsafe {
            let idx = self.frame_index.load(Ordering::Relaxed);
            self.command_allocators[idx].Reset().map_err(|e| format!("{:?}", e))?;
            self.command_list.Reset(&self.command_allocators[idx], None).map_err(|e| format!("{:?}", e))?;
            
            // Set RTV
            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr + idx * self.rtv_descriptor_size as usize,
            };
            self.command_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None);
            self.command_list.ClearRenderTargetView(rtv_handle, &[0.0, 0.0, 0.0, 1.0], None);
            
            // Set Viewport and Scissor
            let viewport = D3D12_VIEWPORT { 
                TopLeftX: 0.0, TopLeftY: 0.0, Width: self.width as f32, Height: self.height as f32, 
                MinDepth: 0.0, MaxDepth: 1.0 
            };
            let scissor = RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 };
            self.command_list.RSSetViewports(&[viewport]);
            self.command_list.RSSetScissorRects(&[scissor]);
            
            Ok(())
        }
    }

    fn end_execute(&self) -> StdResult<(), String> {
        unsafe {
            self.command_list.Close().map_err(|e| format!("{:?}", e))?;
            let cmds = [Some(self.command_list.cast().unwrap())];
            self.command_queue.ExecuteCommandLists(&cmds);
            Ok(())
        }
    }

    fn draw(
        &self,
        pipeline: &Self::RenderPipeline,
        bind_group: Option<&Self::BindGroup>,
        vertex_buffer: &Self::Buffer,
        vertex_count: u32,
        _uniform_data: &[u8],
    ) -> StdResult<(), String> {
        unsafe {
            self.command_list.SetGraphicsRootSignature(&pipeline.root_signature);
            self.command_list.SetPipelineState(&pipeline.pso);
            
            if let Some(bg) = bind_group {
                let heaps = [Some(bg.heap.clone())];
                self.command_list.SetDescriptorHeaps(&heaps);
                self.command_list.SetGraphicsRootDescriptorTable(1, bg.start_handle);
            }
            
            let vb_view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: vertex_buffer.0.GetGPUVirtualAddress(),
                SizeInBytes: vertex_buffer.0.GetDesc().Width as u32,
                StrideInBytes: std::mem::size_of::<Vertex>() as u32,
            };

            self.command_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.command_list.IASetVertexBuffers(0, Some(&[vb_view]));
            self.command_list.DrawInstanced(vertex_count, 1, 0, 0);
            Ok(())
        }
    }

    fn dispatch(
        &self,
        pipeline: &Self::ComputePipeline,
        _bind_group_layout: Option<&Self::BindGroupLayout>,
        groups: [u32; 3],
        _push_constants: &[u8],
    ) -> StdResult<(), String> {
        unsafe {
            self.command_list.SetComputeRootSignature(&pipeline.root_signature);
            self.command_list.SetPipelineState(&pipeline.pso);
            
            self.command_list.Dispatch(groups[0], groups[1], groups[2]);
            Ok(())
        }
    }

    fn copy_texture(&self, src: &Self::Texture, dst: &Self::Texture) -> StdResult<(), String> {
        unsafe {
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(src.0.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 }
            };
            let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(dst.0.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 }
            };
            self.command_list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
            Ok(())
        }
    }

    fn generate_mipmaps(&self, _texture: &Self::Texture) -> StdResult<(), String> {
        Ok(())
    }

    fn create_bind_group(
        &self,
        _layout: &Self::BindGroupLayout,
        buffers: &[&Self::Buffer],
        textures: &[&Self::TextureView],
        _samplers: &[&Self::Sampler],
    ) -> StdResult<Self::BindGroup, String> {
        unsafe {
            let num_descriptors = (buffers.len() + textures.len()) as u32;
            let idx = self.cbv_srv_uav_ptr.fetch_add(num_descriptors as usize, Ordering::Relaxed);
            
            let mut cpu_handle = self.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            cpu_handle.ptr += idx * self.cbv_srv_uav_descriptor_size as usize;
            
            let mut gpu_handle = self.cbv_srv_uav_heap.GetGPUDescriptorHandleForHeapStart();
            gpu_handle.ptr += (idx * self.cbv_srv_uav_descriptor_size as usize) as u64;

            // 1. Create CBVs for buffers
            for (i, &buf) in buffers.iter().enumerate() {
                let desc = buf.0.GetDesc();
                let cbv_desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                    BufferLocation: buf.0.GetGPUVirtualAddress(),
                    SizeInBytes: (desc.Width as u32 + 255) & !255, // DX12 alignment
                };
                let mut handle = cpu_handle;
                handle.ptr += i * self.cbv_srv_uav_descriptor_size as usize;
                self.device.CreateConstantBufferView(Some(&cbv_desc), handle);
            }

            // 2. Copy SRVs for textures
            let srv_start_idx = buffers.len();
            for (i, &view) in textures.iter().enumerate() {
                let mut handle = cpu_handle;
                handle.ptr += (srv_start_idx + i) * self.cbv_srv_uav_descriptor_size as usize;
                self.device.CopyDescriptorsSimple(1, handle, view.0, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }

            Ok(Dx12BindGroup {
                heap: self.cbv_srv_uav_heap.clone(),
                start_handle: gpu_handle,
            })
        }
    }

    fn get_font_view(&self) -> &Self::TextureView { 
        static DUMMY: D3D12_CPU_DESCRIPTOR_HANDLE = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 };
        unsafe { &*(self.font_view.as_ref().unwrap_or(&DUMMY) as *const D3D12_CPU_DESCRIPTOR_HANDLE as *const Dx12TextureView) }
    }
    fn get_backdrop_view(&self) -> &Self::TextureView { 
        static DUMMY: D3D12_CPU_DESCRIPTOR_HANDLE = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 };
        unsafe { &*(self.backdrop_view.as_ref().unwrap_or(&DUMMY) as *const D3D12_CPU_DESCRIPTOR_HANDLE as *const Dx12TextureView) }
    }
    fn get_default_bind_group_layout(&self) -> &Self::BindGroupLayout { self.default_layout.as_ref().unwrap() }
    fn get_default_render_pipeline(&self) -> &Self::RenderPipeline { self.default_pipeline.as_ref().unwrap() }
    fn get_default_sampler(&self) -> &Self::Sampler { 
        static DUMMY: D3D12_CPU_DESCRIPTOR_HANDLE = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 };
        unsafe { &*(self.default_sampler.as_ref().unwrap_or(&DUMMY) as *const D3D12_CPU_DESCRIPTOR_HANDLE as *const Dx12Sampler) }
    }

    fn resolve(&mut self) -> StdResult<(), String> {
        unsafe {
            let idx = self.frame_index.load(Ordering::Relaxed);
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(self.render_targets[idx].clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        StateAfter: D3D12_RESOURCE_STATE_PRESENT,
                    })
                }
            };
            self.command_list.ResourceBarrier(&[barrier]);
        }
        Ok(())
    }

    fn present(&self) -> StdResult<(), String> {
        unsafe {
            self.swap_chain.Present(1, 0).ok().map_err(|e| format!("{:?}", e))?;
            self.move_to_next_frame();
            Ok(())
        }
    }
}

impl Dx12Backend {
    pub unsafe fn new(hwnd: HWND, width: u32, height: u32) -> Result<Self> {
        #[cfg(debug_assertions)]
        {
            let mut debug: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut debug).is_ok() {
                if let Some(debug) = debug {
                    debug.EnableDebugLayer();
                }
            }
        }

        let factory: IDXGIFactory4 = CreateDXGIFactory2(0)?;
        let adapter = Self::get_hardware_adapter(&factory)?;
        
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        let device = device.unwrap();

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        };
        let command_queue: ID3D12CommandQueue = device.CreateCommandQueue(&queue_desc)?;

        let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
            BufferCount: FRAME_COUNT as u32,
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };

        let swap_chain: IDXGISwapChain1 = factory.CreateSwapChainForHwnd(&command_queue, hwnd, &swap_chain_desc, None, None)?;
        let swap_chain: IDXGISwapChain3 = swap_chain.cast()?;
        let frame_index = swap_chain.GetCurrentBackBufferIndex() as usize;

        // RTV Heap
        let rtv_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            NumDescriptors: FRAME_COUNT as u32,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            ..Default::default()
        })?;
        let rtv_descriptor_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        let mut resources_vec = Vec::with_capacity(FRAME_COUNT);
        let mut rtv_handle = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        for i in 0..FRAME_COUNT {
            let resource: ID3D12Resource = swap_chain.GetBuffer(i as u32)?;
            device.CreateRenderTargetView(&resource, None, rtv_handle);
            resources_vec.push(resource);
            rtv_handle.ptr += rtv_descriptor_size as usize;
        }
        let render_targets: [ID3D12Resource; FRAME_COUNT] = [
            resources_vec.remove(0),
            resources_vec.remove(0),
        ];

        // CBV_SRV_UAV Heap
        let cbv_srv_uav_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            NumDescriptors: 1024,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })?;
        let cbv_srv_uav_descriptor_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        let command_allocators: [ID3D12CommandAllocator; FRAME_COUNT] = [
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?,
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?,
        ];

        let command_list: ID3D12GraphicsCommandList = device.CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, &command_allocators[0], None
        )?;
        command_list.Close()?;

        // Vertex Buffer
        let vb_size = 65536 * std::mem::size_of::<Vertex>() as u64;
        let vertex_buffer = resources::create_buffer(&device, vb_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vertex_buffer.GetGPUVirtualAddress(),
            SizeInBytes: vb_size as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };

        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let fence_event = CreateEventA(None, false, false, PCSTR::null()).map_err(|e| Error::new(E_FAIL, HSTRING::from(e.to_string().as_str())))?;

        let mut backend = Self {
            device, command_queue, swap_chain,
            command_allocators, command_list,
            rtv_heap, rtv_descriptor_size,
            cbv_srv_uav_heap, cbv_srv_uav_descriptor_size,
            render_targets,
            fence, 
            fence_values: [AtomicU64::new(0), AtomicU64::new(0)], 
            fence_event,
            frame_index: AtomicUsize::new(frame_index),
            vertex_buffer, vertex_buffer_view,
            font_texture: None, font_view: None,
            backdrop_texture: None, backdrop_view: None,
            default_sampler: None,
            default_pipeline: None,
            default_layout: None,
            width, height,
            cbv_srv_uav_ptr: AtomicUsize::new(0),
        };

        // Initialize defaults
        {
            let layout = Dx12BindGroupLayout(pipelines::create_root_signature(&backend.device).map_err(|e| Error::new(E_FAIL, HSTRING::from(e.as_str())))?);
            let wgsl_source = include_str!("../wgpu_shader.wgsl");
            let pipeline = backend.create_render_pipeline("Default UI", wgsl_source, Some(&layout)).map_err(|e| Error::new(E_FAIL, HSTRING::from(e.as_str())))?;
            
            backend.default_layout = Some(layout);
            backend.default_pipeline = Some(pipeline);
            backend.default_sampler = Some(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 });
        }

        Ok(backend)
    }

    unsafe fn get_hardware_adapter(factory: &IDXGIFactory4) -> Result<IDXGIAdapter1> {
        for i in 0.. {
            let adapter = match factory.EnumAdapters1(i) {
                Ok(adapter) => adapter,
                Err(_) => break,
            };
            let mut desc = DXGI_ADAPTER_DESC1::default();
            adapter.GetDesc1(&mut desc)?;
            
            // Handle DXGI_ADAPTER_FLAG type mismatch across windows-rs versions
            let flags = desc.Flags;
            // DXGI_ADAPTER_FLAG_SOFTWARE is usually 2
            if (flags & 2) != 0 { continue; }
            
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, std::ptr::null_mut::<Option<ID3D12Device>>()).is_ok() {
                return Ok(adapter);
            }
        }
        Err(Error::from(E_FAIL))
    }

    unsafe fn wait_for_gpu(&self) {
        let frame_idx = self.frame_index.load(Ordering::Relaxed);
        let fence_value = self.fence_values[frame_idx].fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.command_queue.Signal(&self.fence, fence_value);

        if self.fence.GetCompletedValue() < fence_value {
            let _ = self.fence.SetEventOnCompletion(fence_value, self.fence_event);
            let _ = WaitForSingleObject(self.fence_event, u32::MAX);
        }
    }

    unsafe fn move_to_next_frame(&self) {
        let frame_idx = self.frame_index.load(Ordering::Relaxed);
        let current_fence_value = self.fence_values[frame_idx].load(Ordering::Relaxed);
        let _ = self.command_queue.Signal(&self.fence, current_fence_value);

        let new_frame_idx = self.swap_chain.GetCurrentBackBufferIndex() as usize;
        self.frame_index.store(new_frame_idx, Ordering::Relaxed);

        let wait_value = self.fence_values[new_frame_idx].load(Ordering::Relaxed);
        if self.fence.GetCompletedValue() < wait_value {
            let _ = self.fence.SetEventOnCompletion(wait_value, self.fence_event);
            let _ = WaitForSingleObject(self.fence_event, u32::MAX);
        }

        self.fence_values[new_frame_idx].store(current_fence_value + 1, Ordering::Relaxed);
    }
}

impl crate::backend::GraphicsBackend for Dx12Backend {
    fn name(&self) -> &str { "DirectX 12" }

    fn update_font_texture(&mut self, width: u32, height: u32, data: &[u8]) {
        unsafe {
            let tex = resources::create_texture(
                &self.device, width, height, DXGI_FORMAT_R8_UNORM, 
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST
            ).unwrap();

            let row_pitch = (width + 255) & !255;
            let size = (row_pitch * height) as u64;
            let staging = resources::create_buffer(
                &self.device, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ
            ).unwrap();

            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            staging.Map(0, None, Some(&mut ptr)).unwrap();
            let dest = ptr as *mut u8;
            for y in 0..height {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add((y * width) as usize),
                    dest.add((y * row_pitch) as usize),
                    width as usize
                );
            }
            staging.Unmap(0, None);

            let idx = self.frame_index.load(Ordering::Relaxed);
            self.command_allocators[idx].Reset().unwrap();
            self.command_list.Reset(&self.command_allocators[idx], None).unwrap();
            
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(staging.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8_UNORM,
                            Width: width, Height: height, Depth: 1, RowPitch: row_pitch,
                        }
                    }
                }
            };
            let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(tex.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 }
            };
            self.command_list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
            
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(tex.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                        StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    })
                }
            };
            self.command_list.ResourceBarrier(&[barrier]);
            self.command_list.Close().unwrap();
            
            let cmds = [Some(self.command_list.cast().unwrap())];
            self.command_queue.ExecuteCommandLists(&cmds);
            self.wait_for_gpu();

            let mut srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            srv_desc.Anonymous.Texture2D.MipLevels = 1;

            let idx_srv = self.cbv_srv_uav_ptr.fetch_add(1, Ordering::Relaxed);
            let mut handle = self.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += idx_srv * self.cbv_srv_uav_descriptor_size as usize;
            self.device.CreateShaderResourceView(&tex, Some(&srv_desc), handle);

            self.font_texture = Some(tex);
            self.font_view = Some(handle);
        }
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        let mut orch = crate::renderer::orchestrator::RenderOrchestrator::new();
        let tasks = orch.plan(dl);
        orch.execute(self, tasks.as_slice(), 0.0, width, height).unwrap();
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}
