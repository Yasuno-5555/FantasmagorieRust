//! DirectX 12 backend - Windows-native GPU rendering
//!
//! Full implementation using windows-rs crate for DirectX 12 API on Windows 10+.

use crate::core::{ColorF, Vec2};
use crate::draw::{DrawCommand, DrawList};
use std::cell::Cell;
use std::ffi::c_void;
use std::mem::ManuallyDrop;

use windows::{
    core::*, Win32::Foundation::*, Win32::Graphics::Direct3D::Fxc::*, Win32::Graphics::Direct3D::*,
    Win32::Graphics::Direct3D12::*, Win32::Graphics::Dxgi::Common::*, Win32::Graphics::Dxgi::*,
    Win32::System::Threading::*,
};

const FRAME_COUNT: usize = 2;

/// DirectX 12-based rendering backend
pub struct Dx12Backend {
    device: ID3D12Device,
    command_queue: ID3D12CommandQueue,
    swap_chain: IDXGISwapChain3,
    rtv_heap: ID3D12DescriptorHeap,
    rtv_descriptor_size: u32,
    render_targets: [ID3D12Resource; FRAME_COUNT],
    command_allocators: [ID3D12CommandAllocator; FRAME_COUNT],
    command_list: ID3D12GraphicsCommandList,

    // Root signature and pipeline
    root_signature: ID3D12RootSignature,
    pipeline_state: Option<ID3D12PipelineState>,

    // Resources
    vertex_buffer: ID3D12Resource,
    vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    constant_buffer: ID3D12Resource,
    #[allow(dead_code)]
    cbv_heap: ID3D12DescriptorHeap,

    // Synchronization (using Cell for interior mutability)
    fence: ID3D12Fence,
    fence_values: Cell<[u64; FRAME_COUNT]>,
    fence_event: HANDLE,
    frame_index: Cell<usize>,

    width: u32,
    height: u32,
}

/// Vertex format for DirectX 12
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

/// Constant buffer data
#[repr(C, align(256))]
#[derive(Copy, Clone, Debug)]
struct ConstantBuffer {
    projection: [[f32; 4]; 4],
    rect: [f32; 4],
    radii: [f32; 4],
    border_color: [f32; 4],
    glow_color: [f32; 4], // Added
    mode: i32,
    border_width: f32,
    elevation: f32,
    is_squircle: i32,   // Added
    glow_strength: f32, // Added
    _padding: [f32; 27], // Fill to 256 bytes (64 floats total). 40 used. 24 left?
                        // Wait, 40 floats * 4 = 160 bytes.
                        // 256 - 160 = 96 bytes. 96/4 = 24 floats.
                        // Let's verify align. 160 is 16-byte aligned.
}

impl Dx12Backend {
    /// Create a new DirectX 12 backend
    ///
    /// # Safety
    /// Requires a valid HWND window handle
    pub unsafe fn new(hwnd: HWND, width: u32, height: u32) -> Result<Self> {
        // Enable debug layer in debug builds
        #[cfg(debug_assertions)]
        {
            let mut debug: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut debug).is_ok() {
                if let Some(debug) = debug {
                    debug.EnableDebugLayer();
                }
            }
        }

        // Create DXGI Factory
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG)?;

        // Find adapter
        let adapter = Self::get_hardware_adapter(&factory)?;

        // Create D3D12 Device
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        let device = device.unwrap();

        println!("✅ DirectX 12 device created");

        // Create Command Queue
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            ..Default::default()
        };
        let command_queue: ID3D12CommandQueue = device.CreateCommandQueue(&queue_desc)?;

        // Create Swap Chain
        let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
            BufferCount: FRAME_COUNT as u32,
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };

        let swap_chain: IDXGISwapChain1 =
            factory.CreateSwapChainForHwnd(&command_queue, hwnd, &swap_chain_desc, None, None)?;

        // Disable Alt+Enter fullscreen
        factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER)?;

        let swap_chain: IDXGISwapChain3 = swap_chain.cast()?;
        let frame_index = swap_chain.GetCurrentBackBufferIndex() as usize;

        // Create RTV Descriptor Heap
        let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            NumDescriptors: FRAME_COUNT as u32,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let rtv_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&rtv_heap_desc)?;
        let rtv_descriptor_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        // Create Render Targets
        let mut render_targets: [ID3D12Resource; FRAME_COUNT] = std::mem::zeroed();
        let mut rtv_handle = rtv_heap.GetCPUDescriptorHandleForHeapStart();

        for i in 0..FRAME_COUNT {
            let resource: ID3D12Resource = swap_chain.GetBuffer(i as u32)?;
            device.CreateRenderTargetView(&resource, None, rtv_handle);
            render_targets[i] = resource;
            rtv_handle.ptr += rtv_descriptor_size as usize;
        }

        // Create Command Allocators
        let command_allocators: [ID3D12CommandAllocator; FRAME_COUNT] = [
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?,
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?,
        ];

        // Create Root Signature
        let root_signature = Self::create_root_signature(&device)?;

        // Create Command List
        let command_list: ID3D12GraphicsCommandList = device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &command_allocators[0],
            None,
        )?;
        command_list.Close()?;

        // Create CBV Descriptor Heap
        let cbv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            NumDescriptors: 1,
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let cbv_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&cbv_heap_desc)?;

        // Create Constant Buffer
        let cb_size = std::mem::size_of::<ConstantBuffer>() as u64;
        let constant_buffer = Self::create_upload_buffer(&device, cb_size)?;

        // Create CBV
        let cbv_desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: constant_buffer.GetGPUVirtualAddress(),
            SizeInBytes: cb_size as u32,
        };
        device.CreateConstantBufferView(
            Some(&cbv_desc),
            cbv_heap.GetCPUDescriptorHandleForHeapStart(),
        );

        // Create Vertex Buffer (64KB)
        let vb_size = 65536 * std::mem::size_of::<Vertex>() as u64;
        let vertex_buffer = Self::create_upload_buffer(&device, vb_size)?;

        let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vertex_buffer.GetGPUVirtualAddress(),
            SizeInBytes: vb_size as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };

        // Compile Shader & Pipeline
        let pipeline_state = Some(Self::create_pipeline_state(&device, &root_signature)?);

        // Create Fence
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let fence_values = Cell::new([0u64; FRAME_COUNT]);

        // Create event handle using Windows API
        let fence_event = CreateEventA(None, false, false, None)?;

        println!("   ✅ DirectX 12 backend initialized");

        Ok(Self {
            device,
            command_queue,
            swap_chain,
            rtv_heap,
            rtv_descriptor_size,
            render_targets,
            command_allocators,
            command_list,
            root_signature,
            pipeline_state,
            vertex_buffer,
            vertex_buffer_view,
            constant_buffer,
            cbv_heap,
            fence,
            fence_values,
            fence_event,
            frame_index: Cell::new(frame_index),
            width,
            height,
        })
    }

    unsafe fn get_hardware_adapter(factory: &IDXGIFactory4) -> Result<IDXGIAdapter1> {
        for i in 0.. {
            let adapter = match factory.EnumAdapters1(i) {
                Ok(adapter) => adapter,
                Err(_) => break,
            };

            let mut desc = DXGI_ADAPTER_DESC1::default();
            adapter.GetDesc1(&mut desc)?;

            // Skip software adapters
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }

            // Check D3D12 support
            if D3D12CreateDevice(
                &adapter,
                D3D_FEATURE_LEVEL_12_0,
                std::ptr::null_mut::<Option<ID3D12Device>>(),
            )
            .is_ok()
            {
                let name = String::from_utf16_lossy(&desc.Description);
                println!("🎮 DirectX 12 adapter: {}", name.trim_end_matches('\0'));
                return Ok(adapter);
            }
        }
        Err(Error::from_win32())
    }

    unsafe fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
        // Root parameter for CBV
        let root_parameters = [D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 64, // 256 bytes
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];

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
        )?;

        let signature_blob = signature_blob.unwrap();

        let root_signature: ID3D12RootSignature = device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(
                signature_blob.GetBufferPointer() as *const u8,
                signature_blob.GetBufferSize(),
            ),
        )?;

        Ok(root_signature)
    }

    unsafe fn create_pipeline_state(
        device: &ID3D12Device,
        root_signature: &ID3D12RootSignature,
    ) -> Result<ID3D12PipelineState> {
        let shader_src = include_str!("dx12_shader.hlsl");

        let mut vs_blob: Option<ID3DBlob> = None;
        let mut ps_blob: Option<ID3DBlob> = None;
        let mut error_blob: Option<ID3DBlob> = None;

        let entry_vs = s!("VSMain");
        let profile_vs = s!("vs_5_0");
        let entry_ps = s!("PSMain");
        let profile_ps = s!("ps_5_0");
        let flags = D3DCOMPILE_ENABLE_STRICTNESS;

        // Compile VS
        D3DCompile(
            shader_src.as_ptr() as *const c_void,
            shader_src.len(),
            None,
            None,
            None,
            entry_vs,
            profile_vs,
            flags,
            0,
            &mut vs_blob,
            Some(&mut error_blob),
        )
        .map_err(|e| {
            if let Some(err) = error_blob.as_ref() {
                let msg = std::slice::from_raw_parts(
                    err.GetBufferPointer() as *const u8,
                    err.GetBufferSize(),
                );
                let s = String::from_utf8_lossy(msg);
                println!("VS Compile Error: {}", s);
            }
            e
        })?;

        // Compile PS
        D3DCompile(
            shader_src.as_ptr() as *const c_void,
            shader_src.len(),
            None,
            None,
            None,
            entry_ps,
            profile_ps,
            flags,
            0,
            &mut ps_blob,
            Some(&mut error_blob),
        )
        .map_err(|e| {
            if let Some(err) = error_blob.as_ref() {
                let msg = std::slice::from_raw_parts(
                    err.GetBufferPointer() as *const u8,
                    err.GetBufferSize(),
                );
                let s = String::from_utf8_lossy(msg);
                println!("PS Compile Error: {}", s);
            }
            e
        })?;

        let input_element_descs = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("TEXCOORD"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 8,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("COLOR"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 16,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];

        let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(root_signature), // Unsafe copy for C compatible struct?
            // Wait, windows-rs expects Option<ID3D12RootSignature> reference logic?
            // Actually it takes ManuallyDrop or raw interface usually.
            // Let's use simple assignment if compatible.
            // checking Struct definition... it expects a raw pointer usually or interface.
            // In windows-rs 0.52 pRootSignature is ManuallyDrop<Option<ID3D12RootSignature>>.
            // So:
            // pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs_blob.as_ref().unwrap().GetBufferPointer(),
                BytecodeLength: vs_blob.as_ref().unwrap().GetBufferSize(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps_blob.as_ref().unwrap().GetBufferPointer(),
                BytecodeLength: ps_blob.as_ref().unwrap().GetBufferSize(),
            },
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                ..Default::default()
            },
            BlendState: D3D12_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [
                    D3D12_RENDER_TARGET_BLEND_DESC {
                        BlendEnable: true.into(),
                        LogicOpEnable: false.into(),
                        SrcBlend: D3D12_BLEND_SRC_ALPHA,
                        DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                        BlendOp: D3D12_BLEND_OP_ADD,
                        SrcBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                        DestBlendAlpha: D3D12_BLEND_ZERO,
                        BlendOpAlpha: D3D12_BLEND_OP_ADD,
                        LogicOp: D3D12_LOGIC_OP_NOOP,
                        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    },
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                ],
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: false.into(),
                StencilEnable: false.into(),
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: input_element_descs.as_ptr(),
                NumElements: input_element_descs.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: [
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
            ],
            SampleMask: u32::MAX,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };

        pso_desc.pRootSignature = ManuallyDrop::new(Some(root_signature.clone()));

        device.CreateGraphicsPipelineState(&pso_desc)
    }

    unsafe fn create_upload_buffer(device: &ID3D12Device, size: u64) -> Result<ID3D12Resource> {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };

        let buffer_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };

        let mut resource: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut resource,
        )?;

        Ok(resource.unwrap())
    }

    fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        [
            [2.0 / (right - left), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (top - bottom), 0.0, 0.0],
            [0.0, 0.0, -2.0 / (far - near), 0.0],
            [tx, ty, tz, 1.0],
        ]
    }

    fn quad_vertices(pos: Vec2, size: Vec2, color: ColorF) -> [Vertex; 6] {
        Self::quad_vertices_uv(pos, size, [0.0, 0.0, 1.0, 1.0], color)
    }

    fn quad_vertices_uv(pos: Vec2, size: Vec2, uv: [f32; 4], color: ColorF) -> [Vertex; 6] {
        let (x0, y0) = (pos.x, pos.y);
        let (x1, y1) = (pos.x + size.x, pos.y + size.y);
        let c = [color.r, color.g, color.b, color.a];
        let (u0, v0, u1, v1) = (uv[0], uv[1], uv[2], uv[3]);

        [
            Vertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color: c,
            },
            Vertex {
                pos: [x0, y1],
                uv: [u0, v1],
                color: c,
            },
            Vertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color: c,
            },
            Vertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color: c,
            },
            Vertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color: c,
            },
            Vertex {
                pos: [x1, y0],
                uv: [u1, v0],
                color: c,
            },
        ]
    }

    unsafe fn wait_for_gpu(&self) {
        let frame_idx = self.frame_index.get();
        let mut values = self.fence_values.get();
        let fence_value = values[frame_idx] + 1;
        self.command_queue.Signal(&self.fence, fence_value).unwrap();
        values[frame_idx] = fence_value;
        self.fence_values.set(values);

        if self.fence.GetCompletedValue() < fence_value {
            self.fence
                .SetEventOnCompletion(fence_value, self.fence_event)
                .unwrap();
            let _ = WaitForSingleObject(self.fence_event, u32::MAX);
        }
    }

    unsafe fn move_to_next_frame(&self) {
        let frame_idx = self.frame_index.get();
        let mut values = self.fence_values.get();
        let current_fence_value = values[frame_idx];
        self.command_queue
            .Signal(&self.fence, current_fence_value)
            .unwrap();

        let new_frame_idx = self.swap_chain.GetCurrentBackBufferIndex() as usize;
        self.frame_index.set(new_frame_idx);

        if self.fence.GetCompletedValue() < values[new_frame_idx] {
            self.fence
                .SetEventOnCompletion(values[new_frame_idx], self.fence_event)
                .unwrap();
            let _ = WaitForSingleObject(self.fence_event, u32::MAX);
        }

        values[new_frame_idx] = current_fence_value + 1;
        self.fence_values.set(values);
    }
}

impl super::Backend for Dx12Backend {
    fn name(&self) -> &str {
        "DirectX 12"
    }

    fn render(&mut self, dl: &DrawList, width: u32, height: u32) {
        unsafe {
            let frame_idx = self.frame_index.get();

            // Reset command allocator and command list
            self.command_allocators[frame_idx].Reset().unwrap();
            self.command_list
                .Reset(&self.command_allocators[frame_idx], None)
                .unwrap();

            // Transition render target to render target state
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(self.render_targets[frame_idx].clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_PRESENT,
                        StateAfter: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    }),
                },
            };
            self.command_list.ResourceBarrier(&[barrier]);

            // Get RTV handle
            let mut rtv_handle = self.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            rtv_handle.ptr += frame_idx * self.rtv_descriptor_size as usize;

            // Clear render target
            let clear_color: [f32; 4] = [0.08, 0.08, 0.1, 1.0];
            self.command_list
                .ClearRenderTargetView(rtv_handle, &clear_color, None);

            // Set render target
            self.command_list
                .OMSetRenderTargets(1, Some(&rtv_handle), false, None);

            // Set viewport and scissor
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            self.command_list.RSSetViewports(&[viewport]);

            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            self.command_list.RSSetScissorRects(&[scissor]);

            // Map vertex buffer once for writing
            let mut vb_ptr: *mut c_void = std::ptr::null_mut();
            self.vertex_buffer.Map(0, None, Some(&mut vb_ptr)).unwrap();
            let vb_base = vb_ptr as *mut Vertex;
            let mut vb_offset = 0;

            // Set pipeline
            if let Some(ref pso) = self.pipeline_state {
                self.command_list.SetPipelineState(pso);
                self.command_list
                    .SetGraphicsRootSignature(&self.root_signature);
                self.command_list
                    .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                self.command_list
                    .IASetVertexBuffers(0, Some(&[self.vertex_buffer_view]));

                let projection = Self::ortho(0.0, width as f32, height as f32, 0.0, -1.0, 1.0);

                // Iterate commands
                for cmd in dl.commands() {
                    let mut vertices: [Vertex; 6] = [Vertex {
                        pos: [0.0; 2],
                        uv: [0.0; 2],
                        color: [0.0; 4],
                    }; 6];
                    let mut has_draw = false;

                    let mut cb_data = ConstantBuffer {
                        projection,
                        rect: [0.0; 4],
                        radii: [0.0; 4],
                        border_color: [0.0; 4],
                        glow_color: [0.0; 4],
                        mode: 0,
                        border_width: 0.0,
                        elevation: 0.0,
                        is_squircle: 0,
                        glow_strength: 0.0,
                        _padding: [0.0; 27],
                    };

                    match cmd {
                        DrawCommand::RoundedRect {
                            pos,
                            size,
                            radii,
                            color,
                            elevation,
                            is_squircle,
                            border_width,
                            border_color,
                            glow_strength,
                            glow_color,
                            ..
                        } => {
                            vertices = Self::quad_vertices(*pos, *size, *color);
                            has_draw = true;
                            cb_data.rect = [pos.x, pos.y, size.x, size.y];
                            cb_data.radii = *radii;
                            cb_data.border_color = [
                                border_color.r,
                                border_color.g,
                                border_color.b,
                                border_color.a,
                            ];
                            cb_data.glow_color =
                                [glow_color.r, glow_color.g, glow_color.b, glow_color.a];
                            cb_data.mode = 2; // Shape
                            cb_data.border_width = *border_width;
                            cb_data.elevation = *elevation;
                            cb_data.is_squircle = if *is_squircle { 1 } else { 0 };
                            cb_data.glow_strength = *glow_strength;
                        }
                        DrawCommand::Text {
                            pos,
                            size,
                            uv,
                            color,
                        } => {
                            vertices = Self::quad_vertices_uv(*pos, *size, *uv, *color);
                            has_draw = true;
                            cb_data.mode = 1; // Text
                        }
                        _ => {}
                    }

                    if has_draw {
                        // Copy vertices to mapped buffer at current offset
                        let dest = vb_base.add(vb_offset);
                        std::ptr::copy_nonoverlapping(vertices.as_ptr(), dest, 6);

                        // Set constants
                        let constants = std::slice::from_raw_parts(
                            &cb_data as *const _ as *const u32,
                            std::mem::size_of::<ConstantBuffer>() / 4,
                        );
                        // Note: ConstantBuffer is 256 bytes = 64 u32s.
                        self.command_list.SetGraphicsRoot32BitConstants(
                            0,
                            constants.len() as u32,
                            constants.as_ptr() as *const _,
                            0,
                        );

                        // Draw
                        self.command_list.DrawInstanced(6, 1, vb_offset as u32, 0);

                        vb_offset += 6;
                    }
                }
            }

            self.vertex_buffer.Unmap(0, None);

            // Transition back to present
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(self.render_targets[frame_idx].clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        StateAfter: D3D12_RESOURCE_STATE_PRESENT,
                    }),
                },
            };
            self.command_list.ResourceBarrier(&[barrier]);

            self.command_list.Close().unwrap();
            let command_lists = [Some(self.command_list.cast::<ID3D12CommandList>().unwrap())];
            self.command_queue.ExecuteCommandLists(&command_lists);
            let _ = self.swap_chain.Present(1, 0);
            self.move_to_next_frame();
        }
    }
}

impl Drop for Dx12Backend {
    fn drop(&mut self) {
        unsafe {
            self.wait_for_gpu();
            CloseHandle(self.fence_event).unwrap();
        }
    }
}
