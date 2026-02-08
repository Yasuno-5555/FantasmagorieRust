use crate::backend::hal::GpuExecutor;
use crate::backend::GraphicsBackend;
use crate::renderer::graph::{
    RenderGraph, 
    HDR_HANDLE, BACKDROP_HANDLE, VELOCITY_HANDLE, AUX_HANDLE, 
    DEPTH_HANDLE, REFLECTION_HANDLE, EXTRA_HANDLE, SDF_HANDLE,
    SDF_TEMP_A_HANDLE, SDF_TEMP_B_HANDLE, 
};
use crate::renderer::nodes::geometry::{GeometryNode};
use crate::renderer::nodes::postprocess::{ResolveNode};
use crate::renderer::nodes::ssr::SSRNode;
use crate::renderer::nodes::particles::{ParticleNode, ParticleSystem};
use std::sync::{Arc, Mutex};

pub struct Orchestrator<E: GpuExecutor> {
    pub graph: RenderGraph<E>,
    particle_system: Arc<Mutex<ParticleSystem>>,
    pub frame_count: u64,
    camera_cut: bool,
}

impl<E: GpuExecutor + 'static> Orchestrator<E> {
    pub fn new() -> Self {
        Self {
            graph: RenderGraph::new(),
            particle_system: Arc::new(Mutex::new(ParticleSystem::new())),
            frame_count: 0,
            camera_cut: false,
        }
    }

    /// Mark a camera cut for this frame (used by temporal effects like MetalFX)
    pub fn mark_camera_cut(&mut self) {
        self.camera_cut = true;
    }

    pub fn plan(&mut self, dl: &crate::draw::DrawList, width: u32, height: u32, scale: f32) {
        self.graph = RenderGraph::new();

        // 1. Process Draw List into Geometry Nodes
        let mut current_batch = Vec::new();
        for command in dl.commands() {
            current_batch.push(command.clone());
        }
        
        if !current_batch.is_empty() {
            self.graph.add_node(GeometryNode::new(current_batch)
                .with_aux(AUX_HANDLE)
                .with_velocity(VELOCITY_HANDLE)
                .with_depth(DEPTH_HANDLE));
        }

        // 2. Add SSR Node (Aux, Depth, HDR)
        self.graph.add_node(SSRNode::new(
            AUX_HANDLE,
            DEPTH_HANDLE,
            HDR_HANDLE,
        ));

        // 3. JFA SDF Generation
        // Declare transient resources for JFA
        self.graph.resources.insert(SDF_TEMP_A_HANDLE, crate::renderer::graph::GraphResourceDesc::Texture(crate::backend::hal::TextureDescriptor {
            label: Some("SDF Temp A"), width, height, depth: 1,
            format: crate::backend::hal::TextureFormat::Rgba16Float, // Rgba16Float is usually enough for packed coords
            usage: crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::STORAGE_BINDING,
        }));
        self.graph.resources.insert(SDF_TEMP_B_HANDLE, crate::renderer::graph::GraphResourceDesc::Texture(crate::backend::hal::TextureDescriptor {
            label: Some("SDF Temp B"), width, height, depth: 1,
            format: crate::backend::hal::TextureFormat::Rgba16Float,
            usage: crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::STORAGE_BINDING,
        }));
        self.graph.resources.insert(SDF_HANDLE, crate::renderer::graph::GraphResourceDesc::Texture(crate::backend::hal::TextureDescriptor {
            label: Some("SDF Output"), width, height, depth: 1,
            format: crate::backend::hal::TextureFormat::Rgba16Float,
            usage: crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::STORAGE_BINDING,
        }));
        
        self.graph.resources.insert(REFLECTION_HANDLE, crate::renderer::graph::GraphResourceDesc::Texture(crate::backend::hal::TextureDescriptor {
            label: Some("Reflection Buffer"), width, height, depth: 1,
            format: crate::backend::hal::TextureFormat::Rgba16Float,
            usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::COPY_SRC,
        }));

        self.graph.add_node(crate::renderer::nodes::jfa::JfaSdfNode::new(EXTRA_HANDLE));
        // 4. Particle System
        self.graph.add_node(ParticleNode::new(self.particle_system.clone()));

        let internal_width = (width as f32 * scale) as u32;
        let internal_height = (height as f32 * scale) as u32;

        // 5. Lighting Pass
        self.graph.resources.insert(crate::renderer::graph::HDR_LOW_RES_HANDLE, crate::renderer::graph::GraphResourceDesc::Texture(crate::backend::hal::TextureDescriptor {
            label: Some("HDR Low Res"), width: internal_width, height: internal_height, depth: 1,
            format: crate::backend::hal::TextureFormat::Rgba16Float,
            usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING,
        }));
        
        self.graph.resources.insert(crate::renderer::graph::HDR_HIGH_RES_HANDLE, crate::renderer::graph::GraphResourceDesc::Texture(crate::backend::hal::TextureDescriptor {
            label: Some("HDR High Res"), width: width, height: height, depth: 1, // Output of upscale is native
            format: crate::backend::hal::TextureFormat::Rgba16Float,
            usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING,
        }));

        self.graph.add_node(crate::renderer::nodes::lighting::LightingNode);
        self.graph.add_node(crate::renderer::nodes::upscale::UpscaleNode);
        self.graph.add_node(crate::renderer::nodes::bloom::BloomNode);
        self.graph.add_node(crate::renderer::nodes::post::PostProcessNode);
    }

    pub fn execute(&mut self, backend: &mut E, time: f32, width: u32, height: u32, jitter: (f32, f32)) -> Result<(), String> {
        let mut external = std::collections::HashMap::new();
        
        // Pass backend-owned textures as external resources to the graph if they exist
        if let Some(hdr_tex) = backend.get_hdr_texture() {
            let hdr_desc = crate::backend::hal::TextureDescriptor {
                 label: Some("HDR Buffer"), width, height, depth: 1,
                 format: crate::backend::hal::TextureFormat::Rgba16Float,
                 usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::COPY_SRC,
            };
            external.insert(HDR_HANDLE, crate::renderer::graph::GraphResource::Texture(hdr_desc, hdr_tex));
        }

        if let Some(backdrop_tex) = backend.get_backdrop_texture() {
            let backdrop_desc = crate::backend::hal::TextureDescriptor {
                 label: Some("Backdrop Buffer"), width, height, depth: 1,
                 format: crate::backend::hal::TextureFormat::Rgba16Float,
                 usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING | crate::backend::hal::TextureUsage::COPY_DST,
            };
            external.insert(BACKDROP_HANDLE, crate::renderer::graph::GraphResource::Texture(backdrop_desc, backdrop_tex));
        }

        if let Some(extra_tex) = backend.get_extra_texture() {
            let extra_desc = crate::backend::hal::TextureDescriptor {
                 label: Some("Extra Buffer"), width, height, depth: 1,
                 format: crate::backend::hal::TextureFormat::Rgba16Float,
                 usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING,
            };
            external.insert(EXTRA_HANDLE, crate::renderer::graph::GraphResource::Texture(extra_desc, extra_tex));
        }

        if let Some(aux_tex) = backend.get_aux_texture() {
            let aux_desc = crate::backend::hal::TextureDescriptor {
                 label: Some("Aux Buffer"), width, height, depth: 1,
                 format: crate::backend::hal::TextureFormat::Rgba16Float,
                 usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING,
            };
            external.insert(AUX_HANDLE, crate::renderer::graph::GraphResource::Texture(aux_desc, aux_tex));
        }

        if let Some(vel_tex) = backend.get_velocity_texture() {
            let vel_desc = crate::backend::hal::TextureDescriptor {
                 label: Some("Velocity Buffer"), width, height, depth: 1,
                 format: crate::backend::hal::TextureFormat::Rgba16Float,
                 usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING,
            };
            external.insert(VELOCITY_HANDLE, crate::renderer::graph::GraphResource::Texture(vel_desc, vel_tex));
        }

        if let Some(depth_tex) = backend.get_depth_texture() {
            let depth_desc = crate::backend::hal::TextureDescriptor {
                 label: Some("Depth Buffer"), width, height, depth: 1,
                 format: crate::backend::hal::TextureFormat::Depth32Float,
                 usage: crate::backend::hal::TextureUsage::RENDER_ATTACHMENT | crate::backend::hal::TextureUsage::TEXTURE_BINDING,
            };
            external.insert(DEPTH_HANDLE, crate::renderer::graph::GraphResource::Texture(depth_desc, depth_tex));
        }

        if let Some(lut_tex) = backend.get_lut_texture() {
            // Assume 32x32x32 for now, or infer? HAL Texture doesn't have getters, so we trust backend or use fixed size
            let lut_desc = crate::backend::hal::TextureDescriptor {
                 label: Some("LUT"), width: 32, height: 32, depth: 32,
                 format: crate::backend::hal::TextureFormat::Rgba8Unorm, // Or Rgba16Float
                 usage: crate::backend::hal::TextureUsage::TEXTURE_BINDING,
            };
            external.insert(crate::renderer::graph::LUT_HANDLE, crate::renderer::graph::GraphResource::Texture(lut_desc, lut_tex));
        }

        // FXAA / LDR Intermediate
        // We always allocate it if using RenderGraph, or check config?
        // Allocating it every frame as "External" via backend.get_... is not right if it's transient.
        // It should be internal "Transient".
        // BUT Orchestrator logic here populates "external" resource map from backend resources.
        // Internal transient resources are allocated by RenderGraph execution based on node requirements.
        // Wait, does RenderGraph allocate transient handles?
        // Standard RenderGraph: Nodes declare outputs, Graph allocates them.
        // `PostProcessNode` outputs to `LDR_HANDLE`.
        // So I don't need to manually insert it here IF RenderGraph handles internal resources.
        // Let's check RenderGraph implementation.

        let result = self.graph.execute(backend, external, time, width, height, jitter, self.camera_cut);
        self.camera_cut = false; // Reset after frame
        result
    }
}
