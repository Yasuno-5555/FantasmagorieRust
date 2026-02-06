use metal::*;
use std::sync::Arc;


#[derive(Debug, Clone)]
pub struct MetalBindGroupLayout {
    pub entries: Vec<u32>, // Binding indices
}

#[derive(Debug, Clone)]
pub enum MetalResource {
    Buffer(Buffer),
    Texture(Texture),
    Sampler(SamplerState),
}

#[derive(Debug, Clone)]
pub struct MetalBindGroupEntry {
    pub binding: u32,
    pub resource: MetalResource,
}

#[derive(Debug, Clone)]
pub struct MetalBindGroup {
    pub entries: Vec<MetalBindGroupEntry>,
}

pub struct MetalPipelineProvider {
    device: Device,
    library: Library,
}

impl MetalPipelineProvider {
    pub fn new(device: Device, msl_source: &str) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device.new_library_with_source(msl_source, &options)
            .map_err(|e| format!("Failed to compile Metal shaders: {}", e))?;
        
        Ok(Self { device, library })
    }

    pub fn create_render_pipeline(
        &self,
        label: &str,
        source_or_funcs: &str,
        _layout: Option<&MetalBindGroupLayout>,
    ) -> Result<RenderPipelineState, String> {
        let is_source = source_or_funcs.contains('{') || source_or_funcs.contains('#');
        
        let (vs_name, fs_name, library) = if is_source {
            // Compile temporary library
            let options = CompileOptions::new();
            let lib = self.device.new_library_with_source(source_or_funcs, &options)
                .map_err(|e| format!("Failed to compile MSL for '{}': {}", label, e))?;
                
            // Detect entry points
            let (vs_n, fs_n) = if let Some(line) = source_or_funcs.lines().find(|l| l.contains("ENTRY:")) {
                let parts: Vec<&str> = line.split("ENTRY:").last().unwrap().split_whitespace().collect();
                if parts.len() >= 2 {
                    (parts[0].to_string(), parts[1].to_string())
                } else {
                    ("vs_main".to_string(), "fs_main".to_string())
                }
            } else {
                // Heuristic for common names if not specified
                let vs = if source_or_funcs.contains("vs_particles") { "vs_particles" } 
                         else if source_or_funcs.contains("vs_main") { "vs_main" }
                         else { "vs_main" };
                let fs = if source_or_funcs.contains("fs_particles") { "fs_particles" }
                         else if source_or_funcs.contains("fs_main") { "fs_main" }
                         else { "fs_main" };
                (vs.to_string(), fs.to_string())
            };
            (vs_n, fs_n, lib)
        } else {
            let parts: Vec<&str> = source_or_funcs.split_whitespace().collect();
            if parts.len() < 2 {
                return Err(format!("Expected 'vs_func fs_func', got '{}'", source_or_funcs));
            }
            (parts[0].to_string(), parts[1].to_string(), self.library.clone())
        };
        
        let vs = library.get_function(&vs_name, None).map_err(|e| format!("VS '{}' not found in '{}': {}", vs_name, label, e))?;
        let fs = library.get_function(&fs_name, None).map_err(|e| format!("FS '{}' not found in '{}': {}", fs_name, label, e))?;
        
        let desc = RenderPipelineDescriptor::new();
        desc.set_label(label);
        desc.set_vertex_function(Some(&vs));
        desc.set_fragment_function(Some(&fs));
        
        let color_attachment = desc.color_attachments().object_at(0).unwrap();
        
        // HEURISTIC: Use RGBA16Float for offscreen/HDR passes, BGRA8Unorm for final ones
        if label.contains("Main") || label.contains("Bright") || label.contains("Blur") || label.contains("Instanced") || label.contains("Particle") {
             color_attachment.set_pixel_format(MTLPixelFormat::RGBA16Float);
        } else {
             color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        }

        color_attachment.set_blending_enabled(true);
        color_attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        color_attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        
        // Resolve pass doesn't need blending (it's full screen)
        if label.contains("Resolve") || label.contains("Blur") || label.contains("Bright") {
             color_attachment.set_blending_enabled(false);
        }

        // Determine if we need a vertex descriptor based on the vertex function name
        // (vs_main, vs_instanced need it; vs_resolve, vs_particles, vs_post, vs_ssr don't)
        let vs_n_lower = vs_name.to_lowercase();
        let needs_vertex_desc = vs_n_lower.contains("main") || vs_n_lower.contains("instanced");
        
        if needs_vertex_desc {
            let vertex_desc = VertexDescriptor::new();
            
            // Attribute 0: pos (float2)
            let attr0 = vertex_desc.attributes().object_at(0).unwrap();
            attr0.set_format(MTLVertexFormat::Float2);
            attr0.set_offset(0);
            attr0.set_buffer_index(1); // Use slot 1 for vertices
            
            // Attribute 1: uv (float2)
            let attr1 = vertex_desc.attributes().object_at(1).unwrap();
            attr1.set_format(MTLVertexFormat::Float2);
            attr1.set_offset(8);
            attr1.set_buffer_index(1);
            
            // Attribute 2: color (float4)
            let attr2 = vertex_desc.attributes().object_at(2).unwrap();
            attr2.set_format(MTLVertexFormat::Float4);
            attr2.set_offset(16);
            attr2.set_buffer_index(1);
            
            let layout1 = vertex_desc.layouts().object_at(1).unwrap(); // Use layout 1
            layout1.set_stride(32);
            
            desc.set_vertex_descriptor(Some(&vertex_desc));
        } else if vs_n_lower.contains("ssr") || vs_n_lower.contains("post") || vs_n_lower.contains("resolve") {
            // Vertex attributes for SSR fullscreen pass (or other fullscreen passes)
            let vertex_desc = VertexDescriptor::new();
            let attr0 = vertex_desc.attributes().object_at(0).unwrap();
            attr0.set_format(MTLVertexFormat::Float2);
            attr0.set_offset(0);
            attr0.set_buffer_index(0); // Use slot 0 for fullscreen vertices
            
            let attr1 = vertex_desc.attributes().object_at(1).unwrap();
            attr1.set_format(MTLVertexFormat::Float2);
            attr1.set_offset(8);
            attr1.set_buffer_index(0); // Use slot 0 for fullscreen vertices
            
            let layout0 = vertex_desc.layouts().object_at(0).unwrap();
            layout0.set_stride(16); // float2 pos + float2 uv = 16 bytes
            desc.set_vertex_descriptor(Some(&vertex_desc));
        }
        self.device.new_render_pipeline_state(&desc).map_err(|e| e.to_string())
    }

    pub fn create_gbuffer_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
    ) -> Result<RenderPipelineState, String> {
        let parts: Vec<&str> = wgsl_source.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("Expected 'vs_func fs_func', got '{}'", wgsl_source));
        }
        
        let vs_name = parts[0];
        let fs_name = parts[1];

        let vs = self.library.get_function(vs_name, None).map_err(|e| format!("VS '{}' not found: {}", vs_name, e))?;
        let fs = self.library.get_function(fs_name, None).map_err(|e| format!("FS '{}' not found: {}", fs_name, e))?;
        
        let desc = RenderPipelineDescriptor::new();
        desc.set_label(label);
        desc.set_vertex_function(Some(&vs));
        desc.set_fragment_function(Some(&fs));
        
        // Attachment 0: HDR (Blend enabled)
        let color0 = desc.color_attachments().object_at(0).unwrap();
        color0.set_pixel_format(MTLPixelFormat::RGBA16Float);
        color0.set_blending_enabled(true);
        color0.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        color0.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        // Attachment 1: Aux (Replace)
        let color1 = desc.color_attachments().object_at(1).unwrap();
        color1.set_pixel_format(MTLPixelFormat::RGBA16Float);
        color1.set_blending_enabled(false); 

        // Attachment 2: Velocity (Replace)
        let color2 = desc.color_attachments().object_at(2).unwrap();
        color2.set_pixel_format(MTLPixelFormat::RG16Float);
        color2.set_blending_enabled(false);

        let vertex_desc = VertexDescriptor::new();
        
        // Attribute 0: pos (float2)
        let attr0 = vertex_desc.attributes().object_at(0).unwrap();
        attr0.set_format(MTLVertexFormat::Float2); attr0.set_offset(0); attr0.set_buffer_index(0);
        
        // Attribute 1: uv (float2)
        let attr1 = vertex_desc.attributes().object_at(1).unwrap();
        attr1.set_format(MTLVertexFormat::Float2); attr1.set_offset(8); attr1.set_buffer_index(0);
        
        // Attribute 2: color (float4)
        let attr2 = vertex_desc.attributes().object_at(2).unwrap();
        attr2.set_format(MTLVertexFormat::Float4); attr2.set_offset(16); attr2.set_buffer_index(0);
        
        let layout0 = vertex_desc.layouts().object_at(0).unwrap();
        layout0.set_stride(32);
        
        desc.set_vertex_descriptor(Some(&vertex_desc));
        
        // Depth Stencil
        desc.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
        
        self.device.new_render_pipeline_state(&desc).map_err(|e| e.to_string())
    }

    pub fn create_ssr_pipeline(
    &self,
        label: &str,
        wgsl_source: &str,
    ) -> Result<RenderPipelineState, String> {
        let parts: Vec<&str> = wgsl_source.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("Expected 'vs_func fs_func', got '{}'", wgsl_source));
        }
        
        let vs_name = parts[0];
        let fs_name = parts[1];

        let vs = self.library.get_function(vs_name, None).map_err(|e| format!("VS '{}' not found: {}", vs_name, e))?;
        let fs = self.library.get_function(fs_name, None).map_err(|e| format!("FS '{}' not found: {}", fs_name, e))?;
        
        let desc = RenderPipelineDescriptor::new();
        desc.set_label(label);
        desc.set_vertex_function(Some(&vs));
        desc.set_fragment_function(Some(&fs));
        
        let color0 = desc.color_attachments().object_at(0).unwrap();
        color0.set_pixel_format(MTLPixelFormat::RGBA16Float);
        color0.set_blending_enabled(true);
        color0.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        color0.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        self.device.new_render_pipeline_state(&desc).map_err(|e| e.to_string())
    }

    pub fn create_compute_pipeline(
        &self,
        label: &str,
        shader_source: &str, // Assumed MSL if feature=metal
        entry_point: Option<&str>,
    ) -> Result<ComputePipelineState, String> {
        let options = CompileOptions::new();
        let library = self.device.new_library_with_source(shader_source, &options)
            .map_err(|e| format!("Failed to compile compute shader '{}': {}", label, e))?;
            
        let name = entry_point.unwrap_or("main");
        let kernel = library.get_function(name, None)
            .map_err(|e| format!("Kernel '{}' not found in '{}': {}", name, label, e))?;
            
        let desc = ComputePipelineDescriptor::new();
        desc.set_label(label);
        desc.set_compute_function(Some(&kernel));
        
        self.device.new_compute_pipeline_state(&desc)
            .map_err(|e| format!("Failed to create compute pipeline state '{}': {}", label, e))
    }

    pub fn destroy_bind_group(&self, _bind_group: MetalBindGroup) {
        // Automatic via drop
    }
}
