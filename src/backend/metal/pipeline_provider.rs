use metal::*;
use std::sync::Arc;


pub struct MetalBindGroupLayout {
    pub entries: Vec<u32>, // Binding indices
}

pub struct MetalBindGroup {
    pub buffers: Vec<Buffer>,
    pub textures: Vec<Texture>,
    pub samplers: Vec<SamplerState>,
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
        wgsl_source: &str,
        _layout: Option<&MetalBindGroupLayout>,
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
        
        let color_attachment = desc.color_attachments().object_at(0).unwrap();
        
        // HEURISTIC: Use RGBA16Float for offscreen/HDR passes, BGRA8Unorm for final ones
        if label.contains("Main") || label.contains("Bright") || label.contains("Blur") || label.contains("Instanced") {
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

        let vertex_desc = VertexDescriptor::new();
        
        // Attribute 0: pos (float2)
        let attr0 = vertex_desc.attributes().object_at(0).unwrap();
        attr0.set_format(MTLVertexFormat::Float2);
        attr0.set_offset(0);
        attr0.set_buffer_index(0);
        
        // Attribute 1: uv (float2)
        let attr1 = vertex_desc.attributes().object_at(1).unwrap();
        attr1.set_format(MTLVertexFormat::Float2);
        attr1.set_offset(8);
        attr1.set_buffer_index(0);
        
        // Attribute 2: color (float4)
        let attr2 = vertex_desc.attributes().object_at(2).unwrap();
        attr2.set_format(MTLVertexFormat::Float4);
        attr2.set_offset(16);
        attr2.set_buffer_index(0);
        
        let layout0 = vertex_desc.layouts().object_at(0).unwrap();
        layout0.set_stride(32);
        
        desc.set_vertex_descriptor(Some(&vertex_desc));
        
        self.device.new_render_pipeline_state(&desc).map_err(|e| e.to_string())
    }

    pub fn create_compute_pipeline(
        &self,
        label: &str,
        _wgsl_source: &str,
        _layout: Option<&MetalBindGroupLayout>,
    ) -> Result<ComputePipelineState, String> {
        Err("WGSL to MSL transpilation NOT implemented for Metal HAL yet".into())
    }

    pub fn destroy_bind_group(&self, _bind_group: MetalBindGroup) {
        // Automatic via drop
    }
}
