use metal::*;
use std::sync::Arc;

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
        vs_name: &str,
        fs_name: &str,
        pixel_format: MTLPixelFormat,
    ) -> Result<RenderPipelineState, String> {
        let vs = self.library.get_function(vs_name, None)
            .map_err(|_| format!("Vertex function {} not found", vs_name))?;
        let fs = self.library.get_function(fs_name, None)
            .map_err(|_| format!("Fragment function {} not found", fs_name))?;

        let desc = RenderPipelineDescriptor::new();
        desc.set_label(label);
        desc.set_vertex_function(Some(&vs));
        desc.set_fragment_function(Some(&fs));
        
        let vertex_desc = VertexDescriptor::new();
        
        // Attribute 0: pos (float2)
        let attr0 = vertex_desc.attributes().object_at(0).ok_or("No attribute 0")?;
        attr0.set_format(MTLVertexFormat::Float2);
        attr0.set_offset(0);
        attr0.set_buffer_index(0);
        
        // Attribute 1: uv (float2)
        let attr1 = vertex_desc.attributes().object_at(1).ok_or("No attribute 1")?;
        attr1.set_format(MTLVertexFormat::Float2);
        attr1.set_offset(8);
        attr1.set_buffer_index(0);
        
        // Attribute 2: color (float4)
        let attr2 = vertex_desc.attributes().object_at(2).ok_or("No attribute 2")?;
        attr2.set_format(MTLVertexFormat::Float4);
        attr2.set_offset(16);
        attr2.set_buffer_index(0);
        
        let layout = vertex_desc.layouts().object_at(0).ok_or("No layout 0")?;
        layout.set_stride(32);
        layout.set_step_function(MTLVertexStepFunction::PerVertex);
        
        desc.set_vertex_descriptor(Some(&vertex_desc));
        
        let color_attachment = desc.color_attachments().object_at(0).ok_or("No color attachment")?;
        color_attachment.set_pixel_format(pixel_format);
        color_attachment.set_blending_enabled(true);
        color_attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        color_attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        self.device.new_render_pipeline_state(&desc)
            .map_err(|e| format!("Failed to create Metal render pipeline: {}", e))
    }
}
