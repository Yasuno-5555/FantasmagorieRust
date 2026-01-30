use super::types::*;

/// A complete, immutable snapshot of a draw call for the GPU.
/// "Optimization is forbidden here."
pub struct DrawPacket {
    pub pipeline: PipelineHandle,
    pub vertex_buffer: BufferHandle,
    pub index_buffer: Option<BufferHandle>,
    // In real implementation, might be a list of descriptors or a specific set handle
    pub descriptor_set: DescriptorSetHandle,
    pub draw_range: DrawRange,
}

impl DrawPacket {
    // Helper to identify empty packets if needed
    pub fn is_empty(&self) -> bool {
        self.draw_range.count == 0
    }
}
