pub struct WgpuProfiler {
    pub query_set: Option<wgpu::QuerySet>,
    pub resolve_buffer: Option<wgpu::Buffer>,
    pub readback_buffer: Option<wgpu::Buffer>,
}

impl WgpuProfiler {
    pub fn new(device: &wgpu::Device) -> Self {
        let query_set = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Main Query Set"),
                count: 8,
                ty: wgpu::QueryType::Timestamp,
            }))
        } else {
            None
        };

        let resolve_buffer = query_set.as_ref().map(|_| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Resolve Buffer"),
            size: 8 * 8, // 8 queries * 8 bytes
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let readback_buffer = query_set.as_ref().map(|_| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Readback Buffer"),
            size: 8 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(qs), Some(resolve_buf), Some(readback_buf)) = (&self.query_set, &self.resolve_buffer, &self.readback_buffer) {
            self.timestamp(encoder, 1); // End of Frame Timestamp
            
            encoder.resolve_query_set(
                qs,
                0..8,
                resolve_buf,
                0,
            );
            
            encoder.copy_buffer_to_buffer(
                resolve_buf,
                0,
                readback_buf,
                0,
                resolve_buf.size(),
            );
        }
    }

    pub fn timestamp(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, index);
        }
    }

    pub fn get_results(&self) -> Option<Vec<u64>> {
        if let Some(readback_buf) = &self.readback_buffer {
            let slice = readback_buf.slice(..);
            // In a real scenario, we'd check if the mapping is ready.
            // For now, we assume this is called after map_readback and a poll.
            let data = slice.get_mapped_range();
            let timestamps: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buf.unmap();
            return Some(timestamps);
        }
        None
    }

    pub fn map_readback(&self) {
        if let Some(readback_buf) = &self.readback_buffer {
            let slice = readback_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
        }
    }
}
