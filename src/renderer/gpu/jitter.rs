/// Halton sequence generator for temporal anti-aliasing and upscaling.
pub struct Halton {
    pub x: f32,
    pub y: f32,
}

impl Halton {
    pub fn new(index: u32) -> Self {
        Self {
            x: halton(index + 1, 2) - 0.5,
            y: halton(index + 1, 3) - 0.5,
        }
    }
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut f = 1.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

/// Returns a set of jitter offsets for a sequence of frames.
pub fn get_jitter_offset(frame_index: u32, width: u32, height: u32) -> (f32, f32) {
    let h = Halton::new(frame_index % 16);
    (h.x / width as f32, h.y / height as f32)
}
