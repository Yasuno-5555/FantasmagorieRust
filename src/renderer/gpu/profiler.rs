//! Tracea GPU Profiler Module
//! 
//! Handles GPU high-resolution timestamps and per-kernel execution metrics.

use std::sync::atomic::{AtomicU64, Ordering};

/// Kernel execution metrics shared between GPU and CPU
/// Synchronized via buffer mapping or atomics
#[repr(C)]
#[derive(Debug, Default)]
pub struct KernelStats {
    pub k1_sprite_batch: u64,
    pub k4_cinematic_resolver: u64,
    pub k5_sdf_lighting: u64,
    pub k6_particle_lifecycle: u64,
    pub k8_visibility_culling: u64,
    pub k10_temporal_history: u64,
    pub k12_audio_reactive: u64,
    pub k13_indirect_dispatch: u64,
    
    pub frame_time_ns: u64,
    pub draw_call_count: u32,
    pub dispatch_call_count: u32,
}

/// Thread-safe accumulator for profiling data on the CPU side
pub struct ProfilerRegistry {
    pub stats: KernelStats,
}

impl ProfilerRegistry {
    pub fn new() -> Self {
        Self {
            stats: KernelStats::default(),
        }
    }
    
    pub fn reset(&mut self) {
        self.stats = KernelStats::default();
    }
}
