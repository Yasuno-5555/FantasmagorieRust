use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};

/// A lock-free signal for synchronizing f32 values between threads (e.g. UI and Audio)
#[derive(Clone, Debug)]
pub struct SignalF32 {
    inner: Arc<AtomicU32>,
}

impl SignalF32 {
    pub fn new(val: f32) -> Self {
        Self {
            inner: Arc::new(AtomicU32::new(val.to_bits())),
        }
    }

    pub fn get(&self) -> f32 {
        f32::from_bits(self.inner.load(Ordering::Relaxed))
    }

    pub fn set(&self, val: f32) {
        self.inner.store(val.to_bits(), Ordering::Relaxed);
    }
}

/// A lock-free signal for synchronizing boolean values
#[derive(Clone, Debug)]
pub struct SignalBool {
    inner: Arc<AtomicBool>,
}

impl SignalBool {
    pub fn new(val: bool) -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(val)),
        }
    }

    pub fn get(&self) -> bool {
        self.inner.load(Ordering::Relaxed)
    }

    pub fn set(&self, val: bool) {
        self.inner.store(val, Ordering::Relaxed);
    }
}

/// Registry for signals to allow ID-based lookup in the Inspector
pub struct SignalRegistry {
    pub f32_signals: std::collections::HashMap<u64, SignalF32>,
    pub bool_signals: std::collections::HashMap<u64, SignalBool>,
}

impl SignalRegistry {
    pub fn new() -> Self {
        Self {
            f32_signals: std::collections::HashMap::new(),
            bool_signals: std::collections::HashMap::new(),
        }
    }
}
