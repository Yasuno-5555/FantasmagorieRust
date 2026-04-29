//! Tracea Context Management
//!
//! Manages the Tracea runtime context and device selection.

use std::sync::Arc;

/// Wrapper around Tracea's runtime context
pub struct TraceaContext {
    #[cfg(feature = "metal")]
    device: Option<metal::Device>,
    
    #[cfg(feature = "wgpu")]
    wgpu_device: Option<Arc<wgpu::Device>>,
    #[cfg(feature = "wgpu")]
    wgpu_queue: Option<Arc<wgpu::Queue>>,
    
    initialized: bool,
}

impl TraceaContext {
    /// Create a new Tracea context
    /// If device is provided, it uses it. Otherwise, autodetects.
    #[cfg(feature = "metal")]
    pub fn new(shared_metal_device: Option<Arc<metal::Device>>) -> Result<Self, String> {
        let mut ctx = Self {
            device: None,
            #[cfg(feature = "wgpu")]
            wgpu_device: None,
            #[cfg(feature = "wgpu")]
            wgpu_queue: None,
            initialized: false,
        };

        if let Some(d) = shared_metal_device {
            ctx.device = Some((*d).clone());
            ctx.initialized = true;
            println!("[TraceaContext] Initialized with Shared Metal Device");
        } else if let Some(d) = metal::Device::system_default() {
             ctx.device = Some(d);
             ctx.initialized = true;
             println!("[TraceaContext] Initialized with Default Metal Device");
        }
        
        Ok(ctx)
    }

    #[cfg(not(feature = "metal"))]
    pub fn new(_: Option<()>) -> Result<Self, String> {
        Ok(Self {
            #[cfg(feature = "wgpu")]
            wgpu_device: None,
            #[cfg(feature = "wgpu")]
            wgpu_queue: None,
            initialized: false,
        })
    }

    #[cfg(feature = "wgpu")]
    pub fn set_wgpu(&mut self, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
        self.wgpu_device = Some(device);
        self.wgpu_queue = Some(queue);
        self.initialized = true;
        println!("[TraceaContext] WGPU Backend Configured");
    }
    
    /// Check if context is ready for compute
    pub fn is_ready(&self) -> bool {
        self.initialized
    }
    
    #[cfg(feature = "metal")]
    pub fn device(&self) -> Option<&metal::Device> {
        self.device.as_ref()
    }

    #[cfg(feature = "metal")]
    pub fn has_metal_device(&self) -> bool {
        self.device.is_some()
    }

    #[cfg(feature = "wgpu")]
    pub fn wgpu_device(&self) -> Option<&wgpu::Device> {
        self.wgpu_device.as_ref().map(|v| &**v)
    }

    #[cfg(feature = "wgpu")]
    pub fn wgpu_queue(&self) -> Option<&wgpu::Queue> {
        self.wgpu_queue.as_ref().map(|v| &**v)
    }
}

impl Default for TraceaContext {
    fn default() -> Self {
        Self {
            #[cfg(feature = "metal")]
            device: None,
            #[cfg(feature = "wgpu")]
            wgpu_device: None,
            #[cfg(feature = "wgpu")]
            wgpu_queue: None,
            initialized: false,
        }
    }
}
