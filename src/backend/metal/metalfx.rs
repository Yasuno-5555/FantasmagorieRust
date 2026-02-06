use metal::{DeviceRef, CommandBufferRef, TextureRef, MTLPixelFormat};
use objc::runtime::{Object, Class};
use objc::msg_send;
use objc::sel;
use objc::sel_impl;
use objc::class;
use std::ffi::c_void;

pub struct TemporalScaler {
    scaler: *mut Object,
}

unsafe impl Send for TemporalScaler {} // ObjC objects are ref counted and usually thread safe? Metal objects are.
unsafe impl Sync for TemporalScaler {}

impl TemporalScaler {
    pub fn new(
        device: &DeviceRef,
        input_width: usize,
        input_height: usize,
        output_width: usize,
        output_height: usize,
        color_format: MTLPixelFormat,
        depth_format: MTLPixelFormat,
        motion_format: MTLPixelFormat,
        output_format: MTLPixelFormat
    ) -> Option<Self> {
        unsafe {
            // Try to load MetalFX framework bundle dynamically
            // /System/Library/Frameworks/MetalFX.framework
            let bundle_path_str = "/System/Library/Frameworks/MetalFX.framework";
            // Create NSString (manual or via crate? metal crate has NSString/NSObject? No, usually separate)
            // We use simple generic way.
            // Actually, dlopen might be easier?
            // "link" arg in Cargo works?
            // To be safe, we assume it's available or we fail.
            // Let's rely on Class::get returning None if not loaded.
            // If strictly needed, we can use `NSBundle`.
            
            let desc_cls_name = "MTLFXTemporalScalerDescriptor";
            let mut desc_class = Class::get(desc_cls_name);
            
            if desc_class.is_none() {
                 // Try loading bundle
                 let bundle_cls = Class::get("NSBundle");
                 if let Some(bundle_cls) = bundle_cls {
                     let path = std::ffi::CString::new(bundle_path_str).unwrap();
                     let path_str: *mut Object = msg_send![class!(NSString), stringWithUTF8String: path.as_ptr()];
                     let bundle: *mut Object = msg_send![bundle_cls, bundleWithPath: path_str];
                     if !bundle.is_null() {
                         let _: () = msg_send![bundle, load];
                     }
                 }
                 desc_class = Class::get(desc_cls_name);
            }

            let desc_class = desc_class?;
            
            let desc: *mut Object = msg_send![desc_class, new]; // or alloc init
            
            let _: () = msg_send![desc, setInputWidth: input_width as u64];
            let _: () = msg_send![desc, setInputHeight: input_height as u64];
            let _: () = msg_send![desc, setOutputWidth: output_width as u64];
            let _: () = msg_send![desc, setOutputHeight: output_height as u64];
            let _: () = msg_send![desc, setColorTextureFormat: color_format as u64];
            let _: () = msg_send![desc, setDepthTextureFormat: depth_format as u64];
            let _: () = msg_send![desc, setMotionTextureFormat: motion_format as u64];
            let _: () = msg_send![desc, setOutputTextureFormat: output_format as u64];
            
            // let _: () = msg_send![desc, setInputContentProperties: ...]; // Optional properties
            
            let scaler: *mut Object = msg_send![desc, newTemporalScalerWithDevice: device];
            
            // Release desc? Rust side doesn't own it nicely?
            // `new` usually autoreleases or returns +1?
            // `newTemporalScaler...` returns +1 retained object.
            // We should ensure desc is released if we allocated it. `new` implies alloc+init.
            // But we don't have AutoreleasePool here easily.
            // Let's assume standard Cocoa memory management: if we `alloc` we `release`.
            // `new` is `alloc + init`. So we own it.
            // But usually convenient constructors `scalerWithDevice` are autoreleased.
            // This is `newTemporalScaler...`, starts with new, so it returns retained.
            
            // desc was created with `new`. We should release it?
            // Yes.
            let _: () = msg_send![desc, release]; // Manual release
            
            if scaler.is_null() {
                return None;
            }
            
            Some(Self { scaler })
        }
    }
    
    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        color: &TextureRef,
        depth: &TextureRef,
        motion: &TextureRef,
        output: &TextureRef,
        jitter_x: f32,
        jitter_y: f32,
        reset_history: bool
    ) {
        unsafe {
            if reset_history {
                let _: () = msg_send![self.scaler, setReset: true];
            } else {
                 let _: () = msg_send![self.scaler, setReset: false];
            }
            
            let _: () = msg_send![self.scaler, setJitterOffsetX: jitter_x];
            let _: () = msg_send![self.scaler, setJitterOffsetY: jitter_y];
            
            // Motion Vector scale is implicitly 1.0 (pixels) if not set?
            // Usually we set it to match the motion texture format.
            // Assuming pixel space motion vectors for now.
            
            let _: () = msg_send![self.scaler, setColorTexture: color];
            let _: () = msg_send![self.scaler, setDepthTexture: depth];
            let _: () = msg_send![self.scaler, setMotionTexture: motion];
            let _: () = msg_send![self.scaler, setOutputTexture: output];
            
            let _: () = msg_send![self.scaler, encodeToCommandBuffer: command_buffer];
        }
    }
}

impl Drop for TemporalScaler {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.scaler, release];
        }
    }
}
