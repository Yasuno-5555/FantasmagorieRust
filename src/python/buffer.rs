use crate::python::dlpack::*;
use glow::HasContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyCapsule;
use std::ffi::c_void;
use std::ffi::CString;
use std::ptr::null;

#[repr(C)]
struct ManagedDLPackCtx {
    tensor: DLManagedTensor,
    shape: [i64; 1],
    strides: [i64; 1],
}

unsafe extern "C" fn dlpack_deleter(handle: *mut DLManagedTensor) {
    if !handle.is_null() {
        // Cast back to ManagedDLPackCtx
        // Since tensor is the first field, the pointers are identical
        let ctx = Box::from_raw(handle as *mut ManagedDLPackCtx);
        drop(ctx); // Free memory
    }
}

#[pyclass]
pub struct BufferView {
    pbo: glow::Buffer,
    size: usize,
    mapped_ptr: *mut u8,
}

// Safety: PBO pointer is raw. PyMemoryView handles safety for Python side.
// We must ensure GL context is current for GL calls.
unsafe impl Send for BufferView {}

#[pymethods]
impl BufferView {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        // Access GL from GLOBAL_RESOURCES
        // Only valid if called on the thread where init_frame/backend creation happened
        let gl = crate::core::resource::GLOBAL_RESOURCES
            .with(|res| res.borrow().gl.clone())
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "No active GL context found. Creates BufferView inside the render loop?",
                )
            })?;

        let pbo = unsafe {
            let buffer = gl.create_buffer().map_err(|e| PyRuntimeError::new_err(e))?;
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(buffer));

            // Allocate storage (STREAM_DRAW for frequent updates)
            gl.buffer_data_size(glow::PIXEL_UNPACK_BUFFER, size as i32, glow::STREAM_DRAW);

            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            buffer
        };

        Ok(BufferView {
            pbo,
            size,
            mapped_ptr: std::ptr::null_mut(),
        })
    }

    /// Map the buffer and return a Python MemoryView
    /// This allows direct writing from NumPy: `np.array(view, copy=False)`
    unsafe fn map(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let gl = crate::core::resource::GLOBAL_RESOURCES
            .with(|res| res.borrow().gl.clone())
            .ok_or_else(|| PyRuntimeError::new_err("No active GL context"))?;

        unsafe {
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(self.pbo));
        }

        let ptr = gl.map_buffer_range(
            glow::PIXEL_UNPACK_BUFFER,
            0,
            self.size as i32,
            glow::MAP_WRITE_BIT | glow::MAP_INVALIDATE_BUFFER_BIT,
        );

        unsafe { gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None) };

        if ptr.is_null() {
            return Err(PyRuntimeError::new_err("Failed to map PBO"));
        }

        self.mapped_ptr = ptr;

        // Create PyMemoryView using FFI
        let view_ptr =
            ffi::PyMemoryView_FromMemory(ptr as *mut i8, self.size as isize, ffi::PyBUF_WRITE);

        if view_ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "Failed to create Python MemoryView",
            ));
        }

        // Return owning PyObject
        Ok(PyObject::from_owned_ptr(py, view_ptr))
    }

    fn unmap(&mut self) -> PyResult<()> {
        let gl = crate::core::resource::GLOBAL_RESOURCES
            .with(|res| res.borrow().gl.clone())
            .ok_or_else(|| PyRuntimeError::new_err("No active GL context"))?;

        if self.mapped_ptr.is_null() {
            return Ok(());
        }

        unsafe {
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(self.pbo));
        }
        unsafe { gl.unmap_buffer(glow::PIXEL_UNPACK_BUFFER) };
        unsafe { gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None) };

        self.mapped_ptr = std::ptr::null_mut();

        Ok(())
    }

    fn to_texture(&self, width: u32, height: u32, texture_id: u32) -> PyResult<()> {
        let gl = crate::core::resource::GLOBAL_RESOURCES
            .with(|res| res.borrow().gl.clone())
            .ok_or_else(|| PyRuntimeError::new_err("No active GL context"))?;

        // Bind PBO
        unsafe {
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(self.pbo));
        }

        // Bind Texture
        // Validating texture_id is hard, assuming user passed valid GL ID.
        // glow::Texture is a NativeTexture (often u32).
        // We cast u32 to NativeTexture.

        // This is tricky: PyO3 passes u32, but glow needs NativeTexture.
        // NativeTexture depends on backend. For gl-rs/gl, it is u32/u64.
        // For glow's web support, it's Resource.
        // We assume generic `glow` logic here might need a helper,
        // BUT calling `gl.bind_texture` expects `NativeTexture`.

        // HACK: We assume `NativeTexture` is constructible from raw ID or we wrap it.
        // glow 0.13: `NativeTexture` is an alias. On desktop GL it is `NonZeroU32`.

        let native_tex = unsafe { std::mem::transmute::<u32, glow::NativeTexture>(texture_id) };
        // Check if correct type. If NativeTexture is not u32, this is UB.
        // But fanta_rust is currently Desktop GL (glutin).

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(native_tex));

            // Upload from PBO (offset 0)
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width as i32,
                height as i32,
                glow::RGBA, // Assuming RGBA
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::BufferOffset(0),
            );

            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        Ok(())
    }

    fn __dlpack__(&self, py: Python<'_>, stream: Option<PyObject>) -> PyResult<PyObject> {
        if self.mapped_ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "BufferView must be mapped (call .map()) before converting to DLPack.",
            ));
        }

        // We do not handle stream synchronization here for CPU/Mapped memory (stream=None expected or ignored)
        // If stream is provided, strictly we should ensure sync, but for CPU tensor it's usually implicit.

        // Construct Managed Context
        let shape = [self.size as i64];
        let strides = [1];

        let mut ctx = Box::new(ManagedDLPackCtx {
            tensor: DLManagedTensor {
                dl_tensor: DLTensor {
                    data: self.mapped_ptr as *mut c_void,
                    device: DLDevice {
                        device_type: DLDeviceType::kDLCPU,
                        device_id: 0,
                    },
                    ndim: 1,
                    dtype: DLDataType {
                        code: DLDataTypeCode::kDLUint as u8,
                        bits: 8,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),   // Set below
                    strides: std::ptr::null_mut(), // Set below
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(), // Could store self ref here if we wanted to enforce keep-alive
                deleter: Some(dlpack_deleter),
            },
            shape,
            strides,
        });

        // Patch pointers
        ctx.tensor.dl_tensor.shape = ctx.shape.as_mut_ptr();
        ctx.tensor.dl_tensor.strides = ctx.strides.as_mut_ptr();

        let tensor_ptr = Box::into_raw(ctx) as *mut DLManagedTensor;

        // Create Capsule via FFI to avoid Send requirements on raw pointers
        let name = CString::new("dltensor").unwrap();
        let cap_ptr = unsafe { ffi::PyCapsule_New(tensor_ptr as *mut c_void, name.as_ptr(), None) };

        if cap_ptr.is_null() {
            // Restore box to drop it?
            // If API failed, we should probably clean up tensor_ptr.
            unsafe {
                let _ = Box::from_raw(tensor_ptr as *mut ManagedDLPackCtx);
            }
            return Err(PyRuntimeError::new_err("Failed to create PyCapsule"));
        }

        unsafe { Ok(PyObject::from_owned_ptr(py, cap_ptr)) }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // kDLCPU = 1
        (1, 0)
    }
}
