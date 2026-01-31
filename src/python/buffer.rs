use crate::python::dlpack::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyCapsule;
use std::ffi::c_void;
use std::ffi::CString;
use std::ptr::null;
use wgpu;

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
    buffer: wgpu::Buffer,
    size: usize,
    mapped_ptr: *mut u8,
}

unsafe impl Send for BufferView {}

#[pymethods]
impl BufferView {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let device = crate::core::resource::GLOBAL_RESOURCES.with(|res| res.borrow().device.clone())
            .ok_or_else(|| PyRuntimeError::new_err("No active WGPU device found."))?;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Python BufferView"),
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        Ok(BufferView {
            buffer,
            size,
            mapped_ptr: std::ptr::null_mut(),
        })
    }

    unsafe fn map(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let device = crate::core::resource::GLOBAL_RESOURCES.with(|res| res.borrow().device.clone())
            .ok_or_else(|| PyRuntimeError::new_err("No active WGPU device"))?;

        let buffer_slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Write, move |res| tx.send(res).unwrap());
        device.poll(wgpu::Maintain::Wait);

        if rx.recv().unwrap().is_ok() {
            let ptr = buffer_slice.get_mapped_range_mut().as_mut_ptr();
            self.mapped_ptr = ptr;

            let view_ptr = pyo3::ffi::PyMemoryView_FromMemory(ptr as *mut i8, self.size as isize, pyo3::ffi::PyBUF_WRITE);
            if view_ptr.is_null() {
                return Err(PyRuntimeError::new_err("Failed to create Python MemoryView"));
            }
            Ok(PyObject::from_owned_ptr(py, view_ptr))
        } else {
            Err(PyRuntimeError::new_err("Failed to map WGPU buffer"))
        }
    }

    fn unmap(&mut self) -> PyResult<()> {
        if !self.mapped_ptr.is_null() {
            self.buffer.unmap();
            self.mapped_ptr = std::ptr::null_mut();
        }
        Ok(())
    }

    fn to_texture(&self, _width: u32, _height: u32, _texture_id: u64) -> PyResult<()> {
        // Implementation for WGPU texture upload from buffer would go here
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
