use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

/// A minimal block_on implementation to replace the pollster dependency.
/// This is used primarily for one-off async calls like wgpu adapter/device requests.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);

    fn raw_waker_clone(_: *const ()) -> RawWaker {
        raw_waker()
    }
    fn raw_waker_wake(_: *const ()) {}
    fn raw_waker_wake_by_ref(_: *const ()) {}
    fn raw_waker_drop(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(
        raw_waker_clone,
        raw_waker_wake,
        raw_waker_wake_by_ref,
        raw_waker_drop,
    );

    fn raw_waker() -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }

    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let mut cx = Context::from_waker(&waker);

    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => {
                // Yield to allow background tasks (like wgpu callbacks) to progress
                std::thread::yield_now();
            }
        }
    }
}
