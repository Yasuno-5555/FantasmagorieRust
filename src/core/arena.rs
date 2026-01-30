//! FrameArena - Linear allocator reset every frame
//! Ported from memory.hpp
//!
//! Uses bumpalo for zero-cost bump allocation.
//! All View allocations happen here and are freed in bulk at frame end.

use bumpalo::Bump;

/// Frame-local linear allocator
///
/// Design: Fixed capacity, reset every frame. Zero allocation cost.
/// Safety: Fail-fast on overflow (bumpalo handles this).
pub struct FrameArena {
    bump: Bump,
}

impl Default for FrameArena {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameArena {
    /// Default capacity: 4MB (plenty for 10k+ widgets)
    pub const DEFAULT_CAPACITY: usize = 4 * 1024 * 1024;

    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: Bump::with_capacity(capacity),
        }
    }

    /// Reset the arena for a new frame
    /// All previous allocations become invalid
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Allocate a value on the arena
    pub fn alloc<T>(&self, val: T) -> &mut T {
        self.bump.alloc(val)
    }

    /// Allocate a slice on the arena
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T] {
        self.bump.alloc_slice_copy(slice)
    }

    /// Allocate a string on the arena
    pub fn alloc_str(&self, s: &str) -> &str {
        self.bump.alloc_str(s)
    }

    /// Allocate with default value
    pub fn alloc_default<T: Default>(&self) -> &mut T {
        self.bump.alloc(T::default())
    }

    /// Allocate a zeroed slice
    pub fn alloc_slice_fill_default<T: Default + Copy>(&self, len: usize) -> &mut [T] {
        self.bump.alloc_slice_fill_default(len)
    }

    /// Get current memory usage
    pub fn used(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Get underlying bump allocator (for advanced usage)
    pub fn bump(&self) -> &Bump {
        &self.bump
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc() {
        let arena = FrameArena::new();
        let x = arena.alloc(42i32);
        assert_eq!(*x, 42);
        *x = 100;
        assert_eq!(*x, 100);
    }

    #[test]
    fn test_arena_string() {
        let arena = FrameArena::new();
        let s = arena.alloc_str("Hello, Fantasmagorie!");
        assert_eq!(s, "Hello, Fantasmagorie!");
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = FrameArena::new();
        let _ = arena.alloc(123i32);
        let _ = arena.alloc_str("test");
        let used_before = arena.used();
        assert!(used_before > 0);

        arena.reset();
        // After reset, allocations start fresh
        let _ = arena.alloc(456i32);
        // Used bytes should be similar to before (just the new allocation)
    }
}
