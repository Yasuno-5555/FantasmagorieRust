//! ID type for widget identification
//! Ported from fanta_id.h

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Unique identifier for UI elements
/// Used for hit-testing, focus tracking, and persistent state lookup
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ID(pub u64);

impl ID {
    pub const NONE: ID = ID(0);

    /// Create ID from string (FNV-1a style hash)
    pub fn from_str(s: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        ID(hasher.finish())
    }

    /// Create ID from integer
    pub const fn from_u64(v: u64) -> Self {
        ID(v)
    }

    /// Combine two IDs (for hierarchical IDs)
    pub fn combine(self, other: ID) -> Self {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        other.0.hash(&mut hasher);
        ID(hasher.finish())
    }

    /// Combine with string
    pub fn with_str(self, s: &str) -> Self {
        self.combine(ID::from_str(s))
    }

    /// Combine with index
    pub fn with_index(self, idx: usize) -> Self {
        self.combine(ID::from_u64(idx as u64))
    }

    pub fn is_none(&self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Debug for ID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ID({:#018x})", self.0)
    }
}

impl std::fmt::Display for ID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#018x}", self.0)
    }
}

impl From<&str> for ID {
    fn from(s: &str) -> Self {
        ID::from_str(s)
    }
}

impl From<u64> for ID {
    fn from(v: u64) -> Self {
        ID::from_u64(v)
    }
}

/// ID stack for hierarchical widget identification
#[derive(Default)]
pub struct IDStack {
    stack: Vec<ID>,
    current: ID,
}

impl IDStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, id: ID) {
        self.stack.push(self.current);
        self.current = self.current.combine(id);
    }

    pub fn push_str(&mut self, s: &str) {
        self.push(ID::from_str(s));
    }

    pub fn push_index(&mut self, idx: usize) {
        self.push(ID::from_u64(idx as u64));
    }

    pub fn pop(&mut self) {
        if let Some(prev) = self.stack.pop() {
            self.current = prev;
        }
    }

    pub fn current(&self) -> ID {
        self.current
    }

    pub fn get(&self, local: ID) -> ID {
        self.current.combine(local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_from_str() {
        let id1 = ID::from_str("button");
        let id2 = ID::from_str("button");
        let id3 = ID::from_str("label");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_id_stack() {
        let mut stack = IDStack::new();
        let base = stack.current();

        stack.push_str("panel");
        let panel_id = stack.current();
        assert_ne!(base, panel_id);

        stack.push_index(0);
        let item_id = stack.current();
        assert_ne!(panel_id, item_id);

        stack.pop();
        assert_eq!(stack.current(), panel_id);

        stack.pop();
        assert_eq!(stack.current(), base);
    }
}
