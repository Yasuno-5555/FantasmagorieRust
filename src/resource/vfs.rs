//! Virtual File System (VFS) and Asset Management
//! Ported from vfs.hpp

use std::cell::RefCell;
use std::collections::HashMap;

pub struct Vfs {
    pub files: HashMap<String, Vec<u8>>,
}

impl Vfs {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    pub fn mount(&mut self, path: &str, data: Vec<u8>) {
        self.files.insert(path.to_string(), data);
    }

    pub fn read(&self, path: &str) -> Option<&[u8]> {
        // Check VFS
        if let Some(data) = self.files.get(path) {
            return Some(data.as_slice());
        }
        None
    }
}

thread_local! {
    pub static VFS: RefCell<Vfs> = RefCell::new(Vfs::new());
}
