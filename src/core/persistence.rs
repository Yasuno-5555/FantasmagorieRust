use crate::core::ID;
use serde::{de::DeserializeOwned, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;

/// Trait for types that can be stored/restored
pub trait Store: Serialize + DeserializeOwned + Clone + Default + 'static {}
impl<T> Store for T where T: Serialize + DeserializeOwned + Clone + Default + 'static {}

/// Manages persistence for UI state
#[derive(Default)]
pub struct PersistenceManager {
    /// Map of Widget/State ID -> Serialized Bincode Data
    data: RefCell<HashMap<u64, Vec<u8>>>,
}

impl PersistenceManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Save state T for a given ID
    pub fn save<T: Store>(&self, id: ID, state: &T) {
        if let Ok(bytes) = bincode::serialize(state) {
            self.data.borrow_mut().insert(id.0, bytes);
        }
    }

    /// Load state T for a given ID
    pub fn load<T: Store>(&self, id: ID) -> Option<T> {
        let binding = self.data.borrow();
        if let Some(bytes) = binding.get(&id.0) {
            bincode::deserialize(bytes).ok()
        } else {
            None
        }
    }

    /// Serialize entire store to bytes (e.g. for saving to file)
    pub fn export_blob(&self) -> Option<Vec<u8>> {
        bincode::serialize(&*self.data.borrow()).ok()
    }

    /// Deserialize entire store from bytes
    pub fn import_blob(&self, blob: &[u8]) {
        if let Ok(data) = bincode::deserialize(blob) {
            *self.data.borrow_mut() = data;
        }
    }
}
