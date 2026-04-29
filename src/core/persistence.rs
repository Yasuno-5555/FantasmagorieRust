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
        if let Ok(json) = serde_json::to_vec(state) {
            self.data.borrow_mut().insert(id.0, json);
        }
    }

    /// Load state T for a given ID
    pub fn load<T: Store>(&self, id: ID) -> Option<T> {
        let binding = self.data.borrow();
        if let Some(bytes) = binding.get(&id.0) {
            serde_json::from_slice(bytes).ok()
        } else {
            None
        }
    }

    /// Serialize entire store to bytes (e.g. for saving to file)
    pub fn export_blob(&self) -> Option<Vec<u8>> {
        serde_json::to_vec(&*self.data.borrow()).ok()
    }

    /// Deserialize entire store from bytes
    pub fn import_blob(&self, blob: &[u8]) {
        if let Ok(data) = serde_json::from_slice(blob) {
            *self.data.borrow_mut() = data;
        }
    }
}
