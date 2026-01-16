use std::collections::HashMap;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use image::GenericImageView;

pub type TextureId = u64;
pub mod vfs;

pub struct TextureData {
    pub id: TextureId,
    pub width: u32,
    pub height: u32,
    pub pixels: Option<Vec<u8>>, // Cleared after upload
    pub gl_texture: Option<u32>, // GL handle (cast to u32, stored as generic)
    pub dirty: bool,
}

pub struct TextureManager {
    textures: HashMap<TextureId, TextureData>,
    path_cache: HashMap<String, TextureId>,
    next_id: AtomicU64,
}

impl TextureManager {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            path_cache: HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn load_from_path(&mut self, path: &str) -> Option<(TextureId, u32, u32)> {
        if let Some(&id) = self.path_cache.get(path) {
            let tex = self.textures.get(&id)?;
            return Some((id, tex.width, tex.height));
        }

        // Try VFS first
        let data = vfs::VFS.with(|v| v.borrow().read(path).map(|d| d.to_vec()));
        
        let img = if let Some(d) = data {
            image::load_from_memory(&d).ok()?
        } else {
            image::open(path).ok()?
        };

        let (width, height) = img.dimensions();
        let pixels = img.to_rgba8().into_raw();

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        self.textures.insert(id, TextureData {
            id,
            width,
            height,
            pixels: Some(pixels),
            gl_texture: None,
            dirty: true,
        });
        
        self.path_cache.insert(path.to_string(), id);
        Some((id, width, height))
    }

    pub fn get(&self, id: TextureId) -> Option<&TextureData> {
        self.textures.get(&id)
    }

    pub fn get_mut(&mut self, id: TextureId) -> Option<&mut TextureData> {
        self.textures.get_mut(&id)
    }
}

thread_local! {
    pub static TEXTURE_MANAGER: RefCell<TextureManager> = RefCell::new(TextureManager::new());
}
