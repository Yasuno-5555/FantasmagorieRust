//! Plugin system for extensibility
//!
//! Provides:
//! - Plugin trait for custom extensions
//! - Plugin registry and lifecycle
//! - Hot-reload support

use std::any::Any;
use std::collections::HashMap;

/// Plugin information
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Unique plugin ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Version string
    pub version: String,
    /// Author name
    pub author: String,
    /// Description
    pub description: String,
    /// Plugin dependencies (plugin IDs)
    pub dependencies: Vec<String>,
}

impl PluginInfo {
    pub fn new(id: &str, name: &str, version: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            version: version.to_string(),
            author: String::new(),
            description: String::new(),
            dependencies: Vec::new(),
        }
    }

    pub fn with_author(mut self, author: &str) -> Self {
        self.author = author.to_string();
        self
    }

    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    pub fn with_dependency(mut self, dep_id: &str) -> Self {
        self.dependencies.push(dep_id.to_string());
        self
    }
}

/// Plugin lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is registered but not loaded
    Registered,
    /// Plugin is loaded and ready
    Loaded,
    /// Plugin is active
    Active,
    /// Plugin is disabled
    Disabled,
    /// Plugin failed to load
    Error,
}

/// Plugin capability flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginCapabilities {
    /// Can render custom widgets
    pub custom_widgets: bool,
    /// Can add menu items
    pub menu_items: bool,
    /// Can modify input handling
    pub input_hooks: bool,
    /// Can access file system
    pub file_access: bool,
    /// Can make network requests
    pub network_access: bool,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            custom_widgets: false,
            menu_items: false,
            input_hooks: false,
            file_access: false,
            network_access: false,
        }
    }
}

/// Plugin context provided to plugins
pub struct PluginContext {
    /// Plugin ID
    pub id: String,
    /// Configuration storage
    config: HashMap<String, String>,
}

impl PluginContext {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            config: HashMap::new(),
        }
    }

    /// Get a config value
    pub fn get_config(&self, key: &str) -> Option<&str> {
        self.config.get(key).map(|s| s.as_str())
    }

    /// Set a config value
    pub fn set_config(&mut self, key: &str, value: &str) {
        self.config.insert(key.to_string(), value.to_string());
    }

    /// Store arbitrary data
    pub fn store<T: Any + Send + Sync + 'static>(&self, _key: &str, _value: T) {
        // Storage implementation would go here
    }

    /// Retrieve stored data
    pub fn retrieve<T: Any + Clone>(&self, _key: &str) -> Option<T> {
        // Retrieval implementation would go here
        None
    }
}

/// Plugin trait that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Get plugin information
    fn info(&self) -> PluginInfo;
    
    /// Get plugin capabilities
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::default()
    }

    /// Called when plugin is loaded
    fn on_load(&mut self, _ctx: &mut PluginContext) -> Result<(), String> {
        Ok(())
    }

    /// Called when plugin is unloaded
    fn on_unload(&mut self, _ctx: &mut PluginContext) {
        // Default: do nothing
    }

    /// Called when plugin is enabled
    fn on_enable(&mut self, _ctx: &mut PluginContext) -> Result<(), String> {
        Ok(())
    }

    /// Called when plugin is disabled
    fn on_disable(&mut self, _ctx: &mut PluginContext) {
        // Default: do nothing
    }

    /// Called each frame (if plugin is active)
    fn on_update(&mut self, _ctx: &mut PluginContext, _delta_ms: f32) {
        // Default: do nothing
    }

    /// Called after rendering (for overlays)
    fn on_render(&mut self, _ctx: &mut PluginContext) {
        // Default: do nothing
    }
}

/// Plugin registry entry
struct PluginEntry {
    plugin: Box<dyn Plugin>,
    state: PluginState,
    context: PluginContext,
}

/// Plugin manager
pub struct PluginManager {
    plugins: HashMap<String, PluginEntry>,
    load_order: Vec<String>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            load_order: Vec::new(),
        }
    }

    /// Register a plugin
    pub fn register<P: Plugin + 'static>(&mut self, plugin: P) -> Result<(), String> {
        let info = plugin.info();
        let id = info.id.clone();

        if self.plugins.contains_key(&id) {
            return Err(format!("Plugin '{}' is already registered", id));
        }

        // Check dependencies
        for dep in &info.dependencies {
            if !self.plugins.contains_key(dep) {
                return Err(format!("Plugin '{}' requires unregistered plugin '{}'", id, dep));
            }
        }

        let context = PluginContext::new(&id);
        self.plugins.insert(id.clone(), PluginEntry {
            plugin: Box::new(plugin),
            state: PluginState::Registered,
            context,
        });
        self.load_order.push(id);

        Ok(())
    }

    /// Load a plugin
    pub fn load(&mut self, id: &str) -> Result<(), String> {
        let entry = self.plugins.get_mut(id)
            .ok_or_else(|| format!("Plugin '{}' not found", id))?;

        if entry.state != PluginState::Registered && entry.state != PluginState::Disabled {
            return Err(format!("Plugin '{}' cannot be loaded from state {:?}", id, entry.state));
        }

        match entry.plugin.on_load(&mut entry.context) {
            Ok(()) => {
                entry.state = PluginState::Loaded;
                Ok(())
            }
            Err(e) => {
                entry.state = PluginState::Error;
                Err(e)
            }
        }
    }

    /// Enable a plugin
    pub fn enable(&mut self, id: &str) -> Result<(), String> {
        let entry = self.plugins.get_mut(id)
            .ok_or_else(|| format!("Plugin '{}' not found", id))?;

        if entry.state != PluginState::Loaded {
            return Err(format!("Plugin '{}' must be loaded before enabling", id));
        }

        match entry.plugin.on_enable(&mut entry.context) {
            Ok(()) => {
                entry.state = PluginState::Active;
                Ok(())
            }
            Err(e) => {
                Err(e)
            }
        }
    }

    /// Disable a plugin
    pub fn disable(&mut self, id: &str) -> Result<(), String> {
        let entry = self.plugins.get_mut(id)
            .ok_or_else(|| format!("Plugin '{}' not found", id))?;

        if entry.state != PluginState::Active {
            return Ok(()); // Already disabled
        }

        entry.plugin.on_disable(&mut entry.context);
        entry.state = PluginState::Disabled;
        Ok(())
    }

    /// Unload a plugin
    pub fn unload(&mut self, id: &str) -> Result<(), String> {
        let entry = self.plugins.get_mut(id)
            .ok_or_else(|| format!("Plugin '{}' not found", id))?;

        if entry.state == PluginState::Active {
            self.disable(id)?;
        }

        if let Some(entry) = self.plugins.get_mut(id) {
            entry.plugin.on_unload(&mut entry.context);
            entry.state = PluginState::Registered;
        }
        Ok(())
    }

    /// Update all active plugins
    pub fn update(&mut self, delta_ms: f32) {
        for id in &self.load_order {
            if let Some(entry) = self.plugins.get_mut(id) {
                if entry.state == PluginState::Active {
                    entry.plugin.on_update(&mut entry.context, delta_ms);
                }
            }
        }
    }

    /// Render all active plugins
    pub fn render(&mut self) {
        for id in &self.load_order {
            if let Some(entry) = self.plugins.get_mut(id) {
                if entry.state == PluginState::Active {
                    entry.plugin.on_render(&mut entry.context);
                }
            }
        }
    }

    /// Get plugin state
    pub fn state(&self, id: &str) -> Option<PluginState> {
        self.plugins.get(id).map(|e| e.state)
    }

    /// Get plugin info
    pub fn info(&self, id: &str) -> Option<PluginInfo> {
        self.plugins.get(id).map(|e| e.plugin.info())
    }

    /// List all plugins
    pub fn list(&self) -> Vec<PluginInfo> {
        self.load_order.iter()
            .filter_map(|id| self.plugins.get(id))
            .map(|e| e.plugin.info())
            .collect()
    }

    /// List active plugins
    pub fn active(&self) -> Vec<&str> {
        self.load_order.iter()
            .filter(|id| {
                self.plugins.get(*id)
                    .map(|e| e.state == PluginState::Active)
                    .unwrap_or(false)
            })
            .map(|s| s.as_str())
            .collect()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin {
        loaded: bool,
        enabled: bool,
    }

    impl TestPlugin {
        fn new() -> Self {
            Self { loaded: false, enabled: false }
        }
    }

    impl Plugin for TestPlugin {
        fn info(&self) -> PluginInfo {
            PluginInfo::new("test", "Test Plugin", "1.0.0")
        }

        fn on_load(&mut self, _ctx: &mut PluginContext) -> Result<(), String> {
            self.loaded = true;
            Ok(())
        }

        fn on_enable(&mut self, _ctx: &mut PluginContext) -> Result<(), String> {
            self.enabled = true;
            Ok(())
        }
    }

    #[test]
    fn test_plugin_lifecycle() {
        let mut manager = PluginManager::new();
        
        manager.register(TestPlugin::new()).unwrap();
        assert_eq!(manager.state("test"), Some(PluginState::Registered));

        manager.load("test").unwrap();
        assert_eq!(manager.state("test"), Some(PluginState::Loaded));

        manager.enable("test").unwrap();
        assert_eq!(manager.state("test"), Some(PluginState::Active));

        manager.disable("test").unwrap();
        assert_eq!(manager.state("test"), Some(PluginState::Disabled));
    }
}
