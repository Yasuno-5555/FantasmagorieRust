//! Accessibility (A11y) Support
//!
//! Provides:
//! - Screen reader integration
//! - Focus management
//! - High contrast mode detection
//! - Accessible widget information

use std::collections::HashMap;

/// Accessible information for a widget
#[derive(Debug, Clone, Default)]
pub struct AccessibleInfo {
    /// Accessible name (read by screen readers)
    pub name: String,
    /// Longer description
    pub description: String,
    /// Role (button, slider, checkbox, etc.)
    pub role: AccessibleRole,
    /// Current value (for sliders, inputs)
    pub value: String,
    /// Is focusable
    pub focusable: bool,
    /// Is disabled
    pub disabled: bool,
    /// Tab index for keyboard navigation
    pub tab_index: Option<i32>,
}

impl AccessibleInfo {
    pub fn new(name: &str, role: AccessibleRole) -> Self {
        Self {
            name: name.to_string(),
            role,
            focusable: true,
            ..Default::default()
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_value(mut self, value: &str) -> Self {
        self.value = value.to_string();
        self
    }

    pub fn with_tab_index(mut self, index: i32) -> Self {
        self.tab_index = Some(index);
        self
    }

    pub fn disabled(mut self) -> Self {
        self.disabled = true;
        self
    }
}

/// Widget role for accessibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessibleRole {
    #[default]
    None,
    Button,
    Checkbox,
    Radio,
    Slider,
    TextInput,
    Label,
    Image,
    Link,
    List,
    ListItem,
    Menu,
    MenuItem,
    Tab,
    TabPanel,
    Dialog,
    Alert,
    Progressbar,
    Scrollbar,
    Separator,
    Group,
    Region,
}

impl AccessibleRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            AccessibleRole::None => "none",
            AccessibleRole::Button => "button",
            AccessibleRole::Checkbox => "checkbox",
            AccessibleRole::Radio => "radio",
            AccessibleRole::Slider => "slider",
            AccessibleRole::TextInput => "textbox",
            AccessibleRole::Label => "label",
            AccessibleRole::Image => "img",
            AccessibleRole::Link => "link",
            AccessibleRole::List => "list",
            AccessibleRole::ListItem => "listitem",
            AccessibleRole::Menu => "menu",
            AccessibleRole::MenuItem => "menuitem",
            AccessibleRole::Tab => "tab",
            AccessibleRole::TabPanel => "tabpanel",
            AccessibleRole::Dialog => "dialog",
            AccessibleRole::Alert => "alert",
            AccessibleRole::Progressbar => "progressbar",
            AccessibleRole::Scrollbar => "scrollbar",
            AccessibleRole::Separator => "separator",
            AccessibleRole::Group => "group",
            AccessibleRole::Region => "region",
        }
    }
}

/// Focusable node entry
#[derive(Debug, Clone)]
struct FocusableNode {
    id: usize,
    tab_index: i32,
}

/// Focus manager for keyboard navigation
pub struct FocusManager {
    focusables: Vec<FocusableNode>,
    current_focus: Option<usize>,
    next_index: i32,
    announcements: Vec<String>,
}

impl FocusManager {
    pub fn new() -> Self {
        Self {
            focusables: Vec::new(),
            current_focus: None,
            next_index: 0,
            announcements: Vec::new(),
        }
    }

    /// Register a focusable element
    pub fn register(&mut self, id: usize, tab_index: Option<i32>) {
        let index = tab_index.unwrap_or_else(|| {
            let i = self.next_index;
            self.next_index += 1;
            i
        });
        self.focusables.push(FocusableNode { id, tab_index: index });
    }

    /// Clear all registered focusables (call at frame start)
    pub fn clear(&mut self) {
        self.focusables.clear();
        self.next_index = 0;
    }

    /// Get currently focused element
    pub fn current(&self) -> Option<usize> {
        self.current_focus
    }

    /// Set focus to a specific element
    pub fn focus(&mut self, id: usize) {
        self.current_focus = Some(id);
    }

    /// Clear focus
    pub fn blur(&mut self) {
        self.current_focus = None;
    }

    /// Move focus to next element
    pub fn focus_next(&mut self) {
        if self.focusables.is_empty() {
            return;
        }

        // Sort by tab index
        self.focusables.sort_by_key(|n| n.tab_index);

        if let Some(current) = self.current_focus {
            // Find current and move to next
            for (i, node) in self.focusables.iter().enumerate() {
                if node.id == current {
                    let next = (i + 1) % self.focusables.len();
                    self.current_focus = Some(self.focusables[next].id);
                    return;
                }
            }
        }

        // No current focus, focus first
        self.current_focus = self.focusables.first().map(|n| n.id);
    }

    /// Move focus to previous element
    pub fn focus_prev(&mut self) {
        if self.focusables.is_empty() {
            return;
        }

        self.focusables.sort_by_key(|n| n.tab_index);

        if let Some(current) = self.current_focus {
            for (i, node) in self.focusables.iter().enumerate() {
                if node.id == current {
                    let prev = if i == 0 { self.focusables.len() - 1 } else { i - 1 };
                    self.current_focus = Some(self.focusables[prev].id);
                    return;
                }
            }
        }

        // No current focus, focus last
        self.current_focus = self.focusables.last().map(|n| n.id);
    }

    /// Check if element is focused
    pub fn is_focused(&self, id: usize) -> bool {
        self.current_focus == Some(id)
    }

    /// Announce to screen readers
    pub fn announce(&mut self, text: &str) {
        self.announcements.push(text.to_string());
    }

    /// Get and clear pending announcements
    pub fn take_announcements(&mut self) -> Vec<String> {
        std::mem::take(&mut self.announcements)
    }
}

impl Default for FocusManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Accessibility store for widget information
pub struct AccessibleStore {
    infos: HashMap<usize, AccessibleInfo>,
}

impl AccessibleStore {
    pub fn new() -> Self {
        Self {
            infos: HashMap::new(),
        }
    }

    /// Set accessibility info for a widget
    pub fn set(&mut self, id: usize, info: AccessibleInfo) {
        self.infos.insert(id, info);
    }

    /// Get accessibility info for a widget
    pub fn get(&self, id: usize) -> Option<&AccessibleInfo> {
        self.infos.get(&id)
    }

    /// Clear all info (call at frame start)
    pub fn clear(&mut self) {
        self.infos.clear();
    }
}

impl Default for AccessibleStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if system is in high contrast mode (Windows)
#[cfg(windows)]
pub fn is_high_contrast_mode() -> bool {
    use std::mem::size_of;
    
    #[repr(C)]
    struct HighContrastW {
        cb_size: u32,
        dw_flags: u32,
        lpsz_default_scheme: *mut u16,
    }
    
    const HCF_HIGHCONTRASTON: u32 = 0x00000001;
    const SPI_GETHIGHCONTRAST: u32 = 0x0042;
    
    extern "system" {
        fn SystemParametersInfoW(
            uiAction: u32,
            uiParam: u32,
            pvParam: *mut std::ffi::c_void,
            fWinIni: u32,
        ) -> i32;
    }
    
    unsafe {
        let mut hc = HighContrastW {
            cb_size: size_of::<HighContrastW>() as u32,
            dw_flags: 0,
            lpsz_default_scheme: std::ptr::null_mut(),
        };
        
        if SystemParametersInfoW(
            SPI_GETHIGHCONTRAST,
            size_of::<HighContrastW>() as u32,
            &mut hc as *mut _ as *mut std::ffi::c_void,
            0,
        ) != 0 {
            return (hc.dw_flags & HCF_HIGHCONTRASTON) != 0;
        }
    }
    false
}

#[cfg(not(windows))]
pub fn is_high_contrast_mode() -> bool {
    // Check environment variable on other platforms
    std::env::var("HIGH_CONTRAST").map(|v| v == "1").unwrap_or(false)
}

/// Reduced motion preference (respects system settings)
pub fn prefers_reduced_motion() -> bool {
    // Check environment variable
    std::env::var("REDUCED_MOTION").map(|v| v == "1").unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_manager() {
        let mut fm = FocusManager::new();
        
        fm.register(1, Some(0));
        fm.register(2, Some(1));
        fm.register(3, Some(2));

        fm.focus_next();
        assert_eq!(fm.current(), Some(1));

        fm.focus_next();
        assert_eq!(fm.current(), Some(2));

        fm.focus_prev();
        assert_eq!(fm.current(), Some(1));
    }

    #[test]
    fn test_accessible_info() {
        let info = AccessibleInfo::new("Submit", AccessibleRole::Button)
            .with_description("Submit the form")
            .with_tab_index(5);

        assert_eq!(info.name, "Submit");
        assert_eq!(info.role, AccessibleRole::Button);
        assert_eq!(info.tab_index, Some(5));
    }
}
