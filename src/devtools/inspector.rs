//! UI Inspector for debugging and development
//!
//! Provides runtime inspection of:
//! - Widget tree hierarchy
//! - Property values
//! - Layout bounds
//! - Event handling

use std::collections::HashMap;

/// Inspectable widget information
#[derive(Debug, Clone)]
pub struct WidgetInfo {
    /// Unique identifier
    pub id: String,
    /// Widget type name
    pub widget_type: String,
    /// Display name (for tree view)
    pub display_name: String,
    /// Parent widget ID
    pub parent_id: Option<String>,
    /// Child widget IDs
    pub children: Vec<String>,
    /// Widget properties
    pub properties: HashMap<String, PropertyValue>,
    /// Computed layout bounds
    pub bounds: LayoutBounds,
    /// Is this widget visible
    pub visible: bool,
    /// Is this widget enabled
    pub enabled: bool,
    /// Is this widget focused
    pub focused: bool,
    /// Is this widget hovered
    pub hovered: bool,
}

impl WidgetInfo {
    pub fn new(id: &str, widget_type: &str) -> Self {
        Self {
            id: id.to_string(),
            widget_type: widget_type.to_string(),
            display_name: widget_type.to_string(),
            parent_id: None,
            children: Vec::new(),
            properties: HashMap::new(),
            bounds: LayoutBounds::default(),
            visible: true,
            enabled: true,
            focused: false,
            hovered: false,
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.display_name = name.to_string();
        self
    }

    pub fn set_property(&mut self, name: &str, value: PropertyValue) {
        self.properties.insert(name.to_string(), value);
    }
}

/// Layout bounds information
#[derive(Debug, Clone, Copy, Default)]
pub struct LayoutBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub padding_top: f32,
    pub padding_right: f32,
    pub padding_bottom: f32,
    pub padding_left: f32,
    pub margin_top: f32,
    pub margin_right: f32,
    pub margin_bottom: f32,
    pub margin_left: f32,
}

impl LayoutBounds {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x, y, width, height,
            ..Default::default()
        }
    }

    pub fn content_rect(&self) -> (f32, f32, f32, f32) {
        (
            self.x + self.padding_left,
            self.y + self.padding_top,
            self.width - self.padding_left - self.padding_right,
            self.height - self.padding_top - self.padding_bottom,
        )
    }
}

/// Property value types
#[derive(Debug, Clone)]
pub enum PropertyValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Color { r: f32, g: f32, b: f32, a: f32 },
    Vec2 { x: f32, y: f32 },
    Enum { value: String, options: Vec<String> },
    Array(Vec<PropertyValue>),
}

impl PropertyValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            PropertyValue::String(_) => "String",
            PropertyValue::Int(_) => "Int",
            PropertyValue::Float(_) => "Float",
            PropertyValue::Bool(_) => "Bool",
            PropertyValue::Color { .. } => "Color",
            PropertyValue::Vec2 { .. } => "Vec2",
            PropertyValue::Enum { .. } => "Enum",
            PropertyValue::Array(_) => "Array",
        }
    }

    pub fn as_string(&self) -> String {
        match self {
            PropertyValue::String(s) => s.clone(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => format!("{:.2}", f),
            PropertyValue::Bool(b) => b.to_string(),
            PropertyValue::Color { r, g, b, a } => format!("rgba({:.0}, {:.0}, {:.0}, {:.2})", r*255.0, g*255.0, b*255.0, a),
            PropertyValue::Vec2 { x, y } => format!("({:.1}, {:.1})", x, y),
            PropertyValue::Enum { value, .. } => value.clone(),
            PropertyValue::Array(arr) => format!("[{} items]", arr.len()),
        }
    }
}

/// Inspector state
pub struct Inspector {
    /// All registered widgets
    widgets: HashMap<String, WidgetInfo>,
    /// Root widget IDs
    roots: Vec<String>,
    /// Currently selected widget
    selected: Option<String>,
    /// Is inspector panel visible
    visible: bool,
    /// Inspector panel position
    panel_x: f32,
    panel_y: f32,
    panel_width: f32,
    panel_height: f32,
    /// Tree expansion state
    expanded: HashMap<String, bool>,
    /// Search filter
    search_filter: String,
    /// Show only visible widgets
    show_only_visible: bool,
    /// Highlight hovered widget
    highlight_on_hover: bool,
}

impl Inspector {
    pub fn new() -> Self {
        Self {
            widgets: HashMap::new(),
            roots: Vec::new(),
            selected: None,
            visible: false,
            panel_x: 0.0,
            panel_y: 0.0,
            panel_width: 300.0,
            panel_height: 600.0,
            expanded: HashMap::new(),
            search_filter: String::new(),
            show_only_visible: false,
            highlight_on_hover: true,
        }
    }

    /// Toggle inspector visibility
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Show the inspector
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hide the inspector
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Check if visible
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Clear all widgets (call at start of frame)
    pub fn clear(&mut self) {
        self.widgets.clear();
        self.roots.clear();
    }

    /// Register a widget
    pub fn register(&mut self, info: WidgetInfo) {
        let id = info.id.clone();
        let parent_id = info.parent_id.clone();
        
        self.widgets.insert(id.clone(), info);
        
        if parent_id.is_none() {
            self.roots.push(id);
        } else if let Some(parent) = parent_id {
            if let Some(parent_info) = self.widgets.get_mut(&parent) {
                parent_info.children.push(id);
            }
        }
    }

    /// Select a widget by ID
    pub fn select(&mut self, id: &str) {
        if self.widgets.contains_key(id) {
            self.selected = Some(id.to_string());
            // Expand parent chain
            self.expand_to(id);
        }
    }

    /// Deselect
    pub fn deselect(&mut self) {
        self.selected = None;
    }

    /// Get selected widget
    pub fn selected(&self) -> Option<&WidgetInfo> {
        self.selected.as_ref().and_then(|id| self.widgets.get(id))
    }

    /// Expand tree to show a widget
    fn expand_to(&mut self, id: &str) {
        let mut current = self.widgets.get(id).and_then(|w| w.parent_id.clone());
        while let Some(parent_id) = current {
            self.expanded.insert(parent_id.clone(), true);
            current = self.widgets.get(&parent_id).and_then(|w| w.parent_id.clone());
        }
    }

    /// Toggle expansion state
    pub fn toggle_expanded(&mut self, id: &str) {
        let current = self.expanded.get(id).copied().unwrap_or(false);
        self.expanded.insert(id.to_string(), !current);
    }

    /// Check if expanded
    pub fn is_expanded(&self, id: &str) -> bool {
        self.expanded.get(id).copied().unwrap_or(false)
    }

    /// Get widget by ID
    pub fn get(&self, id: &str) -> Option<&WidgetInfo> {
        self.widgets.get(id)
    }

    /// Get root widgets
    pub fn roots(&self) -> &[String] {
        &self.roots
    }

    /// Set search filter
    pub fn set_filter(&mut self, filter: &str) {
        self.search_filter = filter.to_lowercase();
    }

    /// Check if widget matches filter
    fn matches_filter(&self, info: &WidgetInfo) -> bool {
        if self.search_filter.is_empty() {
            return true;
        }
        info.display_name.to_lowercase().contains(&self.search_filter) ||
        info.widget_type.to_lowercase().contains(&self.search_filter) ||
        info.id.to_lowercase().contains(&self.search_filter)
    }

    /// Get filtered widgets for display
    pub fn get_visible_widgets(&self) -> Vec<&WidgetInfo> {
        self.widgets.values()
            .filter(|w| {
                (!self.show_only_visible || w.visible) && self.matches_filter(w)
            })
            .collect()
    }

    /// Find widget at position (for hover highlighting)
    pub fn widget_at(&self, x: f32, y: f32) -> Option<&WidgetInfo> {
        // Find deepest widget containing point
        self.widgets.values()
            .filter(|w| w.visible)
            .filter(|w| {
                let b = &w.bounds;
                x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height
            })
            .max_by(|a, b| {
                // Prefer deeper (smaller area) widgets
                let area_a = a.bounds.width * a.bounds.height;
                let area_b = b.bounds.width * b.bounds.height;
                area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get tree structure for rendering
    pub fn tree_items(&self) -> Vec<TreeItem> {
        let mut items = Vec::new();
        for root_id in &self.roots {
            self.build_tree_items(root_id, 0, &mut items);
        }
        items
    }

    fn build_tree_items(&self, id: &str, depth: usize, items: &mut Vec<TreeItem>) {
        if let Some(info) = self.widgets.get(id) {
            if self.show_only_visible && !info.visible {
                return;
            }
            if !self.matches_filter(info) {
                return;
            }

            items.push(TreeItem {
                id: id.to_string(),
                display_name: info.display_name.clone(),
                widget_type: info.widget_type.clone(),
                depth,
                has_children: !info.children.is_empty(),
                is_expanded: self.is_expanded(id),
                is_selected: self.selected.as_ref() == Some(&id.to_string()),
            });

            if self.is_expanded(id) {
                for child_id in &info.children {
                    self.build_tree_items(child_id, depth + 1, items);
                }
            }
        }
    }
}

impl Default for Inspector {
    fn default() -> Self {
        Self::new()
    }
}

/// Tree item for rendering
#[derive(Debug, Clone)]
pub struct TreeItem {
    pub id: String,
    pub display_name: String,
    pub widget_type: String,
    pub depth: usize,
    pub has_children: bool,
    pub is_expanded: bool,
    pub is_selected: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspector_basics() {
        let mut inspector = Inspector::new();
        
        let mut root = WidgetInfo::new("root", "Column");
        root.bounds = LayoutBounds::new(0.0, 0.0, 100.0, 100.0);
        inspector.register(root);

        let mut child = WidgetInfo::new("child", "Button");
        child.parent_id = Some("root".to_string());
        child.bounds = LayoutBounds::new(10.0, 10.0, 80.0, 30.0);
        inspector.register(child);

        assert_eq!(inspector.roots().len(), 1);
        inspector.select("child");
        assert!(inspector.selected().is_some());
    }

    #[test]
    fn test_property_values() {
        let color = PropertyValue::Color { r: 1.0, g: 0.5, b: 0.0, a: 1.0 };
        assert_eq!(color.type_name(), "Color");
        assert!(color.as_string().contains("rgba"));
    }
}
