use super::types::{Align, Color, CursorType, LayoutDir, NodeID, Transform};
use crate::widgets::WidgetKind;

// ============================================================================
// Component Structs
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutConstraints {
    pub width: f32,
    pub height: f32,
    pub grow: f32,
    pub shrink: f32,
    pub padding: f32,
    // position type etc. if needed later
}

impl Default for LayoutConstraints {
    fn default() -> Self {
        Self {
            width: -1.0,
            height: -1.0,
            grow: 0.0,
            shrink: 1.0,
            padding: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayoutData {
    pub dir: Option<LayoutDir>, // Default Column if root/container? C++ says Column default.
    pub justify: Option<Align>,
    pub align: Option<Align>,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedStyle {
    pub bg: Color,
    pub bg_hover: Color,
    pub bg_active: Color,
    pub text: Color,
    pub border: Color,
    pub corner_radius: f32,
    pub border_width: f32,
    pub show_focus_ring: bool,
    pub focus_ring_color: Color,
    pub focus_ring_width: f32,
    pub cursor: CursorType,
}

impl Default for ResolvedStyle {
    fn default() -> Self {
        Self {
            bg: Color::TRANSPARENT,
            bg_hover: Color::TRANSPARENT,
            bg_active: Color::TRANSPARENT,
            text: Color::WHITE,
            border: Color::TRANSPARENT,
            corner_radius: 0.0,
            border_width: 0.0,
            show_focus_ring: false,
            focus_ring_color: Color::hex(0x4A90D9FF),
            focus_ring_width: 2.0,
            cursor: CursorType::Arrow,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputState {
    pub hoverable: bool,
    pub clickable: bool,
    pub focusable: bool,
    pub disabled: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScrollState {
    pub scroll_x: f32,
    pub scroll_y: f32,
    pub max_scroll_x: f32,
    pub max_scroll_y: f32,
    pub scrollable: bool,
    pub clip_content: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderData {
    pub text: String,
    pub is_text: bool,
    // texture: Option<TextureId> logic later
}

// ============================================================================
// Node Store (SoA)
// ============================================================================

pub struct NodeStore {
    // Tree
    pub parents: Vec<Option<NodeID>>,
    pub children: Vec<Vec<NodeID>>,

    // Data
    pub constraints: Vec<LayoutConstraints>,
    pub layout: Vec<LayoutData>,
    pub style: Vec<ResolvedStyle>,
    pub input: Vec<InputState>,
    pub scroll: Vec<ScrollState>,
    pub render: Vec<RenderData>,
    pub transform: Vec<Transform>,
    
    // Type info
    pub widget: Vec<WidgetKind>,
    // Persistent ID (for ID mixing and mapping)
    pub ids: Vec<super::types::StableID>,
}

impl NodeStore {
    pub fn new() -> Self {
        // Reserve index 0 as INVALID/Root placeholder
        let mut store = Self {
            parents: Vec::new(),
            children: Vec::new(),
            constraints: Vec::new(),
            layout: Vec::new(),
            style: Vec::new(),
            input: Vec::new(),
            scroll: Vec::new(),
            render: Vec::new(),
            transform: Vec::new(),
            widget: Vec::new(),
            ids: Vec::new(),
        };
        // Add dummy entry for index 0 (INVALID_NODE)
        store.push_dummy();
        store
    }

    fn push_dummy(&mut self) {
        self.parents.push(None);
        self.children.push(Vec::new());
        self.constraints.push(LayoutConstraints::default());
        self.layout.push(LayoutData::default());
        self.style.push(ResolvedStyle::default());
        self.input.push(InputState::default());
        self.scroll.push(ScrollState::default());
        self.render.push(RenderData::default());
        self.transform.push(Transform::default());
        self.widget.push(WidgetKind::None);
        self.ids.push(0);
    }

    pub fn add_node(&mut self, kind: WidgetKind, stable_id: super::types::StableID) -> NodeID {
        let id = self.parents.len();
        self.parents.push(None); // No parent initially
        self.children.push(Vec::new());
        self.constraints.push(LayoutConstraints::default());
        self.layout.push(LayoutData::default());
        self.style.push(ResolvedStyle::default());
        self.input.push(InputState::default());
        self.scroll.push(ScrollState::default());
        self.render.push(RenderData::default());
        self.transform.push(Transform::default());
        self.widget.push(kind);
        self.ids.push(stable_id);
        id
    }

    pub fn set_parent(&mut self, child: NodeID, parent: NodeID) {
        if child == 0 || child >= self.parents.len() { return; }
        // Remove from old parent children list if exists (simplified: assume fresh or handle move)
        if let Some(old_parent) = self.parents[child] {
            if let Some(siblings) = self.children.get_mut(old_parent) {
                siblings.retain(|&x| x != child);
            }
        }

        self.parents[child] = Some(parent);
        if parent != 0 && parent < self.children.len() {
            self.children[parent].push(child);
        }
    }

    pub fn clear(&mut self) {
        self.parents.clear();
        self.children.clear();
        self.constraints.clear();
        self.layout.clear();
        self.style.clear();
        self.input.clear();
        self.scroll.clear();
        self.render.clear();
        self.transform.clear();
        self.widget.clear();
        self.push_dummy();
    }
}
