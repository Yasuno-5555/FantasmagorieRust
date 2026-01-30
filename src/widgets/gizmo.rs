use crate::core::{Vec3, ID};
use crate::view::gizmo::{GizmoData, GizmoMode};
use crate::view::header::{ViewHeader, ViewType};
use crate::widgets::UIContext;

pub struct TransformGizmoBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    id: ID,
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    mode: GizmoMode,
    snap_enabled: bool,
}

impl<'b, 'a> TransformGizmoBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, id: &str) -> Self {
        Self {
            ui,
            id: ID::from_str(id),
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            mode: GizmoMode::Translate,
            snap_enabled: false,
        }
    }

    pub fn position(mut self, p: Vec3) -> Self {
        self.position = p;
        self
    }

    pub fn rotation(mut self, r: Vec3) -> Self {
        self.rotation = r;
        self
    }

    pub fn scale(mut self, s: Vec3) -> Self {
        self.scale = s;
        self
    }

    pub fn mode(mut self, m: GizmoMode) -> Self {
        self.mode = m;
        self
    }

    pub fn snap(mut self, enabled: bool) -> Self {
        self.snap_enabled = enabled;
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let view = self.ui.arena.alloc(ViewHeader {
            view_type: ViewType::TransformGizmo,
            id: std::cell::Cell::new(self.id),
            ..Default::default()
        });

        // Gizmo fills space but renders as overlay
        view.width.set(f32::NAN);
        view.height.set(f32::NAN);
        view.flex_grow.set(1.0);

        let data = self.ui.arena.alloc(GizmoData::new());
        data.position.set(self.position);
        data.rotation.set(self.rotation);
        data.scale.set(self.scale);
        data.mode.set(self.mode);

        let mut snap = data.snap_context.get();
        snap.enabled = self.snap_enabled;
        data.snap_context.set(snap);

        view.gizmo_data.set(Some(data));

        self.ui.push_child(view);
        view
    }
}
