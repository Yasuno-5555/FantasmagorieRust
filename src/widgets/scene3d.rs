use crate::core::{Vec3, ID};
use crate::view::header::{ViewHeader, ViewType};
use crate::view::scene3d::Scene3DData;
use crate::widgets::UIContext;

pub struct Scene3DBuilder<'b, 'a> {
    ui: &'b mut UIContext<'a>,
    scene_id: ID,
    texture_id: u64,
    camera_pos: Vec3,
    camera_target: Vec3,
    fov: f32,
    height: f32,
    width_fill: bool,
    lut_id: u64,
    lut_intensity: f32,
}

impl<'b, 'a> Scene3DBuilder<'b, 'a> {
    pub fn new(ui: &'b mut UIContext<'a>, id: &str) -> Self {
        Self {
            ui,
            scene_id: ID::from_str(id),
            texture_id: 0,
            camera_pos: Vec3::new(0.0, 0.0, 5.0),
            camera_target: Vec3::ZERO,
            fov: 45.0,
            height: 300.0,
            width_fill: true,
            lut_id: 0,
            lut_intensity: 1.0,
        }
    }

    pub fn lut(mut self, id: u64, intensity: f32) -> Self {
        self.lut_id = id;
        self.lut_intensity = intensity;
        self
    }

    pub fn texture(mut self, id: u64) -> Self {
        self.texture_id = id;
        self
    }

    pub fn camera(mut self, pos: Vec3, target: Vec3) -> Self {
        self.camera_pos = pos;
        self.camera_target = target;
        self
    }

    pub fn fov(mut self, f: f32) -> Self {
        self.fov = f;
        self
    }

    pub fn height(mut self, h: f32) -> Self {
        self.height = h;
        self
    }

    pub fn build(self) -> &'a ViewHeader<'a> {
        let view = self.ui.arena.alloc(ViewHeader {
            view_type: ViewType::Scene3D,
            id: std::cell::Cell::new(self.scene_id),
            clip: true.into(),
            ..Default::default()
        });

        view.height.set(self.height);
        if self.width_fill {
            view.flex_grow.set(1.0);
            view.width.set(f32::NAN);
        }

        let scene_data = self.ui.arena.alloc(Scene3DData::new(self.scene_id));
        scene_data.texture_id.set(self.texture_id);
        scene_data.camera_pos.set(self.camera_pos);
        scene_data.camera_target.set(self.camera_target);
        scene_data.fov.set(self.fov);
        scene_data.lut_id.set(self.lut_id);
        scene_data.lut_intensity.set(self.lut_intensity);

        view.scene_data.set(Some(scene_data));
        self.ui.push_child(view);
        view
    }
}
