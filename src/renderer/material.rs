use crate::core::{ColorF, Vec2};

pub type TextureId = u64;

#[derive(Clone, Debug)]
pub struct Material {
    pub albedo_color: ColorF,
    pub albedo_map: Option<TextureId>,
    pub normal_map: Option<TextureId>,
    pub roughness: f32,
    pub metallic: f32,
    pub reflectivity: f32,
    pub emissive_intensity: f32,
    pub emissive_color: ColorF,
    pub parallax_factor: f32,
    pub distortion_strength: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo_color: ColorF::WHITE,
            albedo_map: None,
            normal_map: None,
            roughness: 0.5,
            metallic: 0.0,
            reflectivity: 0.0,
            emissive_intensity: 0.0,
            emissive_color: ColorF::WHITE,
            parallax_factor: 0.0,
            distortion_strength: 0.0,
        }
    }
}

impl Material {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_albedo(mut self, color: ColorF) -> Self {
        self.albedo_color = color;
        self
    }

    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }

    pub fn with_emissive(mut self, color: ColorF, intensity: f32) -> Self {
        self.emissive_color = color;
        self.emissive_intensity = intensity;
        self
    }
}
