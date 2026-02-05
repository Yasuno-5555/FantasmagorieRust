use super::api::FrameContext;
use super::frame::RenderCommand;
use super::types::*;

/// A builder for constructing high-feature SDF shapes.
///
/// This provides a fluent API for adding rounded corners, borders, 
/// glows, and elevation to UI rectangles.
#[must_use = "submit() must be called to record the draw command"]
pub struct DrawBuilder<'a> {
    frame: &'a mut FrameContext,
    rect: Rect,
    color: Color,
    radii: CornerRadii,
    border: Option<Border>,
    glow: Option<Glow>,
    elevation: f32,
    is_squircle: bool,
    morph: f32,
    id: Option<crate::core::ID>,
    reflectivity: f32,
    roughness: f32,
    normal_map: Option<u64>,
    distortion_strength: f32,
    emissive_intensity: f32,
    parallax_factor: f32,
}

impl<'a> DrawBuilder<'a> {
    pub(crate) fn new(frame: &'a mut FrameContext, rect: Rect, color: Color) -> Self {
        Self {
            frame,
            rect,
            color,
            radii: CornerRadii::default(),
            border: None,
            glow: None,
            elevation: 0.0,
            is_squircle: false,
            morph: 0.0,
            id: None,
            reflectivity: 0.0,
            roughness: 0.5,
            normal_map: None,
            distortion_strength: 0.0,
            emissive_intensity: 0.0,
            parallax_factor: 0.0,
        }
    }

    /// Set a uniform corner radius for all corners.
    pub fn rounded(mut self, radius: f32) -> Self {
        self.radii = CornerRadii::uniform(radius);
        self
    }

    /// Set individual corner radii.
    pub fn radii(mut self, tl: f32, tr: f32, br: f32, bl: f32) -> Self {
        self.radii = CornerRadii::new(tl, tr, br, bl);
        self
    }

    /// Add a border to the shape.
    pub fn border(mut self, width: f32, color: Color) -> Self {
        self.border = Some(Border { width, color });
        self
    }

    /// Add a glow (outer highlight) or glow-based shadow.
    pub fn glow(mut self, strength: f32, color: Color) -> Self {
        self.glow = Some(Glow { strength, color });
        self
    }

    /// Set the elevation for drop-shadow effects.
    pub fn elevation(mut self, elevation: f32) -> Self {
        self.elevation = elevation;
        self
    }

    /// Enable squircle (continuous curvature) corners instead of standard rounded ones.
    pub fn squircle(mut self) -> Self {
        self.is_squircle = true;
        self
    }

    /// Set the motion morph weight (0.0 to 1.0).
    pub fn morph(mut self, weight: f32) -> Self {
        self.morph = weight;
        self
    }

    /// Set a persistent ID for motion tracking.
    pub fn id(mut self, id: crate::core::ID) -> Self {
        self.id = Some(id);
        self
    }

    /// Set the reflectivity for SSR (0.0 to 1.0).
    pub fn reflectivity(mut self, value: f32) -> Self {
        self.reflectivity = value;
        self
    }

    /// Set the roughness (0.0: shiny/mirror, 1.0: rough/matte).
    pub fn roughness(mut self, value: f32) -> Self {
        self.roughness = value;
        self
    }

    /// Set a normal map (texture ID).
    pub fn normal_map(mut self, id: u64) -> Self {
        self.normal_map = Some(id);
        self
    }

    /// Set the distortion/refraction strength.
    pub fn distortion(mut self, strength: f32) -> Self {
        self.distortion_strength = strength;
        self
    }

    /// Set the emissive brightness (0.0+: triggers bloom).
    pub fn emissive(mut self, intensity: f32) -> Self {
        self.emissive_intensity = intensity;
        self
    }

    /// Set the parallax depth factor.
    pub fn parallax(mut self, factor: f32) -> Self {
        self.parallax_factor = factor;
        self
    }

    /// Submit the command to the frame.
    ///
    /// This consumes the builder and pushes a `RenderCommand::DrawShape` 
    /// to the current frame context.
    pub fn submit(self) {
        self.frame.push_command(RenderCommand::DrawShape {
            rect: self.rect,
            color: self.color,
            radii: self.radii,
            border: self.border,
            glow: self.glow,
            elevation: self.elevation,
            is_squircle: self.is_squircle,
            morph: self.morph,
            id: self.id,
            reflectivity: self.reflectivity,
            roughness: self.roughness,
            normal_map: self.normal_map,
            distortion_strength: self.distortion_strength,
            emissive_intensity: self.emissive_intensity,
            parallax_factor: self.parallax_factor,
        });
    }
}
