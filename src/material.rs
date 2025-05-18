use crate::color::Color;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub color: Color,
    pub reflectivity: f32, // mirror-like reflections
    pub specular_intensity: f32, // strength of the specular highlight
    pub shininess: f32,      // exponent for specular highlight size
}

impl Material {
    pub fn new(color: Color, reflectivity: f32, specular_intensity: f32, shininess: f32) -> Self {
        Self {
            color,
            reflectivity,
            specular_intensity,
            shininess,
        }
    }
}
