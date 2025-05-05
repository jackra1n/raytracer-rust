use crate::color::Color;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub color: Color,
    pub reflectivity: f32,
}

impl Material {
    pub fn new(color: Color, reflectivity: f32) -> Self {
        Self {
            color,
            reflectivity,
        }
    }
}
