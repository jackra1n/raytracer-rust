use crate::color::Color;
use crate::vec3::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub position: Vec3,
    pub color: Color,
    pub strength: f32,
}

impl Light {
    pub fn new(pos: Vec3, color: Color, strength: f32) -> Self {
        Self {
            position: pos,
            color,
            strength,
        }
    }
}
