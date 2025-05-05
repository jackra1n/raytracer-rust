use crate::material::Material;
use crate::vec3::Vec3;

pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub normal: Vec3,
    pub material: Material,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material: Material) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalized();
        Self {
            v0,
            v1,
            v2,
            normal,
            material,
        }
    }
}
