use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3;

pub trait Object {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitData>;
}

pub struct HitData {
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub material: Material,
}
