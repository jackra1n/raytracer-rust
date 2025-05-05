use crate::hittable::HitData;
use crate::hittable::Object;
use crate::material::Material;
use crate::ray::Ray;
use crate::renderer::EPSILON;
use crate::vec3::Vec3;

pub struct Plane {
    p1: Vec3,
    normal: Vec3,
    material: Material,
}

impl Plane {
    pub fn new(point: Vec3, normal: Vec3, material: Material) -> Self {
        Self {
            p1: point,
            normal: normal.normalized(),
            material,
        }
    }
}

impl Object for Plane {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitData> {
        let denom = self.normal.dot(ray.dir);

        if denom.abs() < EPSILON {
            return None;
        }

        let t = self.normal.dot(self.p1 - ray.start) / denom;

        if t <= t_min || t >= t_max {
            return None;
        }

        let position = ray.at(t);
        Some(HitData {
            t,
            position,
            normal: if denom < 0.0 {
                self.normal
            } else {
                self.normal * -1.0
            },
            material: self.material,
        })
    }
}
