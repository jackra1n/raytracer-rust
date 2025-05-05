use crate::hittable::HitData;
use crate::hittable::Object;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3;

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Material,
}

impl Object for Sphere {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitData> {
        let oc = ray.start - self.center;
        let a = ray.dir.dot(ray.dir);
        let half_b = oc.dot(ray.dir);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root <= t_min || root >= t_max {
            root = (-half_b + sqrtd) / a;
            if root <= t_min || root >= t_max {
                return None;
            }
        }

        let t = root;
        let position = ray.at(t);
        let normal = (position - self.center) / self.radius;

        Some(HitData {
            t,
            position,
            normal,
            material: self.material,
        })
    }
}
