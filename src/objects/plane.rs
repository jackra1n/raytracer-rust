use crate::hittable::HitRecord;
use crate::hittable::Hittable;
use crate::material::Material;
use crate::ray::Ray;
use crate::renderer::EPSILON;
use crate::vec3::Vec3;
use std::sync::Arc;

pub struct Plane {
    p1: Vec3,
    normal: Vec3,
    material: Arc<dyn Material>,
}

impl Plane {
    pub fn new(point: Vec3, normal: Vec3, material: Arc<dyn Material>) -> Self {
        Self {
            p1: point,
            normal: normal.normalized(),
            material,
        }
    }
}

impl Hittable for Plane {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);

        if denom.abs() < EPSILON {
            return None;
        }

        let t = self.normal.dot(self.p1 - ray.origin) / denom;

        if t <= t_min || t >= t_max {
            return None;
        }

        let position = ray.at(t);

        let outward_normal = self.normal;
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        let hit_record_normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };

        Some(HitRecord {
            t,
            position,
            normal: hit_record_normal,
            material: self.material.clone(),
            front_face,
        })
    }
}
