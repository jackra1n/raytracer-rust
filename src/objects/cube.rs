use crate::hittable::HitRecord;
use crate::hittable::Hittable;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3;
use std::sync::Arc;

use crate::renderer::EPSILON;

pub struct Cube {
    pub min: Vec3,
    pub max: Vec3,
    pub material: Arc<dyn Material>,
}

impl Cube {
    pub fn new_pos_size(bottom_center: Vec3, size: Vec3, material: Arc<dyn Material>) -> Self {
        let half_size = size * 0.5;
        let center = bottom_center + Vec3::new(0.0, half_size.y, 0.0);
        let min = center - half_size;
        let max = center + half_size;
        Self { min, max, material }
    }
}

impl Hittable for Cube {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let inv_dir = Vec3::new(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
        let tx1 = (self.min.x - ray.origin.x) * inv_dir.x;
        let tx2 = (self.max.x - ray.origin.x) * inv_dir.x;

        let mut t_enter = tx1.min(tx2);
        let mut t_exit = tx1.max(tx2);

        let ty1 = (self.min.y - ray.origin.y) * inv_dir.y;
        let ty2 = (self.max.y - ray.origin.y) * inv_dir.y;

        t_enter = t_enter.max(ty1.min(ty2));
        t_exit = t_exit.min(ty1.max(ty2));

        let tz1 = (self.min.z - ray.origin.z) * inv_dir.z;
        let tz2 = (self.max.z - ray.origin.z) * inv_dir.z;

        t_enter = t_enter.max(tz1.min(tz2));
        t_exit = t_exit.min(tz1.max(tz2));

        if t_exit < t_enter || t_exit <= t_min || t_enter >= t_max {
            return None;
        }

        let t = if t_enter > t_min { t_enter } else { t_exit };

        if t >= t_max {
            return None;
        }

        let position = ray.at(t);

        let center = (self.min + self.max) * 0.5;
        let dimensions = self.max - self.min;

        let rel_scaled = Vec3::new(
            if dimensions.x.abs() < EPSILON {
                0.0
            } else {
                (position.x - center.x) / (dimensions.x * 0.5)
            },
            if dimensions.y.abs() < EPSILON {
                0.0
            } else {
                (position.y - center.y) / (dimensions.y * 0.5)
            },
            if dimensions.z.abs() < EPSILON {
                0.0
            } else {
                (position.z - center.z) / (dimensions.z * 0.5)
            },
        );

        let abs_x = rel_scaled.x.abs();
        let abs_y = rel_scaled.y.abs();
        let abs_z = rel_scaled.z.abs();

        let normal = if abs_x > abs_y && abs_x > abs_z {
            Vec3::new(rel_scaled.x.signum(), 0.0, 0.0)
        } else if abs_y > abs_z {
            Vec3::new(0.0, rel_scaled.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, rel_scaled.z.signum())
        };

        Some(HitRecord {
            t,
            position,
            normal: normal.normalized(),
            material: self.material.clone(),
            front_face: true,
        })
    }
}
