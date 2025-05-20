use crate::hittable::HitRecord;
use crate::hittable::Hittable;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3 as CrateVec3;
use std::sync::Arc;
use glam::{Mat4, Vec3, Vec4, Affine3A};

use crate::renderer::EPSILON;

pub struct Cube {
    object_to_world: Mat4,
    world_to_object: Mat4,
    material: Arc<dyn Material>,
    local_min: Vec3,
    local_max: Vec3,
}

impl Cube {
    pub fn new_transformed(object_to_world: Mat4, material: Arc<dyn Material>) -> Self {
        let world_to_object = object_to_world.inverse();
        Self {
            object_to_world,
            world_to_object,
            material,
            local_min: Vec3::new(-0.5, -0.5, -0.5),
            local_max: Vec3::new(0.5, 0.5, 0.5),
        }
    }

    #[allow(dead_code)]
    pub fn new_pos_size(bottom_center: CrateVec3, size: CrateVec3, material: Arc<dyn Material>) -> Self {
        let half_size_glam = Vec3::new(size.x * 0.5, size.y * 0.5, size.z * 0.5);
        let center_glam = Vec3::new(bottom_center.x, bottom_center.y + half_size_glam.y, bottom_center.z);
        
        let min_glam = center_glam - half_size_glam;
        let max_glam = center_glam + half_size_glam;

        let translation = Mat4::from_translation(center_glam);
        let scale_mat = Mat4::from_scale(Vec3::new(size.x, size.y, size.z));
        let object_to_world = translation * scale_mat;

        Self {
            object_to_world,
            world_to_object: object_to_world.inverse(),
            material,
            local_min: Vec3::new(-0.5, -0.5, -0.5),
            local_max: Vec3::new(0.5, 0.5, 0.5),
        }
    }
}

impl Hittable for Cube {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let ray_origin_crate = ray.origin;
        let ray_direction_crate = ray.direction;

        let ray_origin_obj_h = self.world_to_object * Vec4::new(ray_origin_crate.x, ray_origin_crate.y, ray_origin_crate.z, 1.0);
        let ray_direction_obj_h = self.world_to_object * Vec4::new(ray_direction_crate.x, ray_direction_crate.y, ray_direction_crate.z, 0.0);

        let ray_origin_obj = Vec3::new(ray_origin_obj_h.x, ray_origin_obj_h.y, ray_origin_obj_h.z);
        let ray_direction_obj = Vec3::new(ray_direction_obj_h.x, ray_direction_obj_h.y, ray_direction_obj_h.z);

        let inv_dir = Vec3::ONE / ray_direction_obj;
        let t1 = (self.local_min - ray_origin_obj) * inv_dir;
        let t2 = (self.local_max - ray_origin_obj) * inv_dir;

        let t_enter_vec = t1.min(t2);
        let t_exit_vec = t1.max(t2);

        let mut t_enter = t_enter_vec.x.max(t_enter_vec.y.max(t_enter_vec.z));
        let mut t_exit = t_exit_vec.x.min(t_exit_vec.y.min(t_exit_vec.z));
        
        if t_exit < t_enter || t_exit <= 0.0 {
            return None;
        }

        let t_hit_obj = if t_enter > 0.0 { t_enter } else { t_exit };

        if t_hit_obj >= t_max || t_hit_obj <= t_min || t_hit_obj < EPSILON {
            return None;
        }
        
        let position_obj = ray_origin_obj + ray_direction_obj * t_hit_obj;

        let mut normal_obj = Vec3::ZERO;
        let abs_pos_obj = position_obj.abs();
        let tolerance = 1e-4;

        if (abs_pos_obj.x - 0.5).abs() < tolerance {
            normal_obj.x = position_obj.x.signum();
        } else if (abs_pos_obj.y - 0.5).abs() < tolerance {
            normal_obj.y = position_obj.y.signum();
        } else if (abs_pos_obj.z - 0.5).abs() < tolerance {
            normal_obj.z = position_obj.z.signum();
        } else {
            if abs_pos_obj.x > abs_pos_obj.y && abs_pos_obj.x > abs_pos_obj.z {
                 normal_obj.x = position_obj.x.signum();
            } else if abs_pos_obj.y > abs_pos_obj.z {
                 normal_obj.y = position_obj.y.signum();
            } else {
                 normal_obj.z = position_obj.z.signum();
            }
        }
        normal_obj = normal_obj.normalize_or_zero();

        let position_world_h = self.object_to_world * Vec4::new(position_obj.x, position_obj.y, position_obj.z, 1.0);
        let position_world = CrateVec3::new(position_world_h.x, position_world_h.y, position_world_h.z);

        let normal_world_h = self.world_to_object.transpose() * Vec4::new(normal_obj.x, normal_obj.y, normal_obj.z, 0.0);
        let mut normal_world = CrateVec3::new(normal_world_h.x, normal_world_h.y, normal_world_h.z).normalized();

        let mut hit_record = HitRecord {
            t: t_hit_obj,
            position: position_world,
            normal: normal_world,
            material: self.material.clone(),
            front_face: false,
        };
        
        let p_minus_o = hit_record.position - ray.origin;
        if p_minus_o.dot(ray.direction) < 0.0 {
             return None; 
        }
        let t_world = (hit_record.position - ray.origin).dot(ray.direction);

        if t_world < t_min || t_world > t_max {
            return None;
        }
        hit_record.t = t_world;

        hit_record.set_face_normal(&ray, normal_world);

        Some(hit_record)
    }
}
