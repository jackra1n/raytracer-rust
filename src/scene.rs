use crate::color::Color;
use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::material::{Dielectric, EmissiveLight, Lambertian, Material, Metal};
use crate::mesh::mesh_object::Mesh;
use crate::objects::cube::Cube;
use crate::objects::plane::Plane;
use crate::objects::sphere::Sphere;
use crate::ray::Ray;
use crate::vec3::Vec3;
use glam::{Mat4, Quat, Vec3 as GlamVec3};
use image::DynamicImage;
use image::{ImageBuffer, Rgb};
use std::sync::Arc;

pub struct Scene {
    pub object_list: HittableList,
    pub skybox_image: Option<DynamicImage>,
    pub skybox_hdr_image: Option<ImageBuffer<Rgb<f32>, Vec<f32>>>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            object_list: HittableList::new(),
            skybox_image: None,
            skybox_hdr_image: None,
        }
    }

    pub fn add_object(&mut self, obj: Box<dyn Hittable + Sync>) {
        self.object_list.add(obj);
    }
}

impl Hittable for Scene {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        self.object_list.hit(ray, t_min, t_max)
    }
}
