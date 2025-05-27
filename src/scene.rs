use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::ray::Ray;
use image::DynamicImage;
use image::{ImageBuffer, Rgb};

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
