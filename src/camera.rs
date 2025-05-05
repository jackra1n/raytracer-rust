use crate::ray::Ray;
use crate::vec3::Vec3;

pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub true_up: Vec3,
    pub half_width: f32,
    pub half_height: f32,
}

impl Camera {
    pub fn new(position: Vec3, look_at: Vec3, world_up: Vec3, fov: f32, aspect: f32) -> Self {
        let forward = (look_at - position).normalized();
        let right = forward.cross(world_up.normalized()).normalized();
        let true_up = right.cross(forward).normalized();

        let fov_rad = fov * std::f32::consts::PI / 180.0;
        let half_height = (fov_rad / 2.0).tan();
        let half_width = half_height * aspect;

        Camera {
            position,
            forward,
            right,
            true_up,
            half_width,
            half_height,
        }
    }

    pub fn get_ray_uv(&self, u: f32, v: f32) -> Ray {
        let ndc_x = 2.0 * u - 1.0;
        let ndc_y = 1.0 - 2.0 * v;

        let offset =
            self.right * (ndc_x * self.half_width) + self.true_up * (ndc_y * self.half_height);
        let ray_dir = (self.forward + offset).normalized();

        Ray::new(self.position, ray_dir)
    }
}
