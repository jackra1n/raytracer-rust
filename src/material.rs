use crate::color::Color;
use crate::ray::Ray;
use crate::hittable::HitRecord;
use crate::vec3::Vec3;
use rand::{Rng, RngCore};


const EPSILON: f32 = 1e-4;

pub trait Material: Send + Sync {
    fn scatter(&self, ray_in: &Ray, hit_record: &HitRecord, rng: &mut dyn RngCore) -> Option<(Ray, Color)>;
    fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color {
        Color::BLACK
    }
}

pub struct Lambertian {
    pub albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self { Self { albedo } }
}

impl Material for Lambertian {
    fn scatter(&self, _ray_in: &Ray, hit_record: &HitRecord, rng: &mut dyn RngCore) -> Option<(Ray, Color)> {
        let mut scatter_direction = hit_record.normal + Vec3::random_in_unit_sphere(rng).normalized();

        if scatter_direction.near_zero() {
            scatter_direction = hit_record.normal;
        }

        let scattered_origin = hit_record.position + hit_record.normal * EPSILON;
        let scattered_ray = Ray::new(scattered_origin, scatter_direction.normalized());
        Some((scattered_ray, self.albedo))
    }
}

pub struct Metal {
    pub albedo: Color,
    pub fuzz: f32,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f32) -> Self { Self { albedo, fuzz: fuzz.clamp(0.0, 1.0) } }
}

impl Material for Metal {
    fn scatter(&self, ray_in: &Ray, hit_record: &HitRecord, rng: &mut dyn RngCore) -> Option<(Ray, Color)> {
        let reflected_direction = reflect(ray_in.direction.normalized(), hit_record.normal);
        
        let fuzzed_direction = if self.fuzz > 0.0 {
            reflected_direction + Vec3::random_in_unit_sphere(rng) * self.fuzz
        } else {
            reflected_direction
        };

        if fuzzed_direction.dot(hit_record.normal) > 0.0 {
            let scattered_origin = hit_record.position + hit_record.normal * EPSILON;
            let scattered_ray = Ray::new(scattered_origin, fuzzed_direction.normalized());
            Some((scattered_ray, self.albedo))
        } else {
            None
        }
    }
}

pub struct Dielectric {
    pub refractive_index: f32,
}

impl Dielectric {
    pub fn new(refractive_index: f32) -> Self { Self { refractive_index } }
}

impl Material for Dielectric {
    fn scatter(&self, ray_in: &Ray, hit_record: &HitRecord, rng: &mut dyn RngCore) -> Option<(Ray, Color)> {
        let attenuation = Color::new(1.0, 1.0, 1.0);

        let refraction_ratio = if hit_record.front_face {
            1.0 / self.refractive_index
        } else {
            self.refractive_index / 1.0
        };

        let unit_direction = ray_in.direction.normalized();
        
        let cos_theta = (-unit_direction).dot(hit_record.normal).min(1.0);
        let sin_theta_squared = 1.0 - cos_theta * cos_theta;

        let cannot_refract = refraction_ratio * refraction_ratio * sin_theta_squared > 1.0;
        let scatter_direction: Vec3;

        let r0_intermediate = (1.0 - self.refractive_index) / (1.0 + self.refractive_index);
        let r0 = r0_intermediate * r0_intermediate;
        let reflectance = r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5);


        if cannot_refract || reflectance > rng.random::<f32>() {
            scatter_direction = reflect(unit_direction, hit_record.normal);
        } else {
            scatter_direction = refract(unit_direction, hit_record.normal, refraction_ratio)
                                .expect("Refraction failed unexpectedly after check");
        }
        
        let scattered_origin = if scatter_direction.dot(hit_record.normal) > 0.0 {
             hit_record.position + hit_record.normal * EPSILON 
        } else {
             hit_record.position - hit_record.normal * EPSILON 
        };

        let scattered_ray = Ray::new(scattered_origin, scatter_direction.normalized());
        Some((scattered_ray, attenuation))
    }

    fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color {
        Color::BLACK
    }
}


pub struct EmissiveLight {
    pub color: Color,
}

impl EmissiveLight {
    pub fn new(color: Color) -> Self { Self { color } }
}

impl Material for EmissiveLight {
    fn scatter(&self, _ray_in: &Ray, _hit_record: &HitRecord, _rng: &mut dyn RngCore) -> Option<(Ray, Color)> {
        None
    }

    fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color {
        self.color
    }
}
fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - n * 2.0 * v.dot(n)
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Option<Vec3> {
    let cos_theta = (-uv).dot(n).min(1.0);
    let r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    let r_out_parallel_squared = 1.0 - r_out_perp.length_squared();

    if r_out_parallel_squared < 0.0 {
        None
    } else {
        let r_out_parallel = n * (-r_out_parallel_squared.sqrt());
        Some(r_out_perp + r_out_parallel)
    }
}

fn schlick_reflectance(cosine: f32, ref_idx_ratio: f32) -> f32 {
    let r0_num = 1.0 - ref_idx_ratio;
    let r0_den = 1.0 + ref_idx_ratio;
    let mut r0 = r0_num / r0_den;
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}