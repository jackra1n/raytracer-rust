use crate::color::Color;
use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::vec3::Vec3;
use rand::{Rng, RngCore};
use serde::Deserialize;
use std::f32::consts::PI;
use std::sync::Arc;

const EPSILON: f32 = 1e-4;

pub trait Material: Send + Sync {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit_record: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)>;
    fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color {
        Color::BLACK
    }
}

#[derive(Clone)]
pub enum AlbedoKind {
    Solid(Color),
    Checked(Arc<crate::tungsten::CheckerTexture>),
}

pub struct Lambertian {
    pub albedo_kind: AlbedoKind,
}

impl Lambertian {
    pub fn new_solid(albedo: Color) -> Self {
        Self {
            albedo_kind: AlbedoKind::Solid(albedo),
        }
    }
    pub fn new_checker(texture: Arc<crate::tungsten::CheckerTexture>) -> Self {
        Self {
            albedo_kind: AlbedoKind::Checked(texture),
        }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        _ray_in: &Ray,
        hit_record: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
        let mut scatter_direction =
            hit_record.normal + Vec3::random_in_unit_sphere(rng).normalized();

        if scatter_direction.near_zero() {
            scatter_direction = hit_record.normal;
        }

        let scattered_origin = hit_record.position + hit_record.normal * EPSILON;
        let scattered_ray = Ray::new(scattered_origin, scatter_direction.normalized());

        let surface_albedo = match &self.albedo_kind {
            AlbedoKind::Solid(color) => *color,
            AlbedoKind::Checked(checker_texture) => checker_texture.value(&hit_record.position),
        };

        Some((scattered_ray, surface_albedo))
    }
}

pub struct Metal {
    pub albedo: Color,
    pub fuzz: f32,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f32) -> Self {
        Self {
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
        }
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit_record: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
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
    pub fn new(refractive_index: f32) -> Self {
        Self { refractive_index }
    }
}

impl Material for Dielectric {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit_record: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
        let attenuation = Color::WHITE;

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

        let reflectance = schlick_reflectance(cos_theta, 1.0 / refraction_ratio);

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
    pub fn new(color: Color) -> Self {
        Self { color }
    }
}

impl Material for EmissiveLight {
    fn scatter(
        &self,
        _ray_in: &Ray,
        _hit_record: &HitRecord,
        _rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
        None
    }

    fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color {
        self.color
    }
}

fn reflect(v_in: Vec3, n_reflect: Vec3) -> Vec3 {
    if v_in.x.is_nan() || v_in.y.is_nan() || v_in.z.is_nan() {
        return Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    }
    if n_reflect.x.is_nan()
        || n_reflect.y.is_nan()
        || n_reflect.z.is_nan()
        || (n_reflect.x == 0.0 && n_reflect.y == 0.0 && n_reflect.z == 0.0)
    {
        return Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    }
    v_in - n_reflect * 2.0 * v_in.dot(n_reflect)
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

// --- Null Material ---
#[derive(Debug, Clone, Copy)]
pub struct NullMaterial;

impl NullMaterial {
    pub fn new() -> Self {
        NullMaterial
    }
}

impl Material for NullMaterial {
    fn scatter(
        &self,
        _ray_in: &Ray,
        _hit_record: &HitRecord,
        _rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
        None // null material does not scatter light
    }

    fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color {
        Color::BLACK // null material does not emit light by default
    }
}
