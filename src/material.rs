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
    Checked(Arc<CheckerTexture>),
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
    pub fn new_checker(texture: Arc<CheckerTexture>) -> Self {
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





// from tungsten


#[derive(Clone)]
pub struct PlasticMaterial {
    pub albedo: Color,
    pub ior: f32,
}

impl PlasticMaterial {
    pub fn new(albedo: Color, ior: f32) -> Self {
        PlasticMaterial { albedo, ior }
    }
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0_sq = r0 * r0;
    r0_sq + (1.0 - r0_sq) * (1.0 - cosine).powi(5)
}

impl Material for PlasticMaterial {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit_record: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
        let reflected_color = Color::new(0.9, 0.9, 0.9); // color of the reflection, can be white or tinted by albedo

        let outward_normal;
        let ni_over_nt;
        let cosine;

        if ray_in.direction.dot(hit_record.normal) > 0.0 {
            outward_normal = -hit_record.normal;
            ni_over_nt = self.ior;
            cosine = self.ior * ray_in.direction.dot(hit_record.normal) / ray_in.direction.length();
        } else {
            outward_normal = hit_record.normal;
            ni_over_nt = 1.0 / self.ior;
            cosine = -ray_in.direction.dot(hit_record.normal) / ray_in.direction.length();
        }

        let reflect_prob = schlick(cosine, self.ior);

        if rng.random::<f32>() < reflect_prob {
            // specular reflection (like metal with fuzz 0)
            let reflected_dir = ray_in.direction.reflect(hit_record.normal).normalized();
            let scattered_origin =
                hit_record.position + hit_record.normal * crate::renderer::EPSILON;
            let scattered_ray = Ray::new(scattered_origin, reflected_dir);
            Some((scattered_ray, reflected_color))
        } else {
            // diffuse scatter (like lambertian)
            let mut scatter_direction =
                hit_record.normal + Vec3::random_in_unit_sphere(rng).normalized();
            if scatter_direction.near_zero() {
                scatter_direction = hit_record.normal;
            }
            let scattered_origin =
                hit_record.position + hit_record.normal * crate::renderer::EPSILON;
            let scattered_ray = Ray::new(scattered_origin, scatter_direction.normalized());
            Some((scattered_ray, self.albedo))
        }
    }

    fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color {
        Color::BLACK
    }
}

#[derive(Clone, Debug)]
pub struct CheckerTexture {
    pub on_color: Color,
    pub off_color: Color,
    pub inv_scale: f32,
}

impl CheckerTexture {
    pub fn new(on_color: Color, off_color: Color, scale: f32) -> Self {
        let inv_scale = if scale.abs() < 1e-6 { 1.0 } else { 1.0 / scale };
        Self {
            on_color,
            off_color,
            inv_scale,
        }
    }

    pub fn value(&self, p: &Vec3) -> Color {
        let x_check = (p.x * self.inv_scale).floor() as i32;
        let y_check = (p.y * self.inv_scale).floor() as i32;
        let z_check = (p.z * self.inv_scale).floor() as i32;

        if (x_check + y_check + z_check) % 2 == 0 {
            self.on_color
        } else {
            self.off_color
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub enum MetalType {
    Cu, // Copper
    Au, // Gold
    Ag, // Silver
    Al, // Aluminum
    Ni, // Nickel
    Ti, // Titanium
    Fe, // Iron
    Pb, // Lead
    Custom(Color),
}

impl MetalType {
    pub fn ior_k(&self) -> (Color, Color) {
        match self {
            MetalType::Cu => (
                Color::new(0.200, 1.090, 1.420),
                Color::new(3.910, 2.570, 2.300),
            ),
            MetalType::Au => (
                Color::new(0.170, 0.350, 1.500),
                Color::new(3.140, 2.300, 1.920),
            ),
            MetalType::Ag => (
                Color::new(0.155, 0.145, 0.135),
                Color::new(3.910, 2.610, 2.370),
            ),
            MetalType::Al => (
                Color::new(1.360, 0.965, 0.620),
                Color::new(7.570, 6.690, 5.440),
            ),
            MetalType::Ni => (
                Color::new(1.920, 1.920, 1.920),
                Color::new(3.670, 3.670, 3.670),
            ),
            MetalType::Ti => (
                Color::new(2.740, 2.740, 2.740),
                Color::new(3.170, 3.170, 3.170),
            ),
            MetalType::Fe => (
                Color::new(2.870, 2.870, 2.870),
                Color::new(3.140, 3.140, 3.140),
            ),
            MetalType::Pb => (
                Color::new(1.910, 1.910, 1.910),
                Color::new(3.180, 3.180, 3.180),
            ),
            MetalType::Custom(c) => (*c, Color::new(1.0, 1.0, 1.0)),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub enum MicrofacetDistribution {
    GGX,
    Beckmann,
}

pub struct RoughConductor {
    pub albedo: Color,
    pub roughness: f32,
    pub metal_type: MetalType,
    pub distribution: MicrofacetDistribution,
}

impl RoughConductor {
    pub fn new(
        albedo: Color,
        roughness: f32,
        metal_type: MetalType,
        distribution: MicrofacetDistribution,
    ) -> Self {
        Self {
            albedo,
            roughness: roughness.max(0.01),
            metal_type,
            distribution,
        }
    }
}

fn fresnel_conductor(cos_theta: f32, eta: Color, k: Color) -> Color {
    // schlick-like approximation for conductors
    let cos_theta = cos_theta.clamp(0.0, 1.0);
    let cos2 = Color::splat(cos_theta * cos_theta);
    let sin2 = Color::splat(1.0) - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2plusb2 = (t0 * t0 + Color::splat(4.0) * eta2 * k2).sqrt();
    let t1 = a2plusb2 + cos2;
    let a = (a2plusb2 + t0) * Color::splat(0.5);
    let a = a.sqrt();
    let t2 = Color::splat(2.0 * cos_theta) * a;
    let rs = (t1 - t2) / (t1 + t2);
    let t3 = cos2 * a2plusb2 + sin2 * sin2;
    let t4 = t2;
    let rp = rs * ((t3 - t4) / (t3 + t4));
    (rs + rp) * Color::splat(0.5)
}

fn ggx_d(roughness: f32, n_dot_h: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom + 1e-7)
}

fn beckmann_d(roughness: f32, n_dot_h: f32) -> f32 {
    let a = roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    (-((n_dot_h2 - 1.0) / (a2 * n_dot_h2 + 1e-7))).exp() / (PI * a2 * n_dot_h2 + 1e-7)
}

// replace with a more standard smith G1 formulation for GGX
fn ggx_g1(n_dot_x: f32, roughness: f32) -> f32 {
    if n_dot_x <= 0.0 {
        return 0.0;
    }
    let a = roughness * roughness; // alpha_sq
    let k = a / 2.0;
    let denom = n_dot_x * (1.0 - k) + k;
    if denom < EPSILON {
        return 1.0;
    } // avoid division by zero, effectively G1=1 if denom is tiny
    n_dot_x / denom
}

fn ggx_g(roughness: f32, n_dot_v: f32, n_dot_l: f32) -> f32 {
    // separable smith G term
    ggx_g1(n_dot_v, roughness) * ggx_g1(n_dot_l, roughness)
}

fn beckmann_g(roughness: f32, n_dot_v: f32, n_dot_l: f32) -> f32 {
    let a = roughness;
    let lambda = |x: f32| {
        let t = (a * x).recip();
        if t < 1.6 {
            (1.0 - 1.259 * t + 0.396 * t * t) / (3.535 * t + 2.181 * t * t)
        } else {
            0.0
        }
    };
    1.0 / (1.0 + lambda(n_dot_v) + lambda(n_dot_l))
}

fn sample_ggx(normal: Vec3, roughness: f32, rng: &mut dyn RngCore) -> Vec3 {
    // ---- nan check for sample_ggx inputs ----
    if normal.x.is_nan()
        || normal.y.is_nan()
        || normal.z.is_nan()
        || (normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0)
    {
        return Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    }
    // ---- end nan check ----

    let u1 = rng.random::<f32>().max(1e-6);
    let u2 = rng.random::<f32>();
    let a = roughness * roughness;
    let theta_arg = a * a * (-u1.ln()) / (1.0 - u1);

    if theta_arg.is_nan() || theta_arg.is_infinite() || theta_arg < 0.0 {
        return Vec3::to_world(Vec3::new(0.0, 0.0, 1.0), normal);
    }
    let theta = theta_arg.sqrt().atan();
    let phi = 2.0 * PI * u2;
    let (sin_theta, cos_theta) = theta.sin_cos();
    let h_local = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

    if h_local.x.is_nan() || h_local.y.is_nan() || h_local.z.is_nan() {
        return Vec3::to_world(Vec3::new(0.0, 0.0, 1.0), normal);
    }
    Vec3::to_world(h_local, normal)
}

fn sample_beckmann(normal: Vec3, roughness: f32, rng: &mut dyn RngCore) -> Vec3 {
    if normal.x.is_nan()
        || normal.y.is_nan()
        || normal.z.is_nan()
        || (normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0)
    {
        return Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    }

    let u1 = rng.random::<f32>().max(1e-6);
    let u2 = rng.random::<f32>();
    let theta_arg = -(roughness * roughness * u1.ln());
    if theta_arg.is_nan() || theta_arg.is_infinite() || theta_arg < 0.0 {
        return Vec3::to_world(Vec3::new(0.0, 0.0, 1.0), normal);
    }
    let theta = theta_arg.sqrt().atan();
    let phi = 2.0 * PI * u2;
    let (sin_theta, cos_theta) = theta.sin_cos();
    let h_local = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

    if h_local.x.is_nan() || h_local.y.is_nan() || h_local.z.is_nan() {
        return Vec3::to_world(Vec3::new(0.0, 0.0, 1.0), normal);
    }
    Vec3::to_world(h_local, normal)
}

impl Material for RoughConductor {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit_record: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
        if ray_in.direction.x.is_nan() || ray_in.direction.y.is_nan() || ray_in.direction.z.is_nan()
        {
            return None;
        }
        if hit_record.normal.x.is_nan()
            || hit_record.normal.y.is_nan()
            || hit_record.normal.z.is_nan()
            || (hit_record.normal.x == 0.0
                && hit_record.normal.y == 0.0
                && hit_record.normal.z == 0.0)
        {
            return None;
        }

        let n = hit_record.normal;
        let v = -ray_in.direction.normalized();
        if v.x.is_nan() || v.y.is_nan() || v.z.is_nan() {
            return None;
        }

        let (eta, k) = self.metal_type.ior_k();
        let rough = self.roughness;

        let h = match self.distribution {
            MicrofacetDistribution::GGX => sample_ggx(n, rough, rng),
            MicrofacetDistribution::Beckmann => sample_beckmann(n, rough, rng),
        };
        if h.x.is_nan() || h.y.is_nan() || h.z.is_nan() {
            return None;
        }

        let l = reflect(-v, h);
        if l.x.is_nan() || l.y.is_nan() || l.z.is_nan() {
            return None;
        }

        if l.dot(n) <= 0.0 {
            return None;
        }

        let n_dot_l = n.dot(l).max(0.0);
        let n_dot_v = n.dot(v).max(0.0);
        let n_dot_h = n.dot(h).max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        let d = match self.distribution {
            MicrofacetDistribution::GGX => ggx_d(rough, n_dot_h),
            MicrofacetDistribution::Beckmann => beckmann_d(rough, n_dot_h),
        };
        let g = match self.distribution {
            MicrofacetDistribution::GGX => ggx_g(rough, n_dot_v, n_dot_l),
            MicrofacetDistribution::Beckmann => beckmann_g(rough, n_dot_v, n_dot_l),
        };
        let f = fresnel_conductor(v_dot_h, eta, k);

        let brdf_numerator = f * g * v_dot_h;
        let brdf_denominator = n_dot_v * n_dot_h + EPSILON;

        let color;
        if brdf_denominator > EPSILON {
            color = self.albedo * (brdf_numerator / brdf_denominator);
        } else {
            color = Color::BLACK;
        }

        if color.r < EPSILON && color.g < EPSILON && color.b < EPSILON && (d * g != 0.0) {}

        let scattered_origin = hit_record.position + n * EPSILON;
        let scattered_ray = Ray::new(scattered_origin, l.normalized());
        Some((scattered_ray, color))
    }
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
