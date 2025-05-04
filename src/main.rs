use chrono::Local;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use minifb::{Key, Window, WindowOptions};
use rand::Rng;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};
use std::path::Path;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const EPSILON: f32 = 1e-5;
const NUM_SHADOW_SAMPLES: usize = 16;
const LIGHT_RADIUS: f32 = 50.0;

const NUM_AA_SAMPLES: usize = 4;
const INV_AA_SAMPLES: f32 = 1.0 / (NUM_AA_SAMPLES as f32);

const MAX_RECURSION_DEPTH: usize = 5;

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(&self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    fn cross(&self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }
    fn normalized(self) -> Vec3 {
        let len = self.length();
        if len < EPSILON {
            self
        } else {
            self * (1.0 / len)
        }
    }

    fn rotate_around_y(&self, angle_degrees: f32) -> Vec3 {
        let angle_rad = angle_degrees * std::f32::consts::PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        Vec3 {
            x: self.x * cos_a + self.z * sin_a,
            y: self.y,
            z: -self.x * sin_a + self.z * cos_a,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        if scalar.abs() < EPSILON {
            panic!("Division by zero in Vec3 division");
        }
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl Color {
    fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
    fn clamp(&mut self) {
        self.r = self.r.clamp(0.0, 1.0);
        self.g = self.g.clamp(0.0, 1.0);
        self.b = self.b.clamp(0.0, 1.0);
    }
}

impl Add for Color {
    type Output = Color;
    fn add(self, other: Color) -> Color {
        Color::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }
}
impl Mul<f32> for Color {
    type Output = Color;
    fn mul(self, s: f32) -> Color {
        Color::new(self.r * s, self.g * s, self.b * s)
    }
}
impl Mul<Color> for Color {
    type Output = Color;
    fn mul(self, o: Color) -> Color {
        Color::new(self.r * o.r, self.g * o.g, self.b * o.b)
    }
}

fn color_to_u32(mut c: Color) -> u32 {
    c.clamp();
    let r = (c.r * 255.0) as u32;
    let g = (c.g * 255.0) as u32;
    let b = (c.b * 255.0) as u32;
    (r << 16) | (g << 8) | b
}

struct Ray {
    start: Vec3,
    dir: Vec3,
}

impl Ray {
    fn new(start: Vec3, dir: Vec3) -> Self {
        Ray {
            start,
            dir: dir.normalized(),
        }
    }
    fn at(&self, t: f32) -> Vec3 {
        self.start + self.dir * t
    }
}

trait Object {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitData>;
}

#[derive(Clone, Copy, Debug)]
struct Material {
    color: Color,
    reflectivity: f32,
}

impl Material {
    fn new(color: Color, reflectivity: f32) -> Self {
        Self {
            color,
            reflectivity,
        }
    }
}

struct HitData {
    t: f32,
    position: Vec3,
    normal: Vec3,
    material: Material,
}

struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
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

struct Cube {
    min: Vec3,
    max: Vec3,
    material: Material,
}

impl Cube {
    fn new_pos_size(bottom_center: Vec3, size: Vec3, material: Material) -> Self {
        let half_size = size * 0.5;
        let center = bottom_center + Vec3::new(0.0, half_size.y, 0.0);
        let min = center - half_size;
        let max = center + half_size;
        Self { min, max, material }
    }
}

impl Object for Cube {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitData> {
        let inv_dir = Vec3::new(1.0 / ray.dir.x, 1.0 / ray.dir.y, 1.0 / ray.dir.z);
        let tx1 = (self.min.x - ray.start.x) * inv_dir.x;
        let tx2 = (self.max.x - ray.start.x) * inv_dir.x;

        let mut t_enter = tx1.min(tx2);
        let mut t_exit = tx1.max(tx2);

        let ty1 = (self.min.y - ray.start.y) * inv_dir.y;
        let ty2 = (self.max.y - ray.start.y) * inv_dir.y;

        t_enter = t_enter.max(ty1.min(ty2));
        t_exit = t_exit.min(ty1.max(ty2));

        let tz1 = (self.min.z - ray.start.z) * inv_dir.z;
        let tz2 = (self.max.z - ray.start.z) * inv_dir.z;

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

        Some(HitData {
            t,
            position,
            normal: normal.normalized(),
            material: self.material,
        })
    }
}

struct Plane {
    p1: Vec3,
    normal: Vec3,
    material: Material,
}

impl Plane {
    fn new(point: Vec3, normal: Vec3, material: Material) -> Self {
        Self {
            p1: point,
            normal: normal.normalized(),
            material,
        }
    }
}

impl Object for Plane {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitData> {
        let denom = self.normal.dot(ray.dir);

        if denom.abs() < EPSILON {
            return None;
        }

        let t = self.normal.dot(self.p1 - ray.start) / denom;

        if t <= t_min || t >= t_max {
            return None;
        }

        let position = ray.at(t);
        Some(HitData {
            t,
            position,
            normal: if denom < 0.0 {
                self.normal
            } else {
                self.normal * -1.0
            },
            material: self.material,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Light {
    pos: Vec3,
    color: Color,
    strength: f32,
}

impl Light {
    fn new(pos: Vec3, color: Color, strength: f32) -> Self {
        Self {
            pos,
            color,
            strength,
        }
    }
}

struct Scene {
    objects: Vec<Box<dyn Object + Sync>>,
    lights: Vec<Light>,
}

impl Scene {
    fn new() -> Self {
        Scene {
            objects: Vec::new(),
            lights: Vec::new(),
        }
    }

    fn add_object(&mut self, obj: Box<dyn Object + Sync>) {
        self.objects.push(obj);
    }
    fn add_light(&mut self, l: Light) {
        self.lights.push(l);
    }
}

fn load_mesh(
    scene: &mut Scene,
    path: &str,
    material: Material,
    scale: f32,
    offset: Vec3,
    rotation_y: f32,
) {
    println!("Attempting to load mesh: {}", path);
    match Mesh::from_obj(path, material, scale, offset, rotation_y) {
        Ok(mesh) => {
            println!(
                "Loaded '{}' with {} triangles. Center: {:?}, Extent: {:?}",
                path,
                mesh.triangles.len(),
                (mesh.bvh.bounds.min + mesh.bvh.bounds.max) * 0.5,
                mesh.bvh.bounds.max - mesh.bvh.bounds.min
            );
            scene.add_object(Box::new(mesh));
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load '{}': {}", path, e);
        }
    }
}

fn init_scene() -> Scene {
    let mut scene = Scene::new();

    let floor_mat = Material::new(Color::new(0.0, 0.3, 0.3), 0.2);
    let blue_mirror_mat = Material::new(Color::new(0.0, 0.5, 1.0), 0.8);
    let yellow_diffuse_mat = Material::new(Color::new(1.0, 1.0, 0.0), 0.0);
    let magenta_mat = Material::new(Color::new(1.0, 0.0, 1.0), 0.5);
    let red_plastic_mat = Material::new(Color::new(1.0, 0.1, 0.1), 0.0);
    let grey_metal_mat = Material::new(Color::new(0.8, 0.8, 0.8), 0.9);

    scene.add_light(Light::new(
        Vec3::new(-500.0, 800.0, -1000.0),
        Color::new(1.0, 1.0, 1.0),
        0.8,
    ));
    scene.add_light(Light::new(
        Vec3::new(700.0, 600.0, -800.0),
        Color::new(1.0, 1.0, 0.8),
        0.6,
    ));
    scene.add_light(Light::new(
        Vec3::new(0.0, 1000.0, 500.0),
        Color::new(0.8, 0.8, 1.0),
        0.5,
    ));

    scene.add_object(Box::new(Plane::new(
        Vec3::new(0.0, -100.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        floor_mat,
    )));

    scene.add_object(Box::new(Sphere {
        center: Vec3::new(-250.0, 50.0, 150.0),
        radius: 150.0,
        material: blue_mirror_mat,
    }));

    scene.add_object(Box::new(Cube::new_pos_size(
        Vec3::new(300.0, -100.0, 0.0),
        Vec3::new(100.0, 400.0, 100.0),
        yellow_diffuse_mat,
    )));

    scene.add_object(Box::new(Cube::new_pos_size(
        Vec3::new(50.0, -50.0, -150.0),
        Vec3::new(100.0, 100.0, 100.0),
        magenta_mat,
    )));

    let amogus_pos = Vec3::new(0.0, -100.0, 200.0);
    let amogus_scale = 3.0;
    load_mesh(
        &mut scene,
        "models/amogus/obj/sus.obj",
        red_plastic_mat,
        amogus_scale,
        amogus_pos,
        180.0,
    );

    let teapot_pos = Vec3::new(300.0, -100.0, 400.0);
    let teapot_scale = 50.0;
    load_mesh(
        &mut scene,
        "models/teapot/teapot.obj",
        grey_metal_mat,
        teapot_scale,
        teapot_pos,
        0.0,
    );

    scene
}

fn trace_ray(ray: &Ray, scene: &Scene, depth: usize) -> Color {
    if depth >= MAX_RECURSION_DEPTH {
        return Color::new(0.0, 0.0, 0.0);
    }
    let mut closest_hit: Option<HitData> = None;
    let mut t_max = f32::INFINITY;

    for obj in &scene.objects {
        if let Some(hit) = obj.intersect(ray, EPSILON, t_max) {
            t_max = hit.t;
            closest_hit = Some(hit);
        }
    }

    if let Some(hd) = closest_hit {
        let ambient_intensity = 0.1;
        let mut local_color = hd.material.color * ambient_intensity;

        let mut rng = rand::rng();

        for light in &scene.lights {
            let mut shadow_factor = 0.0;

            let primary_to_light = light.pos - hd.position;
            let primary_dist_sq = primary_to_light.length_squared();
            let primary_to_light_dir = primary_to_light / primary_dist_sq.sqrt();

            if primary_dist_sq < EPSILON * EPSILON {
                continue;
            }

            let w = primary_to_light_dir;
            let temp_up = if w.x.abs() > 0.9 {
                Vec3::new(0.0, 1.0, 0.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            let u = w.cross(temp_up).normalized();
            let v = w.cross(u).normalized();

            const SHADOW_EPSILON: f32 = 1e-4;
            let shadow_origin = hd.position + hd.normal * SHADOW_EPSILON * 20.0;

            for _ in 0..NUM_SHADOW_SAMPLES {
                let rand1: f32 = rng.random();
                let rand2: f32 = rng.random();

                let radius = LIGHT_RADIUS * rand1.sqrt();
                let angle = 2.0 * std::f32::consts::PI * rand2;

                let offset = u * (radius * angle.cos()) + v * (radius * angle.sin());

                let sample_light_pos = light.pos + offset;

                let sample_to_light_vec = sample_light_pos - hd.position;
                let sample_dist = sample_to_light_vec.length();
                let sample_to_light_dir = sample_to_light_vec / sample_dist;

                let shadow_ray = Ray::new(shadow_origin, sample_to_light_dir);
                let mut is_occluded = false;

                for obj2 in &scene.objects {
                    let t_max_shadow = (sample_dist - SHADOW_EPSILON).max(SHADOW_EPSILON);
                    if obj2
                        .intersect(&shadow_ray, SHADOW_EPSILON, t_max_shadow)
                        .is_some()
                    {
                        is_occluded = true;
                        break;
                    }
                }

                if !is_occluded {
                    shadow_factor += 1.0;
                }
            }

            shadow_factor /= NUM_SHADOW_SAMPLES as f32;

            if shadow_factor > 0.0 {
                let diff = hd.normal.dot(primary_to_light_dir).max(0.0);
                let diffuse_contribution =
                    hd.material.color * light.color * (diff * light.strength * shadow_factor);
                local_color = local_color + diffuse_contribution;
            }
        }

        let reflectivity = hd.material.reflectivity;
        let mut reflected_color = Color::new(0.0, 0.0, 0.0);

        if reflectivity > EPSILON {
            let incoming_dir = ray.dir;
            let normal = hd.normal;
            let reflection_dir =
                (incoming_dir - normal * 2.0 * incoming_dir.dot(normal)).normalized();

            let reflection_origin = hd.position + normal * EPSILON * 10.0;

            let reflection_ray = Ray::new(reflection_origin, reflection_dir);

            reflected_color = trace_ray(&reflection_ray, scene, depth + 1);
        }

        local_color * (1.0 - reflectivity) + reflected_color * reflectivity
    } else {
        Color::new(0.1, 0.1, 0.15)
    }
}

struct Camera {
    position: Vec3,
    look_at: Vec3,
    forward: Vec3,
    right: Vec3,
    true_up: Vec3,
    half_width: f32,
    half_height: f32,
}

impl Camera {
    fn new(position: Vec3, look_at: Vec3, world_up: Vec3, fov: f32, aspect: f32) -> Self {
        let forward = (look_at - position).normalized();
        let right = forward.cross(world_up.normalized()).normalized();
        let true_up = right.cross(forward).normalized();

        let fov_rad = fov * std::f32::consts::PI / 180.0;
        let half_height = (fov_rad / 2.0).tan();
        let half_width = half_height * aspect;

        Camera {
            position,
            look_at,
            forward,
            right,
            true_up,
            half_width,
            half_height,
        }
    }

    fn get_ray_uv(&self, u: f32, v: f32) -> Ray {
        let ndc_x = 2.0 * u - 1.0;
        let ndc_y = 1.0 - 2.0 * v;

        let offset =
            self.right * (ndc_x * self.half_width) + self.true_up * (ndc_y * self.half_height);
        let ray_dir = (self.forward + offset).normalized();

        Ray::new(self.position, ray_dir)
    }
}

struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    normal: Vec3,
    material: Material,
}

impl Triangle {
    fn new(v0: Vec3, v1: Vec3, v2: Vec3, material: Material) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalized();
        Self {
            v0,
            v1,
            v2,
            normal,
            material,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    fn add_point(&mut self, p: Vec3) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    fn intersect(&self, ray: &Ray, mut t_min: f32, mut t_max: f32) -> bool {
        for axis in 0..3 {
            let inv_d = 1.0 / ray.dir[axis];
            let mut t0 = (self.min[axis] - ray.start[axis]) * inv_d;
            let mut t1 = (self.max[axis] - ray.start[axis]) * inv_d;

            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }

            t_min = t_min.max(t0);
            t_max = t_max.min(t1);

            if t_max <= t_min {
                return false;
            }
        }
        true
    }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Invalid index for Vec3"),
        }
    }
}

struct BVHNode {
    bounds: Aabb,
    left: Option<Box<BVHNode>>,
    right: Option<Box<BVHNode>>,
    triangle_indices: Vec<usize>,
}

impl BVHNode {
    fn new(triangles: &[Triangle], indices: &mut [usize], depth: usize) -> Self {
        let num_triangles = indices.len();
        let mut bounds = Aabb::empty();
        for &idx in indices.iter() {
            let tri = &triangles[idx];
            bounds.add_point(tri.v0);
            bounds.add_point(tri.v1);
            bounds.add_point(tri.v2);
        }

        const MAX_DEPTH: usize = 25;
        const MIN_TRIANGLES_PER_LEAF: usize = 4;
        if num_triangles <= MIN_TRIANGLES_PER_LEAF || depth >= MAX_DEPTH {
            return BVHNode {
                bounds,
                left: None,
                right: None,
                triangle_indices: indices.to_vec(),
            };
        }

        let extent = bounds.max - bounds.min;
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };

        indices.sort_unstable_by(|&a, &b| {
            let centroid_a = (triangles[a].v0 + triangles[a].v1 + triangles[a].v2) * (1.0 / 3.0);
            let centroid_b = (triangles[b].v0 + triangles[b].v1 + triangles[b].v2) * (1.0 / 3.0);
            let val_a = centroid_a[axis];
            let val_b = centroid_b[axis];
            val_a
                .partial_cmp(&val_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = num_triangles / 2;
        let (left_indices, right_indices) = indices.split_at_mut(mid);

        if left_indices.is_empty() || right_indices.is_empty() {
            return BVHNode {
                bounds,
                left: None,
                right: None,
                triangle_indices: indices.to_vec(),
            };
        }

        let left_child = Box::new(BVHNode::new(triangles, left_indices, depth + 1));
        let right_child = Box::new(BVHNode::new(triangles, right_indices, depth + 1));

        BVHNode {
            bounds,
            left: Some(left_child),
            right: Some(right_child),
            triangle_indices: Vec::new(),
        }
    }

    fn intersect_recursive<'a>(
        &'a self,
        ray: &Ray,
        triangles: &'a [Triangle],
        t_min: f32,
        mut t_max: f32,
    ) -> Option<HitData> {
        if !self.bounds.intersect(ray, t_min, t_max) {
            return None;
        }

        if self.left.is_none() {
            let mut closest_hit: Option<HitData> = None;
            for &idx in &self.triangle_indices {
                let triangle = &triangles[idx];

                let edge1 = triangle.v1 - triangle.v0;
                let edge2 = triangle.v2 - triangle.v0;
                let h = ray.dir.cross(edge2);
                let a = edge1.dot(h);

                if a.abs() < EPSILON {
                    continue;
                }

                let f = 1.0 / a;
                let s = ray.start - triangle.v0;
                let u = f * s.dot(h);
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }

                let q = s.cross(edge1);
                let v = f * ray.dir.dot(q);
                if v < 0.0 || u + v > 1.0 {
                    continue;
                }

                let t = f * edge2.dot(q);

                if t > t_min && t < t_max {
                    let hit_data = HitData {
                        t,
                        position: ray.at(t),
                        normal: triangle.normal,
                        material: triangle.material,
                    };
                    t_max = t;
                    closest_hit = Some(hit_data);
                }
            }
            return closest_hit;
        }

        let hit_left = self
            .left
            .as_ref()
            .unwrap()
            .intersect_recursive(ray, triangles, t_min, t_max);

        if let Some(ref hit) = hit_left {
            t_max = hit.t;
        }

        let hit_right = self
            .right
            .as_ref()
            .unwrap()
            .intersect_recursive(ray, triangles, t_min, t_max);

        match (hit_left, hit_right) {
            (Some(l), Some(r)) => {
                if l.t < r.t {
                    Some(l)
                } else {
                    Some(r)
                }
            }
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }
}

struct Mesh {
    triangles: Vec<Triangle>,
    bvh: BVHNode,
}

impl Mesh {
    fn from_obj(
        path: &str,
        material: Material,
        scale: f32,
        offset: Vec3,
        rotation_y: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let obj_file = tobj::load_obj(Path::new(path), &tobj::GPU_LOAD_OPTIONS)?;

        let (models, _) = obj_file;
        if models.is_empty() {
            return Err(format!("No models found in OBJ file: {}", path).into());
        }
        let mut triangles = Vec::new();

        for model in models {
            let mesh = model.mesh;
            if mesh.indices.is_empty() || mesh.positions.is_empty() {
                println!(
                    "Warning: Model '{}' in '{}' has no indices or positions. Skipping.",
                    model.name, path
                );
                continue;
            }
            if mesh.positions.len() % 3 != 0 {
                return Err(format!(
                    "Invalid position data length in model '{}' in '{}'",
                    model.name, path
                )
                .into());
            }

            let vertices: Vec<Vec3> = (0..mesh.positions.len() / 3)
                .map(|i| {
                    let idx = i * 3;
                    let unrotated = Vec3::new(
                        mesh.positions[idx] * scale,
                        mesh.positions[idx + 1] * scale,
                        mesh.positions[idx + 2] * scale,
                    );

                    let rotated = unrotated.rotate_around_y(rotation_y);

                    Vec3::new(
                        rotated.x + offset.x,
                        rotated.y + offset.y,
                        rotated.z + offset.z,
                    )
                })
                .collect();

            if mesh.indices.len() % 3 != 0 {
                return Err(format!(
                    "Invalid index data length in model '{}' in '{}'",
                    model.name, path
                )
                .into());
            }

            for i in 0..mesh.indices.len() / 3 {
                let idx = i * 3;

                let v0_idx = mesh.indices[idx] as usize;
                let v1_idx = mesh.indices[idx + 1] as usize;
                let v2_idx = mesh.indices[idx + 2] as usize;

                if v0_idx >= vertices.len() || v1_idx >= vertices.len() || v2_idx >= vertices.len()
                {
                    eprintln!("Warning: Vertex index out of bounds (max={}) in OBJ file '{}'. Indices: ({}, {}, {}). Skipping triangle.",
                        vertices.len() - 1, path, v0_idx, v1_idx, v2_idx);
                    continue;
                }

                let triangle = Triangle::new(
                    vertices[v0_idx],
                    vertices[v1_idx],
                    vertices[v2_idx],
                    material,
                );

                if (triangle.v1 - triangle.v0)
                    .cross(triangle.v2 - triangle.v0)
                    .length_squared()
                    < EPSILON * EPSILON
                {
                    continue;
                }

                triangles.push(triangle);
            }
        }

        if triangles.is_empty() {
            return Err(format!(
                "No valid, non-degenerate triangles loaded from OBJ file '{}'",
                path
            )
            .into());
        }

        println!(
            "Building BVH for {} triangles from '{}'...",
            triangles.len(),
            path
        );
        let start_time = std::time::Instant::now();
        let mut indices: Vec<usize> = (0..triangles.len()).collect();
        let bvh = BVHNode::new(&triangles, &mut indices, 0);
        let build_time = start_time.elapsed().as_millis();
        println!("BVH built in {}ms", build_time);

        Ok(Mesh { triangles, bvh })
    }
}

impl Object for Mesh {
    fn intersect(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitData> {
        self.bvh
            .intersect_recursive(ray, &self.triangles, t_min, t_max)
    }
}

fn main() {
    let mut window = Window::new(
        "Raytracer (Soft Shadows) - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| panic!("Unable to create window: {}", e));

    let mut buffer = vec![0u32; WIDTH * HEIGHT];

    println!("Initializing scene...");
    let scene = init_scene();
    println!(
        "Scene initialized with {} objects and {} lights.",
        scene.objects.len(),
        scene.lights.len()
    );

    let camera = Camera::new(
        Vec3::new(0.0, 250.0, -1200.0),
        Vec3::new(0.0, 50.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        50.0,
        WIDTH as f32 / HEIGHT as f32,
    );

    println!(
        "Rendering frame ({}x{}) with {} shadow samples per light...",
        WIDTH, HEIGHT, NUM_SHADOW_SAMPLES
    );
    let pb = ProgressBar::new(HEIGHT as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} Lines ({per_sec}) {msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
    );

    let start_time = std::time::Instant::now();

    buffer.par_chunks_mut(WIDTH)
        .enumerate()
        .for_each(|(y, row)| {
            let mut rng = rand::rng();
            for (x, pixel) in row.iter_mut().enumerate() {
                let mut accumulated_color = Color::new(0.0, 0.0, 0.0);
                let grid_size = (NUM_AA_SAMPLES as f32).sqrt() as usize;
                if grid_size * grid_size != NUM_AA_SAMPLES {
                    if x == 0 && y == 0 {
                    eprintln!("ERROR: NUM_AA_SAMPLES ({}) is not a perfect square! AA disabled for this run.", NUM_AA_SAMPLES);
                    }
                    let u = (x as f32 + 0.5) / WIDTH as f32;
                    let v = (y as f32 + 0.5) / HEIGHT as f32;
                    let ray = camera.get_ray_uv(u, v);
                    accumulated_color = trace_ray(&ray, &scene, 0);
                    *pixel = color_to_u32(accumulated_color);
                    continue;
                }


                for s_y in 0..grid_size {
                    for s_x in 0..grid_size {
                        let jitter_x: f32 = rng.random();
                        let jitter_y: f32 = rng.random();

                        let u = (x as f32 + (s_x as f32 + jitter_x) / grid_size as f32) / WIDTH as f32;
                        let v = (y as f32 + (s_y as f32 + jitter_y) / grid_size as f32) / HEIGHT as f32;

                        let ray = camera.get_ray_uv(u, v);

                        accumulated_color = accumulated_color + trace_ray(&ray, &scene, 0);
                    }
                }

                let final_color = accumulated_color * INV_AA_SAMPLES;

                *pixel = color_to_u32(final_color);
            }
            pb.inc(1);
        });

    pb.finish_with_message("Render complete!");
    let render_time = start_time.elapsed();
    println!("Rendered in {:.3} seconds", render_time.as_secs_f32());

    save_image(&buffer);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap_or_else(|e| {
                eprintln!("Failed to update window buffer: {}", e);
            });
    }

    println!("Exiting.");
}

fn save_image(buffer: &[u32]) {
    println!("Saving image...");
    let img_start_time = std::time::Instant::now();

    let mut img_buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(WIDTH as u32, HEIGHT as u32);

    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let index = y as usize * WIDTH + x as usize;
        if index < buffer.len() {
            let color_u32 = buffer[index];
            let r = ((color_u32 >> 16) & 0xFF) as u8;
            let g = ((color_u32 >> 8) & 0xFF) as u8;
            let b = (color_u32 & 0xFF) as u8;
            *pixel = Rgb([r, g, b]);
        } else {
            eprintln!("Warning: Buffer access out of bounds at ({}, {})", x, y);
            *pixel = Rgb([255, 0, 255]);
        }
    }

    let date_str = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let filename = format!("render_softshadow_{}.png", date_str);

    match img_buf.save(&filename) {
        Ok(_) => {
            let img_save_time = img_start_time.elapsed();
            println!(
                "Image saved as '{}' in {:.3} seconds",
                filename,
                img_save_time.as_secs_f32()
            );
        }
        Err(e) => eprintln!("Error saving image: {}", e),
    }
}
