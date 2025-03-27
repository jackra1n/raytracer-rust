use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const BACKGROUND: f32 = 1e9;
const EPSILON: f32 = 1e-6;
#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn add(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    fn sub(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    fn mul(&self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    fn dot(&self, o: &Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(&self, o: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    fn normalized(&self) -> Vec3 {
        let len = self.length();
        if len < EPSILON {
            *self
        } else {
            self.mul(1.0 / len)
        }
    }

    fn distance(a: &Vec3, b: &Vec3) -> f32 {
        a.sub(b).length()
    }
}

#[derive(Clone, Copy)]
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
        if self.r < 0.0 { self.r = 0.0; }
        if self.g < 0.0 { self.g = 0.0; }
        if self.b < 0.0 { self.b = 0.0; }
        if self.r > 1.0 { self.r = 1.0; }
        if self.g > 1.0 { self.g = 1.0; }
        if self.b > 1.0 { self.b = 1.0; }
    }
}

impl std::ops::Add for Color {
    type Output = Color;
    fn add(self, other: Color) -> Color {
        Color::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }
}
impl std::ops::Mul<f32> for Color {
    type Output = Color;
    fn mul(self, s: f32) -> Color {
        Color::new(self.r * s, self.g * s, self.b * s)
    }
}
impl std::ops::Mul<Color> for Color {
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
        self.start.add(&self.dir.mul(t))
    }
}

trait Object {
    fn intersect(&self, ray: &Ray) -> Option<f32>;
    fn compute_hit(&self, t: f32, ray: &Ray) -> HitData;
    fn get_color(&self) -> Color;
}

struct HitData {
    position: Vec3,
    normal: Vec3,
    color: Color,
}

struct Sphere {
    center: Vec3,
    radius: f32,
    color: Color,
}

impl Object for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let oc = ray.start.sub(&self.center);
        let a = ray.dir.dot(&ray.dir);
        let b = 2.0 * oc.dot(&ray.dir);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        let eps = 0.0001;
        let t = if t1 > eps && t1 < t2 {
            t1
        } else if t2 > eps {
            t2
        } else {
            return None;
        };
        Some(t)
    }

    fn compute_hit(&self, t: f32, ray: &Ray) -> HitData {
        let pos = ray.at(t);
        let normal = pos.sub(&self.center).normalized();
        HitData {
            position: pos,
            normal,
            color: self.color,
        }
    }

    fn get_color(&self) -> Color {
        self.color
    }
}

struct Cube {
    min: Vec3,
    max: Vec3,
    color: Color,
}

impl Object for Cube {
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let mut tmin = (self.min.x - ray.start.x) / ray.dir.x;
        let mut tmax = (self.max.x - ray.start.x) / ray.dir.x;
        
        if tmin > tmax { std::mem::swap(&mut tmin, &mut tmax); }
        
        let mut tymin = (self.min.y - ray.start.y) / ray.dir.y;
        let mut tymax = (self.max.y - ray.start.y) / ray.dir.y;
        
        if tymin > tymax { std::mem::swap(&mut tymin, &mut tymax); }
        
        if tmin > tymax || tymin > tmax { return None; }
        
        if tymin > tmin { tmin = tymin; }
        if tymax < tmax { tmax = tymax; }
        
        let mut tzmin = (self.min.z - ray.start.z) / ray.dir.z;
        let mut tzmax = (self.max.z - ray.start.z) / ray.dir.z;
        
        if tzmin > tzmax { std::mem::swap(&mut tzmin, &mut tzmax); }
        
        if tmin > tzmax || tzmin > tmax { return None; }
        
        if tzmin > tmin { tmin = tzmin; }
        if tzmax < tmax { tmax = tzmax; }
        
        if tmin > EPSILON { 
            Some(tmin) 
        } else if tmax > EPSILON { 
            Some(tmax) 
        } else { 
            None 
        }
    }
    
    fn compute_hit(&self, t: f32, ray: &Ray) -> HitData {
        let pos = ray.at(t);
        
        let x_dist_min = (pos.x - self.min.x).abs();
        let x_dist_max = (pos.x - self.max.x).abs();
        let y_dist_min = (pos.y - self.min.y).abs();
        let y_dist_max = (pos.y - self.max.y).abs();
        let z_dist_min = (pos.z - self.min.z).abs();
        let z_dist_max = (pos.z - self.max.z).abs();
        
        let min_dist = x_dist_min.min(x_dist_max).min(y_dist_min).min(y_dist_max).min(z_dist_min).min(z_dist_max);
        
        let normal = if min_dist == x_dist_min {
            Vec3::new(-1.0, 0.0, 0.0)
        } else if min_dist == x_dist_max {
            Vec3::new(1.0, 0.0, 0.0)
        } else if min_dist == y_dist_min {
            Vec3::new(0.0, -1.0, 0.0)
        } else if min_dist == y_dist_max {
            Vec3::new(0.0, 1.0, 0.0)
        } else if min_dist == z_dist_min {
            Vec3::new(0.0, 0.0, -1.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };
        
        HitData {
            position: pos,
            normal,
            color: self.color,
        }
    }
    
    fn get_color(&self) -> Color {
        self.color
    }
}

struct Plane {
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    color: Color,
    limited: bool,
}

impl Object for Plane {
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let v1 = self.p2.sub(&self.p1);
        let v2 = self.p3.sub(&self.p1);
        let normal = v1.cross(&v2);
        
        let denom = normal.dot(&ray.dir);
        if denom.abs() < EPSILON {
            return None;
        }
        
        let t = normal.dot(&self.p1.sub(&ray.start)) / denom;
        if t < EPSILON {
            return None;
        }
        
        if !self.limited {
            return Some(t);
        }
        
        let hit_point = ray.at(t);
        
        let edge1 = self.p2.sub(&self.p1);
        let edge2 = self.p3.sub(&self.p1);
        let hit_vec = hit_point.sub(&self.p1);
        
        let dot00 = edge1.dot(&edge1);
        let dot01 = edge1.dot(&edge2);
        let dot02 = edge1.dot(&hit_vec);
        let dot11 = edge2.dot(&edge2);
        let dot12 = edge2.dot(&hit_vec);
        
        let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            Some(t)
        } else {
            None
        }
    }

    fn compute_hit(&self, t: f32, ray: &Ray) -> HitData {
        let pos = ray.at(t);
        let n = self
            .p2
            .sub(&self.p1)
            .cross(&self.p3.sub(&self.p1))
            .normalized();
        HitData {
            position: pos,
            normal: n,
            color: self.color,
        }
    }

    fn get_color(&self) -> Color {
        self.color
    }
}

struct Light {
    pos: Vec3,
    color: Color,
    strength: f32,
}

impl Light {
    fn new(pos: Vec3, color: Color, strength: f32) -> Self {
        Self { pos, color, strength }
    }
}

struct Scene {
    objects: Vec<Box<dyn Object>>,
    lights: Vec<Light>,
}

impl Scene {
    fn new() -> Self {
        Scene {
            objects: Vec::new(),
            lights: Vec::new(),
        }
    }

    fn add_object(&mut self, obj: Box<dyn Object>) {
        self.objects.push(obj);
    }
    fn add_light(&mut self, l: Light) {
        self.lights.push(l);
    }
}

fn init_scene() -> Scene {
    let mut scene = Scene::new();

    scene.add_light(Light::new(Vec3::new(500.0, 0.0, -1000.0), Color::new(0.5, 0.5, 0.5), 0.7));
    scene.add_light(Light::new(Vec3::new(0.0, -1500.0, -500.0), Color::new(0.5, 0.5, 0.5), 1.0));
    scene.add_light(Light::new(Vec3::new(1300.0, 700.0, -1000.0), Color::new(0.5, 0.5, 0.5), 1.0));

    scene.add_object(Box::new(Sphere {
        center: Vec3::new(0.0, 0.0, 800.0),
        radius: 300.0,
        color: Color::new(0.0, 1.0, 0.0),
    }));

    scene.add_object(Box::new(Plane {
        p1: Vec3::new(50.0, 50.0, 50.0),
        p2: Vec3::new(50.0, 50.0, 2000.0),
        p3: Vec3::new(700.0, 50.0, 2000.0),
        color: Color::new(0.0, 1.0, 1.0),
        limited: false,
    }));


    scene.add_object(Box::new(Cube {
        min: Vec3::new(-400.0, -100.0, 500.0),
        max: Vec3::new(-100.0, 200.0, 800.0),
        color: Color::new(1.0, 0.2, 0.2),
    }));

    scene.add_object(Box::new(Cube {
        min: Vec3::new(-400.0, -100.0, 500.0),
        max: Vec3::new(-100.0, 200.0, 800.0),
        color: Color::new(1.0, 0.2, 0.2),
    }));

    scene
}

fn trace_ray(ray: &Ray, scene: &Scene) -> Color {
    let mut nearest_t = BACKGROUND;
    let mut hit_data: Option<HitData> = None;

    for obj in &scene.objects {
        if let Some(t) = obj.intersect(ray) {
            if t < nearest_t {
                nearest_t = t;
                hit_data = Some(obj.compute_hit(t, ray));
            }
        }
    }

    if let Some(hd) = hit_data {
        let ambient = 0.1;
        let mut final_color = Color::new(0.0, 0.0, 0.0);

        final_color = final_color + (hd.color * ambient);

        for light in &scene.lights {
            let to_light = light.pos.sub(&hd.position).normalized();

            let shadow_epsilon = 0.001;
            let shadow_origin = hd.position.add(&to_light.mul(shadow_epsilon));

            let dist_to_light = Vec3::distance(&hd.position, &light.pos);
            let mut in_shadow = false;

            let shadow_ray = Ray::new(shadow_origin, to_light);

            for obj2 in &scene.objects {
                if let Some(t_shadow) = obj2.intersect(&shadow_ray) {
                    if t_shadow < dist_to_light {
                        in_shadow = true;
                        break;
                    }
                }
            }

            if !in_shadow {
                let diff = hd.normal.dot(&to_light).max(0.0);
                final_color = final_color + (hd.color * light.color * (diff * light.strength));
            }
        }

        final_color
    } else {
        Color::new(0.0, 0.0, 0.0)
    }
}

struct Camera {
    position: Vec3,
    look_at: Vec3,
    up: Vec3,
    fov: f32,
    aspect: f32,
}

impl Camera {
    fn new(position: Vec3, look_at: Vec3, up: Vec3, fov: f32, aspect: f32) -> Self {
        Camera {
            position,
            look_at,
            up: up.normalized(),
            fov,
            aspect,
        }
    }
    
    fn get_ray(&self, x: usize, y: usize, width: usize, height: usize) -> Ray {
        let forward = self.look_at.sub(&self.position).normalized();
        let right = forward.cross(&self.up).normalized();
        let true_up = right.cross(&forward).normalized();
        
        let ndc_x = (2.0 * x as f32 / width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * y as f32 / height as f32);
        
        let fov_rad = self.fov * std::f32::consts::PI / 180.0;
        let half_height = (fov_rad / 2.0).tan();
        let half_width = half_height * self.aspect;
        
        let ray_dir = forward.add(
            &right.mul(ndc_x * half_width).add(
                &true_up.mul(ndc_y * half_height)
            )
        ).normalized();
        
        Ray::new(self.position, ray_dir)
    }
}

fn main() {
    let mut window = Window::new(
        "Raytracer - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|_| panic!("Unable to create window"));

    let mut buffer = vec![0u32; WIDTH * HEIGHT];

    let scene = init_scene();

    let camera = Camera::new(
        Vec3::new(0.0, 400.0, -800.0),
        Vec3::new(0.0, 0.0, 500.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        WIDTH as f32 / HEIGHT as f32
    );

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let ray = camera.get_ray(x, y, WIDTH, HEIGHT);
                let color = trace_ray(&ray, &scene);
                buffer[y * WIDTH + x] = color_to_u32(color);
            }
        }
        
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}
