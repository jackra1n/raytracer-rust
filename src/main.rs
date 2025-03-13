use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

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

    fn dot(&self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn normalized(&self) -> Vec3 {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len < 0.00001 {
            return *self; // Avoid division by zero
        }
        Vec3 {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }
    
    fn subtract(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }
    
    fn at(&self, t: f32) -> Vec3 {
        Vec3 {
            x: self.origin.x + self.direction.x * t,
            y: self.origin.y + self.direction.y * t,
            z: self.origin.z + self.direction.z * t,
        }
    }
}

trait Shape {
    fn intersect(&self, ray: &Ray) -> Option<(f32, Vec3)>;
    fn get_color(&self) -> (f32, f32, f32);
}

struct Sphere {
    center: Vec3,
    radius: f32,
    color: (f32, f32, f32),
}

impl Shape for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<(f32, Vec3)> {
        let oc = ray.origin.subtract(&self.center);
        
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        
        let discriminant = b * b - 4.0 * a * c;
        
        if discriminant < 0.0 {
            return None;
        }
        
        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
        
        let t = if t1 > 0.001 {
            t1
        } else if t2 > 0.001 {
            t2
        } else {
            return None;
        };
        
        let hit_point = ray.at(t);
        let normal = hit_point.subtract(&self.center).normalized();
        
        Some((t, normal))
    }
    
    fn get_color(&self) -> (f32, f32, f32) {
        self.color
    }
}

fn rgb_to_u32(r: f32, g: f32, b: f32) -> u32 {
    let r = (r.clamp(0.0, 1.0) * 255.0) as u32;
    let g = (g.clamp(0.0, 1.0) * 255.0) as u32;
    let b = (b.clamp(0.0, 1.0) * 255.0) as u32;
    
    (r << 16) | (g << 8) | b
}

fn main() {
    let mut window = Window::new(
        "Raytracer - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let camera_pos = Vec3::new(0.0, 0.0, -500.0);
    
    let center_x = 0.0;
    let center_y = 0.0;

    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Sphere {
            center: Vec3::new(center_x, center_y, 100.0),
            radius: 200.0,
            color: (0.1, 0.1, 0.8),
        }),
        Box::new(Sphere {
            center: Vec3::new(center_x + 60.0, center_y + 50.0, 20.0),
            radius: 120.0,
            color: (0.0, 0.8, 0.8),
        }),
    ];

    let background_color = rgb_to_u32(0.05, 0.05, 0.05);
    let light_dir = Vec3::new(-0.7, -0.7, -0.7).normalized();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let world_x = -(x as f32 - WIDTH as f32 / 2.0);
                let world_y = (y as f32 - HEIGHT as f32 / 2.0) * -1.0;
                
                let direction = Vec3::new(
                    world_x - camera_pos.x,
                    world_y - camera_pos.y,
                    0.0 - camera_pos.z,
                ).normalized();
                
                let ray = Ray::new(camera_pos, direction);
                
                let mut closest_hit: Option<(f32, Vec3, (f32, f32, f32))> = None;
                let mut closest_t = f32::MAX;
                
                for shape in &shapes {
                    if let Some((t, normal)) = shape.intersect(&ray) {
                        if t < closest_t {
                            closest_t = t;
                            closest_hit = Some((t, normal, shape.get_color()));
                        }
                    }
                }
                
                let color = if let Some((_, normal, (r, g, b))) = closest_hit {
                    // Apply lighting
                    let diffuse = normal.dot(light_dir).max(0.0);
                    let ambient = 0.3;
                    let intensity = (diffuse + ambient).min(1.0);
                    
                    rgb_to_u32(r * intensity, g * intensity, b * intensity)
                } else {
                    background_color
                };

                buffer[y * WIDTH + x] = color;
            }
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}