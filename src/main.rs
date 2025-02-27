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
        let len = self.dot(*self).sqrt();
        Vec3 {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }
}

trait Shape {
    fn intersect(&self, x: f32, y: f32) -> Option<(f32, Vec3)>;
    fn get_color(&self) -> (u8, u8, u8);
}

struct Sphere {
    center: Vec3,
    radius: f32,
    color: (u8, u8, u8),
}

impl Shape for Sphere {
    fn intersect(&self, x: f32, y: f32) -> Option<(f32, Vec3)> {
        let dx = x - self.center.x;
        let dy = y - self.center.y;
        let distance_sq = dx * dx + dy * dy;
        let radius_sq = self.radius * self.radius;
        
        if distance_sq <= radius_sq {
            let dz = (radius_sq - distance_sq).sqrt();
            let normal = Vec3::new(dx, dy, dz).normalized();
            Some((distance_sq, normal))
        } else {
            None
        }
    }
    
    fn get_color(&self) -> (u8, u8, u8) {
        self.color
    }
}

struct Rectangle {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    height: f32,
    color: (u8, u8, u8),
}

impl Shape for Rectangle {
    fn intersect(&self, x: f32, y: f32) -> Option<(f32, Vec3)> {
        if x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y {
            let normal = Vec3::new(0.0, 0.0, 1.0);
            let dx = (x - (self.min_x + self.max_x) / 2.0) / ((self.max_x - self.min_x) / 2.0);
            let dy = (y - (self.min_y + self.max_y) / 2.0) / ((self.max_y - self.min_y) / 2.0);
            let distance_sq = dx * dx + dy * dy + self.height * self.height;
            Some((distance_sq, normal))
        } else {
            None
        }
    }
    
    fn get_color(&self) -> (u8, u8, u8) {
        self.color
    }
}

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
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

    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Sphere {
            center: Vec3::new(200.0, 300.0, 0.0),
            radius: 150.0,
            color: (55, 55, 255),
        }),
        Box::new(Sphere {
            center: Vec3::new(500.0, 250.0, 0.0),
            radius: 100.0,
            color: (255, 55, 55),
        }),
        Box::new(Rectangle {
            min_x: 100.0,
            min_y: 100.0,
            max_x: 300.0,
            max_y: 200.0,
            height: 10.0,
            color: (55, 255, 55),
        }),
    ];

    let light_dir = Vec3::new(0.5, -1.0, 0.5).normalized();
    let background_color = rgb_to_u32(30, 30, 30);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let mut color = background_color;
                let mut min_distance = f32::MAX;
                let mut hit_normal = None;
                let mut hit_color = None;
                
                for shape in &shapes {
                    if let Some((distance, normal)) = shape.intersect(x as f32, y as f32) {
                        if distance < min_distance {
                            min_distance = distance;
                            hit_normal = Some(normal);
                            hit_color = Some(shape.get_color());
                        }
                    }
                }
                
                if let (Some(normal), Some(shape_color)) = (hit_normal, hit_color) {
                    let diffuse = normal.dot(light_dir).max(0.0);
                    let ambient = 0.2;
                    let intensity = (diffuse + ambient).min(1.0);
                    
                    let (r, g, b) = shape_color;
                    color = rgb_to_u32(
                        (r as f32 * intensity) as u8,
                        (g as f32 * intensity) as u8,
                        (b as f32 * intensity) as u8,
                    );
                }

                buffer[y * WIDTH + x] = color;
            }
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}