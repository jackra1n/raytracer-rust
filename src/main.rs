use minifb::{Key, Window, WindowOptions};
use std::f32::consts::PI;

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

    fn length(&self) -> f32 {
        self.dot(*self).sqrt()
    }

    fn normalized(&self) -> Vec3 {
        let len = self.length();
        Vec3 {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    fn sub(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

struct Sphere {
    center: Vec3,
    radius: f32,
    color: (u8, u8, u8),
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

    let sphere = Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.3,
        color: (55, 55, 255),
    };

    let camera_pos = Vec3::new(0.0, 0.0, -3.0);
    let light_dir = Vec3::new(0.5, -1.0, -0.5).normalized();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let px = (x as f32 / WIDTH as f32) * 2.0 - 1.0;
                let py = 1.0 - (y as f32 / HEIGHT as f32) * 2.0;
                let aspect_ratio = WIDTH as f32 / HEIGHT as f32;

                let ray_dir = Vec3::new(px * aspect_ratio, py, 0.0).sub(camera_pos).normalized();

                let oc = camera_pos.sub(sphere.center);
                let a = ray_dir.dot(ray_dir);
                let b = 2.0 * oc.dot(ray_dir);
                let c = oc.dot(oc) - sphere.radius * sphere.radius;
                let discriminant = b * b - 4.0 * a * c;

                let mut color = rgb_to_u32(30, 30, 30); // Dark background

                if discriminant >= 0.0 {
                    let t = (-b - discriminant.sqrt()) / (2.0 * a);
                    let hit_point = Vec3 {
                        x: camera_pos.x + ray_dir.x * t,
                        y: camera_pos.y + ray_dir.y * t,
                        z: camera_pos.z + ray_dir.z * t,
                    };

                    // Calculate normal and lighting
                    let normal = hit_point.sub(sphere.center).normalized();
                    let diffuse = normal.dot(light_dir).max(0.0);

                    // Add some ambient light
                    let ambient = 0.2;
                    let intensity = (diffuse + ambient).min(1.0);

                    // Apply lighting to color
                    let (r, g, b) = sphere.color;
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