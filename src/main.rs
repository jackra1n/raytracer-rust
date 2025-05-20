mod camera;
mod color;
mod hittable;
mod material;
mod mesh;
mod objects;
mod ray;
mod renderer;
mod scene;
mod vec3;

use minifb::{Key, Window, WindowOptions};

use crate::camera::Camera;
use crate::scene::init_scene;
use crate::vec3::Vec3;

use crate::renderer::{render_scene, save_image, HEIGHT, WIDTH};

fn main() {
    let mut window = Window::new(
        "Raytracer - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| panic!("Unable to create window: {}", e));

    println!("Initializing scene...");
    let scene = init_scene();
    println!(
        "Scene initialized with {} objects.",
        scene.object_list.objects.len()
    );

    let camera = Camera::new(
        Vec3::new(0.0, 250.0, -1200.0),
        Vec3::new(0.0, 50.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        WIDTH as f32 / HEIGHT as f32,
    );

    let buffer = render_scene(&scene, &camera);

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
