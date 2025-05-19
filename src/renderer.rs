// In your renderer.rs

use crate::camera::Camera;
use crate::color::{color_to_u32, Color}; // Assuming Color::BLACK is defined
use crate::hittable::{HitRecord, Hittable}; // Hittable trait needs to be in scope
use crate::ray::Ray;
use crate::scene::Scene;
use crate::vec3::Vec3;
// Make sure Material trait is in scope if not already
use crate::material::Material;


use chrono::Local;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

pub const WIDTH: usize = 800;
pub const HEIGHT: usize = 600;

pub const NUM_AA_SAMPLES: usize = 16; // Or 4 for faster, 100+ for quality
pub const INV_AA_SAMPLES: f32 = 1.0 / (NUM_AA_SAMPLES as f32);

// NUM_SHADOW_SAMPLES is less relevant now if using path tracing for soft shadows
// It would be implicitly handled by diffuse bounces from area lights.
// If you still have explicit light sampling for direct illumination, it can stay.

pub const MAX_RECURSION_DEPTH: usize = 10; // Max bounces, 50 is common for path tracing

pub const EPSILON: f32 = 1e-4; // General small float
// SURFACE_BIAS is now handled within material scatter/ray generation
// LIGHT_RADIUS and SHADOW_EPSILON are for the old explicit lighting.

// New trace_ray function (often called ray_color)
pub fn trace_ray(
    ray_in: &Ray,
    scene: &Scene, // scene contains objects and potentially explicit lights
    depth: usize,
    rng: &mut dyn RngCore, // Pass the RNG
) -> Color {
    // 1. Max Depth Check
    if depth == 0 {
        return Color::BLACK; // Max depth reached, contribute no more light
    }

    // 2. Intersection Test
    // Find the closest object hit by the ray.
    // The `obj.hit` method should now also take `rng` if any part of its hit
    // logic could be stochastic (e.g., for motion blur, though unlikely for simple objects).
    // For now, let's assume `obj.hit` doesn't need rng.
    // Your Hittable trait's hit method was:
    // fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>
    // Let's keep it that way for now. If object intersection itself needed randomness,
    // then `Hittable::hit` would need `rng`.
    match scene.object_list.hit(ray_in, EPSILON, f32::INFINITY) {
        Some(hit_record) => {
            // 3. Hit an Object
            let material = hit_record.material.as_ref(); // Get a &dyn Material

            // Calculate emitted light from the surface itself
            // Assuming u,v texture coords aren't used yet for emission.
            // The emitted() method signature in your trait was:
            // fn emitted(&self, _u: f32, _v: f32, _p: &Vec3) -> Color
            let emitted_light = material.emitted(0.0, 0.0, &hit_record.position);

            // Try to scatter the ray
            match material.scatter(ray_in, &hit_record, rng) {
                Some((scattered_ray, attenuation_color)) => {
                    // Ray scattered, get color from the scattered direction
                    let scattered_color = trace_ray(&scattered_ray, scene, depth - 1, rng);
                    emitted_light + attenuation_color * scattered_color
                }
                None => {
                    // Ray was absorbed (or material is purely emissive and doesn't scatter)
                    emitted_light
                }
            }
        }
        None => {
            // Ray missed all objects, return background/sky color
            // Example: Simple blueish gradient sky
            let unit_direction = ray_in.direction.normalized();
            let t = 0.5 * (unit_direction.y + 1.0); // y is up
            Color::new(1.0, 1.0, 1.0) * (1.0 - t) + Color::new(0.5, 0.7, 1.0) * t
            // Or your previous background:
            // Color::new(0.1, 0.1, 0.15)
        }
    }
}


pub fn render_scene(scene: &Scene, camera: &Camera) -> Vec<u32> {
    println!(
        "Rendering frame ({}x{}) with {} AA samples...",
        WIDTH, HEIGHT, NUM_AA_SAMPLES
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

    let start_time = Instant::now();
    let mut image_data = vec![Color::BLACK; WIDTH * HEIGHT]; // Store Color structs first

    // Parallel iteration over rows (scanlines)
    image_data
        .par_chunks_mut(WIDTH)
        .enumerate()
        .for_each(|(y_idx, row_slice)| {
            // Each thread gets its own RNG instance from rand::thread_rng()
            // or the more general rand::rng() which might be equivalent in some contexts.
            // rand::thread_rng() is explicitly for getting a thread-local generator.
            let mut rng = rand::rng(); // Use thread_rng() for explicitness

            for x_idx in 0..WIDTH {
                let mut accumulated_color = Color::BLACK;
                for _s in 0..NUM_AA_SAMPLES {
                    // Generate random u,v within the pixel bounds for antialiasing
                    // rng.gen::<f32>() is available because `rand::thread_rng()` returns a type that implements `Rng`.
                    let u = (x_idx as f32 + rng.random::<f32>()) / (WIDTH as f32);
                    let v = (y_idx as f32 + rng.random::<f32>()) / (HEIGHT as f32);

                    // Pass a mutable reference to this thread-local rng.
                    // It will be coerced to `&mut dyn RngCore`.
                    let ray = camera.get_ray(u, v);
                    accumulated_color = accumulated_color + trace_ray(&ray, scene, MAX_RECURSION_DEPTH, &mut rng);
                }
                row_slice[x_idx] = accumulated_color * INV_AA_SAMPLES;
            }
            pb.inc(1);
        });

    pb.finish_with_message("Render complete!");
    let render_time = start_time.elapsed();
    println!("Rendered in {:.3} seconds", render_time.as_secs_f32());

    // Convert Color buffer to u32 buffer for saving
    // Also apply gamma correction here
    let final_buffer: Vec<u32> = image_data
        .iter()
        .map(|color| {
            let r = color.r.sqrt(); // Gamma correction (gamma 2.0)
            let g = color.g.sqrt();
            let b = color.b.sqrt();
            color_to_u32(Color::new(r, g, b))
        })
        .collect();

    final_buffer
}


// Your save_image function seems fine, assuming color_to_u32 and Color representation are correct.
// Make sure color_to_u32 clamps values between 0.0 and 1.0 before converting to 0-255.
pub fn save_image(buffer: &[u32]) {
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
            // This case should ideally not happen if buffer is correctly sized
            eprintln!("Warning: Buffer access out of bounds at ({}, {})", x, y);
            *pixel = Rgb([255, 0, 255]); // Magenta error color
        }
    }

    let date_str = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    // You might want to remove "softshadow" if it's now a general path tracer
    let filename = format!("render_pt_{}.png", date_str);

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