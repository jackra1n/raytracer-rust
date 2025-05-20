use crate::camera::Camera;
use crate::color::{color_to_u32, Color};
use crate::hittable::{Hittable};
use crate::ray::Ray;
use crate::scene::Scene;


use chrono::Local;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use std::fs;
use std::path::Path;

pub const WIDTH: usize = 800;
pub const HEIGHT: usize = 600;

pub const NUM_AA_SAMPLES: usize = 16; // 16 for fast, 64/100+ for quality
pub const INV_AA_SAMPLES: f32 = 1.0 / (NUM_AA_SAMPLES as f32);

pub const MAX_RECURSION_DEPTH: usize = 10; // 50 common for path tracing

pub const EPSILON: f32 = 1e-4;


pub fn trace_ray(
    ray_in: &Ray,
    scene: &Scene,
    depth: usize,
    rng: &mut dyn RngCore,
) -> Color {
    if depth == 0 {
        return Color::BLACK;
    }

    match scene.object_list.hit(ray_in, EPSILON, f32::INFINITY) {
        Some(hit_record) => {
            let material = hit_record.material.as_ref();

            let emitted_light = material.emitted(0.0, 0.0, &hit_record.position);

            match material.scatter(ray_in, &hit_record, rng) {
                Some((scattered_ray, attenuation_color)) => {
                    let scattered_color = trace_ray(&scattered_ray, scene, depth - 1, rng);
                    emitted_light + attenuation_color * scattered_color
                }
                None => {
                    emitted_light
                }
            }
        }
        None => {
            // ray missed all objects, return background/sky color
            let unit_direction = ray_in.direction.normalized();
            let t = 0.5 * (unit_direction.y + 1.0); // y is up
            Color::new(1.0, 1.0, 1.0) * (1.0 - t) + Color::new(0.5, 0.7, 1.0) * t
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
    let mut image_data = vec![Color::BLACK; WIDTH * HEIGHT];

    image_data
        .par_chunks_mut(WIDTH)
        .enumerate()
        .for_each(|(y_idx, row_slice)| {
            let mut rng = rand::rng();

            for x_idx in 0..WIDTH {
                let mut accumulated_color = Color::BLACK;
                for _s in 0..NUM_AA_SAMPLES {
                    let u = (x_idx as f32 + rng.random::<f32>()) / (WIDTH as f32);
                    let v = (y_idx as f32 + rng.random::<f32>()) / (HEIGHT as f32);

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

    let final_buffer: Vec<u32> = image_data
        .iter()
        .map(|color| {
            let r = color.r.sqrt();
            let g = color.g.sqrt();
            let b = color.b.sqrt();
            color_to_u32(Color::new(r, g, b))
        })
        .collect();

    final_buffer
}


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
            eprintln!("Warning: Buffer access out of bounds at ({}, {})", x, y);
            *pixel = Rgb([255, 0, 255]);
        }
    }

    let date_str = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let output_dir = "render_images";
    
    if !Path::new(output_dir).exists() {
        match fs::create_dir_all(output_dir) {
            Ok(_) => println!("Created directory: {}", output_dir),
            Err(e) => {
                eprintln!("Error creating directory '{}': {}.", output_dir, e);
                return;
            }
        }
    }

    let filename = format!("{}/render_pt_{}.png", output_dir, date_str);

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