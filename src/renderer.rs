use crate::camera::Camera;
use crate::color::{color_to_u32, Color};
use crate::hittable::{Hittable};
use crate::ray::Ray;
use crate::scene::Scene;
use crate::tungsten_parser::RenderSettings;
use crate::vec3::Vec3;

use chrono::Local;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use std::fs;
use std::path::Path;

// Removed global WIDTH and HEIGHT constants, will use render_settings
// pub const WIDTH: usize = 800;
// pub const HEIGHT: usize = 600;

// NUM_AA_SAMPLES and MAX_RECURSION_DEPTH are effectively superseded by render_settings
// but trace_ray directly uses MAX_RECURSION_DEPTH. For now, keep it and INV_AA_SAMPLES.
// If render_settings provides these, trace_ray should take them as args or render_scene pass them.
// For now, render_scene uses render_settings.samples_per_pixel, but trace_ray uses global MAX_RECURSION_DEPTH.
// Let's leave these for a moment and focus on width/height.

pub const NUM_AA_SAMPLES: usize = 16; // This should eventually come from render_settings.samples_per_pixel
pub const INV_AA_SAMPLES: f32 = 1.0 / (NUM_AA_SAMPLES as f32); // Needs to be dynamic if NUM_AA_SAMPLES is

pub const MAX_RECURSION_DEPTH: usize = 10; // This should come from render_settings.max_depth

pub const EPSILON: f32 = 1e-4;


pub fn trace_ray(
    ray_in: &Ray,
    scene: &Scene,
    depth: usize, // This is correctly passed down, originating from render_settings.max_depth
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
                    // Material doesn't scatter (e.g., it's a light source or a perfect absorber).
                    // Return only the light emitted by the material itself.
                    emitted_light
                }
            }
        }
        None => {
            // Ray MISSED all objects. Apply skybox or default background.
            if let Some(hdr_skybox) = &scene.skybox_hdr_image {
                let dir = ray_in.direction.normalized();
                let theta = dir.y.acos();
                let phi = dir.z.atan2(dir.x) + std::f32::consts::PI;
                let u = phi / (2.0 * std::f32::consts::PI);
                let v = theta / std::f32::consts::PI;

                let x_pixel = (u * (hdr_skybox.width() - 1) as f32).max(0.0) as u32;
                let y_pixel = (v * (hdr_skybox.height() - 1) as f32).max(0.0) as u32;
                
                let pixel = hdr_skybox.get_pixel(x_pixel.min(hdr_skybox.width() - 1), y_pixel.min(hdr_skybox.height() - 1));
                Color::new(pixel[0], pixel[1], pixel[2]) // Rgb<f32> values are directly usable
            } else if let Some(skybox) = &scene.skybox_image {
                let skybox_rgba = skybox.to_rgba8(); // Convert LDR to Rgba8Image
                let dir = ray_in.direction.normalized();
                let theta = dir.y.acos();
                let phi = dir.z.atan2(dir.x) + std::f32::consts::PI;
                let u = phi / (2.0 * std::f32::consts::PI);
                let v = theta / std::f32::consts::PI;

                let x_pixel = (u * (skybox_rgba.width() - 1) as f32).max(0.0) as u32;
                let y_pixel = (v * (skybox_rgba.height() - 1) as f32).max(0.0) as u32;
                
                let pixel = skybox_rgba.get_pixel(x_pixel.min(skybox_rgba.width() -1), y_pixel.min(skybox_rgba.height()-1));
                Color::new(pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
            } else {
                // Default procedural background color if no skybox image is loaded
                // let unit_direction = ray_in.direction.normalized();
                // let t = 0.5 * (unit_direction.y + 1.0);
                // Color::new(1.0, 1.0, 1.0).lerp(Color::new(0.5, 0.7, 1.0), t)
                Color::BLACK
            }
        }
    }
}


pub fn render_scene(scene: &Scene, camera: &Camera, render_settings: &RenderSettings) -> Vec<u32> {
    println!(
        "Rendering frame ({}x{}) with {} AA samples...",
        render_settings.width, render_settings.height, render_settings.samples_per_pixel
    );
    let pb = ProgressBar::new(render_settings.height as u64); // Use render_settings.height
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} Lines ({per_sec}) {msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
    );

    let start_time = Instant::now();
    // Use render_settings for image_data dimensions
    let mut image_data = vec![Color::BLACK; render_settings.width * render_settings.height];

    let inv_aa_samples_dynamic = 1.0 / (render_settings.samples_per_pixel as f32);

    image_data
        .par_chunks_mut(render_settings.width) // Use render_settings.width
        .enumerate()
        .for_each(|(y_idx, row_slice)| {
            // Use StdRng seeded per row for consistent noise patterns if rerendering same row
            let mut rng = rand::rngs::StdRng::seed_from_u64(y_idx as u64);

            for x_idx in 0..render_settings.width { // Use render_settings.width
                let mut accumulated_color = Color::BLACK;
                for _s in 0..render_settings.samples_per_pixel { // Use render_settings.samples_per_pixel
                    let u = (x_idx as f32 + rng.gen::<f32>()) / (render_settings.width as f32);
                    let v = (y_idx as f32 + rng.gen::<f32>()) / (render_settings.height as f32);

                    let ray = camera.get_ray(u, v);
                    // Pass render_settings.max_depth to trace_ray
                    accumulated_color = accumulated_color + trace_ray(&ray, scene, render_settings.max_depth, &mut rng);
                }
                row_slice[x_idx] = accumulated_color * inv_aa_samples_dynamic;
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


pub fn save_image(buffer: &[u32], width: usize, height: usize) { // Pass width and height
    println!("Saving image...");
    let img_start_time = std::time::Instant::now();

    // Use passed width and height for ImageBuffer dimensions
    let mut img_buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);

    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let index = y as usize * width + x as usize; // Use passed width
        if index < buffer.len() {
            let color_u32 = buffer[index];
            let r = ((color_u32 >> 16) & 0xFF) as u8;
            let g = ((color_u32 >> 8) & 0xFF) as u8;
            let b = (color_u32 & 0xFF) as u8;
            *pixel = Rgb([r, g, b]);
        } else {
            eprintln!("Warning: Buffer access out of bounds at ({}, {})", x, y);
            *pixel = Rgb([255, 0, 255]); // Magenta for out of bounds
        }
    }

    let date_str = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let output_dir = "render_images";
    
    if !Path::new(output_dir).exists() {
        match fs::create_dir_all(output_dir) {
            Ok(_) => println!("Created directory: {}", output_dir),
            Err(e) => {
                eprintln!("Error creating directory '{}': {}. Image will be saved in current directory.", output_dir, e);
                // Fallback: Try to save in the current directory if dir creation fails
                let filename = format!("render_pt_{}.png", date_str);
                match img_buf.save(&filename) {
                    Ok(_) => println!("Image saved as '{}' in current directory.", filename),
                    Err(e_fb) => eprintln!("Error saving image in current directory: {}", e_fb),
                }
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