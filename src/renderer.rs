use crate::camera::Camera;
use crate::color::color_to_u32;
use crate::color::Color;
use crate::hittable::HitData;
use crate::ray::Ray;
use crate::scene::Scene;
use crate::vec3::Vec3;

use chrono::Local;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

pub const WIDTH: usize = 800;
pub const HEIGHT: usize = 600;

// fast AA
//pub const NUM_AA_SAMPLES: usize = 4;
// quality AA
pub const NUM_AA_SAMPLES: usize = 16;
pub const INV_AA_SAMPLES: f32 = 1.0 / (NUM_AA_SAMPLES as f32);
// fast shadow
//pub const NUM_SHADOW_SAMPLES: usize = 1;
// quality shadow
pub const NUM_SHADOW_SAMPLES: usize = 4;

pub const MAX_RECURSION_DEPTH: usize = 5;

pub const EPSILON: f32 = 1e-5;
const LIGHT_RADIUS: f32 = 50.0;
const SHADOW_EPSILON: f32 = 1e-4;

pub fn trace_ray(ray: &Ray, scene: &Scene, depth: usize) -> Color {
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

        let view_dir = (ray.start - hd.position).normalized();

        for light in &scene.lights {
            let mut shadow_factor = 0.0;

            let primary_to_light = light.pos - hd.position;
            let primary_dist_sq = primary_to_light.length_squared();
            let primary_dist = primary_dist_sq.sqrt();
            let primary_to_light_dir = primary_to_light / primary_dist;

            if primary_dist < EPSILON {
                continue;
            }

            let w = primary_to_light_dir;
            let temp_up = if w.x.abs() > 0.9 {
                Vec3::new(0.0, 1.0, 0.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            let u_light_basis = w.cross(temp_up).normalized();
            let v_light_basis = w.cross(u_light_basis).normalized();

            let shadow_origin = hd.position + hd.normal * SHADOW_EPSILON * 20.0;

            for _ in 0..NUM_SHADOW_SAMPLES {
                let rand1: f32 = rng.random();
                let rand2: f32 = rng.random();

                let radius_sample = LIGHT_RADIUS * rand1.sqrt();
                let angle_sample = 2.0 * std::f32::consts::PI * rand2;

                let offset = u_light_basis * (radius_sample * angle_sample.cos()) + v_light_basis * (radius_sample * angle_sample.sin());
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
                let diff_intensity = hd.normal.dot(primary_to_light_dir).max(0.0);
                let diffuse_contribution =
                    hd.material.color * light.color * (diff_intensity * light.strength * shadow_factor);
                local_color = local_color + diffuse_contribution;

                if hd.material.specular_intensity > EPSILON && hd.material.shininess > EPSILON {
                    let halfway_dir = (primary_to_light_dir + view_dir).normalized();
                    let spec_angle = hd.normal.dot(halfway_dir).max(0.0);
                    let specular_term = spec_angle.powf(hd.material.shininess);
                    
                    let specular_highlight_color = light.color;
                    let specular_contribution = specular_highlight_color * (hd.material.specular_intensity * specular_term * light.strength * shadow_factor);
                    local_color = local_color + specular_contribution;
                }
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

        let transparency = hd.material.transparency;
        let mut refracted_color = Color::new(0.0, 0.0, 0.0);

        if transparency > EPSILON {
            let incident_dir = ray.dir;
            let hit_normal = hd.normal;
            let material_ior = hd.material.refractive_index;

            let n1: f32;
            let n2: f32;
            let snell_normal: Vec3;
            let refraction_origin_offset_normal: Vec3;

            if incident_dir.dot(hit_normal) < 0.0 {
                n1 = 1.0;
                n2 = material_ior;
                snell_normal = hit_normal;
                refraction_origin_offset_normal = hit_normal * -1.0;
            } else {
                n1 = material_ior;
                n2 = 1.0;
                snell_normal = hit_normal * -1.0;
                refraction_origin_offset_normal = hit_normal;
            }

            let eta = n1 / n2;
            let cos_i = -incident_dir.dot(snell_normal);

            let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

            if k >= 0.0 {
                let cos_t = k.sqrt();
                let refraction_dir = (incident_dir * eta + snell_normal * (eta * cos_i - cos_t)).normalized();
                
                let refraction_origin = hd.position + refraction_origin_offset_normal * EPSILON * 10.0;
                let refraction_ray = Ray::new(refraction_origin, refraction_dir);
                refracted_color = trace_ray(&refraction_ray, scene, depth + 1);
            }
        }

        let surface_color_contribution = local_color * (1.0 - reflectivity - transparency).max(0.0);
        let reflection_contribution = reflected_color * reflectivity;
        let refraction_contribution = refracted_color * transparency;

        let total_transmitted_reflected = reflectivity + transparency;
        if total_transmitted_reflected > 1.0 {
            let norm_reflectivity = reflectivity / total_transmitted_reflected;
            let norm_transparency = transparency / total_transmitted_reflected;
            return local_color * (1.0 - norm_reflectivity - norm_transparency).max(0.0) + 
                   reflected_color * norm_reflectivity + 
                   refracted_color * norm_transparency;
        }

        surface_color_contribution + reflection_contribution + refraction_contribution

    } else {
        Color::new(0.1, 0.1, 0.15)
    }
}

pub fn render_scene(scene: &Scene, camera: &Camera) -> Vec<u32> {
    println!(
        "Rendering frame ({}x{}) with {} AA samples and {} shadow samples...",
        WIDTH, HEIGHT, NUM_AA_SAMPLES, NUM_SHADOW_SAMPLES
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
    let mut buffer = vec![0u32; WIDTH * HEIGHT];

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
                    accumulated_color = trace_ray(&ray, scene, 0);
                } else {
                    for s_y in 0..grid_size {
                        for s_x in 0..grid_size {
                            let jitter_x: f32 = rng.random();
                            let jitter_y: f32 = rng.random();
                            let u = (x as f32 + (s_x as f32 + jitter_x) / grid_size as f32) / WIDTH as f32;
                            let v = (y as f32 + (s_y as f32 + jitter_y) / grid_size as f32) / HEIGHT as f32;
                            let ray = camera.get_ray_uv(u, v);
                            accumulated_color = accumulated_color + trace_ray(&ray, scene, 0);
                        }
                    }
                    accumulated_color = accumulated_color * INV_AA_SAMPLES;
                }

                let final_color = accumulated_color;
                *pixel = color_to_u32(final_color);
            }
            pb.inc(1);
        });

    pb.finish_with_message("Render complete!");
    let render_time = start_time.elapsed();
    println!("Rendered in {:.3} seconds", render_time.as_secs_f32());

    buffer
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
