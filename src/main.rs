mod camera;
mod color;
mod hittable;
mod material;
mod ray;
mod scene;
mod vec3;
mod objects;
mod mesh;
mod renderer;
mod tungsten_parser;

use minifb::{Key, Window, WindowOptions};
use indicatif::HumanDuration;
use std::time::Instant;
use std::path::Path;

use crate::renderer::{render_scene, save_image};
use crate::tungsten_parser::RenderSettings;

fn main() {
    let start_time = Instant::now();

    // let scene_file_path_str = "data/scenes/tungsten/cornell-box/scene.json";
    // let scene_file_path_str = "data/scenes/tungsten/teapot/scene.json";
    // let scene_file_path_str = "data/scene_from_rust.json";
    // let scene_file_path_str = "data/scenes/tungsten/dragon/scene.json";
    // let scene_file_path_str = "data/scenes/tungsten/volumetric-caustic/scene.json";
    let scene_path_str = "data/scenes/tungsten/veach-mis/scene.json";
    // let scene_file_path_str = "data/scenes/mitsuba/cornell-box/scene_v3.xml";
    println!("Attempting to load scene from: {}", scene_path_str);
    let scene_path = Path::new(scene_path_str);

    let result = match scene_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some("json") => {
            println!("Detected JSON scene file.");
            tungsten_parser::load_scene_from_json(scene_path_str)
        }
        _ => {
            panic!("Unsupported scene file extension or path error for: {}", scene_path.display());
        }
    };

    match result {
        Ok((scene, camera, mut render_settings)) => {
            // Override max_depth for testing
            println!("Original max_depth from scene file: {}", render_settings.max_depth);
            render_settings.max_depth = 10; // Experiment with a higher max_depth
            println!("Overridden max_depth for rendering: {}", render_settings.max_depth);

            println!(
                "Scene loaded. Objects: {}. Image: {}x{}, Samples: {}, Max Depth: {}",
                scene.object_list.objects.len(),
                render_settings.width,
                render_settings.height,
                render_settings.samples_per_pixel,
                render_settings.max_depth
            );

            let buffer = render_scene(&scene, &camera, &render_settings);
            save_image(&buffer, render_settings.width, render_settings.height);

            let mut window = Window::new(
                "Raytracer - ESC to exit",
                render_settings.width,
                render_settings.height,
                WindowOptions::default(),
            )
            .unwrap_or_else(|e| panic!("Unable to create window: {}", e));

            println!("Displaying render. Press ESC to close.");
            while window.is_open() && !window.is_key_down(Key::Escape) {
                window
                    .update_with_buffer(&buffer, render_settings.width, render_settings.height)
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to update window buffer: {}", e);
                    });
            }
            println!("Window closed.");
        }
        Err(e) => {
            eprintln!("Failed to load and render scene from '{}': {}", scene_path_str, e);
            return;
        }
    }

    let total_time = start_time.elapsed();
    println!("Total execution time: {}", HumanDuration(total_time));
}
