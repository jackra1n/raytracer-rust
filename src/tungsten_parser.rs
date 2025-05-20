// This file will be renamed from scene_config.rs to tungsten_parser.rs
// Its content will be adapted later for the Tungsten JSON format.
// For now, it contains the JSON parsing logic we developed previously.

use serde::Deserialize;
use std::sync::Arc;
use crate::vec3::Vec3;
use crate::color::Color;
use crate::material::{Material, Lambertian, Metal, Dielectric, EmissiveLight, self};
use crate::camera::Camera;
use crate::scene::Scene;
use image::{DynamicImage, GenericImageView, RgbaImage};
use crate::objects::sphere::Sphere;
use crate::objects::plane::Plane;
use crate::mesh::mesh_object::Mesh;
use std::collections::HashMap;
use glam::{Mat4, Vec3 as GlamVec3, Quat};

#[derive(Deserialize, Debug, Copy, Clone)]
pub struct Vec3Config {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<Vec3Config> for Vec3 {
    fn from(v_conf: Vec3Config) -> Self {
        Vec3::new(v_conf.x, v_conf.y, v_conf.z)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ColorConfig(f32, f32, f32);

impl From<ColorConfig> for Color {
    fn from(c_conf: ColorConfig) -> Self {
        Color::new(c_conf.0, c_conf.1, c_conf.2)
    }
}

#[derive(Deserialize, Debug)]
pub struct SkyConfig {
    pub texture: Option<String>,
}

// New struct for the nested camera transform configuration
#[derive(Deserialize, Debug)]
pub struct CameraTransformConfig {
    pub position: Vec3Config, // Corresponds to look_from
    #[serde(rename = "look_at")] // Already matches, but good for clarity or future changes
    pub look_at: Vec3Config,
    pub up: Vec3Config,       // Corresponds to vup
}

#[derive(Deserialize, Debug)]
pub struct CameraConfig {
    // Removed direct look_from, look_at, vup
    pub transform: CameraTransformConfig, // Added nested transform
    #[serde(rename = "fov")] // Match JSON field name "fov"
    pub vfov: f32,           // Kept internal name vfov for consistency with Camera::new
    pub aspect: Option<f32>, // aspect might not be in this specific JSON, but good to keep
    // Add other fields from JSON if needed, e.g., "type", "tonemap", "resolution"
    // For now, focusing on what Camera::new needs.
    pub resolution: Option<[usize; 2]>, // Added resolution from JSON
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
pub enum MaterialTypeConfig {
    Lambertian {
        albedo: ColorConfig,
    },
    Texture {
        albedo: Option<ColorConfig>,
        pixels: String,
        h_offset: Option<f32>,
    },
    Metal {
        albedo: ColorConfig,
        fuzz: f32,
    },
    Glass {
        index_of_refraction: f32,
    },
    Light {
    },
}

// Define a generic transform config for objects if needed, or use specific fields.
// For now, let's add specific fields directly to Quad and Cube if they are simple.
// The JSON shows position, scale, rotation nested under "transform".

#[derive(Deserialize, Debug)]
pub struct ObjectTransformConfig { // Similar to CameraTransformConfig but for objects
    pub position: Option<Vec3Config>,
    pub scale: Option<Vec3Config>,    // Using Vec3 for non-uniform scale
    pub rotation: Option<Vec3Config>, // Euler angles (degrees) a_x, a_y, a_z
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ObjectConfigVariant {
    Sphere {
        center: Vec3Config,
        radius: f32,
        material: MaterialTypeConfig,
    },
    Plane {
        point: Vec3Config,
        normal: Vec3Config,
        material: MaterialTypeConfig,
    },
    Mesh {
        path: String,
        scale: Option<f32>, // Original mesh config
        offset: Option<Vec3Config>,
        rotation_y: Option<f32>,
        material: MaterialTypeConfig,
    },
    // Added Quad variant
    Quad {
        // Quads in Tungsten JSON have a transform and a bsdf reference.
        // We will need to define how this transform maps to our Quad struct.
        transform: ObjectTransformConfig,
        bsdf: String, // BSDF name to be resolved later
        emission: Option<ColorConfig>, // For light quads
    },
    // Added Cube variant
    Cube {
        // Cubes also have a transform and bsdf ref.
        transform: ObjectTransformConfig,
        bsdf: String, // BSDF name to be resolved later
    },
}

#[derive(Deserialize, Debug)]
pub struct IntegratorConfig { // For "integrator" block in JSON
    #[serde(rename = "max_bounces")]
    pub max_depth: Option<usize>, // Using max_depth to match RenderSettings field name
    // Add other integrator fields if needed, e.g., min_bounces, type
}

#[derive(Deserialize, Debug)]
pub struct RendererConfig { // For "renderer" block in JSON
    #[serde(rename = "spp")]
    pub samples_per_pixel: Option<usize>, // Using samples_per_pixel to match RenderSettings
    // Add other renderer fields if needed, e.g., output_file, hdr_output_file
}

// New struct to represent BSDF definitions from the JSON array
#[derive(Deserialize, Debug, Clone)] // Added Clone
pub struct BsdfConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub type_str: String, // e.g., "lambert", "null"
    // Albedo can be a ColorConfig array [r,g,b] or sometimes a single float (e.g. for Light's "null" BSDF)
    // Or it might be absent. Using Option<serde_json::Value> to handle flexibility initially.
    pub albedo: Option<serde_json::Value>, 
    // Add other BSDF properties as needed, e.g., eta for dielectric, roughness for glossy
}

#[derive(Deserialize, Debug)]
pub struct SceneConfig {
    // pub width: Option<usize>, // These will now come from camera.resolution
    // pub height: Option<usize>, // These will now come from camera.resolution
    // pub samples_per_pixel: Option<usize>, // This will come from renderer.spp
    // pub max_depth: Option<usize>, // This will come from integrator.max_bounces
    pub sky: Option<SkyConfig>,
    pub camera: CameraConfig,
    #[serde(rename = "primitives")]
    pub objects: Vec<ObjectConfigVariant>,
    pub integrator: Option<IntegratorConfig>, // Added integrator config
    pub renderer: Option<RendererConfig>,   // Added renderer config
    pub bsdfs: Option<Vec<BsdfConfig>>, // Added for the top-level BSDFs array
    // Note: The top-level "bsdfs" and "media" from JSON are not yet parsed here.
}

pub struct RenderSettings {
    pub width: usize,
    pub height: usize,
    pub samples_per_pixel: usize,
    pub max_depth: usize,
}

pub struct TextureMaterial {
    pub albedo: Color,
    pub texture: Arc<RgbaImage>,
    pub h_offset: f32,
}

impl Material for TextureMaterial {
    fn scatter(&self, _ray_in: &Ray, hit_record: &HitRecord, rng: &mut dyn RngCore) -> Option<(Ray, Color)> {
        let mut scatter_direction = hit_record.normal + Vec3::random_in_unit_sphere(rng).normalized();
        if scatter_direction.near_zero() {
            scatter_direction = hit_record.normal;
        }
        let scattered_origin = hit_record.position + hit_record.normal * crate::renderer::EPSILON;
        let scattered_ray = Ray::new(scattered_origin, scatter_direction.normalized());

        let normal_tex = hit_record.normal;
        let theta = normal_tex.y.acos();
        let phi = normal_tex.z.atan2(normal_tex.x) + std::f32::consts::PI;
        let mut u = phi / (2.0 * std::f32::consts::PI);
        let v = theta / std::f32::consts::PI;
        u = (u + self.h_offset) % 1.0;

        let tex_width = self.texture.width();
        let tex_height = self.texture.height();
        let x_pixel = (u.max(0.0) * (tex_width -1).max(0) as f32) as u32;
        let y_pixel = (v.max(0.0) * (tex_height-1).max(0) as f32) as u32;
        
        let rgba_texel = self.texture.get_pixel(x_pixel.min(tex_width -1), y_pixel.min(tex_height-1));
        let texture_color = Color::new(rgba_texel[0] as f32 / 255.0, rgba_texel[1] as f32 / 255.0, rgba_texel[2] as f32 / 255.0);

        Some((scattered_ray, self.albedo * texture_color))
    }
}

pub fn load_scene_from_json(json_path: &str) -> Result<(Scene, Camera, RenderSettings), Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(json_path)?;
    let config: SceneConfig = serde_json::from_str(&data)?;

    // Default values for render settings
    let mut width = 800;
    let mut height = 600;
    let mut samples_per_pixel = 16;
    let mut max_depth = 10;

    // Get width and height from camera.resolution if available
    if let Some(res) = config.camera.resolution {
        if res.len() == 2 {
            width = res[0];
            height = res[1];
        }
    }

    // Get samples_per_pixel from renderer.spp if available
    if let Some(renderer_conf) = &config.renderer {
        if let Some(spp) = renderer_conf.samples_per_pixel {
            samples_per_pixel = spp;
        }
    }

    // Get max_depth from integrator.max_bounces if available
    if let Some(integrator_conf) = &config.integrator {
        if let Some(md) = integrator_conf.max_depth {
            max_depth = md;
        }
    }

    let render_settings = RenderSettings {
        width,
        height,
        samples_per_pixel,
        max_depth,
    };

    let aspect_ratio = config.camera.aspect.unwrap_or(render_settings.width as f32 / render_settings.height as f32);
    let camera = Camera::new(
        config.camera.transform.position.into(),
        config.camera.transform.look_at.into(),
        config.camera.transform.up.into(),
        config.camera.vfov,
        aspect_ratio,
    );

    let mut scene = Scene::new();

    // 1. Parse global BSDFs into a HashMap
    let mut parsed_bsdfs_map: HashMap<String, Arc<dyn Material>> = HashMap::new();
    if let Some(bsdf_list) = &config.bsdfs {
        for bsdf_conf in bsdf_list {
            let material_type_config_result: Result<MaterialTypeConfig, String> = match bsdf_conf.type_str.as_str() {
                "lambert" => {
                    if let Some(albedo_val) = &bsdf_conf.albedo {
                        match serde_json::from_value::<ColorConfig>(albedo_val.clone()) {
                            Ok(cc) => Ok(MaterialTypeConfig::Lambertian { albedo: cc }),
                            Err(e) => {
                                // Attempt to parse as single float for grayscale Lambertian
                                match serde_json::from_value::<f32>(albedo_val.clone()) {
                                    Ok(gray_val) => Ok(MaterialTypeConfig::Lambertian { albedo: ColorConfig(gray_val, gray_val, gray_val) }),
                                    Err(_) => Err(format!("Failed to parse albedo for Lambertian BSDF '{}' as Color or f32: {}", bsdf_conf.name, e))
                                }
                            }
                        }
                    } else {
                        Err(format!("Lambertian BSDF '{}' missing albedo.", bsdf_conf.name))
                    }
                }
                "null" => {
                    // For "null" BSDF, used by lights. Emission is on the primitive.
                    // Create a placeholder; it will be overridden if the primitive is emissive.
                    Ok(MaterialTypeConfig::Lambertian{ albedo: ColorConfig(0.0, 0.0, 0.0) }) 
                }
                // TODO: Add other BSDF types like "dielectric", "metal", "phong", etc.
                unsupported_type => {
                    eprintln!("Warning: Unsupported BSDF type '{}' for BSDF named '{}'.", unsupported_type, bsdf_conf.name);
                    Err(format!("Unsupported BSDF type: {}", unsupported_type))
                }
            };

            if let Ok(mat_type_conf) = material_type_config_result {
                 // Use a modified parse_material or directly create material here
                 // The original parse_material closure is defined later and captures obj_conf_variant.material
                 // We need a way to parse MaterialTypeConfig to Arc<dyn Material> here.
                 // For now, let's adapt the logic from the existing closure:
                let material_arc: Arc<dyn Material> = match mat_type_conf {
                    MaterialTypeConfig::Lambertian { albedo } => Arc::new(Lambertian::new(albedo.into())),
                    MaterialTypeConfig::Texture { albedo, pixels, h_offset } => { /* ... as before ... */ Arc::new(Lambertian::new(Color::WHITE)) }, // Placeholder
                    MaterialTypeConfig::Metal { albedo, fuzz } => Arc::new(Metal::new(albedo.into(), fuzz)),
                    MaterialTypeConfig::Glass { index_of_refraction } => Arc::new(Dielectric::new(index_of_refraction)),
                    MaterialTypeConfig::Light {} => Arc::new(EmissiveLight::new(Color::new(1.0,1.0,1.0))), // Default for BSDF Light type?
                };
                parsed_bsdfs_map.insert(bsdf_conf.name.clone(), material_arc);
            } else if let Err(e_str) = material_type_config_result {
                eprintln!("Skipping BSDF '{}': {}", bsdf_conf.name, e_str);
            }
        }
    }

    if let Some(sky_conf) = &config.sky {
        if let Some(tex_path) = &sky_conf.texture {
            match image::open(tex_path) {
                Ok(img) => {
                    println!("Successfully loaded skybox image: {}", tex_path);
                    scene.skybox_image = Some(img);
                }
                Err(e) => eprintln!("Error loading skybox image '{}': {}. Using default background.", tex_path, e),
            }
        }
    }

    for obj_conf_variant in config.objects {
        // This parse_material closure is problematic because it uses obj_conf_variant.material, 
        // but Quad/Cube now have obj_conf_variant.bsdf (a string) + obj_conf_variant.emission.
        // We need to resolve the material using parsed_bsdfs_map for Quads/Cubes.
        
        // let parse_material = |mat_conf: MaterialTypeConfig| -> Arc<dyn Material> { ... }; // Original closure

        match obj_conf_variant {
            ObjectConfigVariant::Sphere { center, radius, material } => {
                // Sphere still uses inline material definition as per our current SceneConfig structure
                let sphere_material_type_conf = material; // This is MaterialTypeConfig
                let sphere_material: Arc<dyn Material> = match sphere_material_type_conf {
                    MaterialTypeConfig::Lambertian { albedo } => Arc::new(Lambertian::new(albedo.into())),
                    MaterialTypeConfig::Texture { albedo, pixels, h_offset } => { /* ... */ Arc::new(Lambertian::new(Color::WHITE)) },
                    MaterialTypeConfig::Metal { albedo, fuzz } => Arc::new(Metal::new(albedo.into(), fuzz)),
                    MaterialTypeConfig::Glass { index_of_refraction } => Arc::new(Dielectric::new(index_of_refraction)),
                    MaterialTypeConfig::Light {} => Arc::new(EmissiveLight::new(Color::new(10.0, 10.0, 10.0))),
                };
                let sphere = Sphere {
                    center: center.into(),
                    radius,
                    material: sphere_material,
                };
                scene.add_object(Box::new(sphere));
            }
            ObjectConfigVariant::Plane { point, normal, material } => {
                // Plane also uses inline material definition
                let plane_material_type_conf = material; // This is MaterialTypeConfig
                let plane_material: Arc<dyn Material> = match plane_material_type_conf { /* ... similar to Sphere ... */ MaterialTypeConfig::Lambertian{albedo} => Arc::new(Lambertian::new(albedo.into())), _ => Arc::new(Lambertian::new(Color::WHITE)) }; // Simplified for brevity
                let plane = Plane::new(point.into(), normal.into(), plane_material);
                scene.add_object(Box::new(plane));
            }
            ObjectConfigVariant::Mesh { path, scale, offset, rotation_y, material } => {
                // Mesh also uses inline material definition
                let mesh_material_type_conf = material; // This is MaterialTypeConfig
                let mesh_material: Arc<dyn Material> = match mesh_material_type_conf { /* ... similar to Sphere ... */ _ => Arc::new(Lambertian::new(Color::WHITE)) }; // Simplified
                let s = scale.unwrap_or(1.0);
                let o = offset.map_or(Vec3::new(0.0, 0.0, 0.0), |v_conf| v_conf.into());
                let r_y = rotation_y.unwrap_or(0.0);
                match Mesh::from_obj(&path, mesh_material, s, o, r_y) {
                    Ok(mesh_obj) => scene.add_object(Box::new(mesh_obj)),
                    Err(e) => eprintln!("Error loading mesh '{}': {}", path, e),
                }
            }
            ObjectConfigVariant::Quad { transform, bsdf, emission } => {
                let quad_material: Arc<dyn Material> = if let Some(emission_color_conf) = emission {
                    Arc::new(EmissiveLight::new(emission_color_conf.into()))
                } else {
                    parsed_bsdfs_map.get(&bsdf)
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Warning: BSDF '{}' not found for Quad '{}'. Using default material.", bsdf, bsdf);
                            Arc::new(Lambertian::new(Color::MAGENTA))
                        })
                };

                let center = transform.position.map_or(Vec3::new(0.0,0.0,0.0), |p| p.into());
                // Scale in JSON is full dimensions [width, height, depth] for the object in its local orientation
                let json_scale = transform.scale.map_or(Vec3::new(1.0,1.0,1.0), |s| s.into()); 
                // Rotation is Vec3(ax, ay, az) in degrees. Not used by new_axis_aligned yet.
                // let rotation = transform.rotation.map_or(Vec3::new(0.0,0.0,0.0), |r| r.into());

                // Heuristic for major_axis based on typical Cornell Box names
                // This is a HACK and needs proper transform handling.
                let major_axis = match bsdf.as_str() {
                    "Floor" | "Ceiling" | "Light" => 1, // XZ plane, normal along Y
                    "BackWall" => 2,                 // XY plane, normal along Z (assuming default Tungsten orientation)
                    "LeftWall" | "RightWall" => 0,    // YZ plane, normal along X
                    _ => 1, // Default to XZ plane if name doesn't match
                };
                
                // The `size` for new_axis_aligned depends on the major_axis.
                // If major_axis = 1 (XZ plane), size.x is width, size.z is depth. size.y is ignored (thickness)
                // The JSON scale is likely [width, height, depth] in its own coordinate system.
                // For an XZ quad (Floor/Light), scale.x is width, scale.y is thickness, scale.z is depth.
                // So, size for new_axis_aligned would be Vec3(json_scale.x, thickness, json_scale.z)
                // The current new_axis_aligned uses size directly. This needs careful mapping.

                // Simplification: use json_scale as the size directly. This will work if the quad's local orientation
                // matches the axis it lies on (e.g. scale.y is small for an XZ quad).
                let quad_size = json_scale; 

                let quad = crate::objects::quad::Quad::new_axis_aligned(center, quad_size, major_axis, quad_material);
                scene.add_object(Box::new(quad));
                // eprintln!("Quad loading (with material '{}') partially implemented for rendering.", bsdf);
            }
            ObjectConfigVariant::Cube { transform, bsdf } => {
                let cube_material = parsed_bsdfs_map.get(&bsdf)
                    .cloned()
                    .unwrap_or_else(|| {
                        eprintln!("Warning: BSDF '{}' not found for Cube '{}'. Using default material.", bsdf, bsdf);
                        Arc::new(Lambertian::new(Color::MAGENTA))
                    });

                let translation_vec = transform.position.map_or(GlamVec3::ZERO, |p| GlamVec3::new(p.x, p.y, p.z));
                let scale_vec = transform.scale.map_or(GlamVec3::ONE, |s| GlamVec3::new(s.x, s.y, s.z));
                let rotation_angles_deg_json = transform.rotation.map_or(GlamVec3::ZERO, |r| GlamVec3::new(r.x, r.y, r.z));
                
                // Convert Euler angles (degrees) to Quaternions for rotation
                // Using YXZ order based on Tungsten source code hint
                let rotation_quat = Quat::from_euler(
                    glam::EulerRot::YXZ, 
                    rotation_angles_deg_json.y.to_radians(),   // Y rotation angle
                    rotation_angles_deg_json.x.to_radians(),   // X rotation angle
                    rotation_angles_deg_json.z.to_radians()    // Z rotation angle
                );

                let transform_matrix = Mat4::from_scale_rotation_translation(scale_vec, rotation_quat, translation_vec);

                let cube_obj = crate::objects::cube::Cube::new_transformed(transform_matrix, cube_material);

                scene.add_object(Box::new(cube_obj));
            }
        }
    }

    Ok((scene, camera, render_settings))
}

use crate::ray::Ray;
use rand::RngCore;
use crate::hittable::HitRecord;
