use crate::camera::Camera;
use crate::color::Color;
use crate::hittable::Hittable;
use crate::material::{self, Dielectric, EmissiveLight, Lambertian, Material, Metal, NullMaterial};
use crate::mesh::mesh_object::Mesh;
use crate::objects::plane::Plane;
use crate::objects::sphere::Sphere;
use crate::scene::Scene;
use crate::vec3::Vec3;
use glam::{Mat4, Quat, Vec3 as GlamVec3};
use image::RgbaImage;
use serde::Deserialize;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

#[derive(Deserialize, Debug, Copy, Clone, Default)]
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

#[derive(Deserialize, Debug)]
pub struct CameraTransformConfig {
    pub position: Vec3Config,
    #[serde(rename = "look_at")]
    pub look_at: Vec3Config,
    pub up: Vec3Config,
}

#[derive(Deserialize, Debug)]
pub struct CameraConfig {
    pub transform: CameraTransformConfig,
    #[serde(rename = "fov")]
    pub vfov: f32,
    pub aspect: Option<f32>,
    pub resolution: Option<ResolutionConfig>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ResolutionConfig {
    Square(usize),
    Explicit([usize; 2]),
}

#[derive(Deserialize, Debug, Clone)]
pub struct CheckerTextureConfig {
    pub on_color: ColorConfig,
    pub off_color: ColorConfig,
    pub res_u: Option<f32>,
    pub res_v: Option<f32>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum AlbedoConfig {
    Solid(ColorConfig),
    GrayscaleSolid(f32),
    Checker(CheckerTextureConfig),
}

impl AlbedoConfig {
    fn into_color_or_texture(&self, _scene_directory: &Path) -> Result<Color, String> {
        match self {
            AlbedoConfig::Solid(cc) => Ok((*cc).clone().into()),
            AlbedoConfig::GrayscaleSolid(f) => Ok(Color::new(*f, *f, *f)),
            AlbedoConfig::Checker(_) => Err("AlbedoConfig::Checker cannot be converted to a single Color. Use Lambertian::new_checker.".to_string()),
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
pub enum MaterialTypeConfig {
    Lambertian {
        albedo: AlbedoConfig,
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
    Light {},
    Plastic {
        albedo: ColorConfig,
        ior: f32,
    },
    RoughConductor {
        albedo: ColorConfig,
        roughness: f32,
        metal_type: material::MetalType,
        distribution: material::MicrofacetDistribution,
    },
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ScaleConfig {
    Uniform(f32),
    NonUniform(Vec3Config),
}

#[derive(Deserialize, Debug, Clone)]
pub struct ObjectTransformConfig {
    pub position: Option<Vec3Config>,
    pub scale: Option<ScaleConfig>,
    pub rotation: Option<Vec3Config>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ObjectConfigVariant {
    Sphere {
        transform: ObjectTransformConfig,
        radius: Option<f32>,
        bsdf: String,
        power: Option<f32>,
    },
    Plane {
        point: Vec3Config,
        normal: Vec3Config,
        material: MaterialTypeConfig,
    },
    Mesh {
        transform: ObjectTransformConfig,
        #[serde(rename = "file")]
        path: String,
        bsdf: String,
        smooth: Option<bool>,
        backface_culling: Option<bool>,
        recompute_normals: Option<bool>,
    },
    Quad {
        transform: ObjectTransformConfig,
        bsdf: String,
        emission: Option<serde_json::Value>,
    },
    Cube {
        transform: ObjectTransformConfig,
        bsdf: String,
    },
    #[serde(rename = "infinite_sphere")]
    InfiniteSphere {
        transform: ObjectTransformConfig,
        emission: Option<String>,
        sample: Option<bool>,
    },
    #[serde(rename = "infinite_sphere_cap")]
    InfiniteSphereCap {
        transform: ObjectTransformConfig,
        emission: Option<String>,
        power: Option<f32>,
        sample: Option<bool>,
        cap_angle: Option<f32>,
    },
}

#[derive(Deserialize, Debug)]
pub struct IntegratorConfig {
    #[serde(rename = "max_bounces")]
    pub max_depth: Option<usize>,
}

#[derive(Deserialize, Debug)]
pub struct RendererConfig {
    #[serde(rename = "spp")]
    pub samples_per_pixel: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BsdfConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub type_str: String,
    pub albedo: Option<serde_json::Value>,
    pub ior: Option<f32>,
    pub roughness: Option<f32>,
    pub material: Option<String>,
    pub distribution: Option<String>,
}

#[derive(Debug)]
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
    fn scatter(
        &self,
        _ray_in: &Ray,
        hit_record: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> Option<(Ray, Color)> {
        let mut scatter_direction =
            hit_record.normal + Vec3::random_in_unit_sphere(rng).normalized();
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
        let x_pixel = (u.max(0.0) * (tex_width - 1).max(0) as f32) as u32;
        let y_pixel = (v.max(0.0) * (tex_height - 1).max(0) as f32) as u32;

        let rgba_texel = self
            .texture
            .get_pixel(x_pixel.min(tex_width - 1), y_pixel.min(tex_height - 1));
        let texture_color = Color::new(
            rgba_texel[0] as f32 / 255.0,
            rgba_texel[1] as f32 / 255.0,
            rgba_texel[2] as f32 / 255.0,
        );

        Some((scattered_ray, self.albedo * texture_color))
    }
}

pub fn load_scene_from_json(
    json_path: &str,
) -> Result<(Scene, Camera, RenderSettings), Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(json_path)?;
    let config: SceneConfig = serde_json::from_str(&data)?;

    let scene_dir = Path::new(json_path).parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid scene JSON path")
    })?;

    let mut width = 800;
    let mut height = 600;
    let mut samples_per_pixel = 16;
    let mut max_depth = 10;

    if let Some(res_conf) = config.camera.resolution {
        match res_conf {
            ResolutionConfig::Square(s) => {
                width = s;
                height = s;
            }
            ResolutionConfig::Explicit(arr) => {
                if arr.len() == 2 {
                    width = arr[0];
                    height = arr[1];
                }
            }
        }
    }

    if let Some(renderer_conf) = &config.renderer {
        if let Some(spp) = renderer_conf.samples_per_pixel {
            samples_per_pixel = spp;
        }
    }

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

    let aspect_ratio = config
        .camera
        .aspect
        .unwrap_or(render_settings.width as f32 / render_settings.height as f32);
    let camera = Camera::new(
        config.camera.transform.position.into(),
        config.camera.transform.look_at.into(),
        config.camera.transform.up.into(),
        config.camera.vfov,
        aspect_ratio,
    );

    let mut scene = Scene::new();

    let mut parsed_bsdfs_map: HashMap<String, Arc<dyn Material>> = HashMap::new();
    if let Some(bsdf_list) = &config.bsdfs {
        for bsdf_conf in bsdf_list {
            let material_type_config_result: Result<MaterialTypeConfig, String> = match bsdf_conf
                .type_str
                .as_str()
            {
                "lambert" => {
                    if let Some(albedo_val) = &bsdf_conf.albedo {
                        match serde_json::from_value::<AlbedoConfig>(albedo_val.clone()) {
                            Ok(parsed_albedo_config) => Ok(MaterialTypeConfig::Lambertian { albedo: parsed_albedo_config }),
                            Err(e) => Err(format!("Failed to parse albedo for Lambertian BSDF '{}' as Color, f32, or Checker: {}. Value: {:?}", bsdf_conf.name, e, albedo_val))
                        }
                    } else {
                        Err(format!(
                            "Lambertian BSDF '{}' missing albedo.",
                            bsdf_conf.name
                        ))
                    }
                }
                "plastic" => {
                    let mut parsed_albedo_for_plastic = ColorConfig(0.8, 0.8, 0.8);
                    if let Some(albedo_val) = &bsdf_conf.albedo {
                        match serde_json::from_value::<ColorConfig>(albedo_val.clone()) {
                            Ok(cc) => parsed_albedo_for_plastic = cc,
                            Err(_) => match serde_json::from_value::<f32>(albedo_val.clone()) {
                                Ok(gray_val) => {
                                    parsed_albedo_for_plastic =
                                        ColorConfig(gray_val, gray_val, gray_val)
                                }
                                Err(e_f32) => {
                                    eprintln!("Warning: Failed to parse albedo for Plastic BSDF '{}' as Color or f32: {}. Using default albedo.", bsdf_conf.name, e_f32);
                                }
                            },
                        }
                    } else {
                        eprintln!(
                            "Warning: Plastic BSDF '{}' missing albedo. Using default albedo.",
                            bsdf_conf.name
                        );
                    }

                    let parsed_ior = bsdf_conf.ior.unwrap_or(1.5);

                    Ok(MaterialTypeConfig::Plastic {
                        albedo: parsed_albedo_for_plastic,
                        ior: parsed_ior,
                    })
                }
                "null" => Ok(MaterialTypeConfig::Lambertian {
                    albedo: AlbedoConfig::Solid(ColorConfig(0.0, 0.0, 0.0)),
                }),
                "rough_conductor" => {
                    let mut albedo = Color::new(1.0, 1.0, 1.0);
                    let mut roughness = 0.1;
                    let mut metal_type = material::MetalType::Cu;
                    let mut distribution = material::MicrofacetDistribution::GGX;

                    if let Some(albedo_val) = &bsdf_conf.albedo {
                        if let Ok(cc) = serde_json::from_value::<ColorConfig>(albedo_val.clone()) {
                            albedo = cc.into();
                        } else if let Ok(gray_val) =
                            serde_json::from_value::<f32>(albedo_val.clone())
                        {
                            albedo = Color::new(gray_val, gray_val, gray_val);
                        }
                    }
                    if let Some(r_val) = bsdf_conf.roughness {
                        roughness = r_val;
                    }

                    if let Some(mat_str) = bsdf_conf.material.as_ref() {
                        metal_type = match mat_str.to_lowercase().as_str() {
                            "cu" => material::MetalType::Cu,
                            "au" => material::MetalType::Au,
                            "ag" => material::MetalType::Ag,
                            "al" => material::MetalType::Al,
                            "ni" => material::MetalType::Ni,
                            "ti" => material::MetalType::Ti,
                            "fe" => material::MetalType::Fe,
                            "pb" => material::MetalType::Pb,
                            _ => {
                                eprintln!("Warning: Unknown metal type '{}' for rough_conductor, defaulting to Cu.", mat_str);
                                material::MetalType::Cu
                            }
                        };
                    }
                    if let Some(dist_str) = bsdf_conf.distribution.as_ref() {
                        distribution = match dist_str.to_lowercase().as_str() {
                            "ggx" => material::MicrofacetDistribution::GGX,
                            "beckmann" => material::MicrofacetDistribution::Beckmann,
                            _ => {
                                eprintln!("Warning: Unknown distribution '{}' for rough_conductor, defaulting to GGX.", dist_str);
                                material::MicrofacetDistribution::GGX
                            }
                        };
                    }
                    Ok(MaterialTypeConfig::RoughConductor {
                        albedo: ColorConfig(albedo.r, albedo.g, albedo.b),
                        roughness,
                        metal_type,
                        distribution,
                    })
                }
                unsupported_type => {
                    eprintln!(
                        "Warning: Unsupported BSDF type '{}' for BSDF named '{}'.",
                        unsupported_type, bsdf_conf.name
                    );
                    Err(format!("Unsupported BSDF type: {}", unsupported_type))
                }
            };

            if let Ok(mat_type_conf) = material_type_config_result {
                let material_arc: Arc<dyn Material> = match mat_type_conf {
                    MaterialTypeConfig::Lambertian { albedo } => match albedo {
                        AlbedoConfig::Solid(cc) => Arc::new(Lambertian::new_solid(cc.into())),
                        AlbedoConfig::GrayscaleSolid(f) => {
                            Arc::new(Lambertian::new_solid(Color::new(f, f, f)))
                        }
                        AlbedoConfig::Checker(tex_conf) => {
                            let scale = tex_conf.res_u.or(tex_conf.res_v).unwrap_or(10.0);
                            let checker_tex = Arc::new(material::CheckerTexture::new(
                                tex_conf.on_color.into(),
                                tex_conf.off_color.into(),
                                scale,
                            ));
                            Arc::new(Lambertian::new_checker(checker_tex))
                        }
                    },
                    MaterialTypeConfig::Texture {
                        albedo,
                        pixels,
                        h_offset,
                    } => {
                        let default_albedo = albedo.map_or(Color::WHITE, |c| c.into());
                        let tex_path_abs = scene_dir.join(&pixels);
                        match image::open(&tex_path_abs) {
                            Ok(dynamic_image) => {
                                let rgba_image = dynamic_image.to_rgba8();
                                Arc::new(TextureMaterial {
                                    albedo: default_albedo,
                                    texture: Arc::new(rgba_image),
                                    h_offset: h_offset.unwrap_or(0.0),
                                })
                            }
                            Err(e) => {
                                eprintln!("Error loading texture image for TextureMaterial '{:?}': {}. Using default albedo.", tex_path_abs, e);
                                Arc::new(Lambertian::new_solid(default_albedo))
                            }
                        }
                    }
                    MaterialTypeConfig::Metal { albedo, fuzz } => {
                        Arc::new(Metal::new(albedo.into(), fuzz))
                    }
                    MaterialTypeConfig::Glass {
                        index_of_refraction,
                    } => Arc::new(Dielectric::new(index_of_refraction)),
                    MaterialTypeConfig::Light {} => {
                        Arc::new(EmissiveLight::new(Color::new(1.0, 1.0, 1.0)))
                    }
                    MaterialTypeConfig::Plastic { albedo, ior } => {
                        Arc::new(material::PlasticMaterial::new(albedo.into(), ior))
                    }
                    MaterialTypeConfig::RoughConductor {
                        albedo,
                        roughness,
                        metal_type,
                        distribution,
                    } => Arc::new(material::RoughConductor::new(
                        albedo.into(),
                        roughness,
                        metal_type,
                        distribution,
                    )),
                };
                parsed_bsdfs_map.insert(bsdf_conf.name.clone(), material_arc);
            } else if let Err(e_str) = material_type_config_result {
                eprintln!("Skipping BSDF '{}': {}", bsdf_conf.name, e_str);
            }
        }
    }

    for obj_conf_variant in &config.objects {
        if let ObjectConfigVariant::InfiniteSphereCap {
            emission: Some(emission_path),
            ..
        } = obj_conf_variant
        {
            if emission_path.ends_with(".hdr") {
                let tex_path_abs = scene_dir.join(emission_path);
                match image::open(&tex_path_abs) {
                    Ok(img) => {
                        let hdr_image = img.into_rgb32f();
                        println!("Successfully loaded HDR skybox from InfiniteSphereCap: {:?}", tex_path_abs);
                        scene.skybox_hdr_image = Some(hdr_image);
                        break;
                    }
                    Err(e) => eprintln!("Error loading HDR skybox image from InfiniteSphereCap '{:?}': {}. Proceeding without it.", tex_path_abs, e),
                }
            } else if emission_path.ends_with(".png")
                || emission_path.ends_with(".jpg")
                || emission_path.ends_with(".jpeg")
            {
                let tex_path_abs = scene_dir.join(emission_path);
                match image::open(&tex_path_abs) {
                    Ok(img) => {
                        println!("Successfully loaded LDR skybox from InfiniteSphereCap: {:?}", tex_path_abs);
                        scene.skybox_image = Some(img);
                        break;
                    }
                    Err(e) => eprintln!("Error loading LDR skybox image from InfiniteSphereCap '{:?}': {}. Proceeding without it.", tex_path_abs, e),
                }
            }
        }
    }

    if scene.skybox_hdr_image.is_none() && scene.skybox_image.is_none() {
        if let Some(sky_conf) = &config.sky {
            if let Some(tex_path_relative) = &sky_conf.texture {
                let tex_path_abs = scene_dir.join(tex_path_relative);
                if tex_path_relative.ends_with(".hdr") {
                    match image::open(&tex_path_abs) {
                        Ok(img) => {
                            let hdr_image = img.into_rgb32f();
                            println!("Successfully loaded and converted HDR skybox image from 'sky' config: {:?}", tex_path_abs);
                            scene.skybox_hdr_image = Some(hdr_image);
                        }
                        Err(e) => eprintln!("Error loading HDR skybox image from 'sky' config '{:?}': {}. Using default background.", tex_path_abs, e),
                    }
                } else {
                    match image::open(&tex_path_abs) {
                        Ok(img) => {
                            println!("Successfully loaded LDR skybox image from 'sky' config: {:?}", tex_path_abs);
                            scene.skybox_image = Some(img);
                        }
                        Err(e) => eprintln!("Error loading LDR skybox image from 'sky' config '{:?}': {}. Using default background.", tex_path_abs, e),
                    }
                }
            }
        }
    }

    for obj_conf_variant in config.objects {
        match obj_conf_variant {
            ObjectConfigVariant::Sphere {
                transform,
                radius,
                bsdf,
                power,
            } => {
                let mut sphere_material: Arc<dyn Material>;

                if let Some(p_val) = power {
                    let intensity = (p_val / 300.0).clamp(0.0, 50.0);
                    let emissive_color = Color::new(intensity, intensity, intensity);
                    sphere_material = Arc::new(EmissiveLight::new(emissive_color));
                } else {
                    sphere_material = parsed_bsdfs_map.get(&bsdf)
                        .cloned()
                        .unwrap_or_else(|| {
                            eprintln!("Warning: BSDF '{}' not found for Sphere. Using default (magenta Lambertian) material.", bsdf);
                            Arc::new(Lambertian::new_solid(Color::MAGENTA))
                        });
                }

                let center_pos = transform
                    .position
                    .map_or(Vec3::new(0.0, 0.0, 0.0), |p| p.into());

                let sphere_radius = match radius {
                    Some(r) => r,
                    None => match transform.scale {
                        Some(ScaleConfig::Uniform(s)) => s,
                        Some(ScaleConfig::NonUniform(v_conf)) => {
                            if (v_conf.x - v_conf.y).abs() > 1e-6
                                || (v_conf.x - v_conf.z).abs() > 1e-6
                            {
                                eprintln!("Warning: Sphere parsed with non-uniform scale ({:?}) and no explicit radius. Using x-component for radius.", v_conf);
                            }
                            v_conf.x
                        }
                        None => 1.0,
                    },
                };

                if let Some(p_val) = power {
                    let radiance_val = if sphere_radius > 1e-6 {
                        p_val / (4.0 * PI * PI * sphere_radius * sphere_radius)
                    } else {
                        eprintln!("Warning: Sphere light has zero or very small radius ({}). Emitting no light.", sphere_radius);
                        0.0
                    };

                    let emissive_color = Color::new(radiance_val, radiance_val, radiance_val);
                    sphere_material = Arc::new(EmissiveLight::new(emissive_color));
                }

                let sphere = Sphere {
                    center: center_pos,
                    radius: sphere_radius,
                    material: sphere_material,
                };
                scene.add_object(Box::new(sphere));
            }
            ObjectConfigVariant::Plane {
                point,
                normal,
                material,
            } => {
                let plane_material: Arc<dyn Material> = match material {
                    MaterialTypeConfig::Lambertian { albedo } => match albedo {
                        AlbedoConfig::Solid(cc) => Arc::new(Lambertian::new_solid(cc.into())),
                        AlbedoConfig::GrayscaleSolid(f) => {
                            Arc::new(Lambertian::new_solid(Color::new(f, f, f)))
                        }
                        AlbedoConfig::Checker(tex_conf) => {
                            let scale = tex_conf.res_u.or(tex_conf.res_v).unwrap_or(10.0);
                            let checker_texture = Arc::new(material::CheckerTexture::new(
                                tex_conf.on_color.into(),
                                tex_conf.off_color.into(),
                                scale,
                            ));
                            Arc::new(Lambertian::new_checker(checker_texture))
                        }
                    },
                    MaterialTypeConfig::Metal { albedo, fuzz } => {
                        Arc::new(Metal::new(albedo.into(), fuzz))
                    }
                    MaterialTypeConfig::Glass {
                        index_of_refraction,
                    } => Arc::new(Dielectric::new(index_of_refraction)),
                    MaterialTypeConfig::Plastic { albedo, ior } => {
                        Arc::new(material::PlasticMaterial::new(albedo.into(), ior))
                    }
                    MaterialTypeConfig::RoughConductor {
                        albedo,
                        roughness,
                        metal_type,
                        distribution,
                    } => Arc::new(material::RoughConductor::new(
                        albedo.into(),
                        roughness,
                        metal_type,
                        distribution,
                    )),
                    _ => {
                        eprintln!("Warning: Unsupported inline material type for Plane. Defaulting to white Lambertian.");
                        Arc::new(Lambertian::new_solid(Color::WHITE))
                    }
                };
                let plane = Plane::new(point.into(), normal.into(), plane_material);
                scene.add_object(Box::new(plane));
            }
            ObjectConfigVariant::Mesh {
                transform,
                path,
                bsdf,
                smooth,
                backface_culling,
                recompute_normals,
            } => {
                let mesh_material = parsed_bsdfs_map.get(&bsdf).cloned().unwrap_or_else(|| {
                    eprintln!(
                        "Warning: BSDF '{}' not found for Mesh '{}'. Using default material.",
                        bsdf, path
                    );
                    Arc::new(Lambertian::new_solid(Color::MAGENTA))
                });

                let translation_vec = transform
                    .position
                    .map_or(GlamVec3::ZERO, |p| GlamVec3::new(p.x, p.y, p.z));

                let scale_vec = match transform.scale {
                    Some(ScaleConfig::Uniform(s)) => GlamVec3::splat(s),
                    Some(ScaleConfig::NonUniform(v_conf)) => {
                        GlamVec3::new(v_conf.x, v_conf.y, v_conf.z)
                    }
                    None => GlamVec3::ONE,
                };

                let rotation_angles_deg_json = transform
                    .rotation
                    .map_or(GlamVec3::ZERO, |r| GlamVec3::new(r.x, r.y, r.z));

                let rotation_quat = Quat::from_euler(
                    glam::EulerRot::YXZ,
                    rotation_angles_deg_json.y.to_radians(),
                    rotation_angles_deg_json.x.to_radians(),
                    rotation_angles_deg_json.z.to_radians(),
                );

                let object_to_world_matrix = Mat4::from_scale_rotation_translation(
                    scale_vec,
                    rotation_quat,
                    translation_vec,
                );

                let model_path_abs = scene_dir.join(&path);

                if path.ends_with(".wo3") {
                    match Mesh::from_wo3(
                        model_path_abs.to_str().unwrap_or_default(),
                        mesh_material.clone(),
                        object_to_world_matrix,
                    ) {
                        Ok(mesh_obj) => scene.add_object(Box::new(mesh_obj)),
                        Err(e) => {
                            eprintln!("Error loading .wo3 mesh '{:?}': {}", model_path_abs, e)
                        }
                    }
                } else {
                    match Mesh::from_obj(
                        model_path_abs.to_str().unwrap_or_default(),
                        mesh_material.clone(),
                        object_to_world_matrix,
                    ) {
                        Ok(mesh_obj) => scene.add_object(Box::new(mesh_obj)),
                        Err(e) => {
                            eprintln!("Error loading .obj mesh '{:?}': {}", model_path_abs, e)
                        }
                    }
                }
            }
            ObjectConfigVariant::Quad {
                transform,
                bsdf,
                emission,
            } => {
                let quad_material: Arc<dyn Material> = if let Some(emission_color_conf) = emission {
                    if let Ok(color_conf) =
                        serde_json::from_value::<ColorConfig>(emission_color_conf.clone())
                    {
                        Arc::new(EmissiveLight::new(color_conf.into()))
                    } else if let Ok(texture_path_str) =
                        serde_json::from_value::<String>(emission_color_conf.clone())
                    {
                        eprintln!("Warning: Textured emission for Quad ('{}') not fully supported yet. Treating as bright light.", texture_path_str);
                        Arc::new(EmissiveLight::new(Color::new(5.0, 5.0, 5.0)))
                    } else {
                        eprintln!(
                            "Warning: Could not parse emission for Quad. Using default material."
                        );
                        parsed_bsdfs_map
                            .get(&bsdf)
                            .cloned()
                            .unwrap_or_else(|| Arc::new(Lambertian::new_solid(Color::MAGENTA)))
                    }
                } else {
                    parsed_bsdfs_map.get(&bsdf).cloned().unwrap_or_else(|| {
                        eprintln!(
                            "Warning: BSDF '{}' not found for Quad. Using default material.",
                            bsdf
                        );
                        Arc::new(Lambertian::new_solid(Color::MAGENTA))
                    })
                };

                let translation_vec = transform
                    .position
                    .map_or(GlamVec3::ZERO, |p| GlamVec3::new(p.x, p.y, p.z));

                let scale_vec = match transform.scale {
                    Some(ScaleConfig::Uniform(s)) => GlamVec3::splat(s),
                    Some(ScaleConfig::NonUniform(v_conf)) => {
                        GlamVec3::new(v_conf.x, v_conf.y, v_conf.z)
                    }
                    None => GlamVec3::ONE,
                };

                let rotation_angles_deg_json = transform
                    .rotation
                    .map_or(GlamVec3::ZERO, |r| GlamVec3::new(r.x, r.y, r.z));

                let rotation_quat = Quat::from_euler(
                    glam::EulerRot::YXZ,
                    rotation_angles_deg_json.y.to_radians(),
                    rotation_angles_deg_json.x.to_radians(),
                    rotation_angles_deg_json.z.to_radians(),
                );

                let transform_matrix = Mat4::from_scale_rotation_translation(
                    scale_vec,
                    rotation_quat,
                    translation_vec,
                );

                let quad =
                    crate::objects::quad::Quad::new_transformed(transform_matrix, quad_material);
                scene.add_object(Box::new(quad));
            }
            ObjectConfigVariant::Cube { transform, bsdf } => {
                let cube_material = parsed_bsdfs_map.get(&bsdf).cloned().unwrap_or_else(|| {
                    eprintln!(
                        "Warning: BSDF '{}' not found for Cube '{}'. Using default material.",
                        bsdf, bsdf
                    );
                    Arc::new(Lambertian::new_solid(Color::MAGENTA))
                });

                let translation_vec = transform
                    .position
                    .map_or(GlamVec3::ZERO, |p| GlamVec3::new(p.x, p.y, p.z));

                let scale_vec = match transform.scale {
                    Some(ScaleConfig::Uniform(s)) => GlamVec3::splat(s),
                    Some(ScaleConfig::NonUniform(v_conf)) => {
                        GlamVec3::new(v_conf.x, v_conf.y, v_conf.z)
                    }
                    None => GlamVec3::ONE,
                };

                let rotation_angles_deg_json = transform
                    .rotation
                    .map_or(GlamVec3::ZERO, |r| GlamVec3::new(r.x, r.y, r.z));

                let rotation_quat = Quat::from_euler(
                    glam::EulerRot::YXZ,
                    rotation_angles_deg_json.y.to_radians(),
                    rotation_angles_deg_json.x.to_radians(),
                    rotation_angles_deg_json.z.to_radians(),
                );

                let transform_matrix = Mat4::from_scale_rotation_translation(
                    scale_vec,
                    rotation_quat,
                    translation_vec,
                );

                let cube_obj =
                    crate::objects::cube::Cube::new_transformed(transform_matrix, cube_material);

                scene.add_object(Box::new(cube_obj));
            }
            ObjectConfigVariant::InfiniteSphere {
                transform: obj_transform,
                emission,
                sample: _sample,
            } => {
                if let Some(emission_path) = emission {
                    if scene.skybox_hdr_image.is_some() || scene.skybox_image.is_some() {
                        println!("InfiniteSphere with emission '{}' was likely used as skybox, not adding as separate object.", emission_path);
                    } else {
                        eprintln!("Warning: 'infinite_sphere' object with emission '{}' is not handled as a geometric object.", emission_path);
                    }
                } else {
                    eprintln!("Warning: 'infinite_sphere' object without emission is not handled.");
                }
            }
            ObjectConfigVariant::InfiniteSphereCap {
                transform: obj_transform,
                emission,
                power: _power,
                sample: _sample,
                cap_angle: _cap_angle,
            } => {
                let mut used_for_skybox = false;
                if let Some(emission_path) = emission {
                    if scene.skybox_hdr_image.is_some() || scene.skybox_image.is_some() {
                        let skybox_path_hdr = scene
                            .skybox_hdr_image
                            .as_ref()
                            .and_then(|_| Some(scene_dir.join(&emission_path)));
                        let skybox_path_ldr = scene
                            .skybox_image
                            .as_ref()
                            .and_then(|_| Some(scene_dir.join(&emission_path)));

                        if (skybox_path_hdr.is_some() && emission_path.ends_with(".hdr"))
                            || (skybox_path_ldr.is_some()
                                && (emission_path.ends_with(".png")
                                    || emission_path.ends_with(".jpg")
                                    || emission_path.ends_with(".jpeg")))
                        {
                            println!("InfiniteSphereCap with emission '{}' was likely used as skybox, not adding as separate object.", emission_path);
                            used_for_skybox = true;
                        }
                    }
                }

                if !used_for_skybox {
                    eprintln!("Warning: 'infinite_sphere_cap' object type is recognized but not yet implemented for rendering as a geometric object.");
                }
            }
        }
    }

    Ok((scene, camera, render_settings))
}

use crate::hittable::HitRecord;
use crate::ray::Ray;
use rand::RngCore;

#[derive(Debug, Clone, Default)]
pub struct Transform {
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Vec3,
}

fn parse_transform(_transform_conf: &Option<ObjectTransformConfig>) -> Transform {
    Transform::default()
}

#[derive(Deserialize, Debug, Clone)]
pub struct PrimitiveConfig {
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub type_str: String,
    pub transform: Option<ObjectTransformConfig>,
    pub bsdf: String,
    pub radius: Option<f32>,
}

fn parse_bsdfs(
    bsdfs_json: &serde_json::Value,
    parsed_bsdfs_map: &mut HashMap<String, Arc<dyn Material>>,
    scene_directory: &Path,
) -> Result<(), String> {
    if let Some(bsdfs_array) = bsdfs_json.as_array() {
        for bsdf_conf_val in bsdfs_array {
            let bsdf_conf: BsdfConfig = match serde_json::from_value(bsdf_conf_val.clone()) {
                Ok(bc) => bc,
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to parse BSDF config: {:?}. Error: {}",
                        bsdf_conf_val, e
                    );
                    continue;
                }
            };

            let material_name = bsdf_conf.name.clone();
            let material_type = bsdf_conf.type_str.as_str();

            let material_result: Result<Arc<dyn Material>, String> = match material_type {
                "lambert" => {
                    let albedo_json =
                        bsdf_conf
                            .albedo
                            .clone()
                            .unwrap_or(serde_json::Value::Number(
                                serde_json::Number::from_f64(0.5).unwrap(),
                            ));
                    match serde_json::from_value::<AlbedoConfig>(albedo_json.clone()) {
                        Ok(albedo_config) => {
                            if let AlbedoConfig::Checker(tex_conf) = albedo_config {
                                let scale = tex_conf.res_u.or(tex_conf.res_v).unwrap_or(10.0);
                                let checker_tex = Arc::new(material::CheckerTexture::new(
                                    tex_conf.on_color.into(),
                                    tex_conf.off_color.into(),
                                    scale,
                                ));
                                Ok(Arc::new(Lambertian::new_checker(checker_tex)))
                            } else {
                                Ok(Arc::new(material::Lambertian::new_solid(
                                    albedo_config
                                        .into_color_or_texture(scene_directory)
                                        .map_err(|e| format!("Lambertian albedo error: {}", e))?,
                                )))
                            }
                        }
                        Err(_) => {
                            let color_conf: ColorConfig =
                                serde_json::from_value(albedo_json).map_err(|e| e.to_string())?;
                            Ok(Arc::new(material::Lambertian::new_solid(color_conf.into())))
                        }
                    }
                }
                "metal" => {
                    let albedo_val = bsdf_conf
                        .albedo
                        .clone()
                        .unwrap_or(serde_json::json!([0.8, 0.8, 0.8]));
                    let color_conf: ColorConfig =
                        serde_json::from_value(albedo_val).map_err(|e| e.to_string())?;
                    let fuzz = bsdf_conf.roughness.unwrap_or(0.0);
                    Ok(Arc::new(material::Metal::new(color_conf.into(), fuzz)))
                }
                "dielectric" | "glass" => {
                    let ior = bsdf_conf.ior.unwrap_or(1.5);
                    Ok(Arc::new(material::Dielectric::new(ior)))
                }
                "rough_conductor" => {
                    let albedo_val = bsdf_conf
                        .albedo
                        .clone()
                        .unwrap_or(serde_json::json!([0.7, 0.7, 0.7]));
                    let albedo_color =
                        match serde_json::from_value::<ColorConfig>(albedo_val.clone()) {
                            Ok(cc) => cc.into(),
                            Err(_) => match serde_json::from_value::<f32>(albedo_val) {
                                Ok(f) => Color::new(f, f, f),
                                Err(e) => {
                                    return Err(format!(
                                        "Could not parse albedo for rough_conductor: {}",
                                        e
                                    ))
                                }
                            },
                        };
                    let roughness = bsdf_conf.roughness.unwrap_or(0.1);
                    let metal_str = bsdf_conf.material.as_deref().unwrap_or("cu").to_lowercase();
                    let metal_type = match metal_str.as_str() {
                        "cu" => material::MetalType::Cu,
                        "au" => material::MetalType::Au,
                        "ag" => material::MetalType::Ag,
                        "al" => material::MetalType::Al,
                        "ni" => material::MetalType::Ni,
                        "ti" => material::MetalType::Ti,
                        "fe" => material::MetalType::Fe,
                        "pb" => material::MetalType::Pb,
                        _ => {
                            eprintln!("Warning: Unknown metal type '{}' for rough_conductor, defaulting to Cu.", metal_str);
                            material::MetalType::Cu
                        }
                    };
                    let dist_str = bsdf_conf
                        .distribution
                        .as_deref()
                        .unwrap_or("ggx")
                        .to_lowercase();
                    let distribution = match dist_str.as_str() {
                        "ggx" => material::MicrofacetDistribution::GGX,
                        "beckmann" => material::MicrofacetDistribution::Beckmann,
                        _ => {
                            eprintln!("Warning: Unknown distribution '{}' for rough_conductor, defaulting to GGX.", dist_str);
                            material::MicrofacetDistribution::GGX
                        }
                    };
                    Ok(Arc::new(material::RoughConductor::new(
                        albedo_color,
                        roughness,
                        metal_type,
                        distribution,
                    )))
                }
                "null" => Ok(Arc::new(NullMaterial::new())),
                _ => {
                    let warning_msg = format!(
                        "Unsupported BSDF type '{}' for BSDF named '{}'.",
                        material_type, material_name
                    );
                    eprintln!("Warning: {}", warning_msg);
                    Err(warning_msg)
                }
            };

            match material_result {
                Ok(mat) => {
                    parsed_bsdfs_map.insert(material_name.clone(), mat);
                }
                Err(e) => {
                    eprintln!("Skipping BSDF '{}': {}", material_name, e);
                }
            }
        }
    } else {
        return Err("BSDFs section is not an array".to_string());
    }
    Ok(())
}

fn parse_primitives(
    primitives_json: &serde_json::Value,
    parsed_bsdfs_map: &HashMap<String, Arc<dyn Material>>,
    scene_directory: &Path,
    default_material: Arc<dyn Material>,
) -> Result<Vec<Arc<dyn Hittable>>, String> {
    let mut hittables: Vec<Arc<dyn Hittable>> = Vec::new();
    if let Some(primitives_array) = primitives_json.as_array() {
        for primitive_conf_val in primitives_array {
            let primitive_conf: PrimitiveConfig =
                match serde_json::from_value(primitive_conf_val.clone()) {
                    Ok(pc) => pc,
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to parse primitive config: {:?}. Error: {}",
                            primitive_conf_val, e
                        );
                        continue;
                    }
                };

            let transform = parse_transform(&primitive_conf.transform);
            let mut material_to_use: Option<Arc<dyn Material>> = None;

            if let Some(power_val) = primitive_conf_val.get("power") {
                let power_f32 = match power_val.as_f64() {
                    Some(p) => p as f32,
                    None => {
                        eprintln!("Warning: Could not parse 'power' as f64 for primitive {:?}, defaulting to 0.0.", primitive_conf_val.get("name"));
                        0.0
                    }
                };
                let intensity = power_f32;
                let emissive_color = Color::new(intensity, intensity, intensity);
                material_to_use = Some(Arc::new(EmissiveLight::new(emissive_color)));
            }

            if material_to_use.is_none() {
                material_to_use = match parsed_bsdfs_map.get(&primitive_conf.bsdf) {
                    Some(mat_ref) => Some(mat_ref.clone()),
                    None => {
                        eprintln!(
                            "Warning: BSDF '{}' not found for {} '{}'. Using default material.",
                            primitive_conf.bsdf,
                            primitive_conf.type_str,
                            primitive_conf.name.as_deref().unwrap_or("unnamed")
                        );
                        Some(default_material.clone())
                    }
                };
            }

            let final_material = material_to_use.unwrap_or_else(|| {
                 eprintln!("Critical Error: Material could not be resolved for primitive {:?}. Using default.", primitive_conf_val.get("name"));
                 default_material.clone()
            });

            match primitive_conf.type_str.as_str() {
                "sphere" => {
                    let radius = primitive_conf.radius.unwrap_or_else(|| {
                        if transform.scale.x == 1.0 && transform.scale.y == 1.0 && transform.scale.z == 1.0 {
                            eprintln!("Warning: Sphere {:?} has no explicit radius and no scale in transform. Defaulting to radius 0.5.", primitive_conf.name);
                            0.5
                        } else {
                            (transform.scale.x + transform.scale.y + transform.scale.z) / 3.0
                        }
                    });
                    let center = transform.position;
                    hittables.push(Arc::new(Sphere {
                        center,
                        radius,
                        material: final_material.clone(),
                    }));
                    println!(
                        "Parsed Sphere: name={:?}, center={:?}, radius={}, material_name={:?}",
                        primitive_conf.name, center, radius, primitive_conf.bsdf
                    );
                }
                "cube" => {
                    eprintln!(
                        "Cube parsing not fully implemented yet for {:?}.",
                        primitive_conf.name
                    );
                }
                "quad" => {
                    eprintln!(
                        "Quad parsing not fully implemented yet for {:?}.",
                        primitive_conf.name
                    );
                }
                unsupported_type => {
                    eprintln!("Warning: Unsupported primitive type '{}' for primitive named {:?}. Skipping.", unsupported_type, primitive_conf.name);
                }
            }
        }
    } else {
        return Err("Primitives section is not an array".to_string());
    }
    Ok(hittables)
}

#[derive(Deserialize, Debug)]
pub struct SceneConfig {
    pub sky: Option<SkyConfig>,
    pub camera: CameraConfig,
    #[serde(rename = "primitives")]
    pub objects: Vec<ObjectConfigVariant>,
    pub integrator: Option<IntegratorConfig>,
    pub renderer: Option<RendererConfig>,
    pub bsdfs: Option<Vec<BsdfConfig>>,
}
