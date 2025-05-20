// src/mitsuba_parser.rs

use quick_xml::events::{attributes::Attributes, BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;
use std::error::Error;
use std::str::{self, FromStr};
use std::sync::Arc;

use crate::camera::Camera;
use crate::color::Color;
use crate::hittable::HittableList;
use crate::hittable::{HitRecord, Hittable};
use crate::material::{EmissiveLight, Lambertian, Material};
use crate::objects::sphere::Sphere;
use crate::ray::Ray;
use crate::scene::Scene;
use crate::tungsten_parser::RenderSettings;
use crate::vec3::Vec3;
use glam::{Mat4, Vec3 as GlamVec3, Vec4};

// Helper to parse "value" from a generic named attribute tag, e.g. <integer name="foo" value="1024"/>
fn get_attribute_value(attributes: Attributes<'_>, target_attr_name: &str) -> Result<String, Box<dyn Error>> {
    for attr_result in attributes {
        let attr = attr_result?;
        if attr.key.as_ref() == target_attr_name.as_bytes() {
            return Ok(str::from_utf8(&attr.value)?.to_string());
        }
    }
    Err(format!("Missing '{}' attribute", target_attr_name).into())
}

// Helper to parse "value" from a <tag value="..."/>
fn parse_value_attr(e: &BytesStart) -> Result<String, Box<dyn Error>> {
    get_attribute_value(e.attributes(), "value")
}

// Helper to parse "name" from a <tag name="..."/>
fn parse_name_attr(e: &BytesStart) -> Result<String, Box<dyn Error>> {
    get_attribute_value(e.attributes(), "name")
}

fn parse_vec3_from_str(s: &str) -> Result<Vec3, Box<dyn Error>> {
    let parts: Vec<&str> = s.split(|c| c == ',' || c == ' ').filter(|s| !s.is_empty()).collect();
    if parts.len() != 3 {
        return Err(format!("Invalid Vec3 string: '{}', expected 3 components", s).into());
    }
    Ok(Vec3::new(
        parts[0].parse::<f32>()?,
        parts[1].parse::<f32>()?,
        parts[2].parse::<f32>()?,
    ))
}

fn parse_rgb_color(e: &BytesStart) -> Result<Color, Box<dyn Error>> {
    let value_str = parse_value_attr(e)?;
    let parts: Vec<&str> = value_str.split(|c| c == ',' || c == ' ').filter(|s| !s.is_empty()).collect();
    if parts.len() == 1 {
        let val = parts[0].parse::<f32>()?;
        Ok(Color::new(val, val, val))
    } else if parts.len() == 3 {
        Ok(Color::new(
            parts[0].parse::<f32>()?,
            parts[1].parse::<f32>()?,
            parts[2].parse::<f32>()?,
        ))
    } else {
        Err(format!("Invalid RGB/Color string: '{}'", value_str).into())
    }
}

fn parse_matrix_from_string(matrix_str: &str) -> Result<Mat4, Box<dyn Error>> {
    let numbers: Vec<f32> = matrix_str
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<f32>, _>>()?;

    if numbers.len() != 16 {
        return Err(format!(
            "Matrix string must contain 16 numbers, found {}",
            numbers.len()
        )
        .into());
    }
    // Mitsuba matrices are column-major. glam::Mat4::from_cols_array expects column-major.
    Ok(Mat4::from_cols_array(&numbers.try_into().unwrap()))
}

#[derive(Debug, Clone)]
struct ParsedBsdf {
    id: String,
    bsdf_type: String, // e.g. "diffuse", "twosided"
    nested_bsdf: Option<Box<ParsedBsdf>>, // For "twosided"
    diffuse_reflectance: Option<Color>,
    // Add other BSDF properties as needed
}

// Placeholder for state tracking during parsing
#[derive(Default)]
struct ParserState {
    current_transform_matrix: Option<Mat4>,
    is_parsing_sensor: bool,
    is_parsing_film: bool,
    is_parsing_sampler: bool,
    is_parsing_integrator: bool,
    is_parsing_bsdf: bool,
    is_parsing_shape: bool,
    is_parsing_emitter: bool,
    current_bsdf_id_for_shape: Option<String>,
    current_shape_id: Option<String>,
    current_bsdf_being_parsed: Option<ParsedBsdf>,
    current_emitter_radiance_for_shape: Option<Color>,
    // For nested BSDFs like <bsdf type="twosided" id="foo"><bsdf type="diffuse">...</bsdf></bsdf>
    bsdf_stack: Vec<ParsedBsdf>, 
}

pub fn load_scene_from_xml(xml_path: &str) -> Result<(Scene, Camera, RenderSettings), Box<dyn std::error::Error>> {
    println!("Attempting to load Mitsuba XML scene from: {}", xml_path);
    let xml_content = std::fs::read_to_string(xml_path)?;
    let mut reader = Reader::from_str(&xml_content);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut state = ParserState::default();

    // Default render settings, can be overridden by <default> or specific tags
    let mut spp: usize = 64;
    let mut res_x: usize = 200;
    let mut res_y: usize = 150;
    let mut max_depth: usize = 10;
    let mut defaults_map: HashMap<String, String> = HashMap::new();

    // Camera parameters
    let mut camera_fov = 35.0f32; // Default FoV
    let mut camera_to_world_matrix: Option<Mat4> = None;

    let mut scene = Scene::new();
    let mut bsdfs_map: HashMap<String, Arc<dyn Material>> = HashMap::new();
    // To handle BSDFs that might reference other BSDFs (e.g. masked, mixture - not implemented yet)
    let mut raw_bsdfs_map: HashMap<String, ParsedBsdf> = HashMap::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(event) => {
                match event {
                    Event::Start(ref e_start) => {
                        let tag_name_event = e_start.name();
                        let current_tag_name_str = str::from_utf8(tag_name_event.as_ref()).unwrap_or("UTF8_ERR_ON_TAG_NAME");
                        println!("DEBUG Event::Start for tag: <{}>", current_tag_name_str);

                        match current_tag_name_str {
                            "default" => {
                                if let (Ok(name), Ok(value_str)) = (parse_name_attr(e_start), parse_value_attr(e_start)) {
                                    defaults_map.insert(name.clone(), value_str.clone());
                                    if !value_str.starts_with('$') {
                                        match name.as_str() {
                                            "spp" => spp = value_str.parse().unwrap_or(spp),
                                            "resx" => res_x = value_str.parse().unwrap_or(res_x),
                                            "resy" => res_y = value_str.parse().unwrap_or(res_y),
                                            "max_depth" => max_depth = value_str.parse().unwrap_or(max_depth),
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            "integrator" => {
                                println!("DEBUG: Entering integrator state");
                                state.is_parsing_integrator = true;
                            }
                            "sensor" => {
                                println!("DEBUG: Entering sensor state");
                                state.is_parsing_sensor = true;
                            }
                            "film" => {
                                println!("DEBUG: Entering film state");
                                state.is_parsing_film = true;
                            }
                            "sampler" => {
                                println!("DEBUG: Entering sampler state");
                                state.is_parsing_sampler = true;
                            }
                            "transform" => {
                                println!("DEBUG: Transform encountered. Name: {:?}, is_parsing_sensor: {}", parse_name_attr(e_start).ok(), state.is_parsing_sensor);
                                state.current_transform_matrix = None; 
                            }
                            "matrix" => {
                                println!("DEBUG: Matrix encountered. is_parsing_sensor: {}", state.is_parsing_sensor);
                                if let Ok(matrix_str) = parse_value_attr(e_start) {
                                    match parse_matrix_from_string(&matrix_str) {
                                        Ok(mat) => {
                                            state.current_transform_matrix = Some(mat);
                                            if state.is_parsing_sensor {
                                                println!("DEBUG: Assigning matrix to camera_to_world_matrix");
                                                camera_to_world_matrix = Some(mat);
                                            }
                                        }
                                        Err(err) => eprintln!("Error parsing matrix: {}", err),
                                    }
                                }
                            }
                            "bsdf" => {
                                state.is_parsing_bsdf = true;
                                let id = get_attribute_value(e_start.attributes(), "id").ok();
                                let bsdf_type = get_attribute_value(e_start.attributes(), "type")?;
                                
                                let new_bsdf = ParsedBsdf {
                                    id: id.unwrap_or_else(|| format!("bsdf_{}", bsdfs_map.len() + raw_bsdfs_map.len())), 
                                    bsdf_type,
                                    nested_bsdf: None,
                                    diffuse_reflectance: None,
                                };
                                state.bsdf_stack.push(new_bsdf);
                            }
                            "shape" => {
                                state.is_parsing_shape = true;
                                state.current_transform_matrix = None; 
                                state.current_bsdf_id_for_shape = None;
                                state.current_emitter_radiance_for_shape = None;
                                state.current_shape_id = get_attribute_value(e_start.attributes(), "id").ok();
                            }
                            "emitter" => {
                                state.is_parsing_emitter = true;
                            }
                            "ref" => { 
                                if state.is_parsing_shape {
                                    if let Ok(id_val) = get_attribute_value(e_start.attributes(), "id") {
                                        state.current_bsdf_id_for_shape = Some(id_val);
                                    }
                                }
                            }
                            "integer" | "float" | "string" | "boolean" | "rgb" | "srgb" => {
                                if let Ok(name_attr) = parse_name_attr(e_start) { 
                                    let value_str_resolved = match parse_value_attr(e_start) {
                                        Ok(val_str) => {
                                            if val_str.starts_with('$') {
                                                let var_name = &val_str[1..];
                                                defaults_map.get(var_name).cloned().unwrap_or(val_str)
                                            } else {
                                                val_str
                                            }
                                        }
                                        Err(_) => continue, 
                                    };

                                    println!("DEBUG: GenericPropParse: tag=\"{}\", name=\"{}\", value_resolved=\"{}\", is_film={}, is_sensor={}, is_sampler={}, is_integrator={}",
                                        current_tag_name_str, 
                                        name_attr, 
                                        value_str_resolved,
                                        state.is_parsing_film,
                                        state.is_parsing_sensor,
                                        state.is_parsing_sampler,
                                        state.is_parsing_integrator
                                    );

                                    match name_attr.as_str() {
                                        "max_depth" if state.is_parsing_integrator => {
                                            println!("DEBUG: Setting max_depth from integrator: {}", value_str_resolved);
                                            max_depth = value_str_resolved.parse().unwrap_or(max_depth);
                                        }
                                        "width" if state.is_parsing_film => {
                                            println!("DEBUG: Setting width from film: {}", value_str_resolved);
                                            res_x = value_str_resolved.parse().unwrap_or(res_x);
                                        }
                                        "height" if state.is_parsing_film => {
                                            println!("DEBUG: Setting height from film: {}", value_str_resolved);
                                            res_y = value_str_resolved.parse().unwrap_or(res_y);
                                        }
                                        "fov" if state.is_parsing_sensor => {
                                            println!("DEBUG: Setting fov from sensor: {}", value_str_resolved);
                                            camera_fov = value_str_resolved.parse().unwrap_or(camera_fov);
                                        }
                                        "sample_count" if state.is_parsing_sampler => {
                                            println!("DEBUG: Setting spp from sampler (sample_count): {}", value_str_resolved);
                                            spp = value_str_resolved.parse().unwrap_or(spp);
                                        }
                                        "radius" if state.is_parsing_shape && current_tag_name_str == "float" => {}
                                        "reflectance" if (current_tag_name_str == "rgb" || current_tag_name_str == "srgb") && !state.bsdf_stack.is_empty() => {
                                            if let Some(current_bsdf) = state.bsdf_stack.last_mut() {
                                                if current_bsdf.bsdf_type == "diffuse" { 
                                                     if let Ok(color) = parse_rgb_color(e_start) {
                                                        current_bsdf.diffuse_reflectance = Some(color);
                                                     }
                                                }
                                            }
                                        }
                                        "radiance" if (current_tag_name_str == "rgb" || current_tag_name_str == "srgb") && state.is_parsing_emitter => {
                                            if let Ok(color) = parse_rgb_color(e_start) {
                                                state.current_emitter_radiance_for_shape = Some(color);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            _ => {} 
                        }
                    }
                    Event::End(ref e_end) => {
                        let tag_name_event = e_end.name();
                        let current_tag_name_str = str::from_utf8(tag_name_event.as_ref()).unwrap_or("UTF8_ERR_ON_TAG_NAME");
                        println!("DEBUG Event::End for tag: </{}>", current_tag_name_str);
                        match current_tag_name_str {
                            "integrator" => { println!("DEBUG: Exiting integrator state"); state.is_parsing_integrator = false; }
                            "sensor" => { println!("DEBUG: Exiting sensor state"); state.is_parsing_sensor = false; }
                            "film" => { println!("DEBUG: Exiting film state"); state.is_parsing_film = false; }
                            "sampler" => { println!("DEBUG: Exiting sampler state"); state.is_parsing_sampler = false; }
                            "bsdf" => {
                                if let Some(mut parsed_bsdf) = state.bsdf_stack.pop() {
                                    if parsed_bsdf.bsdf_type == "twosided" {
                                        if let Some(inner_bsdf_box) = parsed_bsdf.nested_bsdf.take() {
                                            let inner_bsdf = *inner_bsdf_box;
                                            if inner_bsdf.bsdf_type == "diffuse" {
                                                parsed_bsdf.diffuse_reflectance = inner_bsdf.diffuse_reflectance;
                                                parsed_bsdf.bsdf_type = "diffuse".to_string();
                                            }
                                        }
                                         if parsed_bsdf.bsdf_type == "diffuse" {
                                            if let Some(color) = parsed_bsdf.diffuse_reflectance {
                                                bsdfs_map.insert(parsed_bsdf.id.clone(), Arc::new(Lambertian::new(color)));
                                                raw_bsdfs_map.insert(parsed_bsdf.id.clone(), parsed_bsdf);
                                            }
                                         } else {
                                            raw_bsdfs_map.insert(parsed_bsdf.id.clone(), parsed_bsdf);
                                         }

                                    } else { 
                                        if parsed_bsdf.bsdf_type == "diffuse" {
                                            if let Some(color) = parsed_bsdf.diffuse_reflectance {
                                                 bsdfs_map.insert(parsed_bsdf.id.clone(), Arc::new(Lambertian::new(color)));
                                                 raw_bsdfs_map.insert(parsed_bsdf.id.clone(), parsed_bsdf);
                                            }
                                        } else {
                                            if let Some(parent_bsdf) = state.bsdf_stack.last_mut() {
                                                parent_bsdf.nested_bsdf = Some(Box::new(parsed_bsdf));
                                            } else {
                                                raw_bsdfs_map.insert(parsed_bsdf.id.clone(), parsed_bsdf);
                                            }
                                        }
                                    }
                                }
                                if state.bsdf_stack.is_empty() {
                                    state.is_parsing_bsdf = false; 
                                }
                            }
                            "shape" => {
                                state.is_parsing_shape = false;
                                state.current_shape_id = None;
                            }
                            "emitter" => state.is_parsing_emitter = false,
                            _ => {}
                        }
                    }
                    Event::Text(e_text) => { 
                        if let Ok(text) = e_text.unescape() {
                            if !text.trim().is_empty() {
                                println!("DEBUG Event::Text: \"{}\"", text);
                            }
                        }
                    }
                    Event::Eof => {println!("DEBUG Event::Eof"); break; }
                    Event::Comment(_) => {println!("DEBUG Event::Comment");}
                    Event::CData(_) => {println!("DEBUG Event::CData");}
                    Event::Decl(_) => {println!("DEBUG Event::Decl");}
                    Event::PI(_) => {println!("DEBUG Event::PI");}
                    Event::DocType(_) => {println!("DEBUG Event::DocType");}
                    Event::Empty(_) => {println!("DEBUG Event::Empty (self-closing tag)");}
                }
            }
            Err(e) => {
                eprintln!(
                    "Error parsing Mitsuba XML at position {}: {:?}",
                    reader.buffer_position(),
                    e
                );
                return Err(Box::new(e));
            }
        }
        buf.clear();
    }

    // Construct Camera from parsed matrix
    let final_camera = if let Some(matrix) = camera_to_world_matrix {
        let eye = Vec3::new(matrix.w_axis.x, matrix.w_axis.y, matrix.w_axis.z);
        // Assuming Z is forward in camera space, so -Z is view direction in world space
        let view_dir_world = GlamVec3::new(matrix.z_axis.x, matrix.z_axis.y, matrix.z_axis.z).normalize_or_zero() * -1.0;
        let target = eye + Vec3::new(view_dir_world.x, view_dir_world.y, view_dir_world.z);
        let up_world = GlamVec3::new(matrix.y_axis.x, matrix.y_axis.y, matrix.y_axis.z).normalize_or_zero();
        
        Camera::new(
            eye,
            target,
            Vec3::new(up_world.x, up_world.y, up_world.z),
            camera_fov,
            res_x as f32 / res_y as f32,
        )
    } else {
        // Fallback if no matrix (should not happen for the provided Cornell Box XML)
        eprintln!("Warning: Camera matrix not found in XML. Using default camera.");
        Camera::new(
            Vec3::new(0.0,1.0,6.8), // Default position similar to Tungsten CBox
            Vec3::new(0.0, 0.0, 0.0), // Replaced Vec3::ZERO
            Vec3::new(0.0, 1.0, 0.0), // Replaced Vec3::Y
            camera_fov,
            res_x as f32 / res_y as f32
        )
    };

    let render_settings = RenderSettings {
        width: res_x,
        height: res_y,
        samples_per_pixel: spp,
        max_depth,
    };
    
    println!("Mitsuba XML parsing (partially) done. Render settings: {:?}, Camera eye: {:?}", render_settings, final_camera.position); // Changed origin to position
    // For now, scene object list will be empty. Shapes parsing is next.

    Ok((scene, final_camera, render_settings))
} 