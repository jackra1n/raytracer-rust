// src/mitsuba_parser.rs

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;
use std::error::Error;
use std::str::FromStr;
use std::sync::Arc;

use crate::camera::Camera;
use crate::color::Color;
use crate::hittable::HittableList;
use crate::hittable::{HitRecord, Hittable};
use crate::material::{Lambertian, Material};
use crate::objects::sphere::Sphere;
use crate::ray::Ray;
use crate::scene::Scene;
use crate::tungsten_parser::RenderSettings;
use crate::vec3::Vec3;

// Helper to parse "value" from a specific tag, e.g. <integer value="1024"/>
fn parse_value_attr(e: &BytesStart) -> Result<String, Box<dyn Error>> {
    for attr in e.attributes() {
        let attr = attr?;
        if attr.key.as_ref() == b"value" {
            return Ok(String::from_utf8(attr.value.to_vec())?);
        }
    }
    Err("Missing 'value' attribute".into())
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

#[derive(Debug, Clone)]
struct MitsubaTransform {
    lookat_origin: Option<Vec3>,
    lookat_target: Option<Vec3>,
    lookat_up: Option<Vec3>,
    translate: Option<Vec3>,
    scale: Option<Vec3>,
}

impl MitsubaTransform {
    fn new() -> Self {
        MitsubaTransform {
            lookat_origin: None,
            lookat_target: None,
            lookat_up: None,
            translate: None,
            scale: None,
        }
    }
}

#[derive(Debug, Clone)]
struct ParsedBsdf {
    id: String,
    bsdf_type: String,
    diffuse_reflectance: Option<Color>,
}

#[derive(Debug, Clone)]
struct MitsubaShape {
    shape_type: String,
    transform: MitsubaTransform,
    bsdf_ref: Option<String>,
    emitter_ref: Option<String>,
    center: Option<Vec3>,
    radius: Option<f64>,
    filename: Option<String>,
}

pub fn load_scene_from_xml(xml_path: &str) -> Result<(Scene, Camera, RenderSettings), Box<dyn std::error::Error>> {
    println!("Attempting to load Mitsuba XML scene from: {}", xml_path);
    let xml_content = std::fs::read_to_string(xml_path)?;
    let mut reader = Reader::from_str(&xml_content);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();

    let mut camera_lookat_origin = Vec3::new(0.0, 0.0, 3.0);
    let mut camera_lookat_target = Vec3::new(0.0, 0.0, 0.0);
    let mut camera_lookat_up = Vec3::new(0.0, 1.0, 0.0);
    let mut camera_fov = 45.0;
    let mut image_width: u32 = 200;
    let mut image_height: u32 = 150;
    let mut samples_per_pixel: u32 = 10;
    let mut max_depth: i32 = 5;

    let mut scene = Scene::new();
    let mut bsdfs_map: HashMap<String, Arc<dyn Material>> = HashMap::new();
    
    let mut current_transform = MitsubaTransform::new();
    let mut current_bsdf_id_ref_for_shape: Option<String> = None;
    let mut current_parsing_bsdf: Option<ParsedBsdf> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"integrator" => {
                        // Assuming path tracer, parse maxDepth
                    }
                    b"sensor" => {
                        // Mark that we are inside a sensor block to parse its children (film, transform)
                        // If type is "perspective", then we look for fov.
                    }
                    b"film" => {
                        // Parse film properties like width, height, pixelformat
                    }
                    b"transform" => {
                        current_transform = MitsubaTransform::new();
                        if e.attributes().any(|a| a.map_or(false, |attr| attr.key.as_ref() == b"name" && attr.value.as_ref() == b"to_world")) {
                            // Further parsing if needed, e.g. for a matrix
                        }
                    }
                    b"lookat" => {
                        let mut origin = None;
                        let mut target = None;
                        let mut up = None;
                        for attr in e.attributes() {
                            let attr = attr?;
                            match attr.key.as_ref() {
                                b"origin" => origin = Some(parse_vec3_from_str(&String::from_utf8(attr.value.to_vec())?)?),
                                b"target" => target = Some(parse_vec3_from_str(&String::from_utf8(attr.value.to_vec())?)?),
                                b"up" => up = Some(parse_vec3_from_str(&String::from_utf8(attr.value.to_vec())?)?),
                                _ => {}
                            }
                        }
                        if let (Some(o), Some(t), Some(u)) = (origin, target, up) {
                            camera_lookat_origin = o;
                            camera_lookat_target = t;
                            camera_lookat_up = u;
                            current_transform.lookat_origin = Some(o);
                            current_transform.lookat_target = Some(t);
                            current_transform.lookat_up = Some(u);
                        }
                    }
                    b"shape" => {
                        current_transform = MitsubaTransform::new();
                        current_bsdf_id_ref_for_shape = None;
                        
                        let shape_type_attr = e
                            .attributes()
                            .find(|a| a.as_ref().map_or(false, |attr| attr.key.as_ref() == b"type"));
                        
                        if let Some(Ok(attr)) = shape_type_attr {
                            let shape_type = String::from_utf8(attr.value.to_vec())?;
                            if shape_type == "sphere" {
                                // Sphere specific parsing will happen with its child properties
                                // (e.g. <float name="radius" value="1"/>)
                                // and BSDF reference (<ref id="mat_id"/>)
                            }
                        }
                    }
                    b"bsdf" => {
                        let mut id_attr = None;
                        let mut type_attr = None;
                        for attr in e.attributes() {
                            let attr = attr?;
                            match attr.key.as_ref() {
                                b"id" => id_attr = Some(String::from_utf8(attr.value.to_vec())?),
                                b"type" => type_attr = Some(String::from_utf8(attr.value.to_vec())?),
                                _ => {}
                            }
                        }
                        if let (Some(id), Some(bsdf_type)) = (id_attr, type_attr) {
                            current_parsing_bsdf = Some(ParsedBsdf {
                                id,
                                bsdf_type,
                                diffuse_reflectance: None,
                            });
                        } else {
                            // eprint error or handle missing id/type for BSDF
                        }
                    }
                     b"ref" => {
                        for attr in e.attributes() {
                            let attr = attr?;
                            if attr.key.as_ref() == b"id" {
                                current_bsdf_id_ref_for_shape = Some(String::from_utf8(attr.value.to_vec())?);
                            }
                        }
                    }
                    b"integer" | b"float" | b"string" | b"boolean" | b"rgb" | b"srgb" => {
                        let current_tag_is_rgb_or_srgb = e.name().as_ref() == b"rgb" || e.name().as_ref() == b"srgb";
                        
                        let mut name_attr_val_opt: Option<String> = None;
                        for attr_res in e.attributes() { // Iterate to find the 'name' attribute's value
                            if let Ok(attr) = attr_res {
                                if attr.key.as_ref() == b"name" {
                                    name_attr_val_opt = String::from_utf8(attr.value.to_vec()).ok();
                                    break; 
                                }
                            }
                        }

                        if let Some(name_attr_val_str) = name_attr_val_opt {
                            match name_attr_val_str.as_str() {
                                "max_depth" => {
                                    if let Ok(val_str) = parse_value_attr(e) {
                                        if let Ok(val) = val_str.parse::<i32>() { max_depth = val; }
                                    }
                                }
                                "width" => {
                                     if let Ok(val_str) = parse_value_attr(e) {
                                        if let Ok(val) = val_str.parse::<u32>() { image_width = val; }
                                    }
                                }
                                "height" => {
                                     if let Ok(val_str) = parse_value_attr(e) {
                                        if let Ok(val) = val_str.parse::<u32>() { image_height = val; }
                                    }
                                }
                                "fov" => {
                                     if let Ok(val_str) = parse_value_attr(e) {
                                        if let Ok(val) = val_str.parse::<f64>() { camera_fov = val; }
                                    }
                                }
                                "samples_per_pixel" | "sample_count" => {
                                    if let Ok(val_str) = parse_value_attr(e) {
                                        if let Ok(val) = val_str.parse::<u32>() { samples_per_pixel = val; }
                                    }
                                }
                                "radius" => { 
                                    if let Ok(val_str) = parse_value_attr(e) {
                                        if let Ok(radius_val) = val_str.parse::<f64>() {
                                            if let Some(bsdf_id_to_use) = &current_bsdf_id_ref_for_shape {
                                                if let Some(material) = bsdfs_map.get(bsdf_id_to_use) {
                                                    let center = current_transform.translate.unwrap_or(Vec3::new(0.0,0.0,0.0));
                                                    scene.object_list.add(Box::new(Sphere { center, radius: radius_val as f32, material: Arc::clone(material) }));
                                                } else {
                                                   // eprintln!("Warning: BSDF id '{}' not found for sphere.", bsdf_id_to_use);
                                                }
                                            } else {
                                               // eprintln!("Warning: No BSDF id specified for sphere with radius {}.", radius_val);
                                            }
                                        }
                                    }
                                }
                                "reflectance" if current_tag_is_rgb_or_srgb => {
                                    if let Some(parsing_bsdf) = &mut current_parsing_bsdf {
                                        if parsing_bsdf.bsdf_type == "diffuse" { 
                                            if let Ok(color_val) = parse_rgb_color(e) {
                                                parsing_bsdf.diffuse_reflectance = Some(color_val);
                                            }
                                        }
                                    }
                                }
                                "translate" => { // Assuming <vector name="translate" value="x,y,z"/> or similar
                                     if e.name().as_ref() == b"vector" { // Check if the tag itself is <vector>
                                        if let Ok(val_str) = parse_value_attr(e) {
                                            if let Ok(trans_vec) = parse_vec3_from_str(&val_str) {
                                                current_transform.translate = Some(trans_vec);
                                            }
                                        }
                                     }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => (),
                }
            }
            Ok(Event::End(ref e)) => {
                 match e.name().as_ref() {
                    b"bsdf" => {
                        if let Some(parsed_bsdf) = current_parsing_bsdf.take() {
                            if parsed_bsdf.bsdf_type == "diffuse" {
                                if let Some(color) = parsed_bsdf.diffuse_reflectance {
                                    bsdfs_map.insert(parsed_bsdf.id, Arc::new(Lambertian::new(color)));
                                    // println!("Finalized and stored BSDF: ID={}, Color={:?}", parsed_bsdf.id, color);
                                } else {
                                    // eprintln!("Warning: Diffuse BSDF '{}' ended without reflectance.", parsed_bsdf.id);
                                }
                            }
                        }
                        current_parsing_bsdf = None;
                    }
                    b"shape" => {
                        current_transform = MitsubaTransform::new();
                        current_bsdf_id_ref_for_shape = None;
                    }
                    _ => {}
                 }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                eprintln!("Error parsing Mitsuba XML at position {}: {:?}", reader.buffer_position(), e);
                return Err(Box::new(e));
            }
            _ => (),
        }
        buf.clear();
    }
    
    let camera_aspect_ratio = image_width as f64 / image_height as f64;
    let camera = Camera::new(
        camera_lookat_origin,
        camera_lookat_target,
        camera_lookat_up,
        camera_fov as f32,
        camera_aspect_ratio as f32,
    );
    let render_settings = RenderSettings {
        width: image_width as usize,
        height: image_height as usize,
        samples_per_pixel: samples_per_pixel as usize,
        max_depth: max_depth as usize,
    };

    Ok((scene, camera, render_settings))
} 