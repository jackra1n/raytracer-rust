#![allow(dead_code)]
#![allow(unused_imports)]
use crate::acceleration::BVHNode;
use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::mesh::triangle::Triangle;
use crate::ray::Ray;
use crate::renderer::EPSILON;
use crate::vec3::Vec3 as CrateVec3;
use byteorder::{LittleEndian, ReadBytesExt};
use glam::{Mat4, Vec3 as GlamVec3, Vec4};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

pub struct Mesh {
    pub triangles: Vec<Triangle>,
    pub bvh: BVHNode,
    object_to_world: Mat4,
    world_to_object: Mat4,
}

impl Mesh {
    fn new_internal(
        triangles: Vec<Triangle>,
        object_to_world: Mat4,
        path_for_debug: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if triangles.is_empty() {
            return Err(format!(
                "No valid, non-degenerate triangles loaded for mesh '{}'",
                path_for_debug
            )
            .into());
        }

        println!(
            "Building BVH for {} object-space triangles from '{}'...",
            triangles.len(),
            path_for_debug
        );
        let start_time = std::time::Instant::now();
        let mut indices: Vec<usize> = (0..triangles.len()).collect();
        let bvh = BVHNode::new(&triangles, &mut indices, 0);
        let build_time = start_time.elapsed().as_millis();
        println!("BVH built in {}ms for '{}'", build_time, path_for_debug);

        let world_to_object = object_to_world.inverse();

        Ok(Mesh {
            triangles,
            bvh,
            object_to_world,
            world_to_object,
        })
    }

    pub fn from_obj(
        path: &str,
        material: Arc<dyn Material>,
        object_to_world: Mat4,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let obj_file = tobj::load_obj(Path::new(path), &tobj::GPU_LOAD_OPTIONS)?;

        let (models, _) = obj_file;
        if models.is_empty() {
            return Err(format!("No models found in OBJ file: {}", path).into());
        }
        let mut triangles_obj_space = Vec::new();

        for model in models {
            let mesh = model.mesh;
            if mesh.indices.is_empty() || mesh.positions.is_empty() {
                println!(
                    "Warning: Model '{}' in '{}' has no indices or positions. Skipping.",
                    model.name, path
                );
                continue;
            }
            if mesh.positions.len() % 3 != 0 {
                return Err(format!(
                    "Invalid position data length in model '{}' in '{}'",
                    model.name, path
                )
                .into());
            }

            let vertices: Vec<CrateVec3> = (0..mesh.positions.len() / 3)
                .map(|i| {
                    let idx = i * 3;
                    CrateVec3::new(
                        mesh.positions[idx],
                        mesh.positions[idx + 1],
                        mesh.positions[idx + 2],
                    )
                })
                .collect();

            if mesh.indices.len() % 3 != 0 {
                return Err(format!(
                    "Invalid index data length in model '{}' in '{}'",
                    model.name, path
                )
                .into());
            }

            for i in 0..mesh.indices.len() / 3 {
                let idx = i * 3;
                let v0_idx = mesh.indices[idx] as usize;
                let v1_idx = mesh.indices[idx + 1] as usize;
                let v2_idx = mesh.indices[idx + 2] as usize;

                if v0_idx >= vertices.len() || v1_idx >= vertices.len() || v2_idx >= vertices.len()
                {
                    eprintln!("Warning: Vertex index out of bounds (max={}) in OBJ file '{}'. Indices: ({}, {}, {}). Skipping triangle.",
                        vertices.len().saturating_sub(1), path, v0_idx, v1_idx, v2_idx);
                    continue;
                }

                let triangle = Triangle::new(
                    vertices[v0_idx],
                    vertices[v1_idx],
                    vertices[v2_idx],
                    material.clone(),
                );

                if (triangle.v1 - triangle.v0)
                    .cross(triangle.v2 - triangle.v0)
                    .length_squared()
                    < EPSILON * EPSILON
                {
                    continue;
                }
                triangles_obj_space.push(triangle);
            }
        }
        Self::new_internal(triangles_obj_space, object_to_world, path)
    }

    pub fn from_wo3(
        path: &str,
        material: Arc<dyn Material>,
        object_to_world: Mat4,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[WO3_LOADER] Attempting to load .wo3 file: {}", path);
        let file = File::open(Path::new(path))?;
        let mut reader = BufReader::new(file);

        let num_verts = reader.read_u64::<LittleEndian>()?;
        println!("[WO3_LOADER] Header: num_verts = {}", num_verts);
        let mut vertices_obj_space = Vec::with_capacity(num_verts as usize);

        println!("[WO3_LOADER] Reading {} vertices...", num_verts);
        for i in 0..num_verts {
            let px = reader.read_f32::<LittleEndian>()?;
            let py = reader.read_f32::<LittleEndian>()?;
            let pz = reader.read_f32::<LittleEndian>()?;
            vertices_obj_space.push(CrateVec3::new(px, py, pz));

            reader.read_f32::<LittleEndian>()?; // nx
            reader.read_f32::<LittleEndian>()?; // ny
            reader.read_f32::<LittleEndian>()?; // nz
            reader.read_f32::<LittleEndian>()?; // u
            reader.read_f32::<LittleEndian>()?; // v

            if i < 5 {
                println!("[WO3_LOADER] Vertex {}: ({}, {}, {})", i, px, py, pz);
            }
        }
        if num_verts == 0 {
            println!("[WO3_LOADER] Warning: No vertices found in .wo3 file header.");
        } else {
            println!(
                "[WO3_LOADER] Finished reading {} vertices.",
                vertices_obj_space.len()
            );
        }

        let num_tris = reader.read_u64::<LittleEndian>()?;
        println!("[WO3_LOADER] Header: num_tris = {}", num_tris);
        let mut triangles_obj_space = Vec::with_capacity(num_tris as usize);

        println!("[WO3_LOADER] Reading {} triangles...", num_tris);
        let mut valid_triangles_count = 0;
        let mut degenerate_triangles_count = 0;
        let mut out_of_bounds_indices_count = 0;

        for i in 0..num_tris {
            let v0_idx = reader.read_u32::<LittleEndian>()? as usize;
            let v1_idx = reader.read_u32::<LittleEndian>()? as usize;
            let v2_idx = reader.read_u32::<LittleEndian>()? as usize;

            if i < 5 {
                println!(
                    "[WO3_LOADER] Triangle indices {}: ({}, {}, {})",
                    i, v0_idx, v1_idx, v2_idx
                );
            }

            // Original logic: Skip if any index is out of bounds
            if v0_idx >= vertices_obj_space.len()
                || v1_idx >= vertices_obj_space.len()
                || v2_idx >= vertices_obj_space.len()
            {
                out_of_bounds_indices_count += 1;
                if out_of_bounds_indices_count <= 5 {
                    eprintln!(
                        "[WO3_LOADER] Warning: Vertex index out of bounds (max={}) in .wo3 file '{}'. Indices: ({}, {}, {}). Skipping triangle.",
                        vertices_obj_space.len().saturating_sub(1), path, v0_idx, v1_idx, v2_idx
                    );
                }
                continue;
            }

            let triangle = Triangle::new(
                vertices_obj_space[v0_idx],
                vertices_obj_space[v1_idx],
                vertices_obj_space[v2_idx],
                material.clone(),
            );

            if (triangle.v1 - triangle.v0)
                .cross(triangle.v2 - triangle.v0)
                .length_squared()
                < EPSILON * EPSILON
            {
                degenerate_triangles_count += 1;
                if degenerate_triangles_count <= 5 {
                    eprintln!("[WO3_LOADER] Warning: Degenerate triangle {} with indices ({}, {}, {}). Vertices: {:?}, {:?}, {:?}. Skipping.", 
                        i, v0_idx, v1_idx, v2_idx,
                        vertices_obj_space[v0_idx], vertices_obj_space[v1_idx], vertices_obj_space[v2_idx]);
                }
                continue;
            }
            triangles_obj_space.push(triangle);
            valid_triangles_count += 1;
        }

        println!(
            "[WO3_LOADER] Finished reading triangles. Valid: {}, Degenerate: {}, Index OOB: {}",
            valid_triangles_count, degenerate_triangles_count, out_of_bounds_indices_count
        );

        if triangles_obj_space.is_empty() {
            eprintln!(
                "[WO3_LOADER] Error: No valid triangles were loaded from .wo3 file '{}'.",
                path
            );
            return Err(format!("[WO3_LOADER] No valid triangles in {}", path).into());
        }

        println!(
            "[WO3_LOADER] Calling Self::new_internal for '{}' with {} triangles.",
            path,
            triangles_obj_space.len()
        );
        Self::new_internal(triangles_obj_space, object_to_world, path)
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray_world: &Ray, t_min_world: f32, t_max_world: f32) -> Option<HitRecord> {
        let ray_origin_obj_h: Vec4 = self.world_to_object
            * Vec4::new(
                ray_world.origin.x,
                ray_world.origin.y,
                ray_world.origin.z,
                1.0,
            );
        let ray_direction_obj_h: Vec4 = self.world_to_object
            * Vec4::new(
                ray_world.direction.x,
                ray_world.direction.y,
                ray_world.direction.z,
                0.0,
            );

        let ray_origin_obj =
            CrateVec3::new(ray_origin_obj_h.x, ray_origin_obj_h.y, ray_origin_obj_h.z);
        let ray_direction_obj = CrateVec3::new(
            ray_direction_obj_h.x,
            ray_direction_obj_h.y,
            ray_direction_obj_h.z,
        );

        let ray_obj = Ray::new(ray_origin_obj, ray_direction_obj.normalized());

        if let Some(mut hit_rec_obj) =
            self.bvh
                .intersect_recursive(&ray_obj, &self.triangles, t_min_world, t_max_world)
        {
            let pos_world_h: Vec4 = self.object_to_world
                * Vec4::new(
                    hit_rec_obj.position.x,
                    hit_rec_obj.position.y,
                    hit_rec_obj.position.z,
                    1.0,
                );
            let normal_world_h: Vec4 = self.world_to_object.transpose()
                * Vec4::new(
                    hit_rec_obj.normal.x,
                    hit_rec_obj.normal.y,
                    hit_rec_obj.normal.z,
                    0.0,
                );

            let pos_world = CrateVec3::new(pos_world_h.x, pos_world_h.y, pos_world_h.z);
            let normal_world =
                CrateVec3::new(normal_world_h.x, normal_world_h.y, normal_world_h.z).normalized();

            let ray_dir_obj_length = ray_direction_obj.length();
            let ray_dir_world_length = ray_world.direction.length();
            let t_world = hit_rec_obj.t * ray_dir_obj_length / ray_dir_world_length;

            if t_world < t_min_world || t_world > t_max_world {
                return None;
            }

            hit_rec_obj.position = pos_world;
            hit_rec_obj.normal = normal_world;
            hit_rec_obj.t = t_world;
            hit_rec_obj.set_face_normal(ray_world, normal_world);

            Some(hit_rec_obj)
        } else {
            None
        }
    }
}
