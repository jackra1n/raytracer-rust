use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::mesh::bvh::BVHNode;
use crate::mesh::triangle::Triangle;
use crate::ray::Ray;
use crate::renderer::EPSILON;
use crate::vec3::Vec3;
use std::sync::Arc;

use std::path::Path;

pub struct Mesh {
    pub triangles: Vec<Triangle>,
    pub bvh: BVHNode,
}

impl Mesh {
    pub fn from_obj(
        path: &str,
        material: Arc<dyn Material>,
        scale: f32,
        offset: Vec3,
        rotation_y: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let obj_file = tobj::load_obj(Path::new(path), &tobj::GPU_LOAD_OPTIONS)?;

        let (models, _) = obj_file;
        if models.is_empty() {
            return Err(format!("No models found in OBJ file: {}", path).into());
        }
        let mut triangles = Vec::new();

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

            let vertices: Vec<Vec3> = (0..mesh.positions.len() / 3)
                .map(|i| {
                    let idx = i * 3;
                    let unrotated = Vec3::new(
                        mesh.positions[idx] * scale,
                        mesh.positions[idx + 1] * scale,
                        mesh.positions[idx + 2] * scale,
                    );

                    let rotated = unrotated.rotate_around_y(rotation_y);

                    Vec3::new(
                        rotated.x + offset.x,
                        rotated.y + offset.y,
                        rotated.z + offset.z,
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
                        vertices.len() - 1, path, v0_idx, v1_idx, v2_idx);
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

                triangles.push(triangle);
            }
        }

        if triangles.is_empty() {
            return Err(format!(
                "No valid, non-degenerate triangles loaded from OBJ file '{}'",
                path
            )
            .into());
        }

        println!(
            "Building BVH for {} triangles from '{}'...",
            triangles.len(),
            path
        );
        let start_time = std::time::Instant::now();
        let mut indices: Vec<usize> = (0..triangles.len()).collect();
        let bvh = BVHNode::new(&triangles, &mut indices, 0);
        let build_time = start_time.elapsed().as_millis();
        println!("BVH built in {}ms", build_time);

        Ok(Mesh { triangles, bvh })
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        self.bvh
            .intersect_recursive(ray, &self.triangles, t_min, t_max)
    }
}
