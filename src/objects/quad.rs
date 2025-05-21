use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3;
use std::sync::Arc;
use crate::renderer::EPSILON; // For comparisons
use glam::{Mat4, Vec3A as GlamVec3A}; // Using Vec3A for SIMD alignment if Mat4 expects it

#[derive(Clone)]
pub struct Quad {
    base: Vec3,      // One corner of the quad (world space, e.g. transformed local (-0.5, 0, -0.5))
    edge0: Vec3,     // Vector along one edge from base (world space)
    edge1: Vec3,     // Vector along the other edge from base (world space)
    normal: Vec3,    // Unit normal vector of the quad's plane
    d: f32,          // Constant in plane equation normal.dot(P) - d = 0
    material: Arc<dyn Material>,
    inv_edge0_len_sq: f32,
    inv_edge1_len_sq: f32,
    // Optional: area for emissive quads
    // pub area: f32,
}

impl Quad {
    // Creates a quad from a transformation matrix applied to a canonical 1x1 quad
    // in the local XZ plane (Y=0), centered at the origin.
    pub fn new_transformed(transform: Mat4, material: Arc<dyn Material>) -> Self {
        // Canonical quad vertices (1x1 in XZ plane, Y=0, centered at origin)
        // A: (-0.5, 0.0, -0.5) - this will be the base
        // B: ( 0.5, 0.0, -0.5) - base + local_edge0
        // D: (-0.5, 0.0,  0.5) - base + local_edge1
        let v_a_local = GlamVec3A::new(-0.5, 0.0, -0.5);
        let v_b_local = GlamVec3A::new( 0.5, 0.0, -0.5);
        let v_d_local = GlamVec3A::new(-0.5, 0.0,  0.5);

        // Transform points to world space
        let base_w_glam = (transform * v_a_local.extend(1.0)).truncate();
        let p_b_w_glam = (transform * v_b_local.extend(1.0)).truncate();
        let p_d_w_glam = (transform * v_d_local.extend(1.0)).truncate();

        let base = Vec3::new(base_w_glam.x, base_w_glam.y, base_w_glam.z);
        let p_b_w = Vec3::new(p_b_w_glam.x, p_b_w_glam.y, p_b_w_glam.z);
        let p_d_w = Vec3::new(p_d_w_glam.x, p_d_w_glam.y, p_d_w_glam.z);

        let edge0 = p_b_w - base;
        let edge1 = p_d_w - base;
        
        // Normal: Tungsten uses edge1.cross(edge0). Our cross product: X.cross(Z) = -Y.
        // If local edge0 is +X (0.5 - (-0.5) = 1,0,0) and local edge1 is +Z (0,0,1),
        // then transformed edge0 and edge1.
        // For CCW winding A, B, C(B+D-A), D, normal is (B-A).cross(D-A)
        // So edge0.cross(edge1)
        let normal = edge0.cross(edge1).normalized();
        let d = normal.dot(base);

        let edge0_len_sq = edge0.length_squared();
        let edge1_len_sq = edge1.length_squared();

        // let area = (edge0.cross(edge1)).length(); // For emissive sampling

        Self {
            base,
            edge0,
            edge1,
            normal,
            d,
            material,
            inv_edge0_len_sq: if edge0_len_sq > EPSILON { 1.0 / edge0_len_sq } else { 0.0 },
            inv_edge1_len_sq: if edge1_len_sq > EPSILON { 1.0 / edge1_len_sq } else { 0.0 },
            // area,
        }
    }
}

impl Hittable for Quad {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);

        // Ray is parallel to the plane
        if denom.abs() < EPSILON {
            return None;
        }

        let t = (self.d - self.normal.dot(ray.origin)) / denom;

        if t <= t_min || t >= t_max {
            return None;
        }

        let hit_pos = ray.at(t);
        let v_hit_to_base = hit_pos - self.base;

        // Parametric check ( barycentric coords for parallelogram)
        // l0 = (v_hit_to_base . edge0) / edge0.length_squared()
        // l1 = (v_hit_to_base . edge1) / edge1.length_squared()
        let l0 = v_hit_to_base.dot(self.edge0) * self.inv_edge0_len_sq;
        let l1 = v_hit_to_base.dot(self.edge1) * self.inv_edge1_len_sq;
        
        // Check if hit is within the parallelogram defined by base, edge0, edge1
        if !(l0 >= -EPSILON && l0 <= 1.0 + EPSILON && l1 >= -EPSILON && l1 <= 1.0 + EPSILON) {
            // Using EPSILON for float comparisons at boundaries
            return None;
        }
        
        let front_face = ray.direction.dot(self.normal) < 0.0; // Standard definition
        let hit_normal = if front_face { self.normal } else { -self.normal };

        // UV coordinates (optional, but good for texturing/emission)
        // let u = l0;
        // let v = l1;

        Some(HitRecord {
            t,
            position: hit_pos,
            normal: hit_normal,
            material: self.material.clone(),
            front_face,
            // u, 
            // v,
        })
    }
} 