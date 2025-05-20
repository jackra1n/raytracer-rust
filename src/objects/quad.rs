use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3;
use std::sync::Arc;
use crate::renderer::EPSILON; // For comparisons

#[derive(Clone)] // Added Clone for potential future use, not strictly needed by Hittable now
pub struct Quad {
    p0: Vec3,       // Min corner (x0, y_k, z0)
    p1: Vec3,       // Max corner (x1, y_k, z1)
    normal: Vec3,   // Normal of the quad's plane
    material: Arc<dyn Material>,
    d: f32,         // Constant in plane equation normal.dot(P) - d = 0
    w: Vec3,        // Normal vector for area calculation, not necessarily unit
}

impl Quad {
    // Creates an axis-aligned quad on a plane parallel to XZ, YZ, or XY
    // center: center of the quad
    // size: Vec3(width, height, depth) - relevant components used based on major_axis
    // major_axis: 0 for YZ plane (normal along X), 1 for XZ plane (normal along Y), 2 for XY plane (normal along Z)
    // material: the material of the quad
    pub fn new_axis_aligned(center: Vec3, size: Vec3, major_axis: usize, material: Arc<dyn Material>) -> Self {
        let half_size = size * 0.5;
        let mut p0 = center - half_size;
        let mut p1 = center + half_size;
        let normal;
        let d;

        match major_axis {
            0 => { // YZ plane, normal along X
                normal = Vec3::new(1.0, 0.0, 0.0);
                p0.x = center.x; // Quad lies on this X plane
                p1.x = center.x;
                d = normal.dot(center);
            }
            1 => { // XZ plane, normal along Y
                normal = Vec3::new(0.0, 1.0, 0.0);
                p0.y = center.y; // Quad lies on this Y plane
                p1.y = center.y;
                d = normal.dot(center);
            }
            2 => { // XY plane, normal along Z
                normal = Vec3::new(0.0, 0.0, 1.0);
                p0.z = center.z; // Quad lies on this Z plane
                p1.z = center.z;
                d = normal.dot(center);
            }
            _ => panic!("Invalid major_axis for Quad, must be 0, 1, or 2"),
        }
        
        // w is for area calculation, related to edges, not necessarily unit normal.
        // For an XZ quad (axis 1): edge0 along X, edge1 along Z.
        // This needs to be set up carefully based on p0, p1 for UVs / area later if needed.
        // For now, a simple placeholder for `w` as it's not used in hit.
        let w_placeholder = p1 - p0; // Not a proper normal for area, but fills the field

        Self { p0, p1, normal, material, d, w: w_placeholder }
    }
}

impl Hittable for Quad {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let denom = self.normal.dot(ray.direction);

        // Ray is parallel to the plane or pointing away from the plane (if one-sided)
        // For two-sided quads, allow hitting from behind by checking abs(denom).
        if denom.abs() < EPSILON { // Ray is parallel
            return None;
        }

        let t = (self.d - self.normal.dot(ray.origin)) / denom;

        if t <= t_min || t >= t_max {
            return None;
        }

        let hit_pos = ray.at(t);

        // Check if hit_pos is within the quad boundaries
        // This depends on which plane the quad lies on.
        // For an XZ quad (normal along Y): check X and Z bounds.
        // For a YZ quad (normal along X): check Y and Z bounds.
        // For an XY quad (normal along Z): check X and Y bounds.
        
        let mut within_bounds = false;
        if self.normal.x.abs() > 0.9 { // YZ plane (normal along X)
            within_bounds = hit_pos.y >= self.p0.y && hit_pos.y <= self.p1.y &&
                            hit_pos.z >= self.p0.z && hit_pos.z <= self.p1.z;
        } else if self.normal.y.abs() > 0.9 { // XZ plane (normal along Y)
            within_bounds = hit_pos.x >= self.p0.x && hit_pos.x <= self.p1.x &&
                            hit_pos.z >= self.p0.z && hit_pos.z <= self.p1.z;
        } else if self.normal.z.abs() > 0.9 { // XY plane (normal along Z)
            within_bounds = hit_pos.x >= self.p0.x && hit_pos.x <= self.p1.x &&
                            hit_pos.y >= self.p0.y && hit_pos.y <= self.p1.y;
        }

        if !within_bounds {
            return None;
        }

        let front_face = ray.direction.dot(self.normal) < 0.0;
        let hit_normal = if front_face { self.normal } else { -self.normal };

        Some(HitRecord {
            t,
            position: hit_pos,
            normal: hit_normal,
            material: self.material.clone(),
            front_face,
        })
    }
} 