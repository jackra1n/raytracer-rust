use crate::acceleration::Aabb;
use crate::hittable::HitRecord;
use crate::mesh::triangle::Triangle;
use crate::ray::Ray;
use crate::renderer::EPSILON;

pub struct BVHNode {
    pub bounds: Aabb,
    left: Option<Box<BVHNode>>,
    right: Option<Box<BVHNode>>,
    triangle_indices: Vec<usize>,
}

impl BVHNode {
    pub fn new(triangles: &[Triangle], indices: &mut [usize], depth: usize) -> Self {
        let num_triangles = indices.len();
        let mut bounds = Aabb::empty();
        for &idx in indices.iter() {
            let tri = &triangles[idx];
            bounds.add_point(tri.v0);
            bounds.add_point(tri.v1);
            bounds.add_point(tri.v2);
        }

        const MAX_DEPTH: usize = 25;
        const MIN_TRIANGLES_PER_LEAF: usize = 4;
        if num_triangles <= MIN_TRIANGLES_PER_LEAF || depth >= MAX_DEPTH {
            return BVHNode {
                bounds,
                left: None,
                right: None,
                triangle_indices: indices.to_vec(),
            };
        }

        let extent = bounds.max - bounds.min;
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };

        indices.sort_unstable_by(|&a, &b| {
            let centroid_a = (triangles[a].v0 + triangles[a].v1 + triangles[a].v2) * (1.0 / 3.0);
            let centroid_b = (triangles[b].v0 + triangles[b].v1 + triangles[b].v2) * (1.0 / 3.0);
            let val_a = centroid_a[axis];
            let val_b = centroid_b[axis];
            val_a
                .partial_cmp(&val_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = num_triangles / 2;
        let (left_indices, right_indices) = indices.split_at_mut(mid);

        if left_indices.is_empty() || right_indices.is_empty() {
            return BVHNode {
                bounds,
                left: None,
                right: None,
                triangle_indices: indices.to_vec(),
            };
        }

        let left_child = Box::new(BVHNode::new(triangles, left_indices, depth + 1));
        let right_child = Box::new(BVHNode::new(triangles, right_indices, depth + 1));

        BVHNode {
            bounds,
            left: Some(left_child),
            right: Some(right_child),
            triangle_indices: Vec::new(),
        }
    }

    pub fn intersect_recursive<'a>(
        &'a self,
        ray: &Ray,
        triangles: &'a [Triangle],
        t_min: f32,
        mut t_max: f32,
    ) -> Option<HitRecord> {
        if !self.bounds.intersect(ray, t_min, t_max) {
            return None;
        }

        if self.left.is_none() {
            let mut closest_hit: Option<HitRecord> = None;
            for &idx in &self.triangle_indices {
                let triangle = &triangles[idx];

                let edge1 = triangle.v1 - triangle.v0;
                let edge2 = triangle.v2 - triangle.v0;
                let h = ray.direction.cross(edge2);
                let a = edge1.dot(h);

                if a.abs() < EPSILON {
                    continue;
                }

                let f = 1.0 / a;
                let s = ray.origin - triangle.v0;
                let u = f * s.dot(h);
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }

                let q = s.cross(edge1);
                let v = f * ray.direction.dot(q);
                if v < 0.0 || u + v > 1.0 {
                    continue;
                }

                let t = f * edge2.dot(q);

                if t > t_min && t < t_max {
                    let position = ray.at(t);
                    let outward_normal = triangle.normal;
                    let front_face = ray.direction.dot(outward_normal) < 0.0;
                    let hit_record_normal = if front_face {
                        outward_normal
                    } else {
                        -outward_normal
                    };

                    let hit_data = HitRecord {
                        t,
                        position,
                        normal: hit_record_normal,
                        material: triangle.material.clone(),
                        front_face,
                    };
                    t_max = t;
                    closest_hit = Some(hit_data);
                }
            }
            return closest_hit;
        }

        let hit_left = self
            .left
            .as_ref()
            .unwrap()
            .intersect_recursive(ray, triangles, t_min, t_max);

        if let Some(ref hit) = hit_left {
            t_max = hit.t;
        }

        let hit_right = self
            .right
            .as_ref()
            .unwrap()
            .intersect_recursive(ray, triangles, t_min, t_max);

        match (hit_left, hit_right) {
            (Some(l), Some(r)) => {
                if l.t < r.t {
                    Some(l)
                } else {
                    Some(r)
                }
            }
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }
}
