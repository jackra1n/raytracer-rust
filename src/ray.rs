use crate::vec3::Vec3;

pub struct Ray {
    pub start: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(start: Vec3, dir: Vec3) -> Self {
        Ray {
            start,
            dir: dir.normalized(),
        }
    }
    pub fn at(&self, t: f32) -> Vec3 {
        self.start + self.dir * t
    }
}
