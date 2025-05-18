use crate::color::Color;
use crate::hittable::Object;
use crate::light::Light;
use crate::material::Material;
use crate::mesh::mesh_object::Mesh;
use crate::objects::cube::Cube;
use crate::objects::plane::Plane;
use crate::objects::sphere::Sphere;
use crate::vec3::Vec3;

pub struct Scene {
    pub objects: Vec<Box<dyn Object + Sync>>,
    pub lights: Vec<Light>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            objects: Vec::new(),
            lights: Vec::new(),
        }
    }

    pub fn add_object(&mut self, obj: Box<dyn Object + Sync>) {
        self.objects.push(obj);
    }

    pub fn add_light(&mut self, l: Light) {
        self.lights.push(l);
    }
}

fn load_mesh(
    scene: &mut Scene,
    path: &str,
    material: Material,
    scale: f32,
    offset: Vec3,
    rotation_y: f32,
) {
    println!("Attempting to load mesh: {}", path);
    match Mesh::from_obj(path, material, scale, offset, rotation_y) {
        Ok(mesh) => {
            println!(
                "Loaded '{}' with {} triangles. Center: {:?}, Extent: {:?}",
                path,
                mesh.triangles.len(),
                (mesh.bvh.bounds.min + mesh.bvh.bounds.max) * 0.5,
                mesh.bvh.bounds.max - mesh.bvh.bounds.min
            );
            scene.add_object(Box::new(mesh));
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load '{}': {}", path, e);
        }
    }
}

pub fn init_scene() -> Scene {
    let mut scene = Scene::new();

    let floor_mat = Material::new(Color::new(0.0, 0.3, 0.3), 0.2, 0.3, 20.0);
    let blue_mirror_mat = Material::new(Color::new(0.0, 0.5, 1.0), 0.8, 0.9, 1000.0);
    let yellow_diffuse_mat = Material::new(Color::YELLOW, 0.7, 0.8, 50.0);
    let magenta_mat = Material::new(Color::MAGENTA, 0.5, 0.6, 100.0);
    let red_plastic_mat = Material::new(Color::RED, 0.0, 0.7, 50.0);

    scene.add_light(Light::new(
        Vec3::new(-400.0, 800.0, -800.0),
        Color::new(1.0, 1.0, 1.0),
        1.0,
    ));
    scene.add_light(Light::new(
        Vec3::new(400.0, 600.0, -500.0),
        Color::new(0.8, 0.8, 1.0),
        0.8,
    ));
    scene.add_light(Light::new(
        Vec3::new(0.0, 400.0, 1000.0),
        Color::new(1.0, 0.8, 0.8),
        0.5,
    ));

    scene.add_object(Box::new(Plane::new(
        Vec3::new(0.0, -100.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        floor_mat,
    )));

    scene.add_object(Box::new(Sphere {
        center: Vec3::new(-250.0, 50.0, 150.0),
        radius: 150.0,
        material: blue_mirror_mat,
    }));

    scene.add_object(Box::new(Cube::new_pos_size(
        Vec3::new(300.0, -100.0, 0.0),
        Vec3::new(100.0, 400.0, 100.0),
        yellow_diffuse_mat,
    )));

    scene.add_object(Box::new(Cube::new_pos_size(
        Vec3::new(50.0, -50.0, -150.0),
        Vec3::new(100.0, 100.0, 100.0),
        magenta_mat,
    )));

    let amogus_pos = Vec3::new(0.0, -100.0, 200.0);
    let amogus_scale = 3.0;
    load_mesh(
        &mut scene,
        "models/amogus/obj/sus.obj",
        red_plastic_mat,
        amogus_scale,
        amogus_pos,
        180.0,
    );

    // let teapot_pos = Vec3::new(300.0, -100.0, 400.0);
    // let teapot_scale = 50.0;
    // load_mesh(
    //     &mut scene,
    //     "models/teapot/teapot.obj",
    //     grey_metal_mat,
    //     teapot_scale,
    //     teapot_pos,
    //     0.0,
    // );

    scene
}
