use crate::color::Color;
use crate::hittable::{Hittable, HittableList};
use crate::material::{Lambertian, Metal, Dielectric, EmissiveLight, Material};
use crate::mesh::mesh_object::Mesh;
use crate::objects::plane::Plane;
use crate::objects::sphere::Sphere;
use crate::vec3::Vec3;
use std::sync::Arc;



pub struct Scene {
    pub object_list: HittableList,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            object_list: HittableList::new(),
        }
    }

    pub fn add_object(&mut self, obj: Box<dyn Hittable + Sync>) {
        self.object_list.add(obj);
    }
}

fn load_mesh(
    scene: &mut Scene,
    path: &str,
    material: Arc<dyn Material>,
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

    let floor_mat = Arc::new(Lambertian::new(Color::new(0.0, 0.3, 0.3)));
    let default_mirror_mat = Arc::new(Metal::new(Color::new(0.9, 0.9, 1.0), 0.02)); // Reduced fuzz from 0.9, 0.9 is very blurry
    let fuzzy_mirror_mat = Arc::new(Metal::new(Color::new(0.0, 0.9, 1.0), 0.9)); // Reduced fuzz from 0.9, 0.9 is very blurry
    let yellow_diffuse_mat = Arc::new(Lambertian::new(Color::YELLOW));
    let magenta_mat = Arc::new(Lambertian::new(Color::MAGENTA));
    let red_plastic_mat = Arc::new(Lambertian::new(Color::RED));
    let grey_metal_mat = Arc::new(Metal::new(Color::GRAY, 0.0)); // Fuzz 1.0 is max blur, 0.0 for perfect mirror
    let glass_mat = Arc::new(Dielectric::new(1.5));


    let bright_light_color = Color::new(10.0, 10.0, 10.0); // Values > 1.0 make it emissive
    let sphere_light_mat = Arc::new(EmissiveLight::new(bright_light_color));
    // scene.add_light(Light::new(
    //     Vec3::new(-400.0, 800.0, -800.0),
    //     Color::new(1.0, 1.0, 1.0),
    //     1.0,
    // ));
    // scene.add_light(Light::new(
    //     Vec3::new(400.0, 600.0, -500.0),
    //     Color::new(0.8, 0.8, 1.0),
    //     0.8,
    // ));
    // scene.add_light(Light::new(
    //     Vec3::new(0.0, 400.0, 1000.0),
    //     Color::new(1.0, 0.8, 0.8),
    //     0.5,
    // ));

    scene.add_object(Box::new(Sphere {
        center: Vec3::new(-400.0, 800.0, -800.0), // Position it above or to the side
        radius: 50.0,                          // Radius of the light
        material: sphere_light_mat.clone(),
    }));

    scene.add_object(Box::new(Sphere {
        center: Vec3::new(400.0, 600.0, -500.0), // Position it above or to the side
        radius: 50.0,                          // Radius of the light
        material: sphere_light_mat.clone(),
    }));

    scene.add_object(Box::new(Sphere {
        center: Vec3::new(0.0, 400.0, 1000.0), // Position it above or to the side
        radius: 50.0,                          // Radius of the light
        material: sphere_light_mat.clone(),
    }));

    scene.add_object(Box::new(Plane::new(
        Vec3::new(0.0, -100.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        floor_mat,
    )));

    // scene.add_object(Box::new(Sphere {
    //     center: Vec3::new(-350.0, 50.0, -150.0),
    //     radius: 150.0,
    //     material: blue_mirror_mat,
    // }));

    scene.add_object(Box::new(Sphere {
        center: Vec3::new(350.0, 50.0, -450.0),
        radius: 100.0,
        material: glass_mat.clone(),
    }));

    // scene.add_object(Box::new(Cube::new_pos_size(
    //     Vec3::new(150.0, 50.0, -450.0),
    //     Vec3::new(100.0, 100.0, 100.0),
    //     Material::glass(),
    // )));

    // scene.add_object(Box::new(Sphere {
    //     center: Vec3::new(250.0, 50.0, -250.0),
    //     radius: 150.0,
    //     material: grey_metal_mat,
    // }));

    // scene.add_object(Box::new(Cube::new_pos_size(
    //     Vec3::new(300.0, -100.0, 0.0),
    //     Vec3::new(100.0, 400.0, 100.0),
    //     yellow_diffuse_mat,
    // )));

    // scene.add_object(Box::new(Cube::new_pos_size(
    //     Vec3::new(50.0, -50.0, -150.0),
    //     Vec3::new(100.0, 100.0, 100.0),
    //     magenta_mat,
    // )));

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

    let amogus_pos = Vec3::new(-350.0, -100.0, 200.0);
    let amogus_scale = 3.0;
    load_mesh(
        &mut scene,
        "models/amogus/obj/sus.obj",
        glass_mat.clone(),
        amogus_scale,
        amogus_pos,
        180.0,
    );


    let amogus_pos = Vec3::new(350.0, -100.0, 200.0);
    let amogus_scale = 3.0;
    load_mesh(
        &mut scene,
        "models/amogus/obj/sus.obj",
        default_mirror_mat.clone(),
        amogus_scale,
        amogus_pos,
        180.0,
    );

    // let amogus_pos = Vec3::new(200.0, -100.0, 200.0);
    // let amogus_scale = 3.0;
    // load_mesh(
    //     &mut scene,
    //     "models/amogus/obj/sus.obj",
    //     Material::mirror(),
    //     amogus_scale,
    //     amogus_pos,
    //     180.0,
    // );

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
