use crate::color::Color;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub color: Color,
    pub reflectivity: f32, // mirror-like reflections
    pub specular_intensity: f32, // strength of the specular highlight
    pub shininess: f32,      // exponent for specular highlight size
    pub emissive_color: Option<Color>, // color emitted by the material
    pub transparency: f32,
    pub refractive_index: f32,
}

impl Material {

    pub fn builder() -> MaterialBuilder {
        MaterialBuilder::new()
    }

    pub fn new_emissive(color: Color) -> Self {
        Self {
            color: Color::new(0.0, 0.0, 0.0),
            reflectivity: 0.0,
            specular_intensity: 0.0,
            shininess: 0.0,
            emissive_color: Some(color),
            transparency: 0.0,
            refractive_index: 1.0,
        }
    }

    pub fn mirror() -> Self {
        Self {
            color: Color::new(0.9, 0.9, 0.9),
            reflectivity: 0.97,
            specular_intensity: 0.0,
            shininess: 0.0,
            emissive_color: None,
            transparency: 0.0,
            refractive_index: 0.0,
        }
    }

    pub fn glass() -> Self {
        Self {
            color: Color::new(0.95, 0.95, 1.0),
            reflectivity: 0.05,
            specular_intensity: 0.8,
            shininess: 200.0,
            emissive_color: None,
            transparency: 0.8,
            refractive_index: 1.5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialBuilder {
    color: Color,
    reflectivity: f32,
    specular_intensity: f32,
    shininess: f32,
    emissive_color: Option<Color>,
    transparency: f32,
    refractive_index: f32,
}

impl MaterialBuilder {
    pub fn new() -> Self {
        Self {
            color: Color::new(0.8, 0.8, 0.8),
            reflectivity: 0.0,
            specular_intensity: 0.1,
            shininess: 10.0,
            emissive_color: None,
            transparency: 0.0,
            refractive_index: 1.0,
        }
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn reflectivity(mut self, reflectivity: f32) -> Self {
        self.reflectivity = reflectivity.clamp(0.0, 1.0);
        self
    }

    pub fn specular(mut self, intensity: f32, shininess: f32) -> Self {
        self.specular_intensity = intensity.clamp(0.0, 1.0);
        self.shininess = shininess.max(1.0);
        self
    }

    pub fn emissive(mut self, color: Color) -> Self {
        self.emissive_color = Some(color);
        self.color = Color::new(0.0, 0.0, 0.0);
        self.reflectivity = 0.0;
        self.specular_intensity = 0.0;
        self.transparency = 0.0;
        self
    }

    pub fn transparency(mut self, transparency: f32, refractive_index: f32) -> Self {
        self.transparency = transparency.clamp(0.0, 1.0);
        self.refractive_index = refractive_index.max(1.0);
        self
    }

    pub fn build(self) -> Material {
        Material {
            color: self.color,
            reflectivity: self.reflectivity,
            specular_intensity: self.specular_intensity,
            shininess: self.shininess,
            emissive_color: self.emissive_color,
            transparency: self.transparency,
            refractive_index: self.refractive_index,
        }
    }
}
