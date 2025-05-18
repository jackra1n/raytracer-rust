use crate::color::Color;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub color: Color,
    pub reflectivity: f32, // mirror-like reflections
    pub specular_intensity: f32, // strength of the specular highlight
    pub shininess: f32,      // exponent for specular highlight size
    pub emissive_color: Option<Color>, // color emitted by the material
}

impl Material {

    pub fn builder() -> MaterialBuilder {
        MaterialBuilder::new()
    }

    // Convenience constructor for a purely emissive material
    pub fn new_emissive(color: Color) -> Self {
        Self {
            color: Color::new(0.0, 0.0, 0.0), // Non-emissive part is black
            reflectivity: 0.0,
            specular_intensity: 0.0,
            shininess: 0.0,
            emissive_color: Some(color),
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
}

impl MaterialBuilder {
    pub fn new() -> Self {
        Self {
            color: Color::new(0.8, 0.8, 0.8),
            reflectivity: 0.0,
            specular_intensity: 0.1,
            shininess: 10.0,
            emissive_color: None,
        }
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn reflectivity(mut self, reflectivity: f32) -> Self {
        self.reflectivity = reflectivity;
        self
    }

    pub fn specular(mut self, intensity: f32, shininess: f32) -> Self {
        self.specular_intensity = intensity;
        self.shininess = shininess;
        self
    }

    pub fn emissive(mut self, color: Color) -> Self {
        self.emissive_color = Some(color);
        self.color = Color::new(0.0, 0.0, 0.0); // Or remove this line if emissive materials can also have a base color
        self.reflectivity = 0.0;
        self.specular_intensity = 0.0;
        self.shininess = 0.0;
        self
    }

    pub fn build(self) -> Material {
        Material {
            color: self.color,
            reflectivity: self.reflectivity,
            specular_intensity: self.specular_intensity,
            shininess: self.shininess,
            emissive_color: self.emissive_color,
        }
    }
}
