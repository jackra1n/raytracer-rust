use std::ops::{Add, Div, Mul, Sub};

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn clamp(&mut self) {
        self.r = self.r.clamp(0.0, 1.0);
        self.g = self.g.clamp(0.0, 1.0);
        self.b = self.b.clamp(0.0, 1.0);
    }

    pub fn splat(v: f32) -> Color {
        Color::new(v, v, v)
    }

    pub fn sqrt(self) -> Color {
        Color::new(self.r.sqrt(), self.g.sqrt(), self.b.sqrt())
    }
}

impl Add for Color {
    type Output = Color;
    fn add(self, other: Color) -> Color {
        Color::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }
}

impl Add<f32> for Color {
    type Output = Color;
    fn add(self, s: f32) -> Color {
        Color::new(self.r + s, self.g + s, self.b + s)
    }
}

impl Mul<f32> for Color {
    type Output = Color;
    fn mul(self, s: f32) -> Color {
        Color::new(self.r * s, self.g * s, self.b * s)
    }
}
impl Mul<Color> for Color {
    type Output = Color;
    fn mul(self, o: Color) -> Color {
        Color::new(self.r * o.r, self.g * o.g, self.b * o.b)
    }
}

impl Sub for Color {
    type Output = Color;
    fn sub(self, other: Color) -> Color {
        Color::new(self.r - other.r, self.g - other.g, self.b - other.b)
    }
}

impl Sub<f32> for Color {
    type Output = Color;
    fn sub(self, s: f32) -> Color {
        Color::new(self.r - s, self.g - s, self.b - s)
    }
}

impl Div<f32> for Color {
    type Output = Color;
    fn div(self, s: f32) -> Color {
        Color::new(self.r / s, self.g / s, self.b / s)
    }
}

impl Div<Color> for Color {
    type Output = Color;
    fn div(self, o: Color) -> Color {
        Color::new(self.r / o.r, self.g / o.g, self.b / o.b)
    }
}

pub fn color_to_u32(mut c: Color) -> u32 {
    c.clamp();
    let r = (c.r * 255.0) as u32;
    let g = (c.g * 255.0) as u32;
    let b = (c.b * 255.0) as u32;
    (r << 16) | (g << 8) | b
}

#[allow(dead_code)]
impl Color {
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };
    pub const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
    };
    pub const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
    };
    pub const BLUE: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
    };
    pub const YELLOW: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 0.0,
    };
    pub const MAGENTA: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
    };
    pub const CYAN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 1.0,
    };
    pub const GRAY: Color = Color {
        r: 0.5,
        g: 0.5,
        b: 0.5,
    };
}
