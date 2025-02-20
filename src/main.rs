use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
}

struct Circle {
    x: f32,
    y: f32,
    radius: f32,
    color: u32,
}

impl Circle {
    fn contains(&self, px: f32, py: f32) -> bool {
        let dx = px - self.x;
        let dy = py - self.y;
        dx * dx + dy * dy <= self.radius * self.radius
    }
}

fn main() {
    let mut window = Window::new(
        "Raytracer - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let circles = vec![
        Circle {
            x: 400.0,
            y: 300.0,
            radius: 100.0,
            color: rgb_to_u32(55, 55, 255),
        },
    ];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let mut red = (x as f32 / WIDTH as f32 * 255.0) as u8;
                let mut green = (y as f32 / HEIGHT as f32 * 255.0) as u8;
                let blue = 128;
                // let mut color = rgb_to_u32(red, green, blue);
                let mut color = rgb_to_u32(0, 0, 0);

                for circle in &circles {
                    if circle.contains(x as f32, y as f32) {
                        color = circle.color;
                    }
                }

                buffer[y * WIDTH + x] = color;
            }
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}