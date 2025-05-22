use std::path::Path;
use std::sync::Arc;

use lyon::math::Point as LyonPoint;
use lyon::path::Path as LyonPath;
use lyon::path::builder::PathBuilder;
use lyon::tessellation::{
    FillTessellator,
    FillOptions,
    BuffersBuilder,
    VertexBuffers,
    geometry_builder::simple_builder,
    FillVertex,
    FillVertexConstructor,
    LineCap,
    LineJoin,
};

// Import types from your project
use crate::mesh::mesh_object::Mesh; // Assuming this is the correct path
use crate::vec3::Vec3 as RaytracerVec3; // Using your project's Vec3, aliased to avoid confusion if glam::Vec3 is also used directly
use crate::material::Material; // Assuming this is your Material trait/struct
use crate::mesh::triangle::Triangle; // Added for constructing triangles
use glam::{Mat4, Vec3A}; // Common glam types for 3D math

// We will need these crates. Please add them to your Cargo.toml if not already:
// rusttype = "0.9"
// lyon = "0.17"
// glam = "0.24" (or your version)

/*
Example of how you might add to Cargo.toml:

[dependencies]
# ... your other dependencies
rusttype = "0.9"
lyon = "0.17"
glam = "0.24"
*/

#[derive(Debug)]
pub enum TextMeshError {
    FontLoadError(String),
    GlyphError(String),
    TessellationError(String),
    IoError(std::io::Error),
    LyonError(lyon::tessellation::TessellationError) // For Lyon specific errors
}

impl From<std::io::Error> for TextMeshError {
    fn from(err: std::io::Error) -> TextMeshError {
        TextMeshError::IoError(err)
    }
}

impl From<lyon::tessellation::TessellationError> for TextMeshError {
    fn from(err: lyon::tessellation::TessellationError) -> TextMeshError {
        TextMeshError::TessellationError(format!("Lyon tessellation error: {:?}", err))
    }
}

fn create_text_mesh(text: &str, font_data: &[u8], font_size: f32, extrusion_depth: f32) -> Result<Mesh, String> {
    // 1. Parse the font
    let settings = FontSettings {
        scale: font_size, // Scale for the output outlines
        ..FontSettings::default()
    };
    let font = Font::from_bytes(font_data, settings).map_err(|e| e.to_string())?;

    let mut combined_mesh = Mesh::new();
    let mut current_x_offset = 0.0;

    // Optional: Use fontdue's Layout for better kerning and positioning
    let mut layout = Layout::new(CoordinateSystem::YUp); // YUp is common for 3D
    layout.append(&[&font], &TextStyle::new(text, font_size, 0)); // 0 is font_index

    for glyph_layout in layout.glyphs() {
        // 2. Get glyph outline for each character
        // `glyph_layout.key.glyph_index` gives the index
        // `glyph_layout.x`, `glyph_layout.y` gives the position
        // `glyph_layout.key.px` is the font size (already set by `font_size`)

        let Some(glyph_raster_info) = font.outline_glyph_raster_info(glyph_layout.key) else {
            // Handle missing glyphs (e.g., space, or unsupported char)
            // For spaces, advance_width is still relevant from metrics
            let metrics = font.metrics_indexed(glyph_layout.key.glyph_index, glyph_layout.key.px);
            current_x_offset += metrics.advance_width;
            continue;
        };

        let glyph = glyph_raster_info.glyph; // This is the `fontdue::glyph::Glyph`

        // 3. Convert fontdue outline to Lyon Path
        let mut path_builder = Path::builder();
        for command in glyph.path_commands() {
            match command {
                fontdue::path::PathCommand::MoveTo(p) => {
                    path_builder.begin(Point::new(p.x, p.y));
                }
                fontdue::path::PathCommand::LineTo(p) => {
                    path_builder.line_to(Point::new(p.x, p.y));
                }
                fontdue::path::PathCommand::QuadTo(p1, p2) => {
                    path_builder.quadratic_bezier_to(Point::new(p1.x, p1.y), Point::new(p2.x, p2.y));
                }
                fontdue::path::PathCommand::CurveTo(p1, p2, p3) => {
                    path_builder.cubic_bezier_to(Point::new(p1.x, p1.y), Point::new(p2.x, p2.y), Point::new(p3.x, p3.y));
                }
                fontdue::path::PathCommand::Close => {
                    path_builder.close();
                }
            }
        }
        let path = path_builder.build();

        // 4. Tessellate the 2D path into triangles using Lyon
        let mut geometry: VertexBuffers<Point, u16> = VertexBuffers::new(); // Use u16 for indices if <65k vertices per char
        let mut tessellator = FillTessellator::new();
        let fill_options = FillOptions::default(); // Default fill rule (NonZero) is usually fine for fonts

        tessellator.tessellate_path(
            &path,
            &fill_options,
            &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| {
                // This closure converts lyon's vertex to your Point format
                // Lyon's FillVertex contains position and normal, but for 2D fill, normal isn't used.
                vertex.position()
            }),
        ).map_err(|e| format!("Lyon tessellation error: {:?}", e))?;

        // --- 5. Extrude 2D triangles to 3D and add to combined_mesh ---
        let base_vertex_index = combined_mesh.vertices.len() as u32;

        // Add front face vertices
        for p_2d in &geometry.vertices {
            combined_mesh.vertices.push(Vertex {
                position: [
                    p_2d.x + glyph_layout.x, // Apply layout offset
                    p_2d.y + glyph_layout.y, // Apply layout offset
                    0.0 // Front face at z=0
                ],
            });
        }
        // Add back face vertices
        for p_2d in &geometry.vertices {
            combined_mesh.vertices.push(Vertex {
                position: [
                    p_2d.x + glyph_layout.x,
                    p_2d.y + glyph_layout.y,
                    -extrusion_depth // Back face at z=-depth (or +depth)
                ],
            });
        }

        let num_2d_vertices = geometry.vertices.len() as u32;

        // Add front face triangles
        for i in (0..geometry.indices.len()).step_by(3) {
            combined_mesh.triangles.push(Triangle {
                v0_idx: base_vertex_index + geometry.indices[i] as u32,
                v1_idx: base_vertex_index + geometry.indices[i + 1] as u32,
                v2_idx: base_vertex_index + geometry.indices[i + 2] as u32,
            });
        }

        // Add back face triangles (reverse winding order)
        for i in (0..geometry.indices.len()).step_by(3) {
            combined_mesh.triangles.push(Triangle {
                v0_idx: base_vertex_index + num_2d_vertices + geometry.indices[i] as u32,
                v1_idx: base_vertex_index + num_2d_vertices + geometry.indices[i + 2] as u32, // Swapped
                v2_idx: base_vertex_index + num_2d_vertices + geometry.indices[i + 1] as u32, // Swapped
            });
        }

        // Add side face triangles
        // For each edge in the 2D outline, create two triangles for the side.
        // Lyon's output is a triangle list, not necessarily an outline.
        // So, we iterate through the edges of the *tessellated* 2D triangles.
        // This creates sides for internal holes too if the font has them.
        for i in (0..geometry.indices.len()).step_by(3) {
            let i0 = geometry.indices[i] as u32;
            let i1 = geometry.indices[i+1] as u32;
            let i2 = geometry.indices[i+2] as u32;

            let edges = [(i0, i1), (i1, i2), (i2, i0)];
            for (u_idx_2d, v_idx_2d) in edges {
                let v0_front = base_vertex_index + u_idx_2d;
                let v1_front = base_vertex_index + v_idx_2d;
                let v0_back  = base_vertex_index + u_idx_2d + num_2d_vertices;
                let v1_back  = base_vertex_index + v_idx_2d + num_2d_vertices;

                // Triangle 1: (v0_front, v1_back, v0_back)
                combined_mesh.triangles.push(Triangle {
                    v0_idx: v0_front,
                    v1_idx: v1_back,
                    v2_idx: v0_back,
                });
                // Triangle 2: (v0_front, v1_front, v1_back)
                combined_mesh.triangles.push(Triangle {
                    v0_idx: v0_front,
                    v1_idx: v1_front,
                    v2_idx: v1_back,
                });
            }
        }
        // No need to manually update `current_x_offset` when using Layout
    }
    Ok(combined_mesh)
}

fn main() -> Result<(), String> {
    // Example: Load a font file (e.g., Arial)
    // Make sure you have a .ttf file (e.g., "arial.ttf") in your project root or specify the correct path.
    // You might need to provide your own font file.
    let font_bytes = std::fs::read("YOUR_FONT.ttf")
        .map_err(|e| format!("Failed to read font file: {}", e))?;

    let text_mesh = create_text_mesh("Rust!", &font_bytes, 60.0, 10.0)?;

    println!("Generated mesh for 'Rust!':");
    println!("  Vertices: {}", text_mesh.vertices.len());
    println!("  Triangles: {}", text_mesh.triangles.len());

    // Now you can add `text_mesh` to your raytracer scene
    // For example, iterate through `text_mesh.triangles`, get the vertex positions
    // using the indices, and create `YourRaytracerTriangle` objects.

    Ok(())
}

// Potentially helper functions:
// fn tessellate_glyph_outline(...) -> (Vec<[f32;2]>, Vec<[u32;3]>) { ... }
// fn extrude_2d_mesh(...) -> (Vec<YourVec3Type>, Vec<[u32;3]>, Vec<YourVec3Type> /*normals*/) { ... } 

// A custom vertex constructor for Lyon to output just the 2D points.
struct SimpleVertexConstructor;
impl FillVertexConstructor<LyonPoint> for SimpleVertexConstructor {
    fn new_vertex(&mut self, vertex: FillVertex) -> LyonPoint {
        vertex.position()
    }
} 