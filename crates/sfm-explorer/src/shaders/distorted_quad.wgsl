// Distorted image quad shader for tessellated frustum far-plane textures.
//
// Used for cameras with lens distortion, where the image quad is tessellated
// into a grid mesh to show the true distorted field of view. Vertex positions
// and UVs are pre-computed on CPU (no bilinear corner interpolation needed).

struct Uniforms {
    view_proj: mat4x4<f32>,
    atlas_cols: u32,
    atlas_rows: u32,
    images_per_page: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var thumbnail_texture: texture_2d_array<f32>;
@group(0) @binding(2) var thumbnail_sampler: sampler;
@group(0) @binding(3) var<storage, read> frustum_colors: array<u32>;

// Pick ID tag for frustum entities (bits 31..24).
const PICK_TAG_FRUSTUM: u32 = 0x01000000u;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) frustum_index: u32,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) frustum_index: u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.frustum_index = in.frustum_index;
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) linear_depth: f32,
    @location(2) pick_id: u32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    // Discard hidden frustums (alpha == 0 in color buffer)
    let color_packed = frustum_colors[in.frustum_index];
    if (color_packed >> 24u) == 0u {
        discard;
    }
    // Compute atlas UV and layer from grid position
    let page = in.frustum_index / uniforms.images_per_page;
    let idx_in_page = in.frustum_index % uniforms.images_per_page;
    let col = idx_in_page % uniforms.atlas_cols;
    let row = idx_in_page / uniforms.atlas_cols;
    let cell_size = vec2<f32>(1.0 / f32(uniforms.atlas_cols), 1.0 / f32(uniforms.atlas_rows));
    let atlas_uv = (vec2<f32>(f32(col), f32(row)) + in.uv) * cell_size;
    let tex_color = textureSample(thumbnail_texture, thumbnail_sampler, atlas_uv, page);

    var out: FragOutput;
    out.color = vec4<f32>(tex_color.rgb, 1.0);
    out.linear_depth = 0.0;
    out.pick_id = PICK_TAG_FRUSTUM | in.frustum_index;
    return out;
}
