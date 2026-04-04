// Image quad shader for frustum far-plane textures.
//
// Renders each camera's image as a textured quad on the frustum's far plane.
// Uses a 2D texture atlas where thumbnails are packed in a grid layout.
// Like frustum wireframes, image quads don't write to the linear depth texture
// (no EDL effect) but do write to the pick buffer.

struct Uniforms {
    view_proj: mat4x4<f32>,
    atlas_cols: u32,
    atlas_rows: u32,
    images_per_page: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var thumbnail_texture: texture_2d_array<f32>;
@group(0) @binding(2) var thumbnail_sampler: sampler;

// Pick ID tag for frustum entities (bits 31..24).
const PICK_TAG_FRUSTUM: u32 = 0x01000000u;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,        // quad corner (-1..1)
    @location(1) corner_tl: vec3<f32>,       // instance: top-left far corner
    @location(2) corner_tr: vec3<f32>,       // instance: top-right far corner
    @location(3) corner_bl: vec3<f32>,       // instance: bottom-left far corner
    @location(4) corner_br: vec3<f32>,       // instance: bottom-right far corner
    @location(5) frustum_index: u32,         // instance: image index
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) frustum_index: u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Map quad corner to UV: (-1,-1)->(0,0), (1,1)->(1,1)
    let u = (in.quad_pos.x + 1.0) * 0.5;
    let v = (in.quad_pos.y + 1.0) * 0.5;

    // Bilinear interpolation of the four corners
    let top = mix(in.corner_tl, in.corner_tr, u);
    let bot = mix(in.corner_bl, in.corner_br, u);
    let world_pos = mix(top, bot, v);

    var out: VertexOutput;
    out.clip_pos = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vec2<f32>(u, v);
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
