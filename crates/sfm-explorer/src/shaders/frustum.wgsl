// Camera frustum wireframe shader.
//
// Renders frustum edges as camera-facing ribbon quads with controllable
// pixel-width lines. Frustums participate in the hardware depth buffer
// (occluded by and occluding points) but write 0.0 to the linear depth
// texture so they don't affect EDL shading or depth readback.
//
// Outputs a pick ID (entity tag + frustum index) so the CPU can identify
// which camera was clicked.

struct FrustumUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    screen_size: vec2<f32>,
    line_half_width: f32,
    hovered_image_index: u32,
}

@group(0) @binding(0) var<uniform> uniforms: FrustumUniforms;
@group(0) @binding(1) var<storage, read> frustum_colors: array<u32>;

// Pick ID tag for frustum entities (bits 31..24).
const PICK_TAG_FRUSTUM: u32 = 0x01000000u;

struct VertexInput {
    // Per-vertex: x in {-1, 1} selects endpoint A/B, y in {-1, 1} selects side
    @location(0) corner: vec2<f32>,
    // Per-instance edge data
    @location(1) endpoint_a: vec3<f32>,
    @location(2) endpoint_b: vec3<f32>,
    @location(3) frustum_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) @interpolate(flat) frustum_index: u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Select which endpoint this vertex belongs to
    let is_b = in.corner.x > 0.0;
    let world_pos = select(in.endpoint_a, in.endpoint_b, is_b);

    // Project both endpoints to clip space
    let clip_a = uniforms.view_proj * vec4<f32>(in.endpoint_a, 1.0);
    let clip_b = uniforms.view_proj * vec4<f32>(in.endpoint_b, 1.0);
    let clip_pos = select(clip_a, clip_b, is_b);

    // Compute edge direction in NDC for perpendicular expansion
    let ndc_a = clip_a.xy / clip_a.w;
    let ndc_b = clip_b.xy / clip_b.w;
    let edge_ndc = ndc_b - ndc_a;
    let edge_len = length(edge_ndc);

    // Perpendicular direction in NDC (rotated 90 degrees)
    var perp: vec2<f32>;
    if edge_len > 0.0001 {
        let edge_dir = edge_ndc / edge_len;
        perp = vec2<f32>(-edge_dir.y, edge_dir.x);
    } else {
        perp = vec2<f32>(0.0, 1.0);
    }

    // Expand by line width in pixels, converted to NDC
    let pixel_to_ndc = vec2<f32>(2.0 / uniforms.screen_size.x, 2.0 / uniforms.screen_size.y);
    let offset_ndc = perp * in.corner.y * uniforms.line_half_width * pixel_to_ndc;

    // Look up color from per-frustum storage buffer
    let color_packed = frustum_colors[in.frustum_index];
    let r = f32(color_packed & 0xFFu) / 255.0;
    let g = f32((color_packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((color_packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((color_packed >> 24u) & 0xFFu) / 255.0;

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);
    out.color = vec3<f32>(r, g, b);
    out.alpha = a;
    out.frustum_index = in.frustum_index;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) linear_depth: f32,
    @location(2) pick_id: u32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var color = in.color;
    var alpha = in.alpha;
    // Highlight hovered frustum (cross-panel hover feedback):
    // override to full-opacity white so it stands out against the
    // default semi-transparent white frustums.
    if in.frustum_index == uniforms.hovered_image_index {
        color = vec3<f32>(1.0, 1.0, 1.0);
        alpha = 1.0;
    }
    var out: FragmentOutput;
    out.color = vec4<f32>(color * alpha, alpha);
    out.linear_depth = 0.0; // do not contribute to EDL or depth readback
    out.pick_id = PICK_TAG_FRUSTUM | in.frustum_index;
    return out;
}
