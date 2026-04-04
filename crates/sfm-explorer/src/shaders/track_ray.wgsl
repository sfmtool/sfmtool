// Track ray shader.
//
// Renders observation rays from camera centers to the selected 3D point as
// semi-transparent glow lines. Each ray is drawn as a camera-facing ribbon
// quad with depth-aware occlusion (discards fragments behind opaque scene
// geometry). Rendered as a post-EDL pass (Pass 2.75) onto edl_output.

struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    screen_size: vec2<f32>,
    line_half_width: f32,
    hovered_image_index: u32, // unused by track rays, matches FrustumUniforms layout
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

// Orange color matching frustum and browser track highlights.
const RAY_COLOR: vec3<f32> = vec3<f32>(1.0, 0.647, 0.0);

struct VertexInput {
    // Per-vertex: x in {-1, 1} selects endpoint A/B, y in {-1, 1} selects side
    @location(0) corner: vec2<f32>,
    // Per-instance: ray endpoints
    @location(1) endpoint_a: vec3<f32>,
    @location(2) endpoint_b: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) perp_coord: f32, // -1..1 across the ribbon width
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
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

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);
    out.perp_coord = in.corner.y;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Load scene depth at this fragment's pixel position (reversed-Z: 1=near, 0=far)
    let pixel = vec2<i32>(in.clip_pos.xy);
    let scene_depth = textureLoad(depth_tex, pixel, 0);

    // Hard occlusion: discard fragments behind opaque geometry.
    // Reversed-Z: scene_depth > 0 means geometry exists, and the fragment
    // is behind it when its depth is less than scene_depth (smaller = farther).
    if scene_depth > 0.0 && in.clip_pos.z < scene_depth {
        discard;
    }

    // Glow falloff from center of the ribbon
    let dist = abs(in.perp_coord);
    let glow = 1.0 - smoothstep(0.3, 1.0, dist);
    let alpha = 0.4 * glow;

    return vec4<f32>(RAY_COLOR * alpha, alpha);
}
