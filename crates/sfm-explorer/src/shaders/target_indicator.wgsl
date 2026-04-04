// Target indicator shader.
//
// Renders the orbit target as a rotating wireframe 3D compass with
// depth-aware transparency. Each edge is drawn as a camera-facing ribbon
// quad for controllable line width. The compass has an elongated vertical
// axis showing world_up, a circular horizontal ring, and radial compass-rose
// spikes.

struct TargetUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    // xyz = target world position, w = indicator radius (world space)
    target_pos_radius: vec4<f32>,
    // rotation matrix column 0 (xyz), w = alpha_scale
    indicator_rot_0: vec4<f32>,
    // rotation matrix column 1 (xyz), w = fog_distance
    indicator_rot_1: vec4<f32>,
    // rotation matrix column 2 (xyz), w = unused
    indicator_rot_2: vec4<f32>,
    // xy = screen_size in pixels, z = point_size, w = line_half_width in pixels
    screen_size_ps: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: TargetUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

struct VertexInput {
    // Per-vertex: x in {-1, 1} selects endpoint A/B, y in {-1, 1} selects side
    @location(0) corner: vec2<f32>,
    // Per-instance: the two endpoints of this edge (xyz = unit compass coords, w = width factor)
    @location(1) endpoint_a: vec4<f32>,
    @location(2) endpoint_b: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) perp_coord: f32, // -1..1 across the ribbon width
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let radius = uniforms.target_pos_radius.w;
    let tgt_pos = uniforms.target_pos_radius.xyz;

    // Rotation matrix from uniforms (aligns Z-up compass to world_up, then spins)
    let rot = mat3x3<f32>(
        uniforms.indicator_rot_0.xyz,
        uniforms.indicator_rot_1.xyz,
        uniforms.indicator_rot_2.xyz,
    );

    // Rotate unit compass vertices, scale, translate
    let a = in.endpoint_a.xyz;
    let b = in.endpoint_b.xyz;
    let rot_a = rot * a;
    let rot_b = rot * b;
    let world_a = tgt_pos + rot_a * radius;
    let world_b = tgt_pos + rot_b * radius;

    // Per-endpoint width factors (w component): 0 = point, 1 = full width
    let width_a = in.endpoint_a.w;
    let width_b = in.endpoint_b.w;

    // Select which endpoint this vertex belongs to
    let is_b = in.corner.x > 0.0;
    let width_factor = select(width_a, width_b, is_b);

    // Project both endpoints to clip space
    let clip_a = uniforms.view_proj * vec4<f32>(world_a, 1.0);
    let clip_b = uniforms.view_proj * vec4<f32>(world_b, 1.0);
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

    // Expand by line width in pixels, scaled by per-endpoint width factor
    let line_half_width = uniforms.screen_size_ps.w * width_factor;
    let pixel_to_ndc = vec2<f32>(2.0 / uniforms.screen_size_ps.x, 2.0 / uniforms.screen_size_ps.y);
    let offset_ndc = perp * in.corner.y * line_half_width * pixel_to_ndc;

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);
    out.perp_coord = in.corner.y;

    return out;
}

// Cyan when unoccluded, warm red-orange when behind geometry
const COLOR_FRONT: vec3<f32> = vec3<f32>(0.0, 1.0, 1.0);
const COLOR_BEHIND: vec3<f32> = vec3<f32>(1.0, 0.3, 0.0);

/// Depth-aware color and opacity. Returns (color, opacity).
/// Reversed-Z: near=1, far=0. Cleared to 0 (no geometry).
fn depth_aware_color(clip_pos: vec4<f32>) -> vec2<f32> {
    let fog_distance = uniforms.indicator_rot_1.w;
    let pixel = vec2<i32>(clip_pos.xy);
    let scene_depth = textureLoad(depth_tex, pixel, 0);

    // scene_depth == 0 means no geometry (clear value). Indicator is in front
    // when its depth is greater than scene depth (reversed-Z: larger = closer).
    if scene_depth <= 0.0 || clip_pos.z > scene_depth {
        return vec2<f32>(0.0, 1.0); // 0 = front color, full opacity
    } else {
        let ndc_behind = scene_depth - clip_pos.z; // positive when behind
        let t = clamp(ndc_behind / fog_distance, 0.0, 1.0);
        return vec2<f32>(1.0, mix(0.2, 0.05, t)); // 1 = behind color, faded
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let alpha_scale = uniforms.indicator_rot_0.w;
    let da = depth_aware_color(in.clip_pos);
    let color = mix(COLOR_FRONT, COLOR_BEHIND, da.x);
    let opacity = da.y;

    // Glow falloff from center of the ribbon
    let dist = abs(in.perp_coord);
    let core_width = 0.4;
    let glow = smoothstep(1.0, core_width, dist);
    let brightness = mix(0.15, 1.0, glow);

    let intensity = brightness * opacity * alpha_scale;
    // Additive blending: output premultiplied color, alpha unused
    return vec4<f32>(color * intensity, 0.0);
}

// ── Filled star polygon ─────────────────────────────────────────────────

struct StarVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_star(@location(0) position: vec3<f32>) -> StarVertexOutput {
    let radius = uniforms.target_pos_radius.w;
    let tgt_pos = uniforms.target_pos_radius.xyz;

    let rot = mat3x3<f32>(
        uniforms.indicator_rot_0.xyz,
        uniforms.indicator_rot_1.xyz,
        uniforms.indicator_rot_2.xyz,
    );

    let world_pos = tgt_pos + rot * position * radius;

    var out: StarVertexOutput;
    out.clip_pos = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_star(in: StarVertexOutput) -> @location(0) vec4<f32> {
    let alpha_scale = uniforms.indicator_rot_0.w;
    let da = depth_aware_color(in.clip_pos);
    let color = mix(COLOR_FRONT, COLOR_BEHIND, da.x);
    let opacity = da.y;

    // Dimmer fill than the wireframe edges so the outline stays visible
    let intensity = 0.5 * opacity * alpha_scale;
    return vec4<f32>(color * intensity, 0.0);
}
