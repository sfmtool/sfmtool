// Bench track shader.
//
// Draws the bench's active track where it stands in the world: the patch
// frame's square, its outward normal with an arrowhead, one mark per
// observation, and the filled centre disc. Each edge is a camera-facing ribbon
// quad carrying its own colour; the disc is a triangle list. Rendered as a
// post-EDL pass onto edl_output.
//
// Two things differ from the target indicator, which this otherwise follows.
// The blend is ordinary alpha rather than additive and there is no glow: the
// three violets carry meaning, and additive light over a bright point cloud
// washes them all to white. And the hue does not shift when occluded, for the
// same reason -- a verdict's colour stays that verdict's colour, and only the
// opacity says the figure is behind something.

struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    screen_size: vec2<f32>,
    line_half_width: f32,
    near: f32,
    // World units: how far behind the scene the figure fades to the floor.
    fog_distance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

// Opacity the moment the figure passes behind scene geometry, before the fog.
const BEHIND_OPACITY: f32 = 0.45;

// The floor it fades to and never below: a handle that cannot be seen cannot
// be grabbed.
const FLOOR_OPACITY: f32 = 0.15;

// Tiny positive NDC depth, so a direction sits just in front of the reversed-Z
// far plane (cleared to 0.0): it reads as behind every finite thing and in
// front of the empty background, which is what a direction is. The same
// constant points.wgsl and patch.wgsl pin their infinity geometry at.
const INF_DEPTH: f32 = 1e-6;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    // 1.0 for a direction, whose occlusion has no distance to fade over.
    @location(1) @interpolate(flat) at_infinity: f32,
}

// A vertex outside every clip plane, which drops its whole primitive.
fn clipped(color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    out.color = color;
    out.at_infinity = 0.0;
    return out;
}

struct EdgeInput {
    // Per-vertex: x in {-1, 1} selects endpoint A/B, y in {-1, 1} selects side
    @location(0) corner: vec2<f32>,
    // Per-instance: the two homogeneous endpoints and the colour
    @location(1) endpoint_a: vec4<f32>,
    @location(2) endpoint_b: vec4<f32>,
    @location(3) color: vec4<f32>,
}

@vertex
fn vs_edge(in: EdgeInput) -> VertexOutput {
    var clip_a: vec4<f32>;
    var clip_b: vec4<f32>;
    var at_infinity = 0.0;

    // A figure is finite throughout or a direction throughout, so one endpoint
    // decides which projection the edge takes.
    if in.endpoint_a.w == 0.0 {
        at_infinity = 1.0;
        // Rotation-only: the translation drops out with w = 0, so the figure
        // has no parallax and stays on the sky as the viewer moves.
        clip_a = uniforms.view_proj * vec4<f32>(in.endpoint_a.xyz, 0.0);
        clip_b = uniforms.view_proj * vec4<f32>(in.endpoint_b.xyz, 0.0);
        if clip_a.w <= 0.0 || clip_b.w <= 0.0 {
            return clipped(in.color);
        }
        clip_a = vec4<f32>(clip_a.xy, INF_DEPTH * clip_a.w, clip_a.w);
        clip_b = vec4<f32>(clip_b.xy, INF_DEPTH * clip_b.w, clip_b.w);
    } else {
        // Clip the edge against the near plane in view space before the manual
        // perspective divide. See frustum.wgsl for a full explanation.
        let near_z = -uniforms.near;
        let view_a_z = (uniforms.view * vec4<f32>(in.endpoint_a.xyz, 1.0)).z;
        let view_b_z = (uniforms.view * vec4<f32>(in.endpoint_b.xyz, 1.0)).z;
        let a_in_front = view_a_z < near_z;
        let b_in_front = view_b_z < near_z;
        if !a_in_front && !b_in_front {
            return clipped(in.color);
        }
        var world_a = in.endpoint_a.xyz;
        var world_b = in.endpoint_b.xyz;
        let t = (near_z - view_a_z) / (view_b_z - view_a_z);
        if !a_in_front {
            world_a = mix(world_a, world_b, t);
        } else if !b_in_front {
            world_b = mix(world_a, world_b, t);
        }
        clip_a = uniforms.view_proj * vec4<f32>(world_a, 1.0);
        clip_b = uniforms.view_proj * vec4<f32>(world_b, 1.0);
    }

    let is_b = in.corner.x > 0.0;
    let clip_pos = select(clip_a, clip_b, is_b);

    // Edge direction in NDC, for the perpendicular the ribbon expands along.
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

    // Expand by the line width in pixels, converted to NDC.
    let pixel_to_ndc = vec2<f32>(2.0 / uniforms.screen_size.x, 2.0 / uniforms.screen_size.y);
    let offset_ndc = perp * in.corner.y * uniforms.line_half_width * pixel_to_ndc;

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);
    out.color = in.color;
    out.at_infinity = at_infinity;
    return out;
}

struct DiscInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_disc(in: DiscInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = in.color;
    out.at_infinity = 0.0;

    if in.position.w == 0.0 {
        let clip = uniforms.view_proj * vec4<f32>(in.position.xyz, 0.0);
        if clip.w <= 0.0 {
            return clipped(in.color);
        }
        out.clip_pos = vec4<f32>(clip.xy, INF_DEPTH * clip.w, clip.w);
        out.at_infinity = 1.0;
        return out;
    }

    // A triangle cannot be clipped a vertex at a time the way a segment can, so
    // one corner behind the near plane drops the whole triangle rather than
    // letting the divide fling it across the screen. The disc is an eighth of
    // the frame's half-length across, so the three corners agree in every view
    // but a grazing one.
    if (uniforms.view * vec4<f32>(in.position.xyz, 1.0)).z >= -uniforms.near {
        return clipped(in.color);
    }
    out.clip_pos = uniforms.view_proj * vec4<f32>(in.position.xyz, 1.0);
    return out;
}

// How strongly this fragment is drawn, given what the scene has at its pixel.
//
// Reversed-Z: near = 1, far = 0, cleared to 0 where there is no geometry. Under
// the infinite reversed-Z projection ndc_z = near / view_depth, so dividing the
// two depths back out states the gap in world units and the fog distance is the
// patch's own size rather than a slice of the depth range.
fn depth_aware_opacity(clip_pos: vec4<f32>, at_infinity: f32) -> f32 {
    let scene_depth = textureLoad(depth_tex, vec2<i32>(clip_pos.xy), 0);
    if scene_depth <= 0.0 || clip_pos.z >= scene_depth {
        return 1.0;
    }
    if at_infinity > 0.0 {
        // A direction is behind every finite thing, so there is no distance to
        // fade over: wherever the scene has depth it is drawn through it at the
        // floor, and at full strength against the background above.
        return FLOOR_OPACITY;
    }
    let behind = uniforms.near / clip_pos.z - uniforms.near / scene_depth;
    return max(BEHIND_OPACITY * exp(-behind / uniforms.fog_distance), FLOOR_OPACITY);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let opacity = depth_aware_opacity(in.clip_pos, in.at_infinity);
    return vec4<f32>(in.color.rgb, in.color.a * opacity);
}
