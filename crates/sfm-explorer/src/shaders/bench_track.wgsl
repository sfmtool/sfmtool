// Bench track shader.
//
// Draws the 3D view's overlay figures out of the overlay ribbon store: the
// focused item where it stands in the world (the patch frame's square, its
// outward normal with an arrowhead, one mark per observation) and the drawn
// measurements' marks and lines; and the bench figure's filled centre disc,
// from a vertex buffer of its own. Each overlay piece is one instance: a
// segment, drawn as a ribbon facing the eye, or a ring or a disc, drawn as a
// quad facing the eye with its shape cut out of it in the fragment shader.
// Sizes are in panel points, and edges are anti-aliased over about one point.
// Rendered as post-EDL passes onto edl_output and into the measurements'
// layer.
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
    proj: mat4x4<f32>,
    screen_size: vec2<f32>,
    near: f32,
    // World units: how far behind the scene the figure fades to the floor.
    fog_distance: f32,
    // Physical pixels to a panel point.
    pixels_per_point: f32,
    // The fraction of its distance to the eye a PULL_TOWARD_EYE piece is
    // drawn toward it.
    pull_fraction: f32,
}

// The pipeline prepends common.wgsl, which declares ReconUniforms, the
// PICK_TAG_* pick ID tags and INF_DEPTH.

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

// The figure's opacity the moment it passes behind scene geometry, before the
// fog, is BEHIND_OPACITY in common.wgsl.

// The floor it fades to and never below: a handle that cannot be seen cannot
// be grabbed.
const FLOOR_OPACITY: f32 = 0.15;

// OverlayInstance::kind.
const KIND_SEGMENT: u32 = 0u;
const KIND_RING: u32 = 1u;

// OverlayInstance::flags. HOLD is FADE's absence.
const FLAG_FADE: u32 = 1u;
const FLAG_PULL_TOWARD_EYE: u32 = 2u;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    // 1.0 for a direction, whose occlusion has no distance to fade over.
    @location(1) @interpolate(flat) at_infinity: f32,
    // 1.0 where the part behind the scene fades with distance (the bench
    // figure), 0.0 where it holds at BEHIND_OPACITY (a measurement).
    @location(2) @interpolate(flat) fade: f32,
    // Where the fragment lies, in pixels: across a segment from its middle in
    // y, and from a ring's or a disc's centre in x and y. Interpolated across
    // the screen, not in perspective: a segment's two ends are at different
    // depths, and a perspective-correct interpolation would move its middle
    // off the line between them.
    @location(3) @interpolate(linear) local: vec2<f32>,
    // The piece's kind, its radius and its half width, the last two in
    // pixels.
    @location(4) @interpolate(flat) shape: vec3<f32>,
}

// A vertex outside every clip plane, which drops its whole primitive.
fn clipped(color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    out.color = color;
    out.at_infinity = 0.0;
    out.fade = 1.0;
    out.local = vec2<f32>(0.0, 0.0);
    out.shape = vec3<f32>(0.0, 0.0, 0.0);
    return out;
}

struct OverlayInput {
    // Per-vertex: x in {-1, 1} selects end A or B of a segment and y in
    // {-1, 1} its side; together they are a ring's or a disc's quad corner.
    @location(0) corner: vec2<f32>,
    // Per-instance: OverlayInstance.
    @location(1) a: vec4<f32>,
    @location(2) b: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) half_width: f32,
    @location(5) radius: f32,
    @location(6) kind: u32,
    @location(7) flags: u32,
}

// A homogeneous world point in view space, drawn toward the eye by the pull
// when `pull` is set. The eye is the view's origin, so scaling a place toward
// it moves the place along its own line of sight and it projects where it
// did; scaling a direction does not move where it projects.
fn to_view(p: vec4<f32>, pull: bool) -> vec4<f32> {
    let v = uniforms.view * p;
    if pull {
        return vec4<f32>(v.xyz * (1.0 - uniforms.pull_fraction), v.w);
    }
    return v;
}

// A view-space homogeneous point in clip space. A direction is pinned at
// INF_DEPTH, just in front of the reversed-Z far plane, as the points and
// patches at infinity are.
fn to_clip(v: vec4<f32>) -> vec4<f32> {
    let clip = uniforms.proj * v;
    if v.w == 0.0 {
        return vec4<f32>(clip.xy, INF_DEPTH * clip.w, clip.w);
    }
    return clip;
}

// How far in front of the near plane a view-space homogeneous point is, times
// its w: positive in front. A direction is in front when it points forward.
fn in_front(v: vec4<f32>) -> f32 {
    return -v.z - uniforms.near * v.w;
}

@vertex
fn vs_overlay(in: OverlayInput) -> VertexOutput {
    let pull = (in.flags & FLAG_PULL_TOWARD_EYE) != 0u;
    let px = uniforms.pixels_per_point;
    let half_px = in.half_width * px;
    let radius_px = in.radius * px;
    // The quad reaches a point past the shape's edge, for the anti-aliasing
    // to fall off over.
    let aa_px = px;
    let pixel_to_ndc = vec2<f32>(2.0 / uniforms.screen_size.x, 2.0 / uniforms.screen_size.y);

    var out: VertexOutput;
    out.color = in.color;
    out.fade = select(0.0, 1.0, (in.flags & FLAG_FADE) != 0u);
    out.shape = vec3<f32>(f32(in.kind), radius_px, half_px);

    if in.kind != KIND_SEGMENT {
        // A ring or a disc: a quad facing the eye about its projected centre.
        // One whose centre is behind the near plane draws nothing.
        let v = to_view(in.a, pull);
        if in_front(v) <= 0.0 {
            return clipped(in.color);
        }
        let centre = to_clip(v);
        let extent_px = radius_px + half_px + aa_px;
        let offset_ndc = in.corner * extent_px * pixel_to_ndc;
        out.clip_pos = vec4<f32>(centre.xy + offset_ndc * centre.w, centre.zw);
        out.at_infinity = select(0.0, 1.0, in.a.w == 0.0);
        out.local = in.corner * extent_px;
        return out;
    }

    var va = to_view(in.a, pull);
    var vb = to_view(in.b, pull);
    var at_infinity = 0.0;
    if in.a.w == 0.0 && in.b.w == 0.0 {
        // A figure at infinity, both ends directions. One pointing behind the
        // eye drops the edge: two directions have no place between them to
        // clip at.
        at_infinity = 1.0;
        if va.z >= 0.0 || vb.z >= 0.0 {
            return clipped(in.color);
        }
    } else {
        // Clipped at the near plane in homogeneous view space, before the
        // perspective divide, so an end with w = 0 stays a direction rather
        // than being divided by zero. See frustum.wgsl for why a segment is
        // clipped here at all.
        let fa = in_front(va);
        let fb = in_front(vb);
        if fa <= 0.0 && fb <= 0.0 {
            return clipped(in.color);
        }
        let t = fa / (fa - fb);
        if fa <= 0.0 {
            va = mix(va, vb, t);
        } else if fb <= 0.0 {
            vb = mix(va, vb, t);
        }
    }
    let clip_a = to_clip(va);
    let clip_b = to_clip(vb);

    let is_b = in.corner.x > 0.0;
    let clip_pos = select(clip_a, clip_b, is_b);

    // The segment's direction in pixels, for the perpendicular the ribbon
    // expands along.
    let ndc_a = clip_a.xy / clip_a.w;
    let ndc_b = clip_b.xy / clip_b.w;
    let edge_px = (ndc_b - ndc_a) * uniforms.screen_size * 0.5;
    let edge_len = length(edge_px);
    var perp: vec2<f32>;
    if edge_len > 0.0001 {
        let edge_dir = edge_px / edge_len;
        perp = vec2<f32>(-edge_dir.y, edge_dir.x);
    } else {
        perp = vec2<f32>(0.0, 1.0);
    }

    let extent_px = half_px + aa_px;
    let offset_ndc = perp * in.corner.y * extent_px * pixel_to_ndc;
    out.clip_pos = vec4<f32>(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);
    out.at_infinity = at_infinity;
    out.local = vec2<f32>(0.0, in.corner.y * extent_px);
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
    out.fade = 1.0;
    out.local = vec2<f32>(0.0, 0.0);
    out.shape = vec3<f32>(0.0, 0.0, 0.0);

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
//
// A piece that does not fade, a measurement's, is drawn at BEHIND_OPACITY
// wherever it is behind, however far, as a highlighted surfel's hidden outline
// is: a measurement far behind the scene stays as findable as one just behind.
fn depth_aware_opacity(clip_pos: vec4<f32>, at_infinity: f32, fade: f32) -> f32 {
    let scene_depth = textureLoad(depth_tex, vec2<i32>(clip_pos.xy), 0);
    if scene_depth <= 0.0 || clip_pos.z >= scene_depth {
        return 1.0;
    }
    if fade == 0.0 {
        return BEHIND_OPACITY;
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

// The centre disc: solid, as its triangles cover it.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let opacity = depth_aware_opacity(in.clip_pos, in.at_infinity, in.fade);
    return vec4<f32>(in.color.rgb, in.color.a * opacity);
}

// How much of its pixel an overlay piece covers: 1 inside its shape and 0 a
// half point past its edge, falling linearly over the point centred on the
// edge.
fn coverage(in: VertexOutput) -> f32 {
    let kind = u32(in.shape.x + 0.5);
    let radius_px = in.shape.y;
    let half_px = in.shape.z;
    var outside: f32;
    if kind == KIND_SEGMENT {
        outside = abs(in.local.y) - half_px;
    } else if kind == KIND_RING {
        outside = abs(length(in.local) - radius_px) - half_px;
    } else {
        // A disc.
        outside = length(in.local) - radius_px;
    }
    return clamp(0.5 - outside / uniforms.pixels_per_point, 0.0, 1.0);
}

// An overlay piece: its shape's coverage, times how strongly the scene lets
// it through.
@fragment
fn fs_overlay(in: VertexOutput) -> @location(0) vec4<f32> {
    let covered = coverage(in);
    if covered <= 0.0 {
        discard;
    }
    let opacity = depth_aware_opacity(in.clip_pos, in.at_infinity, in.fade);
    return vec4<f32>(in.color.rgb, in.color.a * opacity * covered);
}
