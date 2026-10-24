// Point splat rendering shader.
//
// Renders each point as a camera-facing billboard quad. The fragment shader
// discards fragments outside a circular radius to produce smooth circle splats.
// Outputs color, linear view-space depth (for EDL), and a pick ID (for entity
// picking) as three render targets.

struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_right: vec3<f32>,
    _pad0: f32,
    camera_up: vec3<f32>,
    // Global pick indices (recon base + local index), 0xFFFFFFFF = none.
    selected_point_index: u32,
    hovered_point_index: u32,
    screen_width: f32,
    screen_height: f32,
    infinity_point_px: f32,
}

// The pipeline prepends common.wgsl, which declares ReconUniforms, the
// PICK_TAG_* pick ID tags and INF_DEPTH.

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> recon: ReconUniforms;

// Mix this node's tint into a color: a lerp toward the tint by its strength,
// so a == 0 leaves the original untouched and a == 1 flattens it to the tint.
// Every scene shader composites the same way, so points, frustums, image quads
// and patches of one node all read as the same color.
fn tinted(color: vec3<f32>) -> vec3<f32> {
    return mix(color, recon.tint_color.rgb, recon.tint_color.a);
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) quad_pos: vec2<f32>,       // quad corner (-1..1)
    @location(1) world_pos: vec3<f32>,      // instance: point position
    @location(2) color_packed: u32,         // instance: packed RGBA8
    // instance: 0 = the version in view has deleted this point, 1 = it is in
    // the reconstruction, 2 = it is in the reconstruction and the viewport is
    // calling it out, 3 = it is in the reconstruction and the eraser brush has
    // painted it, 4 = it is in the reconstruction and under the eraser brush's
    // circle. Its own buffer, written where an edit or a mark moved and
    // nowhere else, so neither rewrites the instance buffer. A live point the
    // Track Table's filter rejects carries MASK_REJECTED over its word.
    @location(3) alive: u32,
    // 1 for a corner of the Track Table filter's dimmed draw, 0 for the first
    // draw's (common.wgsl).
    @location(15) dimmed_draw: u32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) view_depth: f32,
    // Global pick index of this point: recon.point_pick_base + instance index.
    @location(3) @interpolate(flat) point3d_index: u32,
    // The instance's mask word, 1 to 4 (a deleted point never reaches the
    // fragment shader): which of the viewport's colours, if any, replaces the
    // point's own.
    @location(4) @interpolate(flat) mark: u32,
    // 1, or DIM_ALPHA for a point the Track Table's filter dims.
    @location(5) @interpolate(flat) alpha: f32,
}

fn place(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.quad_pos;
    let word = in.alive & MASK_WORD;
    out.mark = word;
    out.alpha = 1.0;

    // A deleted point emits a clipped vertex, so it draws nothing, occludes
    // nothing and answers no pick — the same treatment a hidden point at
    // infinity gets below. So does a point that is not this draw's: a point
    // the Track Table's filter rejects is drawn only by the dimmed draw, and
    // by neither while it hides them, unless it is the selected or hovered
    // point. That covers a point at infinity too, which takes the same word.
    let global = recon.point_pick_base + in.instance_index;
    let highlighted = global == uniforms.selected_point_index
        || global == uniforms.hovered_point_index;
    if word == MASK_DELETED
        || !in_this_draw(in.alive, in.dimmed_draw != 0u, highlighted, recon.filter_mode) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.view_depth = 0.0;
        out.color = vec3<f32>(0.0);
        out.point3d_index = global;
        return out;
    }
    if in.dimmed_draw != 0u {
        out.alpha = DIM_ALPHA;
    }

    // Unpack color from u32 (R in low byte, then G, B), then tint. The
    // selection and hover overrides in the fragment shader are applied *after*
    // this and so are never tinted: they exist to be told apart.
    out.color = tinted(vec3<f32>(
        f32((in.color_packed >>  0u) & 0xFFu) / 255.0,
        f32((in.color_packed >>  8u) & 0xFFu) / 255.0,
        f32((in.color_packed >> 16u) & 0xFFu) / 255.0,
    ));
    // Global pick index: this node's base plus the local instance index. The
    // instance buffer stores nothing about it, so moving the base is a uniform
    // rewrite and never a buffer rewrite.
    out.point3d_index = recon.point_pick_base + in.instance_index;

    // The alpha byte is the finite/infinity flag: 0 = point at infinity.
    let is_infinity = ((in.color_packed >> 24u) & 0xFFu) == 0u;

    if is_infinity {
        // `world_pos` is a unit direction. Transform with w = 0 so both the
        // node transform's and the camera's translation drop out — a point at
        // infinity has no parallax, and a direction only rotates.
        let clip_c = uniforms.view_proj * (recon.model * vec4<f32>(in.world_pos, 0.0));
        if recon.show_infinity == 0.0 || clip_c.w <= 0.0 {
            // Hidden by the "points at infinity" toggle, or pointing behind the
            // camera: emit a clipped vertex either way.
            out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
            out.view_depth = 0.0;
            return out;
        }
        // Screen-space billboard: offset the projected point by a fixed pixel
        // radius (2 NDC units span the full screen in each axis).
        let ndc = clip_c.xyz / clip_c.w;
        let offset_ndc = in.quad_pos * 2.0 * uniforms.infinity_point_px
            / vec2<f32>(uniforms.screen_width, uniforms.screen_height);
        let ndc_xy = ndc.xy + offset_ndc;
        // Pin depth to just in front of the far plane and undo the divide.
        out.clip_pos = vec4<f32>(ndc_xy * clip_c.w, INF_DEPTH * clip_c.w, clip_c.w);
        // 0.0 view depth = EDL passthrough (a direction has no finite depth).
        out.view_depth = 0.0;
        return out;
    }

    // Finite point: place the point with the node transform, then billboard in
    // the camera-aligned plane of the shared world space.
    let centre = (recon.model * vec4<f32>(in.world_pos, 1.0)).xyz;
    let offset = uniforms.camera_right * in.quad_pos.x * recon.point_size
               + uniforms.camera_up    * in.quad_pos.y * recon.point_size;
    out.clip_pos = uniforms.view_proj * vec4<f32>(centre + offset, 1.0);

    // Linear view-space depth for EDL (positive = in front of camera)
    let view_pos = uniforms.view * vec4<f32>(centre, 1.0);
    out.view_depth = -view_pos.z;

    return out;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    return place(in);
}

// The node a camera model proposal's preview is on: each point is drawn at
// `stored + s * (target - stored)`, normalised for a direction so a point at
// infinity turns along the shorter arc, where its preview row's flag says it
// moves; a point whose row says it loses its position stands where it is and
// takes the failure colour as `s` grows. The pick ID is the same, so a click
// picks a moving point where it is drawn.
@vertex
fn vs_preview(in: VertexInput, @location(5) preview: vec4<f32>) -> VertexOutput {
    var moved = in;
    if preview.w == PREVIEW_MOVES {
        // At s = 1 the target itself, not `mix`'s `a + s·(b − a)`, which can be an
        // ulp off it: the end of the motion is drawn from the very coordinates
        // the version Apply pushes is drawn from.
        let blended = select(
            mix(in.world_pos, preview.xyz, recon.preview_s),
            preview.xyz,
            recon.preview_s >= 1.0,
        );
        let is_infinity = ((in.color_packed >> 24u) & 0xFFu) == 0u;
        if is_infinity {
            // Two opposite directions blend through the origin; keep the
            // target there rather than draw a direction of zero length.
            moved.world_pos = select(
                preview.xyz,
                normalize(blended),
                length(blended) > 1e-6 && recon.preview_s < 1.0,
            );
        } else {
            moved.world_pos = blended;
        }
    }
    var out = place(moved);
    if preview.w == PREVIEW_LOSES {
        out.color = mix(out.color, FAILURE_COLOR, recon.preview_s);
    }
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
    @location(2) pick_id: u32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    // Circle test: discard fragments outside the unit circle.
    // Hard edges here — the EDL post-process provides edge definition,
    // and anti-aliased fringes cause dark halos because the semi-transparent
    // edge fragments occlude solid fragments behind them via the depth buffer.
    let dist_sq = dot(in.uv, in.uv);
    if dist_sq > 1.0 {
        discard;
    }

    var out: FragOutput;
    // Highlight selected point in yellow, hovered point in bright cyan. Both
    // replace the (already tinted) color outright rather than mixing with it:
    // a tint that could drag the highlight toward itself would cost exactly the
    // legibility the highlight exists for.
    // A called-out point takes the hover variant of the same tint: it is the
    // weakest of the three, because the selection and the pointer are about
    // one point and a call-out is about a set.
    // The eraser brush's two reds come first, over the selection too: they
    // say what releasing the stroke will do to the point.
    var color = in.color;
    if in.mark == MASK_PAINTED {
        color = BRUSH_PAINTED_COLOR;
    } else if in.mark == MASK_UNDER_BRUSH {
        color = BRUSH_UNDER_COLOR;
    } else if in.point3d_index == uniforms.selected_point_index {
        color = vec3<f32>(1.0, 1.0, 0.0);
    } else if in.point3d_index == uniforms.hovered_point_index || in.mark == MASK_HIGHLIGHTED {
        color = vec3<f32>(0.0, 1.0, 1.0);
    }
    // Premultiplied, like the pipeline's blend: a dimmed point lets
    // 1 - DIM_ALPHA of what is behind it through.
    out.color = vec4<f32>(color * in.alpha, in.alpha);
    out.depth = in.view_depth;
    // A non-interactive node reads as background rather than passing the pick
    // through to whatever it occludes.
    out.pick_id = select(PICK_TAG_POINT | in.point3d_index, PICK_TAG_NONE, recon.pickable == 0u);
    return out;
}
