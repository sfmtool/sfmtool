// Patch (surfel) shader — textured, world-oriented quads for embedded patches.
//
// Renders each patch-bearing 3D point as a quad expanded in the vertex shader
// from the point position and the patch's u/v half-extent vectors. Bitmaps are
// packed into a 2D texture array atlas (grid of cells per layer), like the
// camera thumbnail atlas. Patches write real hardware depth (correct
// occlusion), the negated view-space depth to the linear depth target (any
// value <= 0 is passed through by the EDL pass, and the depth readback reads
// the magnitude, so Alt+click finds a surfel's depth), and PICK_TAG_POINT |
// point_index so clicking a patch selects its point. A
// selected, hovered or called-out point's surfel draws an outline along its
// quad's edge in the highlight colour, in place of the splat a dot would show,
// and from behind draws that outline alone.
//
// A second pair of entry points, vs_hidden_outline and fs_hidden_outline, draws
// the same outline again after the EDL pass, onto the EDL output, only where the
// scene hides it, at BEHIND_OPACITY: so a highlighted surfel behind other
// geometry still shows where it is. That pipeline tests against the scene's
// depth without writing it and writes no linear depth and no pick id.

struct Uniforms {
    view_proj: mat4x4<f32>,
    atlas_cols: u32,
    atlas_rows: u32,
    patches_per_page: u32,
    patch_scale: f32,      // user scale multiplier on the stored half-vecs
    patch_opacity: f32,    // global opacity multiplier
    alpha_cutoff: f32,     // coverage discard threshold on bitmap alpha
    // Global pick indices of the selected and hovered points, 0xFFFFFFFF =
    // none: their surfels are outlined, never culled as back faces, and never
    // dimmed or hidden by the Track Table's filter.
    selected_point_index: u32,
    hovered_point_index: u32,
    camera_pos: vec3<f32>, // world-space camera position (front-face culling)
}

// The pipeline prepends common.wgsl, which declares ReconUniforms, the
// PICK_TAG_* pick ID tags and INF_DEPTH.

// Width in screen pixels of a highlighted surfel's outline, a band inside the
// quad's edge: the 2.5 px stroke the bench draws the patch being edited with
// (twice `BENCH_LINE_HALF_WIDTH` in `scene_renderer/gpu_types.rs`).
const OUTLINE_WIDTH_PX: f32 = 2.5;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var patch_texture: texture_2d_array<f32>;
@group(0) @binding(2) var patch_sampler: sampler;
@group(0) @binding(3) var<uniform> recon: ReconUniforms;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,     // patch coordinate (s, t) in [-1, 1]
    @location(1) center: vec3<f32>,       // instance: world pos (unit dir when w == 0)
    @location(2) w: f32,                  // instance: 1.0 finite, 0.0 at infinity
    @location(3) u_halfvec: vec3<f32>,    // instance: u axis × half-extent
    @location(4) v_halfvec: vec3<f32>,    // instance: v axis × half-extent
    @location(5) atlas_layer: u32,        // instance: compacted atlas cell index
    @location(6) point_index: u32,        // instance: recon.points index (picking)
    @location(7) alive: u32,              // instance: the point's mask word (common.wgsl)
    // 1 for a corner of the Track Table filter's dimmed draw, 0 for the first
    // draw's (common.wgsl).
    @location(15) dimmed_draw: u32,
}

struct VertexOutput {
    // Invariant, so the hidden-outline pipeline places a surfel's fragments at
    // exactly the depth the surfel pipeline wrote for them: its strict depth
    // test then fails wherever the outline is the nearest thing, and passes
    // only where something nearer hides it.
    @builtin(position) @invariant clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) atlas_layer: u32,
    @location(2) @interpolate(flat) point_index: u32,
    // The mask word: a point the eraser brush has painted, or one under its
    // circle, draws its surfel in the brush's red as its splat does; a
    // called-out one (MASK_HIGHLIGHTED) is outlined in the hover's cyan.
    @location(3) @interpolate(flat) mark: u32,
    // How far toward the failure colour the camera model proposal's preview
    // has taken this surfel: its blend `s` where its point loses its
    // position, 0 otherwise.
    @location(4) @interpolate(flat) failure: f32,
    // 1, or DIM_ALPHA for a surfel whose point the Track Table's filter dims.
    @location(5) @interpolate(flat) alpha: f32,
    // 1 for a surfel seen from behind, which only a highlighted one is drawn
    // as: it draws its outline and none of its texels.
    @location(6) @interpolate(flat) back_facing: u32,
    // The view-space depth of the fragment, positive in front of the camera:
    // the clip-space w of a finite surfel, 0 for one at infinity, which has no
    // finite depth.
    @location(7) view_depth: f32,
}

// Whether the instance's point is selected or hovered, which the Track Table's
// filter never dims or hides.
fn is_selected_or_hovered(in: VertexInput) -> bool {
    let global = recon.point_pick_base + in.point_index;
    return global == uniforms.selected_point_index || global == uniforms.hovered_point_index;
}

// Whether the instance's surfel is outlined: its point is selected, hovered or
// called out (MASK_HIGHLIGHTED).
fn is_outlined(in: VertexInput) -> bool {
    return is_selected_or_hovered(in) || (in.alive & MASK_WORD) == MASK_HIGHLIGHTED;
}

// A vertex outside the clip volume: all four corners of a quad placed here
// collapse to it, so the quad drops out.
fn clipped() -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    return out;
}

fn place(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let word = in.alive & MASK_WORD;
    out.mark = word;
    out.failure = 0.0;
    out.alpha = 1.0;
    out.back_facing = 0u;
    out.view_depth = 0.0;
    // A deleted point's surfel collapses to a clipped vertex, the same way a
    // back-facing one does below. So does one that is not this draw's, by the
    // splats' rule (points.wgsl): a surfel the Track Table's filter rejects is
    // drawn only by the dimmed draw, unless its point is selected or hovered.
    let highlighted = is_selected_or_hovered(in);
    if word == MASK_DELETED
        || !in_this_draw(in.alive, in.dimmed_draw != 0u, highlighted, recon.filter_mode) {
        out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }
    if in.dimmed_draw != 0u {
        out.alpha = DIM_ALPHA;
    }
    // Map patch coordinate to bitmap UV. Columns run with +u: s=-1 -> u=0.
    // Rows run *against* +v: the bitmap was baked with its top scanline toward
    // +v_axis (WarpMap::from_patch steps rows along −v), so t=+1 -> v=0 and
    // t=-1 -> v=1. Hence the y flip (1 - t)/2.
    out.uv = vec2<f32>((in.quad_pos.x + 1.0) * 0.5, (1.0 - in.quad_pos.y) * 0.5);
    out.atlas_layer = in.atlas_layer;
    // Global pick index: this node's base plus the instance's local point index.
    out.point_index = recon.point_pick_base + in.point_index;

    // Move the surfel into the shared world space. The centre transforms with
    // the instance's own w (a point at infinity is a direction, so translation
    // drops out); the half-vecs are directions and always take w = 0.
    let centre = (recon.model * vec4<f32>(in.center, in.w)).xyz;
    let u_halfvec = (recon.model * vec4<f32>(in.u_halfvec, 0.0)).xyz;
    let v_halfvec = (recon.model * vec4<f32>(in.v_halfvec, 0.0)).xyz;

    // Front-face only: cull a patch whose outward normal (u × v) points away
    // from the camera. cross(u_halfvec, v_halfvec) is a positive multiple of
    // the outward normal, so its sign is all we need — no normalize. (The frame
    // is right-handed: outward normal u × v; the raster reverses v, not the
    // normal — see OrientedPatch.) A point at infinity (w == 0) has a normal
    // that faces every observer, so it is never culled and never back-facing.
    // Collapse all 4 corners to a clipped vertex so the quad drops out, except
    // for the surfel of a selected, hovered or called-out point: that one is
    // marked back-facing instead, and the fragment stage draws its outline
    // alone, so the point still shows where it is from behind.
    if in.w != 0.0 {
        let outward = cross(u_halfvec, v_halfvec);
        if dot(outward, uniforms.camera_pos - centre) <= 0.0 {
            if !is_outlined(in) {
                out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
                return out;
            }
            out.back_facing = 1u;
        }
    }

    // Corner in the patch plane: center ± u_halfvec ± v_halfvec (scaled).
    let corner = centre
        + uniforms.patch_scale * (in.quad_pos.x * u_halfvec + in.quad_pos.y * v_halfvec);

    if in.w == 0.0 {
        // Point at infinity: the corner is a direction. Transform with w = 0
        // so the camera translation drops out — no parallax — then pin depth
        // just in front of the far plane, exactly like infinity point splats.
        let clip_c = uniforms.view_proj * vec4<f32>(corner, 0.0);
        if recon.show_infinity == 0.0 || clip_c.w <= 0.0 {
            // Hidden by the "points at infinity" toggle, as the point's splat
            // is (points.wgsl), or pointing behind the camera: emit a clipped
            // vertex either way, so the surfel neither draws nor picks.
            out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
            return out;
        }
        let ndc = clip_c.xyz / clip_c.w;
        out.clip_pos = vec4<f32>(ndc.xy * clip_c.w, INF_DEPTH * clip_c.w, clip_c.w);
        return out;
    }

    // Finite point: plain projective transform of the world-space corner. The
    // projection's w row is (0, 0, -1, 0), so clip w is the view-space depth.
    out.clip_pos = uniforms.view_proj * vec4<f32>(corner, 1.0);
    out.view_depth = out.clip_pos.w;
    return out;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    return place(in);
}

// The node a camera model proposal's preview is on: each surfel moves with its
// point, keeping its orientation, and its half-vectors are scaled by
// `mix(1, size_factor, s)`, so it keeps its angular size and ends at the size
// Apply writes. A surfel whose point loses its position stands where it is and
// takes the failure colour with it.
fn place_preview(in: VertexInput, preview: vec4<f32>, size_factor: f32) -> VertexOutput {
    var moved = in;
    if preview.w == PREVIEW_MOVES {
        // At s = 1 exactly the target, as in points.wgsl.
        let blended = select(
            mix(in.center, preview.xyz, recon.preview_s),
            preview.xyz,
            recon.preview_s >= 1.0,
        );
        if in.w == 0.0 {
            moved.center = select(
                preview.xyz,
                normalize(blended),
                length(blended) > 1e-6 && recon.preview_s < 1.0,
            );
        } else {
            moved.center = blended;
        }
        let scale = select(
            mix(1.0, size_factor, recon.preview_s),
            size_factor,
            recon.preview_s >= 1.0,
        );
        moved.u_halfvec = in.u_halfvec * scale;
        moved.v_halfvec = in.v_halfvec * scale;
    }
    var out = place(moved);
    if preview.w == PREVIEW_LOSES {
        out.failure = recon.preview_s;
    }
    return out;
}

@vertex
fn vs_preview(
    in: VertexInput,
    @location(8) preview: vec4<f32>,
    @location(9) size_factor: f32,
) -> VertexOutput {
    return place_preview(in, preview, size_factor);
}

// The hidden-outline pipeline's vertex stages: the surfel pipeline's own
// placement, for the outlined surfels alone, so each is drawn where and as
// the surfel pipeline drew it (moved by the preview, culled, filtered or
// deleted alike). Every other surfel collapses to a clipped vertex.
@vertex
fn vs_hidden_outline(in: VertexInput) -> VertexOutput {
    if !is_outlined(in) {
        return clipped();
    }
    return place(in);
}

@vertex
fn vs_hidden_outline_preview(
    in: VertexInput,
    @location(8) preview: vec4<f32>,
    @location(9) size_factor: f32,
) -> VertexOutput {
    if !is_outlined(in) {
        return clipped();
    }
    return place_preview(in, preview, size_factor);
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) linear_depth: f32,
    @location(2) pick_id: u32,
}

// How far a fragment is from the quad's nearest edge, in screen pixels: each
// patch coordinate over the length of its screen-space gradient, so the
// outline keeps its width in pixels at any distance and also where the quad is
// foreshortened. Derivatives are defined only in uniform control flow, so both
// fragment stages call this first, before any discard or branch.
fn edge_distance_px(uv: vec2<f32>) -> f32 {
    let du = length(vec2<f32>(dpdx(uv.x), dpdy(uv.x)));
    let dv = length(vec2<f32>(dpdx(uv.y), dpdy(uv.y)));
    return min(min(uv.x, 1.0 - uv.x) / du, min(uv.y, 1.0 - uv.y) / dv);
}

// The colour of a highlighted surfel's outline: the splat's precedence, the
// eraser brush's reds first, then the selection's yellow over the hover's cyan,
// which a call-out takes too.
fn outline_color(mark: u32, selected: bool) -> vec3<f32> {
    if mark == MASK_PAINTED {
        return BRUSH_PAINTED_COLOR;
    } else if mark == MASK_UNDER_BRUSH {
        return BRUSH_UNDER_COLOR;
    }
    return select(HOVERED_COLOR, SELECTED_COLOR, selected);
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let edge_px = edge_distance_px(in.uv);

    // Compute atlas UV and layer from grid position
    let page = in.atlas_layer / uniforms.patches_per_page;
    let idx_in_page = in.atlas_layer % uniforms.patches_per_page;
    let col = idx_in_page % uniforms.atlas_cols;
    let row = idx_in_page / uniforms.atlas_cols;
    let cell_size = vec2<f32>(1.0 / f32(uniforms.atlas_cols), 1.0 / f32(uniforms.atlas_rows));
    let atlas_uv = (vec2<f32>(f32(col), f32(row)) + in.uv) * cell_size;
    let texel = textureSample(patch_texture, patch_sampler, atlas_uv, page);

    // A selected, hovered or called-out surfel draws a band along its quad's
    // edge in the highlight colour.
    let selected = in.point_index == uniforms.selected_point_index;
    let hovered = in.point_index == uniforms.hovered_point_index || in.mark == MASK_HIGHLIGHTED;
    let outline = (selected || hovered) && edge_px < OUTLINE_WIDTH_PX;

    // The bitmap alpha is per-pixel cross-view confidence, not opacity — treat
    // it as coverage so ragged edges and unfilled corners drop out and the quad
    // reads as a fitted surfel, not a hard rectangle. The outline is exempt, so
    // it traces the whole square whatever the edge cutoff carves from it. A
    // surfel seen from behind, which only a highlighted one is drawn as, keeps
    // its outline and none of its texels.
    if !outline && (texel.a < uniforms.alpha_cutoff || in.back_facing != 0u) {
        discard;
    }

    var out: FragOutput;
    // Tint the surfel's texel before the opacity multiply — the tint is a
    // color, `patch_opacity` a translucency, and mixing in the wrong order
    // would make a tinted patch's strength depend on how transparent it is.
    var rgb = mix(texel.rgb, recon.tint_color.rgb, recon.tint_color.a);
    rgb = mix(rgb, FAILURE_COLOR, in.failure);
    // The eraser brush's reds replace the texel outright, as on a splat, and
    // come before the highlight there too, outline included.
    if outline {
        rgb = outline_color(in.mark, selected);
    } else if in.mark == MASK_PAINTED {
        rgb = BRUSH_PAINTED_COLOR;
    } else if in.mark == MASK_UNDER_BRUSH {
        rgb = BRUSH_UNDER_COLOR;
    }
    // Premultiplied alpha: hard coverage from the discard above; the global
    // patch_opacity uniform is the translucency source, and the filter's dim
    // alpha on top of it. The outline is drawn opaque, so it reads over any
    // patch opacity.
    let opacity = select(uniforms.patch_opacity * in.alpha, 1.0, outline);
    out.color = vec4<f32>(rgb * opacity, opacity);
    // Negated, so the EDL pass passes the textured surface through without edge
    // darkening (it shades only depth > 0), while the depth readback, which
    // reads the magnitude, still finds the surfel's depth for Alt+click.
    out.linear_depth = -in.view_depth;
    // A non-interactive node reads as background rather than passing the pick
    // through to whatever it occludes.
    out.pick_id = select(PICK_TAG_POINT | in.point_index, PICK_TAG_NONE, recon.pickable == 0u);
    return out;
}

// The hidden part of a highlighted surfel's outline, drawn after the EDL pass
// onto its output. The pipeline's depth test, `Less` against the scene's
// reversed-Z depth with no write, passes only where something nearer than the
// outline was drawn, so the part fs_main drew in full is not drawn again. The
// band is the one fs_main draws, in its colours, at BEHIND_OPACITY in place of
// opaque, and straight alpha, as the bench figure draws through the scene. It
// writes the colour alone: no linear depth, which would change the EDL pass's
// shading, and no pick id, which would change what a click picks.
@fragment
fn fs_hidden_outline(in: VertexOutput) -> @location(0) vec4<f32> {
    if edge_distance_px(in.uv) >= OUTLINE_WIDTH_PX {
        discard;
    }
    let selected = in.point_index == uniforms.selected_point_index;
    return vec4<f32>(outline_color(in.mark, selected), BEHIND_OPACITY);
}
