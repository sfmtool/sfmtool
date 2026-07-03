// Patch (surfel) shader — textured, world-oriented quads for embedded patches.
//
// Renders each patch-bearing 3D point as a quad expanded in the vertex shader
// from the point position and the patch's u/v half-extent vectors. Bitmaps are
// packed into a 2D texture array atlas (grid of cells per layer), like the
// camera thumbnail atlas. Like image quads, patches write real hardware depth
// (correct occlusion) but 0.0 linear depth (EDL passthrough), and write
// PICK_TAG_POINT | point_index so clicking a patch selects its point.

struct Uniforms {
    view_proj: mat4x4<f32>,
    atlas_cols: u32,
    atlas_rows: u32,
    patches_per_page: u32,
    patch_scale: f32,      // user scale multiplier on the stored half-vecs
    patch_opacity: f32,    // global opacity multiplier
    alpha_cutoff: f32,     // coverage discard threshold on bitmap alpha
    camera_pos: vec3<f32>, // world-space camera position (front-face culling)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var patch_texture: texture_2d_array<f32>;
@group(0) @binding(2) var patch_sampler: sampler;

// Pick ID tag for 3D point entities (bits 31..24). A patch and its point are
// the same entity, so patches ride the existing point pick path.
const PICK_TAG_POINT: u32 = 0x02000000u;

// Tiny positive NDC depth so an infinity patch sits just in front of the
// reversed-Z far plane (cleared to 0.0, compared with Greater): it passes the
// depth test against the cleared background but loses to all finite geometry.
const INF_DEPTH: f32 = 1e-6;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,     // patch coordinate (s, t) in [-1, 1]
    @location(1) center: vec3<f32>,       // instance: world pos (unit dir when w == 0)
    @location(2) w: f32,                  // instance: 1.0 finite, 0.0 at infinity
    @location(3) u_halfvec: vec3<f32>,    // instance: u axis × half-extent
    @location(4) v_halfvec: vec3<f32>,    // instance: v axis × half-extent
    @location(5) atlas_layer: u32,        // instance: compacted atlas cell index
    @location(6) point_index: u32,        // instance: recon.points index (picking)
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) atlas_layer: u32,
    @location(2) @interpolate(flat) point_index: u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Map patch coordinate to bitmap UV: (-1,-1)->(0,0), (1,1)->(1,1)
    out.uv = (in.quad_pos + vec2<f32>(1.0, 1.0)) * 0.5;
    out.atlas_layer = in.atlas_layer;
    out.point_index = in.point_index;

    // Front-face only: cull a patch whose outward normal (v × u) points away
    // from the camera. cross(v_halfvec, u_halfvec) is a positive multiple of
    // the outward normal, so its sign is all we need — no normalize. (The frame
    // is image-raster handed, so the outward normal is v × u, not u × v; see
    // OrientedPatch.) A point at infinity (w == 0) has a normal that faces every
    // observer, so it is never culled. Collapse all 4 corners to a clipped
    // vertex so the quad drops out.
    if in.w != 0.0 {
        let outward = cross(in.v_halfvec, in.u_halfvec);
        if dot(outward, uniforms.camera_pos - in.center) <= 0.0 {
            out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
            return out;
        }
    }

    // Corner in the patch plane: center ± u_halfvec ± v_halfvec (scaled).
    let corner = in.center
        + uniforms.patch_scale * (in.quad_pos.x * in.u_halfvec + in.quad_pos.y * in.v_halfvec);

    if in.w == 0.0 {
        // Point at infinity: the corner is a direction. Transform with w = 0
        // so the camera translation drops out — no parallax — then pin depth
        // just in front of the far plane, exactly like infinity point splats.
        let clip_c = uniforms.view_proj * vec4<f32>(corner, 0.0);
        if clip_c.w <= 0.0 {
            // Direction points behind the camera: emit a clipped vertex.
            out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
            return out;
        }
        let ndc = clip_c.xyz / clip_c.w;
        out.clip_pos = vec4<f32>(ndc.xy * clip_c.w, INF_DEPTH * clip_c.w, clip_c.w);
        return out;
    }

    // Finite point: plain projective transform of the world-space corner.
    out.clip_pos = uniforms.view_proj * vec4<f32>(corner, 1.0);
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
    let page = in.atlas_layer / uniforms.patches_per_page;
    let idx_in_page = in.atlas_layer % uniforms.patches_per_page;
    let col = idx_in_page % uniforms.atlas_cols;
    let row = idx_in_page / uniforms.atlas_cols;
    let cell_size = vec2<f32>(1.0 / f32(uniforms.atlas_cols), 1.0 / f32(uniforms.atlas_rows));
    let atlas_uv = (vec2<f32>(f32(col), f32(row)) + in.uv) * cell_size;
    let texel = textureSample(patch_texture, patch_sampler, atlas_uv, page);

    // The bitmap alpha is per-pixel cross-view confidence, not opacity — treat
    // it as coverage so ragged edges and unfilled corners drop out and the quad
    // reads as a fitted surfel, not a hard rectangle.
    if texel.a < uniforms.alpha_cutoff {
        discard;
    }

    var out: FragOutput;
    // Premultiplied alpha: hard coverage from the discard above; the global
    // patch_opacity uniform is the only translucency source.
    out.color = vec4<f32>(texel.rgb * uniforms.patch_opacity, uniforms.patch_opacity);
    out.linear_depth = 0.0; // EDL passthrough — textured surface, no edge darkening
    out.pick_id = PICK_TAG_POINT | in.point_index;
    return out;
}
