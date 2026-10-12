// Declarations shared by the scene shaders.
//
// The pipelines that use these prepend this file to their own shader source
// (`concat!(include_str!("common.wgsl"), include_str!("<shader>.wgsl"))`), so
// each value is written once. Each shader still declares its own
// `var<uniform> recon: ReconUniforms` binding, because the binding index
// differs between pipelines. A test in `scene_renderer/pipelines/tests.rs`
// checks the `PICK_TAG_*` values here against `scene_renderer/picking.rs`, and
// the `MASK_*` values against `scene_renderer/upload/overlay.rs`.

// Per-reconstruction block: which node this draw belongs to. Mirrors
// `ReconUniforms` in `scene_renderer/gpu_types.rs`; every scene shader reads
// the same per-recon buffer through it.
struct ReconUniforms {
    model: mat4x4<f32>,
    point_size: f32,
    point_pick_base: u32,
    image_pick_base: u32,
    pickable: u32,
    // Node tint: rgb is the palette color, a its strength. a == 0 = original.
    tint_color: vec4<f32>,
    // Effective "points at infinity" visibility for this node: the global HUD
    // toggle AND the node's own ∞ mini-toggle. Only points.wgsl reads it.
    show_infinity: f32,
    // The camera model proposal's preview blend on this node: 0 draws the
    // stored positions, 1 the targets. Read only by the `vs_preview` entry
    // points of points.wgsl and patch.wgsl.
    preview_s: f32,
}

// Pick ID tags (bits 31..30), matching `PICK_TAG_*` in
// `scene_renderer/picking.rs`.
// Pick ID for "nothing" — what a non-pickable node emits.
const PICK_TAG_NONE: u32 = 0x00000000u;
// Tag for frustum and camera image entities.
const PICK_TAG_FRUSTUM: u32 = 0x40000000u;
// Tag for 3D point entities. A patch and its point are the same entity, so
// patches use this tag too.
const PICK_TAG_POINT: u32 = 0x80000000u;

// Tiny positive NDC depth so geometry at infinity sits just in front of the
// reversed-Z far plane (cleared to 0.0, compared with Greater): it passes the
// depth test against the cleared background but loses to all finite geometry.
const INF_DEPTH: f32 = 1e-6;

// Per-point mask words, matching `MASK_*` in `scene_renderer/upload/overlay.rs`:
// the second instance buffer the point and patch pipelines step with their
// instances carries one per point.
const MASK_DELETED: u32 = 0x00000000u;
const MASK_ALIVE: u32 = 0x00000001u;
const MASK_HIGHLIGHTED: u32 = 0x00000002u;
const MASK_PAINTED: u32 = 0x00000003u;
const MASK_UNDER_BRUSH: u32 = 0x00000004u;

// The eraser brush's colours, which replace a point's own outright as the
// selection's yellow and the hover's cyan do. A saturated red for a point the
// stroke being held has painted, and a pale red for one under the circle
// before a press: the same hue, so the preview reads as what a press would do,
// and lighter, so it reads as not done yet.
const BRUSH_PAINTED_COLOR: vec3<f32> = vec3<f32>(1.0, 0.1, 0.1);
const BRUSH_UNDER_COLOR: vec3<f32> = vec3<f32>(1.0, 0.6, 0.6);

// The camera model proposal's preview flags, matching `FLAG_*` in
// `camera_model_proposal/preview.rs`: the `w` of a preview buffer row. A row
// the preview does not move, one it moves to its target, and one that stays
// where it stands and takes the failure colour.
const PREVIEW_STILL: f32 = 0.0;
const PREVIEW_MOVES: f32 = 1.0;
const PREVIEW_LOSES: f32 = 2.0;

// The failure colour, `FAILURE_RGB` in `scene.rs`: a finite point the
// proposed camera's retriangulation turns into a direction is drawn in
// `mix(colour, FAILURE_COLOR, preview_s)`, over its tint.
const FAILURE_COLOR: vec3<f32> = vec3<f32>(235.0 / 255.0, 30.0 / 255.0, 30.0 / 255.0);
