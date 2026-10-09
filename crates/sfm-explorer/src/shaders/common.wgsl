// Declarations shared by the scene shaders.
//
// The pipelines that use these prepend this file to their own shader source
// (`concat!(include_str!("common.wgsl"), include_str!("<shader>.wgsl"))`), so
// each value is written once. Each shader still declares its own
// `var<uniform> recon: ReconUniforms` binding, because the binding index
// differs between pipelines. A test in `scene_renderer/pipelines/tests.rs`
// checks the `PICK_TAG_*` values here against `scene_renderer/picking.rs`.

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

