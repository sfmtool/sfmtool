// Distorted background image shader for camera view mode.
//
// Uses a tessellated mesh with vertex positions as ray directions in the owning
// node's own coordinates (computed via pixel_to_ray and rotated by the
// camera-to-world matrix). This matches the coordinate convention of frustum
// wireframes and image quads — including the per-recon `model` matrix, which the
// node transform ("Align to…") writes and which every other pass applies too,
// using the same view_proj = projection * view transform pipeline.

struct BgUniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> bg: BgUniforms;
@group(0) @binding(1) var bg_texture: texture_2d<f32>;
@group(0) @binding(2) var bg_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    // Vertex positions are ray directions in the node's own coordinates. Using
    // w=0 transforms them as directions through both matrices (no translation,
    // and the model matrix's uniform scale cancels in the perspective divide),
    // then we fix depth to the far plane.
    let clip = bg.view_proj * (bg.model * vec4<f32>(position, 0.0));
    // Force depth to far plane (z = w) so the BG is behind all 3D geometry.
    out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(bg_texture, bg_sampler, in.uv);
}
