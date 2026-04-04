// Distorted background image shader for camera view mode.
//
// Uses a tessellated mesh with vertex positions as ray directions in world
// space (computed via pixel_to_ray and rotated by the camera-to-world matrix).
// This matches the coordinate convention of frustum wireframes and image quads,
// using the same view_proj = projection * view transform pipeline.

struct BgUniforms {
    view_proj: mat4x4<f32>,
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
    // Vertex positions are world-space ray directions. Using w=0 transforms
    // as a direction (no translation), then we fix depth to the far plane.
    let clip = bg.view_proj * vec4<f32>(position, 0.0);
    // Force depth to far plane (z = w) so the BG is behind all 3D geometry.
    out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(bg_texture, bg_sampler, in.uv);
}
