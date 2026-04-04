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
    point_size: f32,
    camera_up: vec3<f32>,
    selected_point_index: u32,
    hovered_point_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Pick ID tag for point entities (bits 31..24).
const PICK_TAG_POINT: u32 = 0x02000000u;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) quad_pos: vec2<f32>,       // quad corner (-1..1)
    @location(1) world_pos: vec3<f32>,      // instance: point position
    @location(2) color_packed: u32,         // instance: packed RGBA8
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) view_depth: f32,
    @location(3) @interpolate(flat) point3d_index: u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Billboard: expand quad corners in the camera-aligned plane
    let offset = uniforms.camera_right * in.quad_pos.x * uniforms.point_size
               + uniforms.camera_up    * in.quad_pos.y * uniforms.point_size;
    let world = vec4<f32>(in.world_pos + offset, 1.0);

    var out: VertexOutput;
    out.clip_pos = uniforms.view_proj * world;
    out.uv = in.quad_pos;

    // Unpack color from u32 (R in low byte, then G, B)
    out.color = vec3<f32>(
        f32((in.color_packed >>  0u) & 0xFFu) / 255.0,
        f32((in.color_packed >>  8u) & 0xFFu) / 255.0,
        f32((in.color_packed >> 16u) & 0xFFu) / 255.0,
    );

    // Linear view-space depth for EDL (positive = in front of camera)
    let view_pos = uniforms.view * vec4<f32>(in.world_pos, 1.0);
    out.view_depth = -view_pos.z;

    out.point3d_index = in.instance_index;

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
    // Highlight selected point in yellow, hovered point in bright cyan.
    var color = in.color;
    if in.point3d_index == uniforms.selected_point_index {
        color = vec3<f32>(1.0, 1.0, 0.0);
    } else if in.point3d_index == uniforms.hovered_point_index {
        color = vec3<f32>(0.0, 1.0, 1.0);
    }
    out.color = vec4<f32>(color, 1.0);
    out.depth = in.view_depth;
    out.pick_id = PICK_TAG_POINT | in.point3d_index;
    return out;
}
