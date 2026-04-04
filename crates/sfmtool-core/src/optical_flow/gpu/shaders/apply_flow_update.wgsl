// Apply Jacobi solver result to the flow field: flow += du.
// Each thread handles one pixel.

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> flow_u: array<f32>;
@group(0) @binding(1) var<storage, read_write> flow_v: array<f32>;
@group(0) @binding(2) var<storage, read> du: array<f32>;
@group(0) @binding(3) var<storage, read> dv: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }
    let idx = row * params.width + col;
    flow_u[idx] += du[idx];
    flow_v[idx] += dv[idx];
}
