// Compute image gradients using central differences.
// Each thread handles one pixel. Boundary pixels use clamped access.
//
// Matches the CPU compute_gradients() in dis.rs:
//   grad_x = (right - left) * 0.5
//   grad_y = (down - up) * 0.5

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_y: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }

    let w = params.width;
    let h = params.height;

    let left_col = select(col - 1u, 0u, col == 0u);
    let right_col = select(col + 1u, w - 1u, col + 1u >= w);
    let up_row = select(row - 1u, 0u, row == 0u);
    let down_row = select(row + 1u, h - 1u, row + 1u >= h);

    let left = image[row * w + left_col];
    let right = image[row * w + right_col];
    let up = image[up_row * w + col];
    let down = image[down_row * w + col];

    let idx = row * w + col;
    grad_x[idx] = (right - left) * 0.5;
    grad_y[idx] = (down - up) * 0.5;
}
