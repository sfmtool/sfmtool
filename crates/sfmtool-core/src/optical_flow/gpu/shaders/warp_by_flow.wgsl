// Warp an image by a flow field using bilinear interpolation.
// Each thread handles one output pixel.
//
// Matches the CPU warp_by_flow() in variational.rs:
//   For pixel (col, row): sample tgt_image at (col + 0.5 + flow_u, row + 0.5 + flow_v)
//   using bilinear interpolation with pixel-center-at-0.5 convention.

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> tgt_image: array<f32>;
@group(0) @binding(1) var<storage, read> flow_u: array<f32>;
@group(0) @binding(2) var<storage, read> flow_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> warped: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

fn sample_bilinear(x: f32, y: f32) -> f32 {
    let w = params.width;
    let h = params.height;

    // Pixel-center-at-0.5: convert to grid coordinates
    let gx = x - 0.5;
    let gy = y - 0.5;

    let x0 = i32(floor(gx));
    let y0 = i32(floor(gy));
    let fx = gx - f32(x0);
    let fy = gy - f32(y0);

    let x0c = u32(clamp(x0, 0, i32(w) - 1));
    let x1c = u32(clamp(x0 + 1, 0, i32(w) - 1));
    let y0c = u32(clamp(y0, 0, i32(h) - 1));
    let y1c = u32(clamp(y0 + 1, 0, i32(h) - 1));

    let v00 = tgt_image[y0c * w + x0c];
    let v10 = tgt_image[y0c * w + x1c];
    let v01 = tgt_image[y1c * w + x0c];
    let v11 = tgt_image[y1c * w + x1c];

    return (1.0 - fx) * (1.0 - fy) * v00
         + fx * (1.0 - fy) * v10
         + (1.0 - fx) * fy * v01
         + fx * fy * v11;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }

    let idx = row * params.width + col;
    let dx = flow_u[idx];
    let dy = flow_v[idx];
    let sx = f32(col) + 0.5 + dx;
    let sy = f32(row) + 0.5 + dy;

    warped[idx] = sample_bilinear(sx, sy);
}
