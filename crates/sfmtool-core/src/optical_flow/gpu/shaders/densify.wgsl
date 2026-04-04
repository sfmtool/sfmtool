// Gather-based flow densification — one thread per output pixel (Option B from spec).
//
// For each output pixel, finds all overlapping patches on the regular grid,
// computes photometric-error-weighted contributions, and normalizes.
//
// Matches the CPU densify_flow() in interp.rs:
//   weight = 1 / max(1, |tgt_val - ref_val|)
//   flow = sum(weight * patch_flow) / sum(weight)

struct Params {
    width: u32,
    height: u32,
    patch_size: u32,
    stride: u32,
    num_patches_x: u32,
    num_patches_y: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> ref_image: array<f32>;
@group(0) @binding(1) var<storage, read> tgt_image: array<f32>;
@group(0) @binding(2) var<storage, read> patch_flow_u: array<f32>;
@group(0) @binding(3) var<storage, read> patch_flow_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> flow_u: array<f32>;
@group(0) @binding(5) var<storage, read_write> flow_v: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

fn sample_bilinear(x: f32, y: f32) -> f32 {
    let w = params.width;
    let h = params.height;

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

    let ps = params.patch_size;
    let stride = params.stride;
    let w = params.width;

    // Find range of patch column indices whose patches can cover this pixel.
    // Patch (pc, pr) has grid_x = pc * stride, covers [grid_x, grid_x + ps).
    // We need: pc * stride <= col AND pc * stride + ps > col.
    var min_pc: u32 = 0u;
    if col >= ps {
        // Ceiling division: ceil((col + 1 - ps) / stride)
        min_pc = (col + 1u - ps + stride - 1u) / stride;
    }
    let max_pc = min(col / stride, params.num_patches_x - 1u);

    var min_pr: u32 = 0u;
    if row >= ps {
        min_pr = (row + 1u - ps + stride - 1u) / stride;
    }
    let max_pr = min(row / stride, params.num_patches_y - 1u);

    var sum_dx: f32 = 0.0;
    var sum_dy: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var pr = min_pr; pr <= max_pr; pr++) {
        for (var pc = min_pc; pc <= max_pc; pc++) {
            let grid_x = pc * stride;
            let grid_y = pr * stride;

            // Verify this patch actually covers (col, row)
            if col >= grid_x && col < grid_x + ps && row >= grid_y && row < grid_y + ps {
                let patch_idx = pr * params.num_patches_x + pc;
                let fdx = patch_flow_u[patch_idx];
                let fdy = patch_flow_v[patch_idx];

                // Photometric error weight
                let ref_val = ref_image[row * w + col];
                let sx = f32(col) + 0.5 + fdx;
                let sy = f32(row) + 0.5 + fdy;
                let tgt_val = sample_bilinear(sx, sy);
                let diff = abs(tgt_val - ref_val);
                let weight = 1.0 / max(1.0, diff);

                sum_dx += weight * fdx;
                sum_dy += weight * fdy;
                weight_sum += weight;
            }
        }
    }

    let idx = row * w + col;
    if weight_sum > 0.0 {
        flow_u[idx] = sum_dx / weight_sum;
        flow_v[idx] = sum_dy / weight_sum;
    } else {
        flow_u[idx] = 0.0;
        flow_v[idx] = 0.0;
    }
}
