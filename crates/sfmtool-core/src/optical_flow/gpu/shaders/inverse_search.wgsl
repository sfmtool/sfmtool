// DIS inverse search — one thread per patch (Option B from spec).
//
// Each thread handles a single patch on the regular grid:
// 1. Read initial flow at patch center
// 2. Precompute template gradients and 2x2 Hessian
// 3. Iterate: bilinear-sample target, compute mean-normalized residuals,
//    solve 2x2 system, update flow
// 4. Outlier rejection: reset to initial flow if update exceeds patch_size
//
// Matches the CPU inverse_search() and compute_iteration_scalar() in dis.rs.

struct Params {
    width: u32,
    height: u32,
    patch_size: u32,
    stride: u32,
    num_patches_x: u32,
    num_patches_y: u32,
    grad_descent_iterations: u32,
    normalize: u32,
}

@group(0) @binding(0) var<storage, read> ref_image: array<f32>;
@group(0) @binding(1) var<storage, read> tgt_image: array<f32>;
@group(0) @binding(2) var<storage, read> grad_x_img: array<f32>;
@group(0) @binding(3) var<storage, read> grad_y_img: array<f32>;
@group(0) @binding(4) var<storage, read> flow_u: array<f32>;
@group(0) @binding(5) var<storage, read> flow_v: array<f32>;
@group(0) @binding(6) var<storage, read_write> patch_flow_u: array<f32>;
@group(0) @binding(7) var<storage, read_write> patch_flow_v: array<f32>;
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let patch_idx = gid.x;
    let total_patches = params.num_patches_x * params.num_patches_y;
    if patch_idx >= total_patches {
        return;
    }

    let w = params.width;
    let ps = params.patch_size;
    let patch_row = patch_idx / params.num_patches_x;
    let patch_col = patch_idx % params.num_patches_x;
    let gx = patch_col * params.stride;
    let gy = patch_row * params.stride;

    // Read initial flow at patch center
    let cx = min(gx + ps / 2u, w - 1u);
    let cy = min(gy + ps / 2u, params.height - 1u);
    let init_u = flow_u[cy * w + cx];
    let init_v = flow_v[cy * w + cx];

    // Precompute Hessian H' = sum(S'^T * S') and template mean
    var h00: f32 = 0.0;
    var h01: f32 = 0.0;
    var h11: f32 = 0.0;
    var template_mean: f32 = 0.0;
    let n = f32(ps * ps);

    for (var py = 0u; py < ps; py++) {
        for (var px = 0u; px < ps; px++) {
            let idx = (gy + py) * w + (gx + px);
            template_mean += ref_image[idx];

            let gx_val = grad_x_img[idx];
            let gy_val = grad_y_img[idx];
            h00 += gx_val * gx_val;
            h01 += gx_val * gy_val;
            h11 += gy_val * gy_val;
        }
    }

    template_mean /= n;

    // Invert 2x2 Hessian
    let det = h00 * h11 - h01 * h01;
    if abs(det) < 1e-10 {
        // Singular Hessian — no gradient structure, keep initial flow
        patch_flow_u[patch_idx] = init_u;
        patch_flow_v[patch_idx] = init_v;
        return;
    }
    let inv_det = 1.0 / det;
    let ih00 = h11 * inv_det;
    let ih01 = -h01 * inv_det;
    let ih11 = h00 * inv_det;

    var u_x = init_u;
    var u_y = init_v;
    let do_normalize = params.normalize != 0u;

    for (var iter = 0u; iter < params.grad_descent_iterations; iter++) {
        // Phase 1: compute warp mean (if normalizing)
        var warp_mean: f32 = 0.0;
        if do_normalize {
            for (var py = 0u; py < ps; py++) {
                for (var px = 0u; px < ps; px++) {
                    let sx = f32(gx + px) + 0.5 + u_x;
                    let sy = f32(gy + py) + 0.5 + u_y;
                    warp_mean += sample_bilinear(sx, sy);
                }
            }
            warp_mean /= n;
        }

        // Phase 2: accumulate gradient-weighted residuals
        var b0: f32 = 0.0;
        var b1: f32 = 0.0;
        for (var py = 0u; py < ps; py++) {
            for (var px = 0u; px < ps; px++) {
                let idx = (gy + py) * w + (gx + px);
                let sx = f32(gx + px) + 0.5 + u_x;
                let sy = f32(gy + py) + 0.5 + u_y;
                let tgt_val = sample_bilinear(sx, sy);

                var residual: f32;
                if do_normalize {
                    residual = (tgt_val - warp_mean) - (ref_image[idx] - template_mean);
                } else {
                    residual = tgt_val - ref_image[idx];
                }

                b0 += grad_x_img[idx] * residual;
                b1 += grad_y_img[idx] * residual;
            }
        }

        // Solve: du = H'^{-1} * b
        let du0 = ih00 * b0 + ih01 * b1;
        let du1 = ih01 * b0 + ih11 * b1;

        // Inverse compositional update: u <- u - du
        u_x -= du0;
        u_y -= du1;
    }

    // Outlier rejection: reset if update exceeds patch_size
    let ddx = u_x - init_u;
    let ddy = u_y - init_v;
    let ps_sq = f32(ps * ps);
    if ddx * ddx + ddy * ddy > ps_sq {
        u_x = init_u;
        u_y = init_v;
    }

    patch_flow_u[patch_idx] = u_x;
    patch_flow_v[patch_idx] = u_y;
}
