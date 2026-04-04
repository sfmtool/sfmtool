// Precompute variational refinement coefficients (a11, a12, a22, b1, b2).
//
// Fuses gradient computation inline to minimize buffer count. Each thread
// computes first and second derivatives of both ref and warped images using
// a two-pass central-difference approach with clamped boundary conditions
// (matching the CPU's separate gradient passes), then assembles the 2x2
// system coefficients for the Jacobi solver.
//
// Matches the CPU coefficient precomputation in variational_refine() lines 93-170.
//
// Output packing: coefficients buffer stores vec4(a11, a12, a22, b1) per pixel,
// b2 is stored in a separate scalar buffer.

struct Params {
    width: u32,
    height: u32,
    delta: f32,
    gamma: f32,
}

@group(0) @binding(0) var<storage, read> ref_image: array<f32>;
@group(0) @binding(1) var<storage, read> warped_image: array<f32>;
@group(0) @binding(2) var<storage, read_write> coefficients: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> b2_buf: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

// Clamped pixel access for an image stored in a storage buffer.
fn ref_at(col: i32, row: i32) -> f32 {
    let c = u32(clamp(col, 0, i32(params.width) - 1));
    let r = u32(clamp(row, 0, i32(params.height) - 1));
    return ref_image[r * params.width + c];
}

fn warped_at(col: i32, row: i32) -> f32 {
    let c = u32(clamp(col, 0, i32(params.width) - 1));
    let r = u32(clamp(row, 0, i32(params.height) - 1));
    return warped_image[r * params.width + c];
}

// First derivatives with clamped boundaries, matching CPU compute_image_gradients.
// These are used as building blocks for second derivatives so that boundary
// handling matches the CPU's two-pass approach exactly.
fn ref_dx(col: i32, row: i32) -> f32 {
    let left = clamp(col - 1, 0, i32(params.width) - 1);
    let right = clamp(col + 1, 0, i32(params.width) - 1);
    return (ref_at(right, row) - ref_at(left, row)) * 0.5;
}

fn ref_dy(col: i32, row: i32) -> f32 {
    let up = clamp(row - 1, 0, i32(params.height) - 1);
    let down = clamp(row + 1, 0, i32(params.height) - 1);
    return (ref_at(col, down) - ref_at(col, up)) * 0.5;
}

fn warped_dx(col: i32, row: i32) -> f32 {
    let left = clamp(col - 1, 0, i32(params.width) - 1);
    let right = clamp(col + 1, 0, i32(params.width) - 1);
    return (warped_at(right, row) - warped_at(left, row)) * 0.5;
}

fn warped_dy(col: i32, row: i32) -> f32 {
    let up = clamp(row - 1, 0, i32(params.height) - 1);
    let down = clamp(row + 1, 0, i32(params.height) - 1);
    return (warped_at(col, down) - warped_at(col, up)) * 0.5;
}

// Robust penalizer derivative: Psi'(s^2) = 1 / (2 * sqrt(s^2 + eps^2))
fn psi_deriv(s_sq: f32) -> f32 {
    let eps_sq = 1e-6;
    return 1.0 / (2.0 * sqrt(s_sq + eps_sq));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = i32(gid.x);
    let row = i32(gid.y);
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let idx = gid.y * params.width + gid.x;
    let w = i32(params.width) - 1;
    let h = i32(params.height) - 1;

    // First derivatives via central differences with clamped boundaries
    let r_ix = ref_dx(col, row);
    let r_iy = ref_dy(col, row);
    let w_ix = warped_dx(col, row);
    let w_iy = warped_dy(col, row);

    // Second derivatives via two-pass central differences (matching CPU).
    // Evaluate first-derivative helpers at clamped neighbor positions, then
    // apply central differences again. This ensures boundary clamping happens
    // at each pass independently, matching the CPU's separate gradient passes.
    let cl = clamp(col - 1, 0, w);
    let cr = clamp(col + 1, 0, w);
    let ru = clamp(row - 1, 0, h);
    let rd = clamp(row + 1, 0, h);

    let r_ixx = (ref_dx(cr, row) - ref_dx(cl, row)) * 0.5;
    let r_ixy = (ref_dx(col, rd) - ref_dx(col, ru)) * 0.5;
    let r_iyx = (ref_dy(cr, row) - ref_dy(cl, row)) * 0.5;
    let r_iyy = (ref_dy(col, rd) - ref_dy(col, ru)) * 0.5;

    let w_ixx = (warped_dx(cr, row) - warped_dx(cl, row)) * 0.5;
    let w_ixy = (warped_dx(col, rd) - warped_dx(col, ru)) * 0.5;
    let w_iyx = (warped_dy(cr, row) - warped_dy(cl, row)) * 0.5;
    let w_iyy = (warped_dy(col, rd) - warped_dy(col, ru)) * 0.5;

    var a11_val = 0.0;
    var a12_val = 0.0;
    var a22_val = 0.0;
    var b1_val = 0.0;
    var b2_val = 0.0;

    // --- Intensity constancy term ---
    let iz = warped_at(col, row) - ref_at(col, row);
    let ix = 0.5 * (r_ix + w_ix);
    let iy = 0.5 * (r_iy + w_iy);
    let grad_sq = ix * ix + iy * iy;
    let beta0 = 1.0 / (grad_sq + 0.01);
    let psi_i = psi_deriv(beta0 * iz * iz);
    let wi = params.delta * beta0 * psi_i;

    a11_val += wi * ix * ix;
    a12_val += wi * ix * iy;
    a22_val += wi * iy * iy;
    b1_val -= wi * ix * iz;
    b2_val -= wi * iy * iz;

    // --- Gradient constancy term (x-derivative) ---
    let ixz = w_ix - r_ix;
    let ixx = 0.5 * (r_ixx + w_ixx);
    let ixy = 0.5 * (r_ixy + w_ixy);
    let grad_sq_x = ixx * ixx + ixy * ixy;
    let beta_x = 1.0 / (grad_sq_x + 0.01);
    let psi_gx = psi_deriv(beta_x * ixz * ixz);
    let wgx = params.gamma * beta_x * psi_gx;

    a11_val += wgx * ixx * ixx;
    a12_val += wgx * ixx * ixy;
    a22_val += wgx * ixy * ixy;
    b1_val -= wgx * ixx * ixz;
    b2_val -= wgx * ixy * ixz;

    // --- Gradient constancy term (y-derivative) ---
    let iyz = w_iy - r_iy;
    let iyx = 0.5 * (r_iyx + w_iyx);
    let iyy = 0.5 * (r_iyy + w_iyy);
    let grad_sq_y = iyx * iyx + iyy * iyy;
    let beta_y = 1.0 / (grad_sq_y + 0.01);
    let psi_gy = psi_deriv(beta_y * iyz * iyz);
    let wgy = params.gamma * beta_y * psi_gy;

    a11_val += wgy * iyx * iyx;
    a12_val += wgy * iyx * iyy;
    a22_val += wgy * iyy * iyy;
    b1_val -= wgy * iyx * iyz;
    b2_val -= wgy * iyy * iyz;

    // Store packed coefficients
    coefficients[idx] = vec4<f32>(a11_val, a12_val, a22_val, b1_val);
    b2_buf[idx] = b2_val;
}
