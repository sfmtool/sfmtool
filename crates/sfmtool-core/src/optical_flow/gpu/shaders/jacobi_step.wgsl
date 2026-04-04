// Jacobi iteration step for variational refinement.
//
// Near-direct transliteration of jacobi_pixel_scalar_to_row() from variational.rs.
// Each thread handles one pixel. The solver uses double-buffered ping-pong:
// reads from du_old/dv_old, writes to du_new/dv_new.
//
// Coefficients are packed as vec4(a11, a12, a22, b1) with b2 separate,
// to stay within the 8-storage-buffer-per-stage default limit.

struct Params {
    width: u32,
    height: u32,
    alpha: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> du_old: array<f32>;
@group(0) @binding(1) var<storage, read> dv_old: array<f32>;
@group(0) @binding(2) var<storage, read_write> du_new: array<f32>;
@group(0) @binding(3) var<storage, read_write> dv_new: array<f32>;
@group(0) @binding(4) var<storage, read> flow_u: array<f32>;
@group(0) @binding(5) var<storage, read> flow_v: array<f32>;
@group(0) @binding(6) var<storage, read> coefficients: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> b2_buf: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }
    let idx = row * params.width + col;
    let cu = flow_u[idx];
    let cv = flow_v[idx];

    // 4-neighbor Laplacian with boundary handling
    var lap_u = 0.0;
    var lap_v = 0.0;
    var nn = 0.0;

    if col > 0u {
        let n = idx - 1u;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }
    if col + 1u < params.width {
        let n = idx + 1u;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }
    if row > 0u {
        let n = idx - params.width;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }
    if row + 1u < params.height {
        let n = idx + params.width;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }

    // Unpack coefficients
    let coeff = coefficients[idx];
    let a11_val = coeff.x;
    let a12_val = coeff.y;
    let a22_val = coeff.z;
    let b1_val = coeff.w;
    let b2_val = b2_buf[idx];

    let diag_u = a11_val + params.alpha * nn;
    let diag_v = a22_val + params.alpha * nn;

    du_new[idx] = select(
        du_old[idx],
        (b1_val + params.alpha * lap_u - a12_val * dv_old[idx]) / diag_u,
        abs(diag_u) > 1e-10
    );
    dv_new[idx] = select(
        dv_old[idx],
        (b2_val + params.alpha * lap_v - a12_val * du_old[idx]) / diag_v,
        abs(diag_v) > 1e-10
    );
}
