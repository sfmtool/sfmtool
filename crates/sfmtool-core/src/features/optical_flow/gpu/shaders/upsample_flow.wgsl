// Flow field 2x upsampling compute shader.
//
// One thread per output pixel. Bilinear-samples the coarser flow field at
// (col + 0.5) * 0.5, (row + 0.5) * 0.5 and multiplies by 2.0 (magnitudes
// double at 2x resolution). Handles arbitrary input/output dimension ratios
// with clamped boundary conditions.

struct Params {
    out_width: u32,
    out_height: u32,
    in_width: u32,
    in_height: u32,
}

@group(0) @binding(0) var<storage, read> flow_u_in: array<f32>;
@group(0) @binding(1) var<storage, read> flow_v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> flow_u_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> flow_v_out: array<f32>;

@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.out_width || row >= params.out_height { return; }

    // Map output pixel center to input coordinates (pixel-center-at-0.5 convention)
    let src_x = (f32(col) + 0.5) * 0.5;
    let src_y = (f32(row) + 0.5) * 0.5;

    // Bilinear interpolation with clamped boundaries
    let gx = src_x - 0.5;
    let gy = src_y - 0.5;
    let x0 = i32(floor(gx));
    let y0 = i32(floor(gy));
    let fx = gx - f32(x0);
    let fy = gy - f32(y0);

    let iw = i32(params.in_width);
    let ih = i32(params.in_height);
    let cx0 = clamp(x0, 0, iw - 1);
    let cy0 = clamp(y0, 0, ih - 1);
    let cx1 = clamp(x0 + 1, 0, iw - 1);
    let cy1 = clamp(y0 + 1, 0, ih - 1);

    let w_in = params.in_width;
    let i00 = u32(cy0) * w_in + u32(cx0);
    let i10 = u32(cy0) * w_in + u32(cx1);
    let i01 = u32(cy1) * w_in + u32(cx0);
    let i11 = u32(cy1) * w_in + u32(cx1);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    // Bilinear sample and multiply by 2.0 (magnitudes double at 2x resolution)
    let u_val = (w00 * flow_u_in[i00] + w10 * flow_u_in[i10]
               + w01 * flow_u_in[i01] + w11 * flow_u_in[i11]) * 2.0;
    let v_val = (w00 * flow_v_in[i00] + w10 * flow_v_in[i10]
               + w01 * flow_v_in[i01] + w11 * flow_v_in[i11]) * 2.0;

    let out_idx = row * params.out_width + col;
    flow_u_out[out_idx] = u_val;
    flow_v_out[out_idx] = v_val;
}
