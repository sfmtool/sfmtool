// Separable Gaussian blur + 2x downsample compute shader.
//
// Two-pass approach: horizontal blur+downsample, then vertical blur+downsample.
// Uses workgroup shared memory to load tiles with halos for efficient access.
//
// 6-tap kernel [0.017560, 0.129748, 0.352692, 0.352692, 0.129748, 0.017560]
// centered between pixels (taps at offsets -2, -1, 0, +1, +2, +3 from base = 2*oc).
//
// This shader handles ONE pass (horizontal or vertical), selected by `params.pass`.
// Pass 0 (horizontal): reads input image (in_w × in_h), writes intermediate (out_w × in_h)
//   where out_w = in_w / 2. Each output pixel applies the kernel horizontally.
// Pass 1 (vertical): reads intermediate (out_w × in_h), writes output (out_w × out_h)
//   where out_h = in_h / 2. Each output pixel applies the kernel vertically.

struct Params {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@group(1) @binding(0) var<uniform> params: Params;

const K0: f32 = 0.017560;
const K1: f32 = 0.129748;
const K2: f32 = 0.352692;
const K3: f32 = 0.352692;
const K4: f32 = 0.129748;
const K5: f32 = 0.017560;

@compute @workgroup_size(16, 16)
fn horiz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oc = gid.x;  // output column
    let row = gid.y;
    if oc >= params.out_width || row >= params.in_height { return; }

    let iw = i32(params.in_width);
    let base = i32(oc) * 2;

    // Tap positions: base-2, base-1, base, base+1, base+2, base+3
    let c0 = clamp(base - 2, 0, iw - 1);
    let c1 = clamp(base - 1, 0, iw - 1);
    let c2 = clamp(base,     0, iw - 1);
    let c3 = clamp(base + 1, 0, iw - 1);
    let c4 = clamp(base + 2, 0, iw - 1);
    let c5 = clamp(base + 3, 0, iw - 1);

    let row_off = row * params.in_width;
    let val = K0 * input[row_off + u32(c0)]
            + K1 * input[row_off + u32(c1)]
            + K2 * input[row_off + u32(c2)]
            + K3 * input[row_off + u32(c3)]
            + K4 * input[row_off + u32(c4)]
            + K5 * input[row_off + u32(c5)];

    output[row * params.out_width + oc] = val;
}

@compute @workgroup_size(16, 16)
fn vert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let or = gid.y;  // output row
    if col >= params.out_width || or >= params.out_height { return; }

    let ih = i32(params.in_height);
    let base = i32(or) * 2;

    // Tap positions: base-2, base-1, base, base+1, base+2, base+3
    let r0 = clamp(base - 2, 0, ih - 1);
    let r1 = clamp(base - 1, 0, ih - 1);
    let r2 = clamp(base,     0, ih - 1);
    let r3 = clamp(base + 1, 0, ih - 1);
    let r4 = clamp(base + 2, 0, ih - 1);
    let r5 = clamp(base + 3, 0, ih - 1);

    // Input width for this pass is out_width (from the horizontal pass)
    let w = params.out_width;
    let val = K0 * input[u32(r0) * w + col]
            + K1 * input[u32(r1) * w + col]
            + K2 * input[u32(r2) * w + col]
            + K3 * input[u32(r3) * w + col]
            + K4 * input[u32(r4) * w + col]
            + K5 * input[u32(r5) * w + col];

    output[or * params.out_width + col] = val;
}
