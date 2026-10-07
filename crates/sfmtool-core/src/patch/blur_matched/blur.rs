// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The isotropic Gaussian blur of a tile, by normalized convolution over the
//! samples that carry data.
//!
//! A round Gaussian is the product of two 1-D Gaussians, one along each grid
//! axis, so it is applied as two 1-D passes: one down the columns, then one
//! along the rows.
//!
//! Samples without data, and positions past the tile's edge, carry neither a
//! value nor a weight: the colour planes are blurred premultiplied by the data
//! mask, the mask with them, and the blurred colour is divided by the blurred
//! mask at the samples that carry data. The planes sit in a buffer padded with
//! zeros wide enough that no tap of either pass reads past it, and the first
//! pass is computed over the band round the tile the second reads, so the two
//! passes compose as one 2-D convolution of the tile extended by zeros. Every
//! pass is then a sum of whole shifted rows, `dst[x] += w · src[x + s]`, which
//! the compiler vectorizes. The same code compiled for AVX2 ran from 13% faster
//! to 13% slower than the default build, by the kernel's shape, so there is no
//! AVX2 form.

/// Buffers [`blur_tile`] reuses between calls.
#[derive(Debug, Default, Clone)]
pub struct BlurScratch {
    a: Vec<f32>,
    b: Vec<f32>,
    columns: Vec<Tap>,
    rows: Vec<Tap>,
    /// One row of reciprocals of the blurred mask.
    inverse: Vec<f32>,
}

/// One tap of a pass: `dst[y][x] += w · src[y + dy][x + dx]`.
#[derive(Debug, Clone, Copy)]
struct Tap {
    dy: isize,
    dx: isize,
    w: f32,
}

/// Blur the planar colour tile `values` (`channels` planes of `side × side`,
/// row-major) by the isotropic Gaussian of width `sigma` grid px, over the
/// samples `data` marks, into `out` (the same layout). A sample without data
/// keeps its value in `out` and takes no part in the blur, and neither does
/// anything past the tile's edge. A width of `1e-3` or less, or one that is
/// not a number, leaves the tile as it is.
///
/// # Panics
///
/// Panics if `values`, `data` or `out` do not hold a value for every sample.
pub fn blur_tile(
    values: &[f32],
    channels: usize,
    side: usize,
    data: &[bool],
    sigma: f64,
    out: &mut [f32],
    scratch: &mut BlurScratch,
) {
    check_shapes(values, channels, side, data, out);
    out.copy_from_slice(values);
    if sigma.is_nan() || sigma <= 1e-3 || side == 0 {
        return;
    }
    let BlurScratch {
        a,
        b,
        columns,
        rows,
        inverse,
    } = scratch;
    plan(sigma, columns, rows);
    let (r1y, r1x) = reach(columns);
    let (r2y, r2x) = reach(rows);
    // The bands each pass writes: the first over the tile and the second's
    // reach round it, the second over the tile alone. A band's width is
    // rounded up to whole 8-lane registers; the columns past the tile it adds
    // are written and never read back.
    let lanes = |w: usize| w.div_ceil(8) * 8;
    let pad_y = r1y + r2y;
    let pad_x = r1x + r2x;
    let band1 = Band {
        rows: pad_y - r2y..pad_y + side + r2y,
        cols: pad_x - r2x..pad_x - r2x + lanes(side + 2 * r2x),
    };
    let band2 = Band {
        rows: pad_y..pad_y + side,
        cols: pad_x..pad_x + lanes(side),
    };
    // Room for every read: the first pass reads `r1x` past its band and the
    // second `r2x` past its own.
    let stride = (band1.cols.end + r1x).max(band2.cols.end + r2x);
    let height = side + 2 * pad_y;
    let planes = channels + 1;
    let g = Geometry {
        stride,
        plane: stride * height,
        planes,
    };
    // `a` holds the padded tile. It is all zeros between calls, whatever the
    // geometry of the last one, since each call clears what it wrote before
    // it returns; a resize keeps those zeros and adds more. `b` is read only
    // where the first pass of this call wrote it.
    a.resize(planes * g.plane, 0.0);
    b.resize(planes * g.plane, 0.0);
    let n = side * side;
    for y in 0..side {
        let row = (pad_y + y) * stride + pad_x;
        let flags = &data[y * side..(y + 1) * side];
        for c in 0..channels {
            let src = &values[c * n + y * side..c * n + (y + 1) * side];
            let dst = &mut a[c * g.plane + row..c * g.plane + row + side];
            for ((d, &v), &f) in dst.iter_mut().zip(src).zip(flags) {
                *d = if f { v } else { 0.0 };
            }
        }
        let dst = &mut a[channels * g.plane + row..channels * g.plane + row + side];
        for (d, &f) in dst.iter_mut().zip(flags) {
            *d = if f { 1.0 } else { 0.0 };
        }
    }
    pass(a, b, columns, &band1, &g);
    pass(b, a, rows, &band2, &g);
    let mask = &a[channels * g.plane..];
    inverse.resize(side, 0.0);
    for y in 0..side {
        let at = (pad_y + y) * stride + pad_x;
        let flags = &data[y * side..(y + 1) * side];
        for ((inv, &m), &f) in inverse.iter_mut().zip(&mask[at..at + side]).zip(flags) {
            *inv = if f && m > 1e-6 { 1.0 / m } else { 0.0 };
        }
        for c in 0..channels {
            let blurred = &a[c * g.plane + at..c * g.plane + at + side];
            let dst = &mut out[c * n + y * side..c * n + (y + 1) * side];
            for ((d, &v), &inv) in dst.iter_mut().zip(blurred).zip(inverse.iter()) {
                if inv > 0.0 {
                    *d = v * inv;
                }
            }
        }
    }
    // This call wrote `a` only on the tile's rows, from its first column:
    // the tile itself, and the second pass its band. Put the zeros back there,
    // so the next call, of any geometry, finds `a` all zero.
    for y in band2.rows.clone() {
        for p in 0..planes {
            let row = p * g.plane + y * stride;
            a[row + pad_x..row + band2.cols.end].fill(0.0);
        }
    }
}

fn check_shapes(values: &[f32], channels: usize, side: usize, data: &[bool], out: &[f32]) {
    let n = side * side;
    assert_eq!(
        values.len(),
        channels * n,
        "blur_tile: values do not cover the tile"
    );
    assert_eq!(data.len(), n, "blur_tile: one data flag per sample");
    assert_eq!(
        out.len(),
        channels * n,
        "blur_tile: out does not cover the tile"
    );
}

/// The normalized taps `exp(−t²/2s²)` for `t = −K ..= K`, `K = ⌈3σ⌉`, with
/// `s` chosen so the taps' variance `Σ w t²` is `σ²`.
///
/// From `σ = 1` up, the taps of `s = σ` have that variance to within 2% (the
/// cut at `±K` takes up to 1.6% at `σ = 3`), and `s` is `σ`. Below, they fall
/// short of it, more the narrower the Gaussian: `σ = 0.5` gives 0.215 for
/// 0.25, and `σ = 0.3` under a tenth of 0.09. Most pairs are blurred by less
/// than 1 grid px, so there `s` is found by bisection, and the blur adds the
/// variance asked for.
fn gaussian_weights(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let taps = |s: f64| -> Vec<f64> {
        let raw: Vec<f64> = (-radius..=radius)
            .map(|t| (-((t * t) as f64) / (2.0 * s * s)).exp())
            .collect();
        let total: f64 = raw.iter().sum();
        raw.into_iter().map(|w| w / total).collect()
    };
    if sigma >= 1.0 {
        return taps(sigma);
    }
    // `Σ w t²` of the normalized taps of `s`, summed over `t > 0` and doubled.
    let variance = |s: f64| -> f64 {
        let (mut total, mut moment) = (1.0, 0.0);
        for t in 1..=radius {
            let t2 = (t * t) as f64;
            let w = (-t2 / (2.0 * s * s)).exp();
            total += 2.0 * w;
            moment += 2.0 * w * t2;
        }
        moment / total
    };
    // The taps' variance grows with `s`; at `s = σ` it is short of `σ²`, and
    // at `σ + 0.5` past it for every `σ` under 1.
    let (mut lo, mut hi) = (sigma, sigma + 0.5);
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        if variance(mid) < sigma * sigma {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    taps(0.5 * (lo + hi))
}

/// The two passes' taps for a blur of width `sigma`: down the columns, then
/// along the rows.
fn plan(sigma: f64, columns: &mut Vec<Tap>, rows: &mut Vec<Tap>) {
    let weights = gaussian_weights(sigma);
    let radius = (weights.len() / 2) as isize;
    columns.clear();
    rows.clear();
    for (i, &w) in weights.iter().enumerate() {
        let t = i as isize - radius;
        let w = w as f32;
        columns.push(Tap { dy: t, dx: 0, w });
        rows.push(Tap { dy: 0, dx: t, w });
    }
}

/// How far a pass's taps reach: `(rows, columns)`.
fn reach(taps: &[Tap]) -> (usize, usize) {
    taps.iter().fold((0, 0), |(ry, rx), t| {
        (ry.max(t.dy.unsigned_abs()), rx.max(t.dx.unsigned_abs()))
    })
}

struct Geometry {
    stride: usize,
    plane: usize,
    planes: usize,
}

struct Band {
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
}

/// `dst[y][x] = Σ_taps w · src[y + dy][x + dx]` over the band, every plane.
fn pass(src: &[f32], dst: &mut [f32], taps: &[Tap], band: &Band, g: &Geometry) {
    let width = band.cols.len();
    let Some((head, rest)) = taps.split_first() else {
        return;
    };
    for p in 0..g.planes {
        let base = p * g.plane;
        for y in band.rows.clone() {
            let row = base + y * g.stride + band.cols.start;
            let d = &mut dst[row..row + width];
            let at = |tap: &Tap| (row as isize + tap.dy * g.stride as isize + tap.dx) as usize;
            let s = &src[at(head)..at(head) + width];
            for (x, &v) in d.iter_mut().zip(s) {
                *x = head.w * v;
            }
            for tap in rest {
                let from = at(tap);
                let s = &src[from..from + width];
                let w = tap.w;
                for (x, &v) in d.iter_mut().zip(s) {
                    *x += w * v;
                }
            }
        }
    }
}

/// The same blur as [`blur_tile`], as one direct 2-D convolution with the
/// sampled kernel `exp(−|d|²/2σ²)` over `|dx|, |dy| ≤ ⌈3σ⌉`: the reference the
/// two passes are checked against, and the cost they are measured against.
/// It samples the kernel at `σ` itself, where the passes match the variance
/// asked for, so the two agree from `σ = 1` up.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn blur_tile_direct(
    values: &[f32],
    channels: usize,
    side: usize,
    data: &[bool],
    sigma: f64,
    out: &mut [f32],
) {
    check_shapes(values, channels, side, data, out);
    out.copy_from_slice(values);
    if sigma.is_nan() || sigma <= 1e-3 {
        return;
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let side_i = side as isize;
    let n = side * side;
    let mut kernel = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let r2 = (dx * dx + dy * dy) as f64;
            kernel.push((dx, dy, (-0.5 * r2 / (sigma * sigma)).exp()));
        }
    }
    for y in 0..side_i {
        for x in 0..side_i {
            let k = (y * side_i + x) as usize;
            if !data[k] {
                continue;
            }
            let mut sums = [0.0f64; 4];
            let mut weight = 0.0f64;
            for &(dx, dy, w) in &kernel {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx >= side_i || yy >= side_i {
                    continue;
                }
                let j = (yy * side_i + xx) as usize;
                if !data[j] {
                    continue;
                }
                weight += w;
                for (c, sum) in sums.iter_mut().enumerate().take(channels) {
                    *sum += w * f64::from(values[c * n + j]);
                }
            }
            for (c, sum) in sums.iter().enumerate().take(channels) {
                out[c * n + k] = (sum / weight) as f32;
            }
        }
    }
}
