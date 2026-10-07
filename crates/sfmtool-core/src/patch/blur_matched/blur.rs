// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The anisotropic Gaussian blur of a tile, by normalized convolution over the
//! samples that carry data.
//!
//! A Gaussian with covariance `Σ = [[a, b], [b, d]]` is split into two 1-D
//! Gaussians (Geusebroek, Smeulders and van de Weijer, "Fast anisotropic Gauss
//! filtering", 2003): one along the slanted line `(b/d, 1)`, with variance `d`
//! in rows, and one along `x`, with the remaining variance `a − b²/d`. The two
//! covariances add up to `Σ`. Where `a > d` the roles of the axes swap, so the
//! line's slope is at most 1 in magnitude. A tap of the slanted pass lands
//! between two samples of its row, and is read by linear interpolation, which
//! adds a variance of `f(1 − f)` along the axis for a tap at fraction `f`; the
//! axis pass subtracts the weighted mean of it from its own variance, so the
//! two passes together keep the covariance asked for.
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

use super::BlurCovariance;

/// Buffers [`blur_tile`] reuses between calls.
#[derive(Debug, Default, Clone)]
pub struct BlurScratch {
    a: Vec<f32>,
    b: Vec<f32>,
    line: Vec<Tap>,
    axis: Vec<Tap>,
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
/// row-major) by the Gaussian `cov`, over the samples `data` marks, into `out`
/// (the same layout). A sample without data keeps its value in `out` and
/// takes no part in the blur, and neither does anything past the tile's edge.
///
/// # Panics
///
/// Panics if `values`, `data` or `out` do not hold a value for every sample.
pub fn blur_tile(
    values: &[f32],
    channels: usize,
    side: usize,
    data: &[bool],
    cov: BlurCovariance,
    out: &mut [f32],
    scratch: &mut BlurScratch,
) {
    blur_with(values, channels, side, data, cov, out, scratch);
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
/// From `σ = 1` up, the taps of `s = σ` have that variance to within half a
/// percent, and `s` is `σ`. Below, they fall short of it, more the narrower
/// the Gaussian: `σ = 0.5` gives 0.215 for 0.25, and `σ = 0.3` under a tenth
/// of 0.09. A pass of the slanted line is often that narrow along its rows,
/// so there `s` is found by bisection. On the tile of sinusoids the passes are
/// tested on (`the_two_passes_match_the_exact_blur_and_the_direct_2d_blur`),
/// over semi-axes of 0.5 to 0.7 at every 5°, matching the variance brings the
/// passes' worst error against the exact blur from 7.8 grey levels to 2.7.
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

/// The two passes' taps for `cov`: the slanted line's, then the axis pass's.
/// Either is empty when its pass has nothing to do.
fn plan(cov: BlurCovariance, line: &mut Vec<Tap>, axis: &mut Vec<Tap>) {
    line.clear();
    axis.clear();
    let BlurCovariance {
        xx: a,
        xy: b,
        yy: d,
    } = cov;
    // `along_rows`: the line steps one row per tap, `(slope, 1)`; otherwise
    // one column per tap, `(1, slope)`.
    let along_rows = d >= a;
    let (major, other, slope) = if along_rows {
        (d, a, if d > 0.0 { b / d } else { 0.0 })
    } else {
        (a, d, if a > 0.0 { b / a } else { 0.0 })
    };
    let line_sigma = major.max(0.0).sqrt();
    let mut interpolation = 0.0;
    if line_sigma > 1e-3 {
        let weights = gaussian_weights(line_sigma);
        let radius = (weights.len() / 2) as isize;
        for (i, &w) in weights.iter().enumerate() {
            let t = i as isize - radius;
            let offset = t as f64 * slope;
            let whole = offset.floor();
            let f = offset - whole;
            interpolation += w * f * (1.0 - f);
            let whole = whole as isize;
            for (shift, wi) in [(whole, w * (1.0 - f)), (whole + 1, w * f)] {
                if wi == 0.0 {
                    continue;
                }
                let (dy, dx) = if along_rows { (t, shift) } else { (shift, t) };
                line.push(Tap {
                    dy,
                    dx,
                    w: wi as f32,
                });
            }
        }
    }
    // What is left across the line once it carries `slope² · major` of it,
    // less what the line's interpolation already added there.
    let rest = other - slope * slope * major - interpolation;
    if rest > 1e-6 {
        let weights = gaussian_weights(rest.sqrt());
        let radius = (weights.len() / 2) as isize;
        for (i, &w) in weights.iter().enumerate() {
            let t = i as isize - radius;
            // Across a line that steps through the rows is along x.
            let (dy, dx) = if along_rows { (0, t) } else { (t, 0) };
            axis.push(Tap {
                dy,
                dx,
                w: w as f32,
            });
        }
    }
}

/// How far a pass's taps reach: `(rows, columns)`.
fn reach(taps: &[Tap]) -> (usize, usize) {
    taps.iter().fold((0, 0), |(ry, rx), t| {
        (ry.max(t.dy.unsigned_abs()), rx.max(t.dx.unsigned_abs()))
    })
}

fn blur_with(
    values: &[f32],
    channels: usize,
    side: usize,
    data: &[bool],
    cov: BlurCovariance,
    out: &mut [f32],
    scratch: &mut BlurScratch,
) {
    check_shapes(values, channels, side, data, out);
    out.copy_from_slice(values);
    if cov.is_zero() || side == 0 {
        return;
    }
    let BlurScratch {
        a,
        b,
        line,
        axis,
        inverse,
    } = scratch;
    plan(cov, line, axis);
    let (first, second): (&[Tap], &[Tap]) = match (line.is_empty(), axis.is_empty()) {
        (false, false) => (line, axis),
        (false, true) => (line, &[]),
        (true, false) => (axis, &[]),
        (true, true) => return,
    };
    let (r1y, r1x) = reach(first);
    let (r2y, r2x) = reach(second);
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
    pass(a, b, first, &band1, &g);
    let result: &[f32] = if second.is_empty() {
        b
    } else {
        pass(b, a, second, &band2, &g);
        a
    };
    let mask = &result[channels * g.plane..];
    inverse.resize(side, 0.0);
    for y in 0..side {
        let at = (pad_y + y) * stride + pad_x;
        let flags = &data[y * side..(y + 1) * side];
        for ((inv, &m), &f) in inverse.iter_mut().zip(&mask[at..at + side]).zip(flags) {
            *inv = if f && m > 1e-6 { 1.0 / m } else { 0.0 };
        }
        for c in 0..channels {
            let blurred = &result[c * g.plane + at..c * g.plane + at + side];
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
    let end = if second.is_empty() {
        pad_x + side
    } else {
        band2.cols.end
    };
    for y in band2.rows.clone() {
        for p in 0..planes {
            let row = p * g.plane + y * stride;
            a[row + pad_x..row + end].fill(0.0);
        }
    }
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
/// sampled kernel `exp(−½ dᵀ Σ⁻¹ d)` over `|dx|, |dy| ≤ ⌈3 √λ_max⌉`: the
/// reference the two passes are checked against, and the cost they are
/// measured against. `Σ` must be positive definite; a 1-D blur has no 2-D
/// sampled form.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn blur_tile_direct(
    values: &[f32],
    channels: usize,
    side: usize,
    data: &[bool],
    cov: BlurCovariance,
    out: &mut [f32],
) {
    check_shapes(values, channels, side, data, out);
    out.copy_from_slice(values);
    if cov.is_zero() {
        return;
    }
    let BlurCovariance {
        xx: a,
        xy: b,
        yy: d,
    } = cov;
    let det = a * d - b * b;
    assert!(
        det > 0.0,
        "blur_tile_direct: the covariance must be positive definite"
    );
    let (i00, i01, i11) = (d / det, -b / det, a / det);
    let lmax = 0.5 * (a + d + ((a - d) * (a - d) + 4.0 * b * b).sqrt());
    let radius = (3.0 * lmax.sqrt()).ceil() as isize;
    let side_i = side as isize;
    let n = side * side;
    let mut kernel = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (x, y) = (dx as f64, dy as f64);
            let w = (-0.5 * (x * x * i00 + 2.0 * x * y * i01 + y * y * i11)).exp();
            kernel.push((dx, dy, w));
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
