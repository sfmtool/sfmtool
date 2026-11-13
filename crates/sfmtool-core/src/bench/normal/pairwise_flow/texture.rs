// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How much each sample's flow says, from the image texture of the two crops
//! it is read between: the Lucas-Kanade measurement model, in which a flow
//! read over a window is measured only along the image gradients in it.
//!
//! - Each crop's **structure tensor** at a sample, `T = Σ ∇I ∇Iᵀ`, is summed
//!   over the `L × L` square centred on it, `L` the correlation length in
//!   samples at the pair's level ([`window_side`]), so the window the
//!   information is read over and the area it counts once per are the same.
//! - Each tensor keeps only what is significant against noise: the
//!   `1 − noise_false_alarm` quantile of the larger eigenvalue of the tensor
//!   of pure white noise, rendered and read the way that crop is
//!   ([`noise_threshold`], in units of the noise variance), times the crop's
//!   noise variance `σ_I²/2`, is subtracted from both eigenvalues, and an
//!   eigenvalue below zero is set to zero ([`significant`]).
//! - The two crops' tensors combine as the parallel sum
//!   `T_i (T_i + T_j)⁺ T_j` ([`parallel_sum`]), the information of a
//!   difference of two positions, each read in one crop.
//! - `W_s = T_s / (σ_I²/2)`, with `σ_I²` the variance of the photometric
//!   residual over the disc, floored at 8-bit quantization; and the floor
//!   on the flow's precision makes the information one sample carries
//!   `w (I + f² L² w)⁻¹`, `w = W_s / L²` ([`floored`]).
//!
//! The grey values, gradients and tensors are `f32`, as the crops are: a box
//! sum of at most 32² products, each summed in a row and then across rows,
//! is bounded by about `64 · 2⁻²⁴` of its terms' sum, far below the
//! precision of a flow's information. The per-sample information and every
//! sum over a disc are `f64`, since they feed the fit's normal equations,
//! which run in `f64` (`super::fit`).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::features::optical_flow::{FlowField, GrayImage, ImagePyramid};
use crate::numeric::quantile_in_place;

use super::crops::PairLevel;

/// The variance of rounding to 8 bits, in the flow's grey units (`1/255` a
/// level): a perfect match does not claim infinite precision.
pub(crate) const QUANTIZATION_VARIANCE: f64 = 1.0 / (255.0 * 255.0 * 12.0);

/// A 2×2 symmetric matrix `[[xx, xy], [xy, yy]]` as `[xx, xy, yy]`.
pub(crate) type Sym2 = [f64; 3];

/// The side of the noise simulation's field at the level it is read, in
/// samples, and how many fields it draws: about 50,000 windows, so the
/// 0.999 quantile rests on some 50 above it.
const NOISE_FIELD_SIDE: usize = 128;
const NOISE_FIELDS: u64 = 4;

/// The deepest level the noise simulation builds its field down to; a
/// deeper level is simulated as this one with the crop's upsampling scaled
/// to match, so the simulated field is at most `128 · 2³` samples a side.
const NOISE_MAX_LEVEL: u32 = 3;

/// The side, in samples at a pair's level, of the square the structure
/// tensors are summed over and the flow's errors are taken to be correlated
/// across: `correlation_length_px · m`, `m` the coarser view's upsampling at
/// the pair's level, rounded to whole samples and at least 1.
pub(crate) fn window_side(correlation_length_px: f64, level: &PairLevel) -> usize {
    (correlation_length_px * level.upsampling).round().max(1.0) as usize
}

/// Per sample of `image`, the structure tensor summed over the
/// `window × window` square centred on it (`[xx, xy, yy]`), clipped at the
/// image's edges. The gradient is the central difference, and zero where
/// either neighbour is off the image or not `valid`, so the photograph's edge
/// adds no false structure.
pub(crate) fn structure_tensors(image: &GrayImage, valid: &[bool], window: usize) -> Vec<[f32; 3]> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let data = image.data();
    let ok = |c: usize, r: usize| valid[r * w + c];
    let mut products = vec![[0.0f32; 3]; w * h];
    for r in 0..h {
        for c in 0..w {
            let gx = if c > 0 && c + 1 < w && ok(c - 1, r) && ok(c + 1, r) {
                0.5 * (data[r * w + c + 1] - data[r * w + c - 1])
            } else {
                0.0
            };
            let gy = if r > 0 && r + 1 < h && ok(c, r - 1) && ok(c, r + 1) {
                0.5 * (data[(r + 1) * w + c] - data[(r - 1) * w + c])
            } else {
                0.0
            };
            products[r * w + c] = [gx * gx, gx * gy, gy * gy];
        }
    }
    box_sum(&products, w, h, window)
}

/// The sum of `values` over the `window × window` square centred on each
/// sample (offsets `−⌊window/2⌋ ..< window − ⌊window/2⌋`), clipped at the
/// edges: along each row, then along each column.
fn box_sum(values: &[[f32; 3]], w: usize, h: usize, window: usize) -> Vec<[f32; 3]> {
    let before = window / 2;
    let after = window - before;
    let span = |i: usize, n: usize| (i.saturating_sub(before), (i + after).min(n));
    let add = |a: [f32; 3], b: [f32; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let mut rows = vec![[0.0f32; 3]; w * h];
    for r in 0..h {
        for c in 0..w {
            let (lo, hi) = span(c, w);
            rows[r * w + c] = values[r * w + lo..r * w + hi]
                .iter()
                .fold([0.0; 3], |s, &v| add(s, v));
        }
    }
    let mut out = vec![[0.0f32; 3]; w * h];
    for r in 0..h {
        let (lo, hi) = span(r, h);
        for c in 0..w {
            out[r * w + c] = (lo..hi).fold([0.0; 3], |s, k| add(s, rows[k * w + c]));
        }
    }
    out
}

/// The eigenvalues of the symmetric `t`, larger first, and the unit
/// eigenvector of the larger.
fn eigen(t: Sym2) -> ([f64; 2], [f64; 2]) {
    let [xx, xy, yy] = t;
    let mean = 0.5 * (xx + yy);
    let half = 0.5 * (xx - yy);
    let root = half.hypot(xy);
    let (l1, l2) = (mean + root, mean - root);
    // Of the two equivalent forms of the eigenvector, the one that adds two
    // terms of the same sign, so neither subtracts nearly equal numbers.
    let v = if half >= 0.0 {
        [half + root, xy]
    } else {
        [xy, root - half]
    };
    let norm = v[0].hypot(v[1]);
    let v = if norm > 0.0 {
        [v[0] / norm, v[1] / norm]
    } else {
        [1.0, 0.0]
    };
    ([l1, l2], v)
}

/// `t` with each eigenvalue `λ` replaced by `f(λ)`.
fn map_eigen(t: Sym2, f: impl Fn(f64) -> f64) -> Sym2 {
    let ([l1, l2], [a, b]) = eigen(t);
    let (m1, m2) = (f(l1), f(l2));
    // m2·I + (m1 − m2)·e₁e₁ᵀ.
    let d = m1 - m2;
    [m2 + d * a * a, d * a * b, m2 + d * b * b]
}

/// The tensor `t` with `threshold` subtracted from both eigenvalues and an
/// eigenvalue below zero set to zero: what is significant against noise.
pub(crate) fn significant(t: Sym2, threshold: f64) -> Sym2 {
    map_eigen(t, |l| (l - threshold).max(0.0))
}

/// The parallel sum `a (a + b)⁺ b` of two symmetric non-negative matrices:
/// the inverse of the sum of their inverses where both are invertible, never
/// more than either along any direction, and zero along a direction where
/// either is. An eigenvalue of `a + b` at or under `10⁻¹²` of its larger
/// counts as zero in the pseudo-inverse.
pub(crate) fn parallel_sum(a: Sym2, b: Sym2) -> Sym2 {
    let s = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let ([l1, _], _) = eigen(s);
    if !(l1.is_finite() && l1 > 0.0) {
        return [0.0; 3];
    }
    let pinv = map_eigen(s, |l| if l > 1e-12 * l1 { 1.0 / l } else { 0.0 });
    let product = mul(mul(full(a), full(pinv)), full(b));
    // The product is symmetric in exact arithmetic; average the two
    // off-diagonal entries it rounds to.
    [
        product[0][0],
        0.5 * (product[0][1] + product[1][0]),
        product[1][1],
    ]
}

fn full(t: Sym2) -> [[f64; 2]; 2] {
    [[t[0], t[1]], [t[1], t[2]]]
}

fn mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut out = [[0.0; 2]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    out
}

/// The information one sample carries, `w (I + f² L² w)⁻¹` for
/// `w = W / L²`: the flow information `W` of one correlated square of
/// `l2 = L²` samples, with that square's covariance `W⁻¹ + f²·I` for the
/// floor `f` on the flow's precision, in samples at the pair's level, shared
/// among its `L²` samples. A floor of zero leaves `W / L²`.
pub(crate) fn floored(information: Sym2, floor: f64, l2: f64) -> Sym2 {
    let f2 = floor * floor;
    map_eigen(information, |l| {
        let l = l.max(0.0);
        l / (1.0 + f2 * l) / l2
    })
}

/// The `1 − false_alarm` quantile of the larger eigenvalue of the structure
/// tensor ([`structure_tensors`]) over `window × window` squares of white
/// Gaussian noise rendered as a crop with `samples_per_pixel` and read at
/// pyramid level `level`, in units of the noise's variance at that level.
///
/// The noise is drawn at the photograph's pixels and sampled bilinearly
/// onto the grid where `samples_per_pixel` is over 1, as the crop's render
/// upsamples it, and drawn at the grid's samples otherwise; it is then taken
/// down `level` levels of the flow's Gaussian pyramid and scaled to unit
/// variance, so the threshold counts the correlation the render and the
/// pyramid leave between neighbouring gradients. A level deeper than
/// [`NOISE_MAX_LEVEL`] is simulated at that level with `samples_per_pixel`
/// scaled down by the levels left out. `samples_per_pixel` is read in
/// quarter steps.
///
/// The simulation draws from a fixed sequence for each setting, so the
/// threshold is the same on every call and every machine; each setting is
/// simulated once per process and kept.
pub(crate) fn noise_threshold(
    window: usize,
    samples_per_pixel: f64,
    level: u32,
    false_alarm: f64,
) -> f64 {
    let (k_quarters, level) = noise_setting(samples_per_pixel, level);
    let key = (window, k_quarters, level, false_alarm.to_bits());
    static CACHE: OnceLock<Mutex<HashMap<NoiseKey, f64>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&tau) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return tau;
    }
    let tau = simulate_noise_threshold(window, k_quarters, level, false_alarm);
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, tau);
    tau
}

/// A simulated noise threshold's setting: the window's side, the samples
/// per pixel in quarters, the level and the false alarm rate's bits.
type NoiseKey = (usize, u32, u32, u64);

/// The setting a crop's noise is simulated at: its samples per pixel in
/// quarters (4 for a crop read at or finer than its photograph's pixels),
/// and the level, both brought within [`NOISE_MAX_LEVEL`].
fn noise_setting(samples_per_pixel: f64, level: u32) -> (u32, u32) {
    let simulated = level.min(NOISE_MAX_LEVEL);
    let k = if samples_per_pixel.is_finite() {
        samples_per_pixel / f64::from(level - simulated).exp2()
    } else {
        1.0
    };
    // Past 64 samples per pixel the field is flat across every window.
    let quarters = (k.clamp(1.0, 64.0) * 4.0).round() as u32;
    (quarters, simulated)
}

/// [`noise_threshold`] computed afresh for the setting [`noise_setting`]
/// gives.
pub(super) fn simulate_noise_threshold(
    window: usize,
    k_quarters: u32,
    level: u32,
    false_alarm: f64,
) -> f64 {
    let k = f64::from(k_quarters) / 4.0;
    let scale = 1usize << level;
    let side = NOISE_FIELD_SIDE * scale;
    let mut eigenvalues = Vec::new();
    for field in 0..NOISE_FIELDS {
        let seed = [
            window as u64,
            u64::from(k_quarters),
            u64::from(level),
            field,
        ]
        .iter()
        .fold(0x5EED_F10E_u64, |h, &v| splitmix(h ^ v));
        let mut normal = Gaussian::new(seed);
        let data: Vec<f32> = if k_quarters > 4 {
            // Photograph pixels under the grid, with a pixel of margin.
            let photo_side = (side as f64 / k).ceil() as usize + 3;
            let photo: Vec<f32> = (0..photo_side * photo_side)
                .map(|_| normal.next() as f32)
                .collect();
            let mut data = Vec::with_capacity(side * side);
            for r in 0..side {
                for c in 0..side {
                    let x = (c as f64 + 0.5) / k + 0.5;
                    let y = (r as f64 + 0.5) / k + 0.5;
                    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
                    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
                    let at = |cc: usize, rr: usize| photo[rr * photo_side + cc];
                    let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
                    let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
                    data.push(top * (1.0 - fy) + bottom * fy);
                }
            }
            data
        } else {
            (0..side * side).map(|_| normal.next() as f32).collect()
        };
        let image = GrayImage::new(side as u32, side as u32, data);
        let pyramid = ImagePyramid::build(&image, level + 1);
        let read = pyramid.level(level as usize);
        let (w, h) = (read.width() as usize, read.height() as usize);
        let variance = f64::from(field_variance(read.data(), w));
        let tensors = structure_tensors(read, &vec![true; w * h], window);
        // Only windows wholly inside the field, and one sample in from its
        // edge, where every gradient in them is a central difference.
        let before = window / 2;
        let after = window - before;
        for r in before + 1..h.saturating_sub(after) {
            for c in before + 1..w.saturating_sub(after) {
                let t = tensors[r * w + c].map(f64::from);
                eigenvalues.push(eigen(t).0[0] / variance);
            }
        }
    }
    quantile_in_place(&mut eigenvalues, 1.0 - false_alarm)
}

/// The variance of a field of `width` columns about its mean, each row summed
/// and then the row sums, in `f32`.
///
/// The field is the level the noise is read at, 128 × 128 samples of noise of
/// about unit variance. Summed by rows, the rounding is bounded by about
/// `(rows + columns) · 2⁻²⁴` of the sum of the terms' magnitudes,
/// `1.5 · 10⁻⁵` at that size, far below the threshold's own sampling error:
/// its 0.999 quantile rests on some 50 windows above it.
fn field_variance(data: &[f32], width: usize) -> f32 {
    let n = data.len() as f32;
    let rows = || data.chunks_exact(width.max(1));
    let mean = rows().map(|row| row.iter().sum::<f32>()).sum::<f32>() / n;
    let squares: f32 = rows()
        .map(|row| row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>())
        .sum();
    squares / n
}

/// SplitMix64's output function: a fixed scramble of one 64-bit state, so the
/// simulation draws the same numbers on every platform and with every
/// version of the `rand` crate.
fn splitmix(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Standard normal draws by Box–Muller on a SplitMix64 sequence.
pub(super) struct Gaussian {
    state: u64,
    spare: Option<f64>,
}

impl Gaussian {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        // 53 random bits in (0, 1]: never 0, so the logarithm is finite.
        ((splitmix(self.state) >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    pub(super) fn next(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

/// One pair's samples at its level: every sample of crop `i` on its
/// photograph whose flow lands on crop `j`'s, with its position, its flow,
/// the two crops' structure tensors and the photometric residual.
pub(crate) struct PairSamples {
    /// Each sample's grid coordinates, grid samples from the crops' centre.
    pub(crate) positions: Vec<[f64; 2]>,
    /// Each sample's measured flow, samples at the pair's level.
    pub(crate) flows: Vec<[f64; 2]>,
    tensors_from: Vec<[f32; 3]>,
    tensors_to: Vec<[f32; 3]>,
    residuals: Vec<f32>,
    /// The side of the correlated square, samples at the pair's level.
    pub(crate) window: usize,
    /// The largest disc radius, grid samples, inside the square the pair
    /// reads.
    pub(crate) reach: f64,
}

/// One pair's samples within a disc, with each one's information.
pub(crate) struct Disc {
    /// Each sample's grid coordinates, grid samples from the crops' centre.
    pub(crate) positions: Vec<[f64; 2]>,
    /// Each sample's measured flow, samples at the pair's level.
    pub(crate) flows: Vec<[f64; 2]>,
    /// The information each sample carries about its flow, per level sample
    /// squared: `w (I + f² L² w)⁻¹` with `w = W_s / L²`.
    pub(crate) information: Vec<Sym2>,
    /// `σ_I²`, the photometric residual's variance over the disc, in the
    /// flow's grey units squared.
    pub(crate) residual_variance: f64,
    /// `L²`, the samples of one correlated square.
    pub(crate) l2: f64,
}

impl PairSamples {
    /// The samples of the pair `from → to`, two crops at the same level
    /// `level` of their pyramids with their valid masks, given the flow
    /// between them there; `half_width` is the crops' half-width in grid
    /// samples and `window` the correlated square's side.
    pub(crate) fn new(
        from: (&GrayImage, &[bool]),
        to: (&GrayImage, &[bool]),
        flow: &FlowField,
        half_width: u32,
        level: &PairLevel,
        window: usize,
    ) -> Self {
        let (image_i, valid_i) = from;
        let (image_j, valid_j) = to;
        let (w, h) = (image_i.width() as usize, image_i.height() as usize);
        let tensors_i = structure_tensors(image_i, valid_i, window);
        let tensors_j = structure_tensors(image_j, valid_j, window);
        let scale = level.scale();
        let centre = f64::from(half_width) / scale;
        let reach = centre
            .min(w as f64 - centre)
            .min(h as f64 - centre)
            .max(0.0)
            * scale;
        let data_j = image_j.data();
        let mut samples = PairSamples {
            positions: Vec::new(),
            flows: Vec::new(),
            tensors_from: Vec::new(),
            tensors_to: Vec::new(),
            residuals: Vec::new(),
            window,
            reach,
        };
        for r in 0..h {
            for c in 0..w {
                let k = r * w + c;
                if !valid_i[k] {
                    continue;
                }
                let (fx, fy) = flow.get(c as u32, r as u32);
                if !(fx.is_finite() && fy.is_finite()) {
                    continue;
                }
                // Index coordinates of where the flow lands in crop `j`.
                let x = c as f64 + f64::from(fx);
                let y = r as f64 + f64::from(fy);
                if !(x >= 0.0 && y >= 0.0 && x < (w - 1) as f64 && y < (h - 1) as f64) {
                    continue;
                }
                let (x0, y0) = (x as usize, y as usize);
                let corners = [
                    y0 * w + x0,
                    y0 * w + x0 + 1,
                    (y0 + 1) * w + x0,
                    (y0 + 1) * w + x0 + 1,
                ];
                if !corners.iter().all(|&i| valid_j[i]) {
                    continue;
                }
                let (ax, ay) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
                let top = data_j[corners[0]] * (1.0 - ax) + data_j[corners[1]] * ax;
                let bottom = data_j[corners[2]] * (1.0 - ax) + data_j[corners[3]] * ax;
                let value_j = top * (1.0 - ay) + bottom * ay;
                let nearest = (y.round() as usize) * w + x.round() as usize;
                samples.positions.push([
                    (c as f64 + 0.5) * scale - f64::from(half_width),
                    (r as f64 + 0.5) * scale - f64::from(half_width),
                ]);
                samples.flows.push([f64::from(fx), f64::from(fy)]);
                samples.tensors_from.push(tensors_i[k]);
                samples.tensors_to.push(tensors_j[nearest]);
                samples.residuals.push(value_j - image_i.data()[k]);
            }
        }
        samples
    }

    /// The samples within `radius` grid samples of the crops' centre with
    /// their information: each crop's tensor thresholded at its
    /// `thresholds` entry (the noise threshold in units of the noise
    /// variance, crop `i`'s first) times `σ_I²/2`, the parallel sum, `W_s`,
    /// and the floor `floor`, samples at the pair's level. `None` where the
    /// disc reaches past the square the pair reads or holds no sample.
    pub(crate) fn disc(&self, radius: f64, thresholds: [f64; 2], floor: f64) -> Option<Disc> {
        if !(radius.is_finite() && radius > 0.0 && radius <= self.reach) {
            return None;
        }
        let inside: Vec<usize> = (0..self.positions.len())
            .filter(|&k| {
                let [x, y] = self.positions[k];
                x * x + y * y <= radius * radius
            })
            .collect();
        if inside.is_empty() {
            return None;
        }
        let n = inside.len() as f64;
        let mean = inside
            .iter()
            .map(|&k| f64::from(self.residuals[k]))
            .sum::<f64>()
            / n;
        let residual_variance = (inside
            .iter()
            .map(|&k| (f64::from(self.residuals[k]) - mean).powi(2))
            .sum::<f64>()
            / n)
            .max(QUANTIZATION_VARIANCE);
        let noise = residual_variance / 2.0;
        // In f64, so a long correlation length cannot overflow the product.
        let l2 = (self.window as f64).powi(2);
        let information = inside
            .iter()
            .map(|&k| {
                let t_i = significant(self.tensors_from[k].map(f64::from), thresholds[0] * noise);
                let t_j = significant(self.tensors_to[k].map(f64::from), thresholds[1] * noise);
                let w = parallel_sum(t_i, t_j).map(|v| v / noise);
                floored(w, floor, l2)
            })
            .collect();
        Some(Disc {
            positions: inside.iter().map(|&k| self.positions[k]).collect(),
            flows: inside.iter().map(|&k| self.flows[k]).collect(),
            information,
            residual_variance,
            l2,
        })
    }
}

#[cfg(test)]
mod tests;
