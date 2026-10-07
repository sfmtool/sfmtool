// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tiles the blur-matching tests share: textured and grained tiles, their
//! self-similarity readings, and constructed ellipses.

use super::{blur_tile, read_tile_ellipse, BlurScratch, TilePlanes};
use crate::patch::normal_refine::{window_weights, PatchWindow};
use crate::patch::self_similarity::{zncc_self_similarity_parts, PatchTile, SelfSimilarityParams};

/// The side of the test tiles.
pub(crate) const SIDE: usize = 24;

/// A small deterministic generator, so a failure reproduces.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Stretch each channel of `values` to 20 ..= 235.
fn stretch(values: &mut [f32], channels: usize) {
    let n = values.len() / channels;
    for c in 0..channels {
        let plane = &mut values[c * n..(c + 1) * n];
        let lo = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in plane.iter_mut() {
            *v = 20.0 + 215.0 * (*v - lo) / (hi - lo);
        }
    }
}

/// A textured three-channel tile: white noise smoothed by `smooth` grid px and
/// stretched over most of the grey range, every sample carrying data.
pub(crate) fn textured(seed: u64, smooth: f64) -> TilePlanes {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let n = SIDE * SIDE;
    let noise: Vec<f32> = (0..3 * n).map(|_| (rng.next() * 255.0) as f32).collect();
    let data = vec![true; n];
    let mut values = vec![0.0f32; 3 * n];
    blur_tile(
        &noise,
        3,
        SIDE,
        &data,
        smooth,
        &mut values,
        &mut BlurScratch::default(),
    );
    stretch(&mut values, 3);
    TilePlanes {
        values,
        data,
        side: SIDE,
        channels: 3,
    }
}

/// A grained three-channel tile: stripes across the unit direction `across`,
/// 5 grid px apart, with a little noise smoothed by 1 grid px on top. Its
/// ellipse is long along the stripes and short across them, with no blur in
/// it.
pub(crate) fn grained(seed: u64, across: [f64; 2]) -> TilePlanes {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let n = SIDE * SIDE;
    let noise: Vec<f32> = (0..3 * n).map(|_| (rng.next() * 60.0) as f32).collect();
    let data = vec![true; n];
    let mut values = vec![0.0f32; 3 * n];
    blur_tile(
        &noise,
        3,
        SIDE,
        &data,
        1.0,
        &mut values,
        &mut BlurScratch::default(),
    );
    for c in 0..3 {
        for k in 0..n {
            let (x, y) = ((k % SIDE) as f64, (k / SIDE) as f64);
            let t = across[0] * x + across[1] * y;
            values[c * n + k] +=
                (100.0 * (std::f64::consts::TAU * t / 5.0 + c as f64).sin()) as f32;
        }
    }
    stretch(&mut values, 3);
    TilePlanes {
        values,
        data,
        side: SIDE,
        channels: 3,
    }
}

/// The whole-tile self-similarity ellipse matrix of `tile`, read over its
/// samples with data.
pub(crate) fn ellipse_of(tile: &TilePlanes) -> [[f64; 2]; 2] {
    let parts = zncc_self_similarity_parts(
        &PatchTile {
            values: &tile.values,
            channels: tile.channels,
            width: tile.side,
            height: tile.side,
        },
        Some(&tile.data),
        &SelfSimilarityParams::default(),
    );
    parts.whole.ellipse.matrix
}

/// [`read_tile_ellipse`] of `t`, which must have one.
pub(crate) fn read(t: &TilePlanes) -> [[f64; 2]; 2] {
    read_tile_ellipse(&t.values, t.channels, t.side, &t.data).unwrap()
}

/// The reference rule's default window over a test tile.
pub(crate) fn window() -> Vec<f64> {
    window_weights(PatchWindow::GaussianDisk { sigma: 0.6 }, SIDE as u32)
}

/// The largest difference between two tiles' values over the samples
/// `data` marks.
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32], data: &[bool], channels: usize) -> f32 {
    let n = data.len();
    let mut worst = 0.0f32;
    for c in 0..channels {
        for k in 0..n {
            if data[k] {
                worst = worst.max((a[c * n + k] - b[c * n + k]).abs());
            }
        }
    }
    worst
}

/// An ellipse matrix of semi-axes `major` and `minor`, the major axis at
/// `deg` degrees from `x`.
pub(crate) fn ellipse(major: f64, minor: f64, deg: f64) -> [[f64; 2]; 2] {
    let (s, c) = f64::to_radians(deg).sin_cos();
    let (l1, l2) = (major * major, minor * minor);
    [
        [l1 * c * c + l2 * s * s, (l1 - l2) * c * s],
        [(l1 - l2) * c * s, l1 * s * s + l2 * c * c],
    ]
}

/// A tile of a few sinusoids, whose blur by a round Gaussian of width `σ` is
/// known exactly: each sinusoid's amplitude scales by `exp(−½ σ² |k|²)`.
pub(crate) fn sinusoids(sigma: Option<f64>) -> TilePlanes {
    let waves = [
        ([0.9, 0.3], 0.4, 40.0),
        ([-0.5, 1.1], 1.3, 30.0),
        ([1.2, -0.8], 2.0, 25.0),
        ([0.2, 0.6], 0.1, 20.0),
    ];
    let n = SIDE * SIDE;
    let mut values = vec![0.0f32; n];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let mut v = 128.0;
            for (k, phase, amp) in waves {
                let gain =
                    sigma.map_or(1.0, |s| (-0.5 * s * s * (k[0] * k[0] + k[1] * k[1])).exp());
                v += amp * gain * (k[0] * x as f64 + k[1] * y as f64 + phase).sin();
            }
            values[y * SIDE + x] = v as f32;
        }
    }
    TilePlanes {
        values,
        data: vec![true; n],
        side: SIDE,
        channels: 1,
    }
}
