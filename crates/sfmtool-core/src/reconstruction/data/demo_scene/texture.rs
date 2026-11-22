// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The demo scene's surfaces and their textures: [`value_noise`], the value
//! noise the bench tests' synthetic planes also carry, and the tinted colour
//! of each [`Surface`] built from it.

/// Value noise: pseudo-random levels on a square lattice `spacing` apart,
/// blended between lattice nodes with the smoothstep `3t² − 2t³`, so that no
/// shift of it resembles another. The levels come from a hash of the lattice
/// node and `seed`, with no random number generator, and the value is in
/// `[-1, 1]`; each caller maps it to its own range.
///
/// It is computed in `f64` from `+`, `−`, `×`, `÷` and `floor`, so the same
/// arguments give the same bits on every platform. `x / spacing` must fit an
/// `i64` lattice index; a coordinate beyond that saturates to the last node.
pub(crate) fn value_noise(x: f64, y: f64, spacing: f64, seed: u64) -> f64 {
    let level = |i: i64, j: i64| {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ seed.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        h ^= h >> 29;
        h = h.wrapping_mul(0x1656_67B1_9E37_79F9);
        h ^= h >> 32;
        (h % 1024) as f64 / 1023.0 * 2.0 - 1.0
    };
    let (u, v) = (x / spacing, y / spacing);
    let (i, j) = (u.floor() as i64, v.floor() as i64);
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let (s, t) = (smooth(u - u.floor()), smooth(v - v.floor()));
    let top = level(i, j) * (1.0 - s) + level(i.wrapping_add(1), j) * s;
    let bottom =
        level(i, j.wrapping_add(1)) * (1.0 - s) + level(i.wrapping_add(1), j.wrapping_add(1)) * s;
    top * (1.0 - t) + bottom * t
}

/// The lattice spacing of every surface's noise, in metres: about 2.8 px a
/// cell on a surface 4 m from a camera of focal length 560 px that faces it.
pub(super) const NOISE_SPACING: f64 = 0.02;

/// The cube's first horizontal axis: the cube is turned about `+z` so that
/// its faces are not square to the camera arc, by a turn of about 36.9°
/// written with no trigonometric function.
pub(super) const CUBE_A: [f64; 3] = [0.8, 0.6, 0.0];
/// The cube's second horizontal axis, `ẑ × a`.
pub(super) const CUBE_B: [f64; 3] = [-0.6, 0.8, 0.0];

/// One surface of the scene: the ground, or a face of the cube named by its
/// outward normal. The discriminant is the surface's index in the scene's
/// noise seeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Surface {
    /// The plane `z = 0`, seen from above.
    Ground = 0,
    /// The cube's top, outward normal `+z`.
    Top = 1,
    /// The side face with outward normal `-a`.
    MinusA = 2,
    /// The side face with outward normal `-b`.
    MinusB = 3,
    /// The side face with outward normal `+a`.
    PlusA = 4,
    /// The side face with outward normal `+b`.
    PlusB = 5,
    /// The cube's base, outward normal `-z`, which rests on the ground.
    Base = 6,
}

impl Surface {
    /// Every surface, in the order their noise seeds count.
    pub(super) const ALL: [Surface; 7] = [
        Surface::Ground,
        Surface::Top,
        Surface::MinusA,
        Surface::MinusB,
        Surface::PlusA,
        Surface::PlusB,
        Surface::Base,
    ];

    /// The surface's tint, RGB. The tints have BT.709 lumas within 5% of each
    /// other, near 180, so the noise has the same grey contrast on every
    /// surface; the base is never seen and takes the ground's.
    pub(super) fn tint(self) -> [u8; 3] {
        match self {
            Surface::Ground | Surface::Base => [215, 180, 130],
            Surface::Top => [150, 200, 150],
            Surface::MinusA => [250, 160, 150],
            Surface::MinusB => [150, 180, 250],
            Surface::PlusA => [235, 160, 235],
            Surface::PlusB => [120, 195, 200],
        }
    }

    /// The seed of the surface's noise under the scene's `seed`: a different
    /// one for every surface, so the cube does not blend into the ground and
    /// two faces meeting at an edge do not continue each other's pattern.
    fn noise_seed(self, seed: u64) -> u64 {
        seed.wrapping_mul(Surface::ALL.len() as u64)
            .wrapping_add(self as u64)
    }

    /// The surface's own 2D coordinates of the world point `x` on it, in
    /// metres: `(x, y)` on the ground, and on a cube face the coordinates
    /// along its two in-face axes from one corner, each in `[0, 1]` on the
    /// face. A side face with outward normal `n` takes `ẑ × n` and `ẑ`, the
    /// top `a` and `b`, and the base `a` and `-b`, so each pair's cross
    /// product is the outward normal.
    pub(super) fn coordinates(self, x: [f64; 3]) -> (f64, f64) {
        let pa = x[0] * CUBE_A[0] + x[1] * CUBE_A[1];
        let pb = x[0] * CUBE_B[0] + x[1] * CUBE_B[1];
        match self {
            Surface::Ground => (x[0], x[1]),
            Surface::Top => (pa + 0.5, pb + 0.5),
            Surface::Base => (pa + 0.5, 0.5 - pb),
            // ẑ × (−a) = −b, ẑ × (−b) = a, ẑ × a = b, ẑ × b = −a.
            Surface::MinusA => (0.5 - pb, x[2]),
            Surface::MinusB => (pa + 0.5, x[2]),
            Surface::PlusA => (pb + 0.5, x[2]),
            Surface::PlusB => (0.5 - pa, x[2]),
        }
    }

    /// The noise factor of the world point `x` on the surface, under the
    /// scene's `seed`: `0.5 + 0.4 · noise`, in `[0.1, 0.9]`, so the largest
    /// tint channel, 250, stays below 255.
    pub(super) fn noise_factor(self, x: [f64; 3], seed: u64) -> f32 {
        let (s, t) = self.coordinates(x);
        let noise = value_noise(s, t, NOISE_SPACING, self.noise_seed(seed));
        0.5 + 0.4 * noise as f32
    }

    /// The surface's colour at the world point `x`, RGB on `[0, 255]`
    /// unrounded: its tint scaled by [`noise_factor`](Self::noise_factor),
    /// one factor for all three channels, with no lighting.
    ///
    /// The colour is `f32`: it is an 8-bit level computed from an `f32`
    /// factor, which `f32`'s 24 bits hold to about `10⁻⁵` of a level.
    pub(super) fn colour(self, x: [f64; 3], seed: u64) -> [f32; 3] {
        let factor = self.noise_factor(x, seed);
        self.tint().map(|c| c as f32 * factor)
    }
}
