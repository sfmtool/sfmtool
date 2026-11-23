// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The demo scene's surfaces and their textures: the noise functions, and the
//! tinted colour of each [`Surface`] built from them. [`perlin_fbm`], a sum of
//! octaves of [`perlin_noise`], is the surfaces' texture; `value_noise`, built
//! only for tests, is the noise the bench tests' synthetic planes carry.
//!
//! Every noise function here is computed in `f64` from `+`, `−`, `×`, `÷` and
//! `floor`, with its lattice's values drawn from [`lattice_hash`] rather than
//! a random number generator, so the same arguments give the same bits on
//! every platform. A coordinate divided by a lattice spacing must fit an
//! `i64` lattice index; a coordinate beyond that saturates to the last node.

/// A hash of the lattice node `(i, j)` and `seed`, from which each noise
/// function draws the node's level or gradient, and the point sampling the
/// offsets of its grid points.
pub(super) fn lattice_hash(i: i64, j: i64, seed: u64) -> u64 {
    let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 29;
    h = h.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 32;
    h
}

/// Value noise: pseudo-random levels on a square lattice `spacing` apart,
/// blended between lattice nodes with the smoothstep `3t² − 2t³`, so that no
/// shift of it resembles another. The levels come from [`lattice_hash`] of
/// the node and `seed`, and the value is in `[-1, 1]`; each caller maps it to
/// its own range.
#[cfg(test)]
pub(crate) fn value_noise(x: f64, y: f64, spacing: f64, seed: u64) -> f64 {
    let level = |i: i64, j: i64| (lattice_hash(i, j, seed) % 1024) as f64 / 1023.0 * 2.0 - 1.0;
    let (u, v) = (x / spacing, y / spacing);
    let (i, j) = (u.floor() as i64, v.floor() as i64);
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let (s, t) = (smooth(u - u.floor()), smooth(v - v.floor()));
    let top = level(i, j) * (1.0 - s) + level(i.wrapping_add(1), j) * s;
    let bottom =
        level(i, j.wrapping_add(1)) * (1.0 - s) + level(i.wrapping_add(1), j.wrapping_add(1)) * s;
    top * (1.0 - t) + bottom * t
}

/// `1 / √2`, the components of the diagonal gradients.
const H: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// The gradients [`perlin_noise`] picks from: the eight unit vectors 45°
/// apart, written as constants rather than computed with `sin` and `cos`.
const GRADIENTS: [[f64; 2]; 8] = [
    [1.0, 0.0],
    [H, H],
    [0.0, 1.0],
    [-H, H],
    [-1.0, 0.0],
    [-H, -H],
    [0.0, -1.0],
    [H, -H],
];

/// The largest magnitude [`perlin_noise`] takes, `1 / √2`. At a cell's
/// centre each corner is `1 / √2` cells away and the fades are `1/2`, so the
/// noise is the mean of the four gradients' dot products with those offsets,
/// `1 / √2` when all four gradients point at the centre; a search over every
/// choice of the four gradients at 41 × 41 points of the cell finds no larger
/// magnitude elsewhere.
pub(super) const PERLIN_BOUND: f64 = H;

/// Perlin's gradient noise in 2D: a pseudo-random unit gradient at each node
/// of a square lattice `spacing` apart, and between nodes the four corners'
/// linear ramps `g · (p − node)` blended with Perlin's quintic fade
/// `6t⁵ − 15t⁴ + 10t³`. It is 0 at every lattice node and has no level of
/// its own there, so unlike value noise its features sit between the nodes
/// rather than on them. Each gradient is one of eight unit vectors 45° apart,
/// picked by [`lattice_hash`] of the node and `seed`. The value is in
/// `[-PERLIN_BOUND, PERLIN_BOUND]`, and is continuous with a continuous first
/// and second derivative.
pub(super) fn perlin_noise(x: f64, y: f64, spacing: f64, seed: u64) -> f64 {
    let (u, v) = (x / spacing, y / spacing);
    let (fi, fj) = (u.floor(), v.floor());
    let (i, j) = (fi as i64, fj as i64);
    let (du, dv) = (u - fi, v - fj);
    let ramp = |i: i64, j: i64, du: f64, dv: f64| {
        let g = GRADIENTS[(lattice_hash(i, j, seed) >> 61) as usize];
        g[0] * du + g[1] * dv
    };
    let (i1, j1) = (i.wrapping_add(1), j.wrapping_add(1));
    let fade = |t: f64| t * t * t * (t * (6.0 * t - 15.0) + 10.0);
    let (s, t) = (fade(du), fade(dv));
    let top = ramp(i, j, du, dv) * (1.0 - s) + ramp(i1, j, du - 1.0, dv) * s;
    let bottom = ramp(i, j1, du, dv - 1.0) * (1.0 - s) + ramp(i1, j1, du - 1.0, dv - 1.0) * s;
    top * (1.0 - t) + bottom * t
}

/// The octaves of a [`perlin_fbm`]: `octaves` layers of [`perlin_noise`],
/// the first with lattice spacing `spacing`, each next one's spacing the
/// last one's divided by `lacunarity` and its amplitude the last one's times
/// `gain`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FbmOptions {
    /// The coarsest octave's lattice spacing, in the caller's units.
    pub(super) spacing: f64,
    /// The number of octaves.
    pub(super) octaves: u32,
    /// The ratio of one octave's lattice frequency to the last one's.
    pub(super) lacunarity: f64,
    /// The ratio of one octave's amplitude to the last one's.
    pub(super) gain: f64,
}

impl FbmOptions {
    /// The sum of the octaves' amplitudes, `1 + gain + … + gainⁿ⁻¹`.
    pub(super) fn amplitude_sum(&self) -> f64 {
        let mut sum = 0.0;
        let mut amplitude = 1.0;
        for _ in 0..self.octaves {
            sum += amplitude;
            amplitude *= self.gain;
        }
        sum
    }
}

/// The seed of octave `k` of a [`perlin_fbm`] under `seed`.
pub(super) fn octave_seed(seed: u64, k: u32) -> u64 {
    lattice_hash(i64::from(k), -1, seed)
}

/// The shift of octave `k`'s lattice in a [`perlin_fbm`], in its own cells
/// along both axes: the fractional part of `k · 0.618034`.
pub(super) fn octave_shift(k: u32) -> f64 {
    (f64::from(k) * 0.618034).fract()
}

/// Fractional Brownian motion (fBm) of [`perlin_noise`]: the sum of the
/// octaves `options` describes, octave `k` with amplitude `gainᵏ`, divided by
/// the sum of the amplitudes, so the value is in `[-PERLIN_BOUND,
/// PERLIN_BOUND]` like one octave's.
///
/// Octave `k` takes its own seed, [`octave_seed`], a hash of `seed` and `k`,
/// and its lattice is shifted by [`octave_shift`] of its own cell along both
/// axes. Without the
/// seeds, every octave would draw the same gradients at the same node
/// indices, a pattern repeated at each scale about the origin; without the
/// shift, with an integer lacunarity every node of a coarse octave would be a
/// node of every finer one, where all of them are 0.
pub(super) fn perlin_fbm(x: f64, y: f64, options: &FbmOptions, seed: u64) -> f64 {
    let mut sum = 0.0;
    let mut spacing = options.spacing;
    let mut amplitude = 1.0;
    for k in 0..options.octaves {
        let shift = octave_shift(k) * spacing;
        sum += amplitude * perlin_noise(x + shift, y + shift, spacing, octave_seed(seed, k));
        spacing /= options.lacunarity;
        amplitude *= options.gain;
    }
    sum / options.amplitude_sum()
}

/// The surfaces' texture, a [`perlin_fbm`] of five octaves from a lattice
/// spacing of 0.32 m down to 0.02 m, each octave's amplitude 0.8 of the last
/// one's.
pub(super) const TEXTURE_FBM: FbmOptions = FbmOptions {
    spacing: 0.32,
    octaves: 5,
    lacunarity: 2.0,
    gain: 0.8,
};

/// The magnitude of [`TEXTURE_FBM`] that the texture maps to the ends of its
/// grey range, 0.665 of [`PERLIN_BOUND`]: a little above the largest
/// magnitude the fBm takes on any surface of the scenes of seeds 0 to 15,
/// 0.654 of the bound, measured at every 2 cm on the ground and every 2 mm on
/// the faces.
pub(super) const TEXTURE_PEAK: f64 = 0.665 * PERLIN_BOUND;

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
    pub(super) fn noise_seed(self, seed: u64) -> u64 {
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

    /// The surface's outward normal: `+z` on the ground and the top, `-z` on
    /// the base, and on a side face the axis it is named by.
    pub(super) fn normal(self) -> [f64; 3] {
        let neg = |v: [f64; 3]| v.map(|c| -c);
        match self {
            Surface::Ground | Surface::Top => [0.0, 0.0, 1.0],
            Surface::Base => [0.0, 0.0, -1.0],
            Surface::MinusA => neg(CUBE_A),
            Surface::MinusB => neg(CUBE_B),
            Surface::PlusA => CUBE_A,
            Surface::PlusB => CUBE_B,
        }
    }

    /// The surface's frame, the inverse of
    /// [`coordinates`](Self::coordinates): the world point where both
    /// coordinates are 0 and the two in-face unit axes `(u, v)` they run
    /// along, so the point at coordinates `(s, t)` is `origin + s·u + t·v`.
    /// `u × v` is the outward normal.
    pub(super) fn frame(self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        const Z: [f64; 3] = [0.0, 0.0, 1.0];
        let neg = |v: [f64; 3]| v.map(|c| -c);
        // The cube's corner at `α·a + β·b + γ·z`.
        let corner = |alpha: f64, beta: f64, gamma: f64| {
            [0, 1, 2].map(|k| alpha * CUBE_A[k] + beta * CUBE_B[k] + gamma * Z[k])
        };
        match self {
            Surface::Ground => ([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            Surface::Top => (corner(-0.5, -0.5, 1.0), CUBE_A, CUBE_B),
            Surface::Base => (corner(-0.5, 0.5, 0.0), CUBE_A, neg(CUBE_B)),
            Surface::MinusA => (corner(-0.5, 0.5, 0.0), neg(CUBE_B), Z),
            Surface::MinusB => (corner(-0.5, -0.5, 0.0), CUBE_A, Z),
            Surface::PlusA => (corner(0.5, -0.5, 0.0), CUBE_B, Z),
            Surface::PlusB => (corner(0.5, 0.5, 0.0), neg(CUBE_A), Z),
        }
    }

    /// The noise factor of the world point `x` on the surface, under the
    /// scene's `seed`: `0.5 + 0.4 · fbm / TEXTURE_PEAK`, in `[0.1, 0.9]`
    /// wherever the fBm is under [`TEXTURE_PEAK`], so the largest tint
    /// channel, 250, stays below 255.
    pub(super) fn noise_factor(self, x: [f64; 3], seed: u64) -> f32 {
        let (s, t) = self.coordinates(x);
        let fbm = perlin_fbm(s, t, &TEXTURE_FBM, self.noise_seed(seed));
        0.5 + (0.4 / TEXTURE_PEAK * fbm) as f32
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
