// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic photographs and warps for the tests of the regions and the
//! render entry points.

use std::sync::LazyLock;

use crate::camera::image::ImageU8;
use crate::camera::{TiledPhotograph, WarpMap};

use super::{RegionPool, Sampler, ViewRegions};

/// A region pool that holds every read, shared by the tests that read
/// photographs in memory and look at no pool statistics.
pub(crate) fn test_pool() -> &'static RegionPool {
    static POOL: LazyLock<RegionPool> = LazyLock::new(|| RegionPool::with_capacity(u64::MAX));
    &POOL
}

/// A view of `source` reading through [`test_pool`].
pub(crate) fn test_regions(source: &TiledPhotograph) -> ViewRegions<'_> {
    ViewRegions::new(source, test_pool())
}

pub(crate) const SAMPLERS: [Sampler; 3] = [
    Sampler::Bilinear,
    Sampler::BilinearMip,
    Sampler::Anisotropic,
];

/// A small deterministic generator.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in `[lo, hi)`.
    pub(crate) fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() as f32 / (1u64 << 31) as f32)
    }
}

/// A textured RGB photograph, so a tap moved by a pixel changes the value.
pub(crate) fn photograph(width: u32, height: u32) -> ImageU8 {
    let mut rng = Lcg(7);
    let data = (0..width * height * 3).map(|_| rng.next() as u8).collect();
    ImageU8::new(width, height, 3, data)
}

/// A `size` x `size` map through the affine `a` (map pixel centre to source
/// pixel), with every `nan_every`-th texel NaN (0 for none).
pub(crate) fn affine_map(size: u32, a: [f32; 6], nan_every: u32) -> WarpMap {
    let mut data = Vec::with_capacity(2 * (size * size) as usize);
    for row in 0..size {
        for col in 0..size {
            let (c, r) = (col as f32 + 0.5, row as f32 + 0.5);
            if nan_every > 0 && (row * size + col) % nan_every == 3 {
                data.extend([f32::NAN, f32::NAN]);
            } else {
                data.extend([a[0] * c + a[1] * r + a[2], a[3] * c + a[4] * r + a[5]]);
            }
        }
    }
    WarpMap::new(size, size, data)
}

/// A map stretched `stretch` times along the diagonal and shrunk `shrink`
/// times across it, centred at `(cx, cy)`.
pub(crate) fn diagonal_map(size: u32, stretch: f32, shrink: f32, cx: f32, cy: f32) -> WarpMap {
    // R · diag(stretch, shrink) · Rᵀ with R a 45° turn.
    let (p, q) = (0.5 * (stretch + shrink), 0.5 * (stretch - shrink));
    let half = size as f32 / 2.0;
    affine_map(
        size,
        [p, q, cx - (p + q) * half, q, p, cy - (q + p) * half],
        0,
    )
}

/// Random warps over a `w` x `h` photograph: shrinks from 0.4 to 6 times,
/// turns, maps off every edge, NaN texels, a view shrunk past the top level,
/// and a footprint stretched 100 times on a diagonal.
pub(crate) fn maps(w: u32, h: u32) -> Vec<WarpMap> {
    let mut rng = Lcg(11);
    let (wf, hf) = (w as f32, h as f32);
    let mut maps = Vec::new();
    for k in 0..6 {
        let s = rng.range(0.4, 6.0);
        let t = rng.range(0.0, std::f32::consts::TAU);
        let (cos, sin) = (t.cos() * s, t.sin() * s);
        let skew = rng.range(-0.3, 0.3);
        let cx = rng.range(0.0, wf);
        let cy = rng.range(0.0, hf);
        let nan = if k % 2 == 0 { 7 } else { 0 };
        maps.push(affine_map(
            24,
            [cos, -sin + skew, cx - 12.0 * cos, sin, cos, cy - 12.0 * sin],
            nan,
        ));
    }
    // Off every edge: the map spans 40 px past each side.
    maps.push(affine_map(
        24,
        [
            (wf + 80.0) / 24.0,
            0.0,
            -40.0,
            0.0,
            (hf + 80.0) / 24.0,
            -40.0,
        ],
        5,
    ));
    // Off the left and top only, and off the right and bottom only.
    maps.push(affine_map(16, [2.0, 0.0, -20.0, 0.0, 2.0, -20.0], 0));
    maps.push(affine_map(
        16,
        [2.0, 0.0, wf - 12.0, 0.0, 2.0, hf - 12.0],
        0,
    ));
    // Shrunk past the top level.
    maps.push(affine_map(8, [500.0, 0.0, -1000.0, 0.0, 500.0, -800.0], 0));
    // Stretched 100 times on a diagonal.
    maps.push(diagonal_map(24, 100.0, 1.0, wf / 2.0, hf / 2.0));
    maps.push(diagonal_map(16, 100.0, 3.0, wf / 3.0, hf / 2.0));
    maps
}

/// A fresh copy of `map`'s positions, with no SVD.
pub(crate) fn copy(map: &WarpMap) -> WarpMap {
    let mut data = Vec::with_capacity(2 * (map.width() * map.height()) as usize);
    for row in 0..map.height() {
        for col in 0..map.width() {
            let (x, y) = map.get(col, row);
            data.extend([x, y]);
        }
    }
    WarpMap::new(map.width(), map.height(), data)
}
