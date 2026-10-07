// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::patch::normal_refine::{window_weights, PatchWindow};
use crate::patch::self_similarity::{zncc_self_similarity_parts, PatchTile, SelfSimilarityParams};
use crate::progress::Progress;

const SIDE: usize = 24;

/// A small deterministic generator, so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A textured three-channel tile: white noise smoothed by `smooth` grid px and
/// stretched over most of the grey range, every sample carrying data.
fn textured(seed: u64, smooth: f64) -> TilePlanes {
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
        BlurCovariance::isotropic(smooth),
        &mut values,
        &mut BlurScratch::default(),
    );
    // Stretch each channel to 20 ..= 235.
    for c in 0..3 {
        let plane = &mut values[c * n..(c + 1) * n];
        let lo = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in plane.iter_mut() {
            *v = 20.0 + 215.0 * (*v - lo) / (hi - lo);
        }
    }
    TilePlanes {
        values,
        data,
        side: SIDE,
        channels: 3,
    }
}

fn ellipse_of(tile: &TilePlanes) -> [[f64; 2]; 2] {
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

fn window() -> Vec<f64> {
    window_weights(PatchWindow::GaussianDisk { sigma: 0.6 }, SIDE as u32)
}

fn max_abs_diff(a: &[f32], b: &[f32], data: &[bool], channels: usize) -> f32 {
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

// --------------------------------------------------------------- the mapping

#[test]
fn blur_sigma_is_zero_without_a_difference_and_grows_with_it() {
    assert_eq!(blur_sigma(0.7, 0.7), 0.0);
    assert_eq!(blur_sigma(0.5, 0.7), 0.0);
    assert_eq!(blur_sigma(f64::NAN, 0.7), 0.0);
    let mut last = 0.0;
    for longer in [0.75, 0.9, 1.2, 1.8, 2.5, 3.0] {
        let s = blur_sigma(longer, 0.7);
        assert!(s > last, "{longer}: {s} after {last}");
        last = s;
    }
    // The fitted values at the mapping's own points.
    let s = blur_sigma(1.0, 0.5);
    assert!((s - 0.9975 * 0.5f64.powf(0.2613) * 3.0f64.powf(0.1744)).abs() < 1e-12);
    // A sharper length under the floor is read as the floor.
    assert_eq!(blur_sigma(1.0, 0.0), blur_sigma(1.0, BLUR_MAP_MIN_LENGTH));
    assert!(blur_sigma(3.0, 0.0) <= MAX_BLUR_SIGMA);
}

#[test]
fn equal_ellipses_blur_nothing() {
    let e = [[0.8, 0.1], [0.1, 0.3]];
    assert!(pair_blur(&e, &e, 1.0).is_none());
    assert!(pair_blur_for(&e, &[[2.0, 0.0], [0.0, 2.0]], PairMatching::Plain).is_none());
    let unread = [[f64::NAN, 0.0], [0.0, 1.0]];
    assert!(pair_blur(&unread, &e, 1.0).is_none());
}

#[test]
fn each_tile_is_blurred_only_along_the_directions_it_is_sharper() {
    // `a` is sharp along x and blurry along y; `b` the other way round.
    let a = [[0.16, 0.0], [0.0, 1.44]];
    let b = [[1.44, 0.0], [0.0, 0.16]];
    let blur = pair_blur(&a, &b, 1.0);
    assert!(
        blur.a.xx > 0.5 && blur.a.yy.abs() < 1e-9 && blur.a.xy.abs() < 1e-9,
        "{blur:?}"
    );
    assert!(
        blur.b.yy > 0.5 && blur.b.xx.abs() < 1e-9 && blur.b.xy.abs() < 1e-9,
        "{blur:?}"
    );
    assert!((blur.a.xx - blur.b.yy).abs() < 1e-12);

    // A rotated pair: `b` is `a` stretched along 30°.
    let (s, c) = 30f64.to_radians().sin_cos();
    let rot = |l1: f64, l2: f64| {
        [
            [l1 * c * c + l2 * s * s, (l1 - l2) * c * s],
            [(l1 - l2) * c * s, l1 * s * s + l2 * c * c],
        ]
    };
    let blur = pair_blur(&rot(0.25, 0.25), &rot(2.25, 0.25), 1.0);
    assert!(blur.b.is_zero());
    // The blur lies along 30°: its covariance has that direction as its only
    // eigenvector with a non-zero eigenvalue.
    let along = blur.a.xx * c * c + 2.0 * blur.a.xy * c * s + blur.a.yy * s * s;
    let across = blur.a.xx * s * s - 2.0 * blur.a.xy * c * s + blur.a.yy * c * c;
    assert!(
        (along - blur_sigma(1.5, 0.5).powi(2)).abs() < 1e-9,
        "{along}"
    );
    assert!(across.abs() < 1e-9, "{across}");
}

#[test]
fn the_skip_ratio_leaves_a_small_difference_plain() {
    let a = [[0.49, 0.0], [0.0, 0.49]]; // 0.7 along both axes
    let b = [[0.64, 0.0], [0.0, 0.64]]; // 0.8: a ratio of 1.14
    assert!(!pair_blur(&a, &b, 1.0).is_none());
    assert!(!pair_blur(&a, &b, 1.1).is_none());
    assert!(pair_blur(&a, &b, 1.2).is_none());
    assert!(pair_blur_for(&a, &b, PairMatching::BlurMatchedAboveRatio(1.2)).is_none());
    assert!(!pair_blur_for(&a, &b, PairMatching::BlurMatched).is_none());
    // A ratio is per direction: one axis past it is blurred, the other not.
    let c = [[0.64, 0.0], [0.0, 1.96]]; // 0.8 and 1.4
    let blur = pair_blur(&a, &c, 1.2);
    assert!(blur.a.yy > 0.0 && blur.a.xx.abs() < 1e-12, "{blur:?}");
}

#[test]
fn matching_names_round_trip() {
    for m in [
        PairMatching::Plain,
        PairMatching::BlurMatched,
        PairMatching::BlurMatchedAboveRatio(1.3),
    ] {
        assert_eq!(PairMatching::from_name(m.name(), 1.3), Some(m));
    }
    assert_eq!(PairMatching::from_name("blurred", 1.0), None);
    assert_eq!(PairMatching::Plain.min_ratio(), None);
    assert_eq!(
        PairMatching::BlurMatchedAboveRatio(0.5).min_ratio(),
        Some(1.0)
    );
    for k in [
        BlurMatchKernel::Anisotropic,
        BlurMatchKernel::IsotropicLadder,
    ] {
        assert_eq!(BlurMatchKernel::from_name(k.name()), Some(k));
    }
}

#[test]
fn ladder_levels_snap_to_the_nearest_in_the_logarithm() {
    assert_eq!(ladder_level(0.2), None);
    assert_eq!(ladder_level(0.5), Some(0));
    assert_eq!(ladder_level(0.7), Some(1));
    assert_eq!(ladder_level(1.0), Some(2));
    assert_eq!(ladder_level(1.45), Some(3));
    assert_eq!(ladder_level(9.0), Some(5));
    assert_eq!(ladder_level(f64::NAN), None);
}

// ------------------------------------------------------------------ the blur

/// The two passes, and the direct 2-D convolution they stand in for, against
/// the exact blur of a tile of sinusoids, over covariances at several sizes,
/// elongations and angles. The interior is compared, where no tap reaches
/// past the tile's edge. The narrow blurs, at widths 0.5 to 0.7, are those
/// the review cases' pairs mostly get.
#[test]
fn the_two_passes_match_the_exact_blur_and_the_direct_2d_blur() {
    let tile = sinusoids(None);
    let interior =
        |k: usize| (7..SIDE - 7).contains(&(k / SIDE)) && (7..SIDE - 7).contains(&(k % SIDE));
    let (mut worst_fast, mut worst_direct) = (0.0f32, 0.0f32);
    for (major, minor) in [
        (0.5, 0.0),
        (0.5, 0.5),
        (0.6, 0.3),
        (0.7, 0.0),
        (0.7, 0.7),
        (1.0, 0.6),
        (1.5, 0.6),
        (2.0, 0.8),
        (1.5, 1.0),
        (2.0, 1.5),
        (1.0, 0.0),
        (1.5, 0.0),
    ] {
        for deg in [0.0, 15.0, 30.0, 45.0, 60.0, 80.0, 100.0, 135.0, 170.0] {
            let cov = ellipse_cov(major, minor, deg);
            let exact = sinusoids(Some(cov));
            let error = |v: &[f32]| {
                (0..SIDE * SIDE)
                    .filter(|&k| interior(k))
                    .map(|k| (v[k] - exact.values[k]).abs())
                    .fold(0.0f32, f32::max)
            };
            let mut fast = vec![0.0f32; tile.values.len()];
            blur_tile(
                &tile.values,
                1,
                SIDE,
                &tile.data,
                cov,
                &mut fast,
                &mut BlurScratch::default(),
            );
            // The sinusoids span about ±115 grey levels.
            let e = error(&fast);
            worst_fast = worst_fast.max(e);
            assert!(
                e < 4.0,
                "σ {major}×{minor} at {deg}°: the passes are {e} off"
            );
            // The direct blur samples the kernel as it is, which falls short
            // of its width under about 0.6.
            if minor >= 0.6 {
                let mut direct = vec![0.0f32; tile.values.len()];
                blur_tile_direct(&tile.values, 1, SIDE, &tile.data, cov, &mut direct);
                let e = error(&direct);
                worst_direct = worst_direct.max(e);
                assert!(
                    e < 1.0,
                    "σ {major}×{minor} at {deg}°: the 2-D blur is {e} off"
                );
            }
        }
    }
    eprintln!("largest error: passes {worst_fast}, direct 2-D {worst_direct} grey levels");
}

/// A covariance of semi-axes `major` and `minor` with the major axis at `deg`
/// degrees from `x`.
fn ellipse_cov(major: f64, minor: f64, deg: f64) -> BlurCovariance {
    let (s, c) = f64::to_radians(deg).sin_cos();
    let (l1, l2) = (major * major, minor * minor);
    BlurCovariance {
        xx: l1 * c * c + l2 * s * s,
        xy: (l1 - l2) * c * s,
        yy: l1 * s * s + l2 * c * c,
    }
}

/// One scratch reused over many blurs of different shapes, tile sides and
/// missing samples gives every blur bit for bit what a fresh scratch gives:
/// nothing a call leaves in the buffers reaches the next.
#[test]
fn a_reused_scratch_blurs_as_a_fresh_one() {
    let mut rng = Rng(12345);
    let mut scratch = BlurScratch::default();
    for side in [13usize, 16, 24, 32, 24, 13] {
        let n = side * side;
        for _ in 0..400 {
            let values: Vec<f32> = (0..3 * n).map(|_| (rng.next() * 255.0) as f32).collect();
            let gaps = rng.next() < 0.5;
            let data: Vec<bool> = (0..n).map(|_| !gaps || rng.next() > 0.2).collect();
            let major = 0.3 + rng.next() * 2.5;
            let minor = if rng.next() < 0.4 {
                0.0
            } else {
                rng.next() * major
            };
            let cov = ellipse_cov(major, minor, rng.next() * 180.0);
            let mut reused = vec![0.0f32; 3 * n];
            let mut fresh = vec![0.0f32; 3 * n];
            blur_tile(&values, 3, side, &data, cov, &mut reused, &mut scratch);
            blur_tile(
                &values,
                3,
                side,
                &data,
                cov,
                &mut fresh,
                &mut BlurScratch::default(),
            );
            let same = reused
                .iter()
                .zip(&fresh)
                .all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "side {side}, {cov:?}: the reused scratch differs");
        }
    }
}

/// A pair's readings in a track are those of the pair read alone, bit for
/// bit, under both kernels: the other pairs, read with the same scratch and
/// ladder, do not reach it.
#[test]
fn a_pair_reads_the_same_in_a_track_as_alone() {
    let mut rng = Rng(777);
    let k = 8;
    let tiles: Vec<TilePlanes> = (0..k).map(|v| textured(100 + v as u64, 0.8)).collect();
    let refs: Vec<&TilePlanes> = tiles.iter().collect();
    let ellipses: Vec<Option<[[f64; 2]; 2]>> = (0..k)
        .map(|_| {
            let major = 0.3 + rng.next() * 2.0;
            let c = ellipse_cov(major, major * (0.3 + 0.7 * rng.next()), rng.next() * 180.0);
            Some([[c.xx, c.xy], [c.xy, c.yy]])
        })
        .collect();
    let window = PatchWindow::GaussianDisk { sigma: 0.6 };
    let bits = |r: &[[f64; 3]; 3]| r.map(|row| row.map(f64::to_bits));
    for kernel in [
        BlurMatchKernel::Anisotropic,
        BlurMatchKernel::IsotropicLadder,
    ] {
        let read = |tiles: &[&TilePlanes], ellipses: &[Option<[[f64; 2]; 2]>]| {
            blur_matched_pairs(
                tiles,
                ellipses,
                PairMatching::BlurMatched,
                kernel,
                window,
                None,
                &Progress::none(),
            )
        };
        let all = read(&refs, &ellipses);
        assert!(all.pairs_blurred > k);
        for a in 0..k {
            for b in (a + 1)..k {
                let one = read(&[refs[a], refs[b]], &[ellipses[a], ellipses[b]]);
                assert_eq!(
                    one.whole[1].to_bits(),
                    all.whole[a * k + b].to_bits(),
                    "{kernel:?}, pair {a}-{b}: whole"
                );
                assert_eq!(
                    bits(&one.grid[1]),
                    bits(&all.grid[a * k + b]),
                    "{kernel:?}, pair {a}-{b}: grid"
                );
            }
        }
    }
}

/// The passes against the direct 2-D blur on a textured tile with samples
/// missing, where there is no exact answer: the two agree to a ZNCC near 1.
#[test]
fn the_two_passes_match_the_direct_2d_blur_round_missing_samples() {
    let mut tile = textured(3, 1.0);
    for k in 0..SIDE * SIDE {
        if (k * 7919) % 11 == 0 || k % SIDE < 2 {
            tile.data[k] = false;
        }
    }
    for (major, minor) in [(1.0, 0.6), (1.5, 0.8), (2.0, 1.2)] {
        for deg in [0.0, 30.0, 45.0, 75.0, 120.0] {
            let (s, c) = f64::to_radians(deg).sin_cos();
            let (l1, l2): (f64, f64) = (major * major, minor * minor);
            let cov = BlurCovariance {
                xx: l1 * c * c + l2 * s * s,
                xy: (l1 - l2) * c * s,
                yy: l1 * s * s + l2 * c * c,
            };
            let mut fast = vec![0.0f32; tile.values.len()];
            let mut direct = vec![0.0f32; tile.values.len()];
            blur_tile(
                &tile.values,
                3,
                SIDE,
                &tile.data,
                cov,
                &mut fast,
                &mut BlurScratch::default(),
            );
            blur_tile_direct(&tile.values, 3, SIDE, &tile.data, cov, &mut direct);
            let a = TilePlanes {
                values: fast,
                ..tile.clone()
            };
            let b = TilePlanes {
                values: direct,
                ..tile.clone()
            };
            let z = pair_zncc_readings(&a, &b, &window()).whole;
            assert!(z > 0.998, "σ {major}×{minor} at {deg}°: ZNCC {z}");
        }
    }
}

// A sample without data takes no part in the blur, whatever it holds, and
/// keeps its own value.
#[test]
fn samples_without_data_neither_give_nor_take() {
    let tile = textured(5, 0.8);
    let mut data = tile.data.clone();
    for (k, d) in data.iter_mut().enumerate() {
        if (k / SIDE + k % SIDE).is_multiple_of(5) {
            *d = false;
        }
    }
    let mut garbage = tile.values.clone();
    for c in 0..3 {
        for k in 0..SIDE * SIDE {
            if !data[k] {
                garbage[c * SIDE * SIDE + k] = 255.0 * ((k % 2) as f32);
            }
        }
    }
    let cov = BlurCovariance {
        xx: 1.2,
        xy: 0.4,
        yy: 0.6,
    };
    let mut clean = vec![0.0f32; tile.values.len()];
    let mut dirty = vec![0.0f32; tile.values.len()];
    blur_tile(
        &tile.values,
        3,
        SIDE,
        &data,
        cov,
        &mut clean,
        &mut BlurScratch::default(),
    );
    blur_tile(
        &garbage,
        3,
        SIDE,
        &data,
        cov,
        &mut dirty,
        &mut BlurScratch::default(),
    );
    assert!(max_abs_diff(&clean, &dirty, &data, 3) < 1e-3);
    for c in 0..3 {
        for k in 0..SIDE * SIDE {
            if !data[k] {
                assert_eq!(dirty[c * SIDE * SIDE + k], garbage[c * SIDE * SIDE + k]);
            }
        }
    }
    // A flat tile stays flat at every sample with data, up to the edge and
    // round the holes, which a blur that leaked zeros from them would darken.
    let flat = vec![100.0f32; 3 * SIDE * SIDE];
    let mut out = vec![0.0f32; flat.len()];
    blur_tile(
        &flat,
        3,
        SIDE,
        &data,
        cov,
        &mut out,
        &mut BlurScratch::default(),
    );
    for (k, &v) in out.iter().enumerate() {
        if data[k % (SIDE * SIDE)] {
            assert!((v - 100.0).abs() < 1e-3, "{k}: {v}");
        }
    }
}

/// A 1-D blur along a direction moves nothing across it: a tile constant
/// along the direction comes back unchanged.
#[test]
fn a_one_dimensional_blur_leaves_a_texture_along_it_alone() {
    // Stripes along 90° (constant down each column).
    let n = SIDE * SIDE;
    let values: Vec<f32> = (0..n)
        .map(|k| if (k % SIDE) % 4 < 2 { 40.0 } else { 200.0 })
        .collect();
    let data = vec![true; n];
    let mut out = vec![0.0f32; n];
    blur_tile(
        &values,
        1,
        SIDE,
        &data,
        BlurCovariance::along([0.0, 1.0], 1.5),
        &mut out,
        &mut BlurScratch::default(),
    );
    assert!(max_abs_diff(&values, &out, &data, 1) < 1e-3);
    // Across the stripes it smooths them.
    blur_tile(
        &values,
        1,
        SIDE,
        &data,
        BlurCovariance::along([1.0, 0.0], 1.5),
        &mut out,
        &mut BlurScratch::default(),
    );
    assert!(max_abs_diff(&values, &out, &data, 1) > 30.0);
}

// --------------------------------------------------------- blur-matched ZNCC

/// A tile and a blurred copy of it: plain ZNCC charges the copy for the detail
/// it lost, and blur matching recovers a ZNCC near 1. The texture's ellipse
/// is about 0.35 grid px, as a sharp view's is.
#[test]
fn blur_matching_a_blurred_copy_recovers_its_zncc() {
    for (seed, cov) in [
        (11, BlurCovariance::isotropic(1.0)),
        (12, BlurCovariance::isotropic(1.6)),
        (13, BlurCovariance::along([0.8, 0.6], 1.8)),
        (14, BlurCovariance::along([0.0, 1.0], 1.4)),
    ] {
        let sharp = textured(seed, 1.4);
        let blurry = sharp.blurred(cov, &mut BlurScratch::default());
        let (es, eb) = (ellipse_of(&sharp), ellipse_of(&blurry));
        let plain = pair_zncc_readings(&sharp, &blurry, &window()).whole;
        let pairs = blur_matched_pairs(
            &[&sharp, &blurry],
            &[Some(es), Some(eb)],
            PairMatching::BlurMatched,
            BlurMatchKernel::Anisotropic,
            PatchWindow::GaussianDisk { sigma: 0.6 },
            None,
            &Progress::none(),
        );
        let matched = pairs.whole[1];
        assert_eq!(pairs.pairs_blurred, 1);
        assert!(
            matched > 0.965,
            "{cov:?}: blur-matched {matched}, plain {plain}"
        );
        assert!(
            matched > plain + 0.02,
            "{cov:?}: blur-matched {matched}, plain {plain}"
        );
    }
}

/// Two equally sharp views are not blurred, and read what plain ZNCC reads.
#[test]
fn equally_sharp_views_read_plain() {
    let a = textured(21, 0.7);
    let b = textured(21, 0.7);
    let e = ellipse_of(&a);
    for kernel in [
        BlurMatchKernel::Anisotropic,
        BlurMatchKernel::IsotropicLadder,
    ] {
        let pairs = blur_matched_pairs(
            &[&a, &b],
            &[Some(e), Some(e)],
            PairMatching::BlurMatched,
            kernel,
            PatchWindow::GaussianDisk { sigma: 0.6 },
            None,
            &Progress::none(),
        );
        assert_eq!(pairs.pairs_blurred, 0);
        let plain = pair_zncc_readings(&a, &b, &window());
        assert_eq!(pairs.whole[1], plain.whole);
        assert_eq!(pairs.grid[1], plain.grid);
        assert!((pairs.whole[1] - 1.0).abs() < 1e-9);
    }
}

/// The ladder blurs each view at most once per level, and on an isotropic
/// blur reads close to the anisotropic kernel.
#[test]
fn the_ladder_reads_close_to_the_anisotropic_kernel_on_an_isotropic_blur() {
    let sharp = textured(31, 1.4);
    let views: Vec<TilePlanes> = [0.0, 1.0, 1.6]
        .iter()
        .map(|&s| sharp.blurred(BlurCovariance::isotropic(s), &mut BlurScratch::default()))
        .collect();
    let refs: Vec<&TilePlanes> = views.iter().collect();
    let ellipses: Vec<_> = views.iter().map(|v| Some(ellipse_of(v))).collect();
    let run = |kernel| {
        blur_matched_pairs(
            &refs,
            &ellipses,
            PairMatching::BlurMatched,
            kernel,
            PatchWindow::GaussianDisk { sigma: 0.6 },
            None,
            &Progress::none(),
        )
    };
    let aniso = run(BlurMatchKernel::Anisotropic);
    let ladder = run(BlurMatchKernel::IsotropicLadder);
    for p in [1, 2] {
        assert!(ladder.whole[p] > 0.95, "ladder {}", ladder.whole[p]);
        assert!(
            (ladder.whole[p] - aniso.whole[p]).abs() < 0.03,
            "ladder {} against anisotropic {}",
            ladder.whole[p],
            aniso.whole[p]
        );
    }
}

/// `rows` limits the pairs read to those with a view in it.
#[test]
fn rows_limit_the_pairs_read() {
    let views: Vec<TilePlanes> = (0..4).map(|s| textured(40 + s, 0.7)).collect();
    let refs: Vec<&TilePlanes> = views.iter().collect();
    let ellipses: Vec<_> = views.iter().map(|v| Some(ellipse_of(v))).collect();
    let pairs = blur_matched_pairs(
        &refs,
        &ellipses,
        PairMatching::BlurMatched,
        BlurMatchKernel::Anisotropic,
        PatchWindow::GaussianDisk { sigma: 0.6 },
        Some(&[false, true, false, false]),
        &Progress::none(),
    );
    assert_eq!(pairs.pairs, 3);
    assert!(pairs.whole[1].is_finite() && pairs.whole[4 + 2].is_finite());
    assert!(pairs.whole[2].is_nan() && pairs.whole[4 * 2 + 3].is_nan());
    assert!(pairs.row_middle(1).is_finite() && pairs.row_middle(0).is_finite());
}

/// A tile of a few sinusoids, whose blur by a Gaussian is known exactly: each
/// sinusoid's amplitude scales by `exp(−½ kᵀ Σ k)`.
fn sinusoids(cov: Option<BlurCovariance>) -> TilePlanes {
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
                let gain = cov.map_or(1.0, |c| {
                    (-0.5 * (k[0] * k[0] * c.xx + 2.0 * k[0] * k[1] * c.xy + k[1] * k[1] * c.yy))
                        .exp()
                });
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

/// The cost of each step on one thread, printed for the spec's cost table.
/// Run with `cargo test -p sfmtool-core --lib blur_matched::tests::timing --
/// --ignored --nocapture`.
#[test]
#[ignore]
fn timing() {
    use std::hint::black_box;
    use std::time::Instant;
    let a = textured(51, 1.4);
    let b = a.blurred(BlurCovariance::isotropic(1.0), &mut BlurScratch::default());
    let w = window();
    let reps = 20_000;
    let time = |label: &str, f: &mut dyn FnMut()| {
        for _ in 0..100 {
            f();
        }
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        eprintln!("{label:58} {us:7.2} µs");
    };
    time("plain pair readings (whole + 9 cells)", &mut || {
        black_box(pair_zncc_readings(black_box(&a), black_box(&b), &w));
    });
    let mut out = vec![0.0f32; a.values.len()];
    let mut scratch = BlurScratch::default();
    for (label, cov) in [
        ("isotropic σ 1", BlurCovariance::isotropic(1.0)),
        (
            "1-D σ 1 along 30°",
            BlurCovariance::along([0.866, 0.5], 1.0),
        ),
        (
            "σ 1.5 × 0.8 at 30°",
            BlurCovariance {
                xx: 1.85,
                xy: 0.71,
                yy: 1.03,
            },
        ),
        ("isotropic σ 2", BlurCovariance::isotropic(2.0)),
    ] {
        time(&format!("blur, passes, {label}"), &mut || {
            blur_tile(&a.values, 3, SIDE, &a.data, cov, &mut out, &mut scratch);
            black_box(&out);
        });
        if !label.starts_with("1-D") {
            time(&format!("blur, direct 2-D, {label}"), &mut || {
                blur_tile_direct(&a.values, 3, SIDE, &a.data, cov, &mut out);
                black_box(&out);
            });
        }
    }
    let (ea, eb) = (ellipse_of(&a), ellipse_of(&b));
    time("ellipse pair to kernels (pair_blur)", &mut || {
        black_box(pair_blur(black_box(&ea), black_box(&eb), 1.0));
    });
    for kernel in [
        BlurMatchKernel::Anisotropic,
        BlurMatchKernel::IsotropicLadder,
    ] {
        time(
            &format!("blur-matched pair, {}, one pair", kernel.name()),
            &mut || {
                black_box(blur_matched_pairs(
                    &[&a, &b],
                    &[Some(ea), Some(eb)],
                    PairMatching::BlurMatched,
                    kernel,
                    PatchWindow::GaussianDisk { sigma: 0.6 },
                    None,
                    &Progress::none(),
                ));
            },
        );
    }
}
