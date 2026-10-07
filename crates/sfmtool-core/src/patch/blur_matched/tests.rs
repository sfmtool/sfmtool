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

// ------------------------------------------------------- the directions

#[test]
fn the_estimate_is_zero_without_a_difference_and_grows_with_it() {
    assert_eq!(estimated_blur_sigma(0.7, 0.7), 0.0);
    assert_eq!(estimated_blur_sigma(0.5, 0.7), 0.0);
    assert_eq!(estimated_blur_sigma(f64::NAN, 0.7), 0.0);
    let mut last = 0.0;
    for longer in [0.75, 0.9, 1.2, 1.8, 2.5, 3.0] {
        let s = estimated_blur_sigma(longer, 0.7);
        assert!(s > last, "{longer}: {s} after {last}");
        last = s;
    }
    // The difference of squares over the growth at the sharper length.
    let s = estimated_blur_sigma(1.0, 0.5);
    let growth = BLUR_GROWTH_SCALE * 0.5f64.powf(BLUR_GROWTH_POWER);
    assert!((s - (0.75 / growth).sqrt()).abs() < 1e-12, "{s}");
    // A sharper length under the floor is read as the floor.
    assert_eq!(
        estimated_blur_sigma(1.0, 0.0),
        estimated_blur_sigma(1.0, MIN_SHARPER_LENGTH)
    );
    assert_eq!(estimated_blur_sigma(3.0, 0.0), MAX_BLUR_SIGMA);
}

#[test]
fn equal_ellipses_blur_nothing() {
    let e = [[0.8, 0.1], [0.1, 0.3]];
    assert!(pair_directions(&e, &e, 1.0).is_none());
    assert!(pair_directions_for(&e, &[[2.0, 0.0], [0.0, 2.0]], PairMatching::Plain).is_none());
    let unread = [[f64::NAN, 0.0], [0.0, 1.0]];
    assert!(pair_directions(&unread, &e, 1.0).is_none());
}

#[test]
fn each_tile_is_blurred_only_along_the_directions_it_is_sharper() {
    // `a` is sharp along x and blurry along y; `b` the other way round.
    let a = [[0.16, 0.0], [0.0, 1.44]];
    let b = [[1.44, 0.0], [0.0, 0.16]];
    let dirs = pair_directions(&a, &b, 1.0);
    let ([da], [db]) = (dirs.a.as_slice(), dirs.b.as_slice()) else {
        panic!("{dirs:?}");
    };
    assert!(da.u[0].abs() > 1.0 - 1e-12, "{da:?}");
    assert!(db.u[1].abs() > 1.0 - 1e-12, "{db:?}");
    for d in [da, db] {
        assert!((d.sharper - 0.4).abs() < 1e-12 && (d.blurrier - 1.2).abs() < 1e-12);
    }
    let blur = dirs.a.blur(None);
    assert!(
        blur.xx > 0.5 && blur.yy.abs() < 1e-9 && blur.xy.abs() < 1e-9,
        "{blur:?}"
    );

    // A rotated pair: `b` is `a` stretched along 30°.
    let (s, c) = 30f64.to_radians().sin_cos();
    let rot = |l1: f64, l2: f64| {
        [
            [l1 * c * c + l2 * s * s, (l1 - l2) * c * s],
            [(l1 - l2) * c * s, l1 * s * s + l2 * c * c],
        ]
    };
    let dirs = pair_directions(&rot(0.25, 0.25), &rot(2.25, 0.25), 1.0);
    assert!(dirs.b.is_empty());
    let [d] = dirs.a.as_slice() else {
        panic!("{dirs:?}");
    };
    assert!((d.u[0] * c + d.u[1] * s).abs() > 1.0 - 1e-9, "{d:?}");
    assert!(
        (d.sharper - 0.5).abs() < 1e-9 && (d.blurrier - 1.5).abs() < 1e-9,
        "{d:?}"
    );
    // The estimate lies along 30° only.
    let blur = dirs.a.blur(None);
    let across = blur.variance_along([-s, c]);
    assert!(
        (blur.variance_along([c, s]) - estimated_blur_sigma(1.5, 0.5).powi(2)).abs() < 1e-9,
        "{blur:?}"
    );
    assert!(across.abs() < 1e-9, "{across}");
}

#[test]
fn the_skip_ratio_leaves_a_small_difference_plain() {
    let a = [[0.49, 0.0], [0.0, 0.49]]; // 0.7 along both axes
    let b = [[0.64, 0.0], [0.0, 0.64]]; // 0.8: a ratio of 1.14
    assert!(!pair_directions(&a, &b, 1.0).is_none());
    assert!(!pair_directions(&a, &b, 1.1).is_none());
    assert!(pair_directions(&a, &b, 1.2).is_none());
    assert!(pair_directions_for(&a, &b, PairMatching::BlurMatchedAboveRatio(1.2)).is_none());
    assert!(!pair_directions_for(&a, &b, PairMatching::BlurMatched).is_none());
    // A ratio is per direction: one axis past it is blurred, the other not.
    let c = [[0.64, 0.0], [0.0, 1.96]]; // 0.8 and 1.4
    let dirs = pair_directions(&a, &c, 1.2);
    let [d] = dirs.a.as_slice() else {
        panic!("{dirs:?}");
    };
    assert!(d.u[1].abs() > 1.0 - 1e-12 && dirs.b.is_empty(), "{dirs:?}");
    // Lengths within the tolerance of each other are matched already, even
    // when every difference is asked for.
    let near = [[0.5184, 0.0], [0.0, 0.5184]]; // 0.72: a ratio of 1.03
    assert!(pair_directions_for(&a, &near, PairMatching::BlurMatched).is_none());
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

// ------------------------------------------------------- the width from the growth

/// The growth of `tile`, read on its whole tile.
fn growth_of(tile: &TilePlanes) -> BlurGrowth {
    let e = read_tile_ellipse(&tile.values, tile.channels, tile.side, &tile.data).unwrap();
    read_growth(
        tile,
        &e,
        |v| read_tile_ellipse(v, tile.channels, tile.side, &tile.data),
        &mut Vec::new(),
        &mut BlurScratch::default(),
    )
    .unwrap()
}

/// The blurry tile is the sharp tile blurred by a known Gaussian. Where the
/// planted width is no wider than the widest probe, the width the sharp
/// tile's growth gives along each grid axis is within 0.1 grid px of it and
/// the blurred tile's ellipse within 5% of the blurry one's along each
/// direction blurred; a planted blur of 0.4, the narrow probe itself, is
/// found exactly. Past the widest probe the width is held at most to the
/// fitted rate's, so it falls short of the planted one (and of its length,
/// by up to 20%) rather than past it. The texture's ellipse is about 0.45
/// grid px, as a sharp view's is. The planted blurs lie along the grid's
/// axes or are isotropic, where the two passes are an exact Gaussian.
#[test]
fn the_growth_recovers_a_known_blur() {
    let read = |t: &TilePlanes| read_tile_ellipse(&t.values, t.channels, t.side, &t.data).unwrap();
    let widest = GROWTH_PROBE_SIGMAS[GROWTH_PROBE_SIGMAS.len() - 1];
    for (seed, planted) in [
        (61, BlurCovariance::along([1.0, 0.0], 0.6)),
        (62, BlurCovariance::along([0.0, 1.0], 1.5)),
        (63, BlurCovariance::isotropic(0.9)),
        (
            64,
            BlurCovariance {
                xx: 1.44,
                xy: 0.0,
                yy: 0.25,
            },
        ),
        (66, BlurCovariance::isotropic(0.4)),
    ] {
        let sharp = textured(seed, 1.6);
        let blurry = sharp.blurred(planted, &mut BlurScratch::default());
        let (es, eb) = (read(&sharp), read(&blurry));
        let dirs = pair_directions(&es, &eb, 1.0);
        assert!(dirs.b.is_empty(), "{seed}: {dirs:?}");
        let growth = growth_of(&sharp);
        let cov = dirs.a.blur(Some(&growth));
        let em = read(&sharp.blurred(cov, &mut BlurScratch::default()));
        for (u, found, want) in [
            ([1.0, 0.0], cov.xx.sqrt(), planted.xx.sqrt()),
            ([0.0, 1.0], cov.yy.sqrt(), planted.yy.sqrt()),
        ] {
            let ratio = length_along(&em, u) / length_along(&eb, u);
            if want <= widest {
                assert!(
                    (found - want).abs() <= 0.1,
                    "{seed}: planted {want}, found {found}"
                );
                assert!(
                    (ratio - 1.0).abs() <= 0.05,
                    "{seed}: length {ratio} of the blurry one's"
                );
            } else {
                assert!(
                    found < want && found > 0.99 * widest,
                    "{seed}: planted {want}, found {found}"
                );
                assert!(
                    (0.8..1.0).contains(&ratio),
                    "{seed}: length {ratio} of the blurry one's"
                );
            }
        }
    }
}

/// The rule blurs short of the length aimed for rather than past it at the
/// widths most pairs need: on sharp tiles matched to copies of themselves
/// blurred along one direction by 0.5 to 0.7 grid px, the blurred tile's
/// ellipse comes out at most 2% longer than its partner's along the
/// direction blurred, and on most shorter. The isotropic probes also blur
/// across that direction, which lengthens the ellipse along it a little more
/// than a 1-D blur does, so the growth read is a little fast.
#[test]
fn the_growth_blurs_short_of_a_one_dimensional_blur_rather_than_past_it() {
    let read = |t: &TilePlanes| read_tile_ellipse(&t.values, t.channels, t.side, &t.data).unwrap();
    let diagonal = std::f64::consts::FRAC_1_SQRT_2;
    let mut ratios = Vec::new();
    for seed in 0..6u64 {
        let sharp = textured(80 + seed, 1.4);
        let growth = growth_of(&sharp);
        let es = read(&sharp);
        for (u, sigma) in [
            ([1.0, 0.0], 0.5),
            ([0.0, 1.0], 0.7),
            ([diagonal, diagonal], 0.6),
        ] {
            let blurry =
                sharp.blurred(BlurCovariance::along(u, sigma), &mut BlurScratch::default());
            let eb = read(&blurry);
            let dirs = pair_directions(&es, &eb, 1.0);
            let em = read(&sharp.blurred(dirs.a.blur(Some(&growth)), &mut BlurScratch::default()));
            for d in dirs.a.as_slice() {
                let ratio = length_along(&em, d.u) / length_along(&eb, d.u);
                ratios.push(ratio);
            }
        }
    }
    ratios.sort_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];
    assert!(ratios.iter().all(|&r| r <= 1.02), "{ratios:?}");
    assert!(median < 1.0, "{ratios:?}");
}

/// The width is read off the readings: on a growth that is linear in `σ²`
/// it is exact; a piece that does not grow is passed over; past the widest
/// probe the line goes on along its last piece, but a direction's width
/// there is at most the fitted rate's.
#[test]
fn the_width_is_read_off_the_probes() {
    let iso = |l2: f64| [[l2, 0.0], [0.0, l2]];
    let [p1, p2] = GROWTH_PROBE_SIGMAS;
    // l² = 0.25 + σ² along every direction.
    let linear = BlurGrowth::from_ellipses(&iso(0.25), &[iso(0.25 + p1 * p1), iso(0.25 + p2 * p2)]);
    let s = linear.sigma_along([1.0, 0.0], 1.0).unwrap();
    assert!((s - 0.75f64.sqrt()).abs() < 1e-12, "{s}");
    assert_eq!(linear.sigma_along([0.0, 1.0], 0.4), Some(0.0));
    assert!((linear.semi_major_after(0.75f64.sqrt()) - 1.0).abs() < 1e-12);
    // Flat up to the narrow probe, then l² = 0.25 + (σ² − p1²) · 2.
    let late = BlurGrowth::from_ellipses(
        &iso(0.25),
        &[iso(0.25), iso(0.25 + 2.0 * (p2 * p2 - p1 * p1))],
    );
    let s = late.sigma_along([1.0, 0.0], 0.6).unwrap();
    let want = (p1 * p1 + (0.36 - 0.25) / 2.0).sqrt();
    assert!((s - want).abs() < 1e-12, "{s} against {want}");
    // No growth at all: no width, and the fitted rate stands in.
    let flat = BlurGrowth::from_ellipses(&iso(0.25), &[iso(0.25), iso(0.25)]);
    assert_eq!(flat.sigma_along([1.0, 0.0], 1.0), None);
    // A growth so slow that the line reaches the target only far past the
    // widest probe: the width is the fitted rate's.
    let slow = BlurGrowth::from_ellipses(&iso(0.04), &[iso(0.041), iso(0.05)]);
    let d = BlurDirection {
        u: [1.0, 0.0],
        sharper: 0.2,
        blurrier: 0.6,
    };
    assert!(slow.sigma_along(d.u, d.blurrier).unwrap() > estimated_blur_sigma(0.6, 0.2));
    assert_eq!(d.sigma(Some(&slow)), estimated_blur_sigma(0.6, 0.2));
}

/// A view whose growth cannot be read, or whose readings do not grow along a
/// direction, is blurred by the width from the rate fitted on real tiles.
#[test]
fn a_view_without_a_growth_falls_back_to_the_estimate() {
    let sharp = textured(71, 0.8);
    let e = [[0.25, 0.0], [0.0, 0.25]];
    let dirs = pair_directions(&e, &[[1.0, 0.0], [0.0, 0.25]], 1.0);
    let mut growths = ViewGrowths::new(1);
    assert_eq!(growths.get(0, &sharp, &e, |_| None), None);
    assert_eq!(growths.reads(), GROWTH_PROBE_SIGMAS.len());
    let [d] = dirs.a.as_slice() else {
        panic!("{dirs:?}");
    };
    let want = estimated_blur_sigma(d.blurrier, d.sharper);
    assert!(want > 0.0);
    assert_eq!(d.sigma(None), want);
    let flat = BlurGrowth::from_ellipses(&e, &[e, e]);
    assert_eq!(d.sigma(Some(&flat)), want);
    assert_eq!(dirs.a.blur(None), BlurCovariance::along(d.u, want));
}

/// Each view's growth is read once, the first time a pair blurs it, and kept
/// for the others: a sharp view blurred against three blurrier ones has one
/// blurred tile read per probe, under both kernels.
#[test]
fn each_views_growth_is_read_once() {
    let sharp = textured(91, 1.4);
    let mut views = vec![sharp.clone()];
    for s in [0.8, 1.2, 1.6] {
        views.push(sharp.blurred(BlurCovariance::isotropic(s), &mut BlurScratch::default()));
    }
    let refs: Vec<&TilePlanes> = views.iter().collect();
    let ellipses: Vec<_> = views.iter().map(|v| Some(ellipse_of(v))).collect();
    for kernel in [
        BlurMatchKernel::Anisotropic,
        BlurMatchKernel::IsotropicLadder,
    ] {
        let only_sharp = blur_matched_pairs(
            &refs,
            &ellipses,
            PairMatching::BlurMatchedAboveRatio(1.25),
            kernel,
            PatchWindow::GaussianDisk { sigma: 0.6 },
            Some(&[true, false, false, false]),
            &Progress::none(),
        );
        assert_eq!(only_sharp.pairs_blurred, 3, "{kernel:?}");
        assert_eq!(
            only_sharp.ellipse_reads,
            GROWTH_PROBE_SIGMAS.len(),
            "{kernel:?}"
        );
        let all = blur_matched_pairs(
            &refs,
            &ellipses,
            PairMatching::BlurMatched,
            kernel,
            PatchWindow::GaussianDisk { sigma: 0.6 },
            None,
            &Progress::none(),
        );
        // Every view but the blurriest is the sharper one of some pair, and
        // is read once however many pairs it is blurred in.
        assert!(all.pairs_blurred >= 5, "{kernel:?}: {}", all.pairs_blurred);
        assert!(
            all.ellipse_reads <= 3 * GROWTH_PROBE_SIGMAS.len(),
            "{kernel:?}: {}",
            all.ellipse_reads
        );
    }
}

/// The readings do not depend on the order the views come in: reversed,
/// every pair reads the same, bit for bit, since each view's growth is a
/// function of its own tile.
#[test]
fn the_readings_do_not_depend_on_the_order_of_the_views() {
    let mut rng = Rng(4242);
    let k = 6;
    let tiles: Vec<TilePlanes> = (0..k)
        .map(|v| {
            let t = textured(300 + v as u64, 1.2);
            let s = 0.2 + rng.next() * 1.5;
            t.blurred(
                BlurCovariance::along([1.0, 0.0], s),
                &mut BlurScratch::default(),
            )
        })
        .collect();
    let ellipses: Vec<_> = tiles.iter().map(|t| Some(ellipse_of(t))).collect();
    let forward: Vec<&TilePlanes> = tiles.iter().collect();
    let backward: Vec<&TilePlanes> = tiles.iter().rev().collect();
    let back_ellipses: Vec<_> = ellipses.iter().rev().copied().collect();
    for kernel in [
        BlurMatchKernel::Anisotropic,
        BlurMatchKernel::IsotropicLadder,
    ] {
        let run = |t: &[&TilePlanes], e: &[Option<[[f64; 2]; 2]>]| {
            blur_matched_pairs(
                t,
                e,
                PairMatching::BlurMatched,
                kernel,
                PatchWindow::GaussianDisk { sigma: 0.6 },
                None,
                &Progress::none(),
            )
        };
        let f = run(&forward, &ellipses);
        let b = run(&backward, &back_ellipses);
        assert!(f.pairs_blurred > 0, "{kernel:?}");
        for i in 0..k {
            for j in 0..k {
                let (bi, bj) = (k - 1 - i, k - 1 - j);
                let (x, y) = (f.whole[i * k + j], b.whole[bi * k + bj]);
                assert!((x - y).abs() < 1e-9, "{kernel:?} {i}, {j}: {x} against {y}");
            }
        }
    }
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
    time("ellipse pair to directions (pair_directions)", &mut || {
        black_box(pair_directions(black_box(&ea), black_box(&eb), 1.0));
    });
    time("whole self-similarity reading of a tile", &mut || {
        black_box(read_tile_ellipse(&b.values, 3, SIDE, &b.data));
    });
    let mut probe = Vec::new();
    time("one view's growth, read once (read_growth)", &mut || {
        black_box(read_growth(
            &a,
            &ea,
            |v| read_tile_ellipse(v, 3, SIDE, &a.data),
            &mut probe,
            &mut scratch,
        ));
    });
    let growth = read_growth(
        &a,
        &ea,
        |v| read_tile_ellipse(v, 3, SIDE, &a.data),
        &mut probe,
        &mut scratch,
    );
    let dirs = pair_directions(&ea, &eb, 1.0);
    let mut blurred = a.clone();
    time(
        "a blurred pair, its view's growth already read",
        &mut || {
            let cov = dirs.a.blur(growth.as_ref());
            blur_tile(
                &a.values,
                3,
                SIDE,
                &a.data,
                cov,
                &mut blurred.values,
                &mut scratch,
            );
            black_box(pair_zncc_readings(&blurred, &b, &w));
        },
    );
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
