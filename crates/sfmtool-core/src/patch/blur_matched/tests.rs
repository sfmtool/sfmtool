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
fn grained(seed: u64, across: [f64; 2]) -> TilePlanes {
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

fn read(t: &TilePlanes) -> [[f64; 2]; 2] {
    read_tile_ellipse(&t.values, t.channels, t.side, &t.data).unwrap()
}

fn window() -> Vec<f64> {
    window_weights(PatchWindow::GaussianDisk { sigma: 0.6 }, SIDE as u32)
}

fn pairs_of(
    tiles: &[&TilePlanes],
    ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching,
) -> BlurMatchedPairs {
    blur_matched_pairs(
        tiles,
        ellipses,
        matching,
        PatchWindow::GaussianDisk { sigma: 0.6 },
        None,
        &Progress::none(),
    )
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

/// An ellipse matrix of semi-axes `major` and `minor`, the major axis at
/// `deg` degrees from `x`.
fn ellipse(major: f64, minor: f64, deg: f64) -> [[f64; 2]; 2] {
    let (s, c) = f64::to_radians(deg).sin_cos();
    let (l1, l2) = (major * major, minor * minor);
    [
        [l1 * c * c + l2 * s * s, (l1 - l2) * c * s],
        [(l1 - l2) * c * s, l1 * s * s + l2 * c * c],
    ]
}

// ------------------------------------------------------- which tile, and to what

#[test]
fn equal_ellipses_and_unread_ones_blur_nothing() {
    let e = ellipse(0.9, 0.5, 20.0);
    assert_eq!(pair_blur(&e, &e, 1.0), None);
    assert_eq!(
        pair_blur_for(&e, &ellipse(2.0, 2.0, 0.0), PairMatching::Plain),
        None
    );
    let unread = [[f64::NAN, 0.0], [0.0, 1.0]];
    assert_eq!(pair_blur(&unread, &e, 1.0), None);
}

/// The sharper tile is the one whose semi-major axis is shorter than the
/// other's semi-minor axis, and the target is that semi-minor axis, at most
/// 2 grid px. Swapping the pair swaps only which tile is named.
#[test]
fn the_sharper_tile_is_blurred_to_the_others_minor_axis() {
    let sharp = ellipse(0.5, 0.3, 40.0);
    let blurry = ellipse(1.6, 1.1, -10.0);
    let ab = pair_blur(&sharp, &blurry, 1.0).unwrap();
    assert_eq!(ab.sharper, 0);
    assert!((ab.major - 0.5).abs() < 1e-12 && (ab.target - 1.1).abs() < 1e-12);
    let ba = pair_blur(&blurry, &sharp, 1.0).unwrap();
    assert_eq!(ba.sharper, 1);
    assert_eq!((ba.major, ba.target), (ab.major, ab.target));
    // A blurrier tile past the cap is aimed at the cap.
    let far = pair_blur(&sharp, &ellipse(3.0, 2.6, 0.0), 1.0).unwrap();
    assert_eq!(far.target, MAX_MATCHED_LENGTH);
    // Against a tile whose semi-minor axis is at the cap, a sharper tile
    // already near it is left alone.
    assert_eq!(
        pair_blur(&ellipse(1.95, 1.9, 0.0), &ellipse(3.0, 2.6, 0.0), 1.0),
        None
    );
}

/// A tile blurry along one direction only, whose semi-minor axis is no
/// longer than the other tile's semi-major axis, is not blurrier along every
/// direction: neither tile is blurred.
#[test]
fn a_tile_blurry_along_one_direction_only_is_not_matched() {
    let sharp = ellipse(0.6, 0.4, 0.0);
    let oblique = ellipse(2.5, 0.55, 80.0);
    assert_eq!(pair_blur(&sharp, &oblique, 1.0), None);
    assert_eq!(pair_blur(&oblique, &sharp, 1.0), None);
    // Grain at two angles: each is long where the other is short.
    assert_eq!(
        pair_blur(&ellipse(1.8, 0.4, 0.0), &ellipse(1.8, 0.4, 90.0), 1.0),
        None
    );
}

#[test]
fn the_skip_ratio_leaves_a_small_difference_plain() {
    let a = ellipse(0.7, 0.6, 0.0);
    let b = ellipse(0.9, 0.8, 30.0); // 0.8 over 0.7: a ratio of 1.14
    assert!(pair_blur(&a, &b, 1.0).is_some());
    assert!(pair_blur(&a, &b, 1.1).is_some());
    assert!(pair_blur(&a, &b, 1.2).is_none());
    assert!(pair_blur_for(&a, &b, PairMatching::BlurMatchedAboveRatio(1.2)).is_none());
    assert!(pair_blur_for(&a, &b, PairMatching::BlurMatched).is_some());
    // Lengths within the tolerance of each other are matched already, even
    // when every difference is asked for.
    let near = ellipse(0.9, 0.72, 0.0); // 0.72 over 0.7: a ratio of 1.03
    assert!(pair_blur_for(&a, &near, PairMatching::BlurMatched).is_none());
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
}

#[test]
fn the_semi_axes_are_read_off_the_matrix() {
    let [major, minor] = semi_axes(&ellipse(1.3, 0.4, 37.0));
    assert!((major - 1.3).abs() < 1e-12 && (minor - 0.4).abs() < 1e-12);
}

// ------------------------------------------------------- the width from the growth

/// The growth of `tile`, read on its whole tile.
fn growth_of(tile: &TilePlanes) -> BlurGrowth {
    read_growth(
        tile,
        &read(tile),
        |v| read_tile_ellipse(v, tile.channels, tile.side, &tile.data),
        &mut Vec::new(),
        &mut BlurScratch::default(),
    )
    .unwrap()
}

/// The width is read off the readings: on a growth that is linear in `σ²`
/// it is exact; a piece that does not grow is passed over; a growth that
/// never grows gives none; a target already reached gives 0.
#[test]
fn the_width_is_read_off_the_probes() {
    let [p1, p2] = GROWTH_PROBE_SIGMAS;
    let of = |l2: [f64; 3]| BlurGrowth {
        semi_major: l2.map(f64::sqrt),
    };
    // l² = 0.25 + σ².
    let linear = of([0.25, 0.25 + p1 * p1, 0.25 + p2 * p2]);
    let s = linear.sigma_for(1.0).unwrap();
    assert!((s - 0.75f64.sqrt()).abs() < 1e-12, "{s}");
    assert_eq!(linear.sigma_for(0.4), Some(0.0));
    // Past the widest probe the line goes on along its last piece.
    let s = linear.sigma_for(1.5).unwrap();
    assert!((s - 2.0f64.sqrt()).abs() < 1e-12, "{s}");
    // Flat up to the narrow probe, then l² = 0.25 + (σ² − p1²) · 2.
    let late = of([0.25, 0.25, 0.25 + 2.0 * (p2 * p2 - p1 * p1)]);
    let s = late.sigma_for(0.6).unwrap();
    let want = (p1 * p1 + (0.36 - 0.25) / 2.0).sqrt();
    assert!((s - want).abs() < 1e-12, "{s} against {want}");
    // No growth at all: no width.
    assert_eq!(of([0.25, 0.25, 0.25]).sigma_for(1.0), None);
    assert_eq!(of([0.25, f64::NAN, 0.3]).sigma_for(1.0), None);
    // The width is capped.
    assert_eq!(
        of([0.04, 0.041, 0.042]).sigma_for(2.0),
        Some(MAX_BLUR_SIGMA)
    );
}

/// The blurry tile is the sharp tile blurred by a known round Gaussian. The
/// rule picks the sharp tile, and the width its growth gives brings its
/// semi-major axis to within 5% of the blurry tile's semi-minor axis. The
/// texture's ellipse is about 0.45 grid px, as a
/// sharp view's is; such a tile barely lengthens under a blur of 0.5, and a
/// difference that small is left plain.
#[test]
fn the_rule_recovers_a_planted_round_blur() {
    let sharp = textured(60, 1.6);
    let blurry = sharp.blurred(0.5, &mut BlurScratch::default());
    assert_eq!(pair_blur(&read(&sharp), &read(&blurry), 1.0), None);
    let mut ratios = Vec::new();
    for (seed, planted) in [(61, 0.8), (62, 1.0), (63, 1.2), (64, 1.6), (65, 2.0)] {
        let sharp = textured(seed, 1.6);
        let blurry = sharp.blurred(planted, &mut BlurScratch::default());
        let (es, eb) = (read(&sharp), read(&blurry));
        let pb = pair_blur(&es, &eb, 1.0).expect("the sharp tile is the sharper");
        assert_eq!(pb.sharper, 0, "{seed}");
        let sigma = growth_of(&sharp).sigma_for(pb.target).unwrap();
        let after = semi_axes(&read(&sharp.blurred(sigma, &mut BlurScratch::default())))[0];
        let ratio = after / semi_axes(&eb)[1];
        assert!(
            (0.95..=1.05).contains(&ratio),
            "{seed}: planted {planted}, width {sigma}, major {after} of the blurry minor ({ratio})"
        );
        // The width is no wider than the planted blur: the blurry tile's
        // semi-major axis is longer than its semi-minor one, and the blur is
        // aimed at the semi-minor.
        assert!(
            sigma <= planted + 0.05,
            "{seed}: planted {planted}, width {sigma}"
        );
        ratios.push(ratio);
    }
    eprintln!("major after / blurry minor: {ratios:?}");
}

/// Grain at two angles holds no blur difference: each tile is long along its
/// stripes and short across them, so neither is shorter than the other
/// along every direction, and the pair reads what plain ZNCC reads.
#[test]
fn grain_at_another_angle_is_not_read_as_blur() {
    let along_x = grained(5, [0.0, 1.0]);
    let along_y = grained(6, [1.0, 0.0]);
    let diagonal = grained(7, [0.6, 0.8]);
    let tiles = [&along_x, &along_y, &diagonal];
    let ellipses: Vec<_> = tiles.iter().map(|t| Some(read(t))).collect();
    for e in ellipses.iter().flatten() {
        let [major, minor] = semi_axes(e);
        assert!(major > 2.0 * minor, "{major} × {minor}: not grained");
    }
    let pairs = pairs_of(&tiles, &ellipses, PairMatching::BlurMatched);
    assert_eq!(pairs.pairs_blurred, 0);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        let plain = pair_zncc_readings(tiles[a], tiles[b], &window());
        assert_eq!(pairs.whole[a * 3 + b].to_bits(), plain.whole.to_bits());
    }
}

/// Only one tile of a pair is blurred, the sharper, and the width is
/// reported against it; reversing the views blurs the same tile by the same
/// width.
#[test]
fn only_the_sharper_tile_of_a_pair_is_blurred() {
    let sharp = textured(17, 1.4);
    let views: Vec<TilePlanes> = [0.0, 0.8, 1.4]
        .iter()
        .map(|&s| sharp.blurred(s, &mut BlurScratch::default()))
        .collect();
    let refs: Vec<&TilePlanes> = views.iter().collect();
    let ellipses: Vec<_> = views.iter().map(|v| Some(ellipse_of(v))).collect();
    let pairs = pairs_of(&refs, &ellipses, PairMatching::BlurMatched);
    let k = 3;
    for a in 0..k {
        for b in (a + 1)..k {
            let (ab, ba) = (pairs.sigma[a * k + b], pairs.sigma[b * k + a]);
            assert!(ab == 0.0 || ba == 0.0, "{a}-{b}: both blurred");
            assert_eq!(pairs.blurred[a * k + b], ab > 0.0 || ba > 0.0);
        }
    }
    // The sharp view is blurred against both others, and the middle one
    // against the blurriest.
    assert!(pairs.sigma[1] > 0.0 && pairs.sigma[2] > 0.0 && pairs.sigma[k + 2] > 0.0);
    assert!(pairs.sigma[2] > pairs.sigma[1], "{:?}", pairs.sigma);
    let back: Vec<&TilePlanes> = refs.iter().rev().copied().collect();
    let back_e: Vec<_> = ellipses.iter().rev().copied().collect();
    let reversed = pairs_of(&back, &back_e, PairMatching::BlurMatched);
    for a in 0..k {
        for b in 0..k {
            let (ra, rb) = (k - 1 - a, k - 1 - b);
            assert_eq!(pairs.sigma[a * k + b], reversed.sigma[ra * k + rb]);
        }
    }
}

/// A view whose growth cannot be read, or does not grow, leaves its pairs
/// plain.
#[test]
fn a_view_without_a_growth_is_read_plain() {
    let sharp = textured(71, 0.8);
    let e = ellipse(0.5, 0.5, 0.0);
    let mut growths = ViewGrowths::new(1);
    assert_eq!(growths.get(0, &sharp, &e, |_| None), None);
    assert_eq!(growths.reads(), GROWTH_PROBE_SIGMAS.len());
    // A view asked for again is not read again.
    assert_eq!(growths.get(0, &sharp, &e, |_| unreachable!()), None);
    assert_eq!(growths.reads(), GROWTH_PROBE_SIGMAS.len());
}

/// Each view's growth is read once, the first time a pair blurs it, and kept
/// for the others: a sharp view blurred against three blurrier ones has one
/// blurred tile read per probe.
#[test]
fn each_views_growth_is_read_once() {
    let sharp = textured(91, 1.4);
    let mut views = vec![sharp.clone()];
    for s in [0.8, 1.2, 1.6] {
        views.push(sharp.blurred(s, &mut BlurScratch::default()));
    }
    let refs: Vec<&TilePlanes> = views.iter().collect();
    let ellipses: Vec<_> = views.iter().map(|v| Some(ellipse_of(v))).collect();
    let only_sharp = blur_matched_pairs(
        &refs,
        &ellipses,
        PairMatching::BlurMatchedAboveRatio(1.25),
        PatchWindow::GaussianDisk { sigma: 0.6 },
        Some(&[true, false, false, false]),
        &Progress::none(),
    );
    assert_eq!(only_sharp.pairs_blurred, 3);
    assert_eq!(only_sharp.ellipse_reads, GROWTH_PROBE_SIGMAS.len());
    let all = pairs_of(&refs, &ellipses, PairMatching::BlurMatched);
    // Every view but the blurriest is the sharper one of some pair, and is
    // read once however many pairs it is blurred in.
    assert!(all.pairs_blurred >= 5, "{}", all.pairs_blurred);
    assert!(
        all.ellipse_reads <= 3 * GROWTH_PROBE_SIGMAS.len(),
        "{}",
        all.ellipse_reads
    );
}

/// The readings do not depend on the order the views come in: reversed,
/// every pair reads the same, since each view's growth is a function of its
/// own tile.
#[test]
fn the_readings_do_not_depend_on_the_order_of_the_views() {
    let mut rng = Rng(4242);
    let k = 6;
    let tiles: Vec<TilePlanes> = (0..k)
        .map(|v| {
            let t = textured(300 + v as u64, 1.2);
            t.blurred(0.2 + rng.next() * 1.5, &mut BlurScratch::default())
        })
        .collect();
    let ellipses: Vec<_> = tiles.iter().map(|t| Some(ellipse_of(t))).collect();
    let forward: Vec<&TilePlanes> = tiles.iter().collect();
    let backward: Vec<&TilePlanes> = tiles.iter().rev().collect();
    let back_ellipses: Vec<_> = ellipses.iter().rev().copied().collect();
    let f = pairs_of(&forward, &ellipses, PairMatching::BlurMatched);
    let b = pairs_of(&backward, &back_ellipses, PairMatching::BlurMatched);
    assert!(f.pairs_blurred > 0);
    for i in 0..k {
        for j in 0..k {
            let (bi, bj) = (k - 1 - i, k - 1 - j);
            let (x, y) = (f.whole[i * k + j], b.whole[bi * k + bj]);
            assert!((x - y).abs() < 1e-9, "{i}, {j}: {x} against {y}");
        }
    }
}

/// A pair's readings in a track are those of the pair read alone, bit for
/// bit: the other pairs, read with the same scratch and growths, do not
/// reach it.
#[test]
fn a_pair_reads_the_same_in_a_track_as_alone() {
    let mut rng = Rng(777);
    let k = 8;
    let tiles: Vec<TilePlanes> = (0..k).map(|v| textured(100 + v as u64, 0.8)).collect();
    let refs: Vec<&TilePlanes> = tiles.iter().collect();
    let ellipses: Vec<Option<[[f64; 2]; 2]>> = (0..k)
        .map(|_| {
            let major = 0.3 + rng.next() * 2.0;
            Some(ellipse(
                major,
                major * (0.6 + 0.4 * rng.next()),
                rng.next() * 180.0,
            ))
        })
        .collect();
    let bits = |r: &[[f64; 3]; 3]| r.map(|row| row.map(f64::to_bits));
    let all = pairs_of(&refs, &ellipses, PairMatching::BlurMatched);
    assert!(all.pairs_blurred > 3, "{}", all.pairs_blurred);
    for a in 0..k {
        for b in (a + 1)..k {
            let one = pairs_of(
                &[refs[a], refs[b]],
                &[ellipses[a], ellipses[b]],
                PairMatching::BlurMatched,
            );
            assert_eq!(
                one.whole[1].to_bits(),
                all.whole[a * k + b].to_bits(),
                "pair {a}-{b}: whole"
            );
            assert_eq!(
                bits(&one.grid[1]),
                bits(&all.grid[a * k + b]),
                "pair {a}-{b}: grid"
            );
        }
    }
}

// ------------------------------------------------------------------ the blur

/// The two passes, and the direct 2-D convolution they stand in for, against
/// the exact blur of a tile of sinusoids, at several widths. The interior is
/// compared, where no tap reaches past the tile's edge.
#[test]
fn the_two_passes_match_the_exact_blur_and_the_direct_2d_blur() {
    let tile = sinusoids(None);
    let interior =
        |k: usize| (7..SIDE - 7).contains(&(k / SIDE)) && (7..SIDE - 7).contains(&(k % SIDE));
    let (mut worst_fast, mut worst_direct) = (0.0f32, 0.0f32);
    for sigma in [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3, 1.6, 2.0] {
        let exact = sinusoids(Some(sigma));
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
            sigma,
            &mut fast,
            &mut BlurScratch::default(),
        );
        // The sinusoids span about ±115 grey levels.
        let e = error(&fast);
        worst_fast = worst_fast.max(e);
        assert!(e < 2.0, "σ {sigma}: the passes are {e} off");
        // The direct blur samples the kernel at σ itself, which falls short
        // of its width under about 0.6.
        if sigma >= 0.6 {
            let mut direct = vec![0.0f32; tile.values.len()];
            blur_tile_direct(&tile.values, 1, SIDE, &tile.data, sigma, &mut direct);
            let e = error(&direct);
            worst_direct = worst_direct.max(e);
            assert!(e < 1.0, "σ {sigma}: the 2-D blur is {e} off");
        }
    }
    eprintln!("largest error: passes {worst_fast}, direct 2-D {worst_direct} grey levels");
}

/// One scratch reused over many blurs of different widths, tile sides and
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
            let sigma = 0.2 + rng.next() * 2.8;
            let mut reused = vec![0.0f32; 3 * n];
            let mut fresh = vec![0.0f32; 3 * n];
            blur_tile(&values, 3, side, &data, sigma, &mut reused, &mut scratch);
            blur_tile(
                &values,
                3,
                side,
                &data,
                sigma,
                &mut fresh,
                &mut BlurScratch::default(),
            );
            let same = reused
                .iter()
                .zip(&fresh)
                .all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "side {side}, σ {sigma}: the reused scratch differs");
        }
    }
}

/// The passes against the direct 2-D blur on a textured tile with samples
/// missing, where there is no exact answer: from `σ = 1` up the two sample
/// the same kernel, and agree to a ZNCC near 1.
#[test]
fn the_two_passes_match_the_direct_2d_blur_round_missing_samples() {
    let mut tile = textured(3, 1.0);
    for k in 0..SIDE * SIDE {
        if (k * 7919) % 11 == 0 || k % SIDE < 2 {
            tile.data[k] = false;
        }
    }
    for sigma in [1.0, 1.5, 2.0, 2.5] {
        let mut fast = vec![0.0f32; tile.values.len()];
        let mut direct = vec![0.0f32; tile.values.len()];
        blur_tile(
            &tile.values,
            3,
            SIDE,
            &tile.data,
            sigma,
            &mut fast,
            &mut BlurScratch::default(),
        );
        blur_tile_direct(&tile.values, 3, SIDE, &tile.data, sigma, &mut direct);
        assert!(
            max_abs_diff(&fast, &direct, &tile.data, 3) < 0.05,
            "σ {sigma}"
        );
        let a = TilePlanes {
            values: fast,
            ..tile.clone()
        };
        let b = TilePlanes {
            values: direct,
            ..tile.clone()
        };
        let z = pair_zncc_readings(&a, &b, &window()).whole;
        assert!(z > 0.9999, "σ {sigma}: ZNCC {z}");
    }
}

/// A sample without data takes no part in the blur, whatever it holds, and
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
    let sigma = 1.1;
    let mut clean = vec![0.0f32; tile.values.len()];
    let mut dirty = vec![0.0f32; tile.values.len()];
    blur_tile(
        &tile.values,
        3,
        SIDE,
        &data,
        sigma,
        &mut clean,
        &mut BlurScratch::default(),
    );
    blur_tile(
        &garbage,
        3,
        SIDE,
        &data,
        sigma,
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
        sigma,
        &mut out,
        &mut BlurScratch::default(),
    );
    for (k, &v) in out.iter().enumerate() {
        if data[k % (SIDE * SIDE)] {
            assert!((v - 100.0).abs() < 1e-3, "{k}: {v}");
        }
    }
}

// --------------------------------------------------------- blur-matched ZNCC

/// A tile and a blurred copy of it: plain ZNCC charges the copy for the detail
/// it lost, and blur matching recovers a ZNCC near 1. The texture's ellipse
/// is about 0.35 grid px, as a sharp view's is.
#[test]
fn blur_matching_a_blurred_copy_recovers_its_zncc() {
    for (seed, sigma) in [(11, 1.0), (12, 1.3), (13, 1.6), (14, 2.2)] {
        let sharp = textured(seed, 1.4);
        let blurry = sharp.blurred(sigma, &mut BlurScratch::default());
        let (es, eb) = (ellipse_of(&sharp), ellipse_of(&blurry));
        let plain = pair_zncc_readings(&sharp, &blurry, &window()).whole;
        let pairs = pairs_of(
            &[&sharp, &blurry],
            &[Some(es), Some(eb)],
            PairMatching::BlurMatched,
        );
        let matched = pairs.whole[1];
        assert_eq!(pairs.pairs_blurred, 1);
        assert!(pairs.sigma[1] > 0.0 && pairs.sigma[2] == 0.0);
        assert!(
            matched > 0.97,
            "σ {sigma}: blur-matched {matched}, plain {plain}"
        );
        assert!(
            matched > plain + 0.02,
            "σ {sigma}: blur-matched {matched}, plain {plain}"
        );
    }
}

/// Two equally sharp views are not blurred, and read what plain ZNCC reads.
#[test]
fn equally_sharp_views_read_plain() {
    let a = textured(21, 0.7);
    let b = textured(21, 0.7);
    let e = ellipse_of(&a);
    let pairs = pairs_of(&[&a, &b], &[Some(e), Some(e)], PairMatching::BlurMatched);
    assert_eq!(pairs.pairs_blurred, 0);
    let plain = pair_zncc_readings(&a, &b, &window());
    assert_eq!(pairs.whole[1], plain.whole);
    assert_eq!(pairs.grid[1], plain.grid);
    assert!((pairs.whole[1] - 1.0).abs() < 1e-9);
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
        PatchWindow::GaussianDisk { sigma: 0.6 },
        Some(&[false, true, false, false]),
        &Progress::none(),
    );
    assert_eq!(pairs.pairs, 3);
    assert!(pairs.whole[1].is_finite() && pairs.whole[4 + 2].is_finite());
    assert!(pairs.whole[2].is_nan() && pairs.whole[4 * 2 + 3].is_nan());
    assert!(pairs.row_middle(1).is_finite() && pairs.row_middle(0).is_finite());
}

/// A tile of a few sinusoids, whose blur by a round Gaussian of width `σ` is
/// known exactly: each sinusoid's amplitude scales by `exp(−½ σ² |k|²)`.
fn sinusoids(sigma: Option<f64>) -> TilePlanes {
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

/// The cost of each step on one thread, printed for the spec's cost table.
/// Run with `cargo test -p sfmtool-core --lib blur_matched::tests::timing --
/// --ignored --nocapture`.
#[test]
#[ignore]
fn timing() {
    use std::hint::black_box;
    use std::time::Instant;
    let a = textured(51, 1.4);
    let b = a.blurred(1.0, &mut BlurScratch::default());
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
    for sigma in [0.5, 1.0, 2.0] {
        time(&format!("blur, two passes, σ {sigma}"), &mut || {
            blur_tile(&a.values, 3, SIDE, &a.data, sigma, &mut out, &mut scratch);
            black_box(&out);
        });
        time(&format!("blur, direct 2-D, σ {sigma}"), &mut || {
            blur_tile_direct(&a.values, 3, SIDE, &a.data, sigma, &mut out);
            black_box(&out);
        });
    }
    let (ea, eb) = (ellipse_of(&a), ellipse_of(&b));
    time("which tile is blurred (pair_blur)", &mut || {
        black_box(pair_blur(black_box(&ea), black_box(&eb), 1.0));
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
    let growth = growth_of(&a);
    let target = pair_blur(&ea, &eb, 1.0).unwrap().target;
    let mut blurred = a.clone();
    time(
        "a blurred pair, its view's growth already read",
        &mut || {
            let sigma = growth.sigma_for(target).unwrap();
            blur_tile(
                &a.values,
                3,
                SIDE,
                &a.data,
                sigma,
                &mut blurred.values,
                &mut scratch,
            );
            black_box(pair_zncc_readings(&blurred, &b, &w));
        },
    );
    time("blur-matched pair, one pair", &mut || {
        black_box(pairs_of(
            &[&a, &b],
            &[Some(ea), Some(eb)],
            PairMatching::BlurMatched,
        ));
    });
}
