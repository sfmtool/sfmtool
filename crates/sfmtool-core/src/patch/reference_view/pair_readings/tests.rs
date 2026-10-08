// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::patch::blur_matched::test_tiles::*;
use crate::patch::blur_matched::{semi_axes, GROWTH_PROBE_SIGMAS};

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
        &Progress::none(),
    )
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

/// Each view some pair blurs is assessed once, however many pairs blur it: a
/// sharp view blurred against three equally blurry ones has one blurred tile
/// read per probe.
#[test]
fn each_view_is_assessed_once() {
    let sharp = textured(91, 1.4);
    let blurry = sharp.blurred(1.4, &mut BlurScratch::default());
    let views = [&sharp, &blurry, &blurry, &blurry];
    let ellipses: Vec<_> = views.iter().map(|v| Some(ellipse_of(v))).collect();
    let only_sharp = pairs_of(&views, &ellipses, PairMatching::BlurMatched);
    assert_eq!((only_sharp.pairs, only_sharp.pairs_blurred), (6, 3));
    assert_eq!(only_sharp.ellipse_reads, GROWTH_PROBE_SIGMAS.len());
    let mut graded = vec![sharp.clone()];
    for s in [0.8, 1.2, 1.6] {
        graded.push(sharp.blurred(s, &mut BlurScratch::default()));
    }
    let refs: Vec<&TilePlanes> = graded.iter().collect();
    let ellipses: Vec<_> = graded.iter().map(|v| Some(ellipse_of(v))).collect();
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
/// every pair reads the same, since each view's assessment is a function of
/// its own tile.
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
/// bit: the other pairs, read with the same scratch and assessments, do not
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
