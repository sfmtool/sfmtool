// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::test_tiles::*;
use super::*;
use crate::patch::reference_view::pair_zncc_readings;

#[test]
fn the_semi_axes_are_read_off_the_matrix() {
    let [major, minor] = semi_axes(&ellipse(1.3, 0.4, 37.0));
    assert!((major - 1.3).abs() < 1e-12 && (minor - 0.4).abs() < 1e-12);
    // A matrix with an entry that is not finite has no semi-axes.
    for bad in [f64::NAN, f64::INFINITY] {
        let [major, minor] = semi_axes(&[[1.0, 0.0], [0.0, bad]]);
        assert!(major.is_nan() && minor.is_nan(), "{bad}: {major}, {minor}");
    }
}

// ------------------------------------------------------------ the assessment

/// The blur assessment of `tile`, read on its whole tile.
fn assessment_of(tile: &TilePlanes) -> BlurAssessment {
    assess_blur(
        tile,
        &read(tile),
        |v| read_tile_ellipse(v, tile.channels, tile.side, &tile.data),
        &mut BlurScratch::default(),
    )
    .unwrap()
}

/// The assessment keeps both semi-axes of the tile's own reading and of each
/// probe's, each what a separate reading of the tile blurred by that probe
/// gives, bit for bit; one scratch reused over several tiles assesses each
/// as a fresh one does.
#[test]
fn the_assessment_keeps_both_axes_of_every_reading() {
    let mut scratch = BlurScratch::default();
    for (seed, smooth) in [(31, 0.8), (32, 1.4), (33, 2.0)] {
        let tile = textured(seed, smooth);
        let a = assessment_of(&tile);
        let bits = |axes: [f64; 2]| axes.map(f64::to_bits);
        assert_eq!(bits(a.semi_axes), bits(semi_axes(&read(&tile))));
        for (axes, &sigma) in a.growth.iter().zip(&GROWTH_PROBE_SIGMAS) {
            let probe = tile.blurred(sigma, &mut BlurScratch::default());
            assert_eq!(bits(*axes), bits(semi_axes(&read(&probe))), "σ {sigma}");
            assert!(axes[1] <= axes[0]);
        }
        // Blurring lengthens both axes.
        assert!(a.growth[1][0] > a.growth[0][0] && a.growth[0][0] > a.semi_axes[0]);
        assert!(a.growth[1][1] > a.semi_axes[1]);
        let reused = assess_blur(
            &tile,
            &read(&tile),
            |v| read_tile_ellipse(v, tile.channels, tile.side, &tile.data),
            &mut scratch,
        );
        assert_eq!(reused, Some(a));
    }
}

/// A tile whose blurred probes cannot be read has no assessment.
#[test]
fn a_tile_whose_probes_cannot_be_read_has_no_assessment() {
    let tile = textured(71, 0.8);
    let e = ellipse(0.5, 0.5, 0.0);
    let mut calls = 0;
    let none = assess_blur(
        &tile,
        &e,
        |_| {
            calls += 1;
            None
        },
        &mut BlurScratch::default(),
    );
    assert_eq!(none, None);
    // The first probe that cannot be read ends the assessment.
    assert_eq!(calls, 1);
}

/// The width is read off the readings: on a growth that is linear in `σ²`
/// it is exact; a piece that does not grow is passed over, and past the
/// widest probe the line goes on along the last piece that grew; a growth
/// that never grows gives none; a length already reached gives 0.
#[test]
fn the_width_is_read_off_the_probes() {
    let [p1, p2] = GROWTH_PROBE_SIGMAS;
    // Only the semi-major axes set the width.
    let of = |l2: [f64; 3]| BlurAssessment {
        semi_axes: [l2[0].sqrt(), 0.1],
        growth: [[l2[1].sqrt(), 0.1], [l2[2].sqrt(), 0.1]],
    };
    // l² = 0.25 + σ².
    let linear = of([0.25, 0.25 + p1 * p1, 0.25 + p2 * p2]);
    let s = linear.sigma_to_reach(1.0).unwrap();
    assert!((s - 0.75f64.sqrt()).abs() < 1e-12, "{s}");
    assert_eq!(linear.sigma_to_reach(0.4), Some(0.0));
    // Past the widest probe the line goes on along its last piece.
    let s = linear.sigma_to_reach(1.5).unwrap();
    assert!((s - 2.0f64.sqrt()).abs() < 1e-12, "{s}");
    // Flat up to the narrow probe, then l² = 0.25 + (σ² − p1²) · 2.
    let late = of([0.25, 0.25, 0.25 + 2.0 * (p2 * p2 - p1 * p1)]);
    let s = late.sigma_to_reach(0.6).unwrap();
    let want = (p1 * p1 + (0.36 - 0.25) / 2.0).sqrt();
    assert!((s - want).abs() < 1e-12, "{s} against {want}");
    // The widest probe reads shorter than the narrow one: it is passed over,
    // and a length past the narrow probe's is reached along the first piece.
    let dip = BlurAssessment {
        semi_axes: [0.5, 0.4],
        growth: [[0.8, 0.5], [0.7, 0.6]],
    };
    let slope = (0.64 - 0.25) / (p1 * p1);
    let s = dip.sigma_to_reach(0.9).unwrap();
    let want = (p1 * p1 + (0.81 - 0.64) / slope).sqrt();
    assert!((s - want).abs() < 1e-12, "{s} against {want}");
    assert!((s - 0.48).abs() < 0.01, "{s}");
    let s = dip.sigma_to_reach(0.6).unwrap();
    let want = ((0.36 - 0.25) / slope).sqrt();
    assert!((s - want).abs() < 1e-12, "{s} against {want}");
    // No growth at all: no width.
    assert_eq!(of([0.25, 0.25, 0.25]).sigma_to_reach(1.0), None);
    assert_eq!(of([0.25, f64::NAN, 0.3]).sigma_to_reach(1.0), None);
    // The width is capped.
    assert_eq!(
        of([0.04, 0.041, 0.042]).sigma_to_reach(2.0),
        Some(MAX_BLUR_SIGMA)
    );
}

/// A copy of a tile blurred by a known round Gaussian: the tile blurred to
/// the copy's semi-minor axis by the width its own assessment gives ends
/// within 5% of that length, by no more than the planted width. The
/// texture's ellipse is about 0.45 grid px, as a sharp view's is.
#[test]
fn blur_to_length_recovers_a_planted_round_blur() {
    let mut ratios = Vec::new();
    for (seed, planted) in [(61, 0.8), (62, 1.0), (63, 1.2), (64, 1.6), (65, 2.0)] {
        let sharp = textured(seed, 1.6);
        let blurry = sharp.blurred(planted, &mut BlurScratch::default());
        let length = semi_axes(&read(&blurry))[1];
        let a = assessment_of(&sharp);
        let sigma = a.sigma_to_reach(length).unwrap();
        let (blurred, width) =
            blur_to_length(&sharp, &a, length, &mut BlurScratch::default()).unwrap();
        assert_eq!(width, sigma);
        assert_eq!(blurred, sharp.blurred(sigma, &mut BlurScratch::default()));
        let ratio = semi_axes(&read(&blurred))[0] / length;
        assert!(
            (0.95..=1.05).contains(&ratio),
            "{seed}: planted {planted}, width {sigma}, ratio {ratio}"
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

/// A tile whose semi-major axis is already as long as asked is not blurred:
/// it comes back as it is, with a width of 0.
#[test]
fn blur_to_length_leaves_a_long_enough_tile_unblurred() {
    let tile = textured(81, 1.2);
    let a = assessment_of(&tile);
    for length in [a.semi_axes[0], a.semi_axes[0] * 0.5, 0.0] {
        let (out, width) = blur_to_length(&tile, &a, length, &mut BlurScratch::default()).unwrap();
        assert_eq!(width, 0.0);
        assert_eq!(out, tile);
    }
    // An assessment that does not grow gives no width, and no tile.
    let flat = BlurAssessment {
        semi_axes: a.semi_axes,
        growth: [a.semi_axes; 2],
    };
    let length = 2.0 * a.semi_axes[0];
    assert_eq!(
        blur_to_length(&tile, &flat, length, &mut BlurScratch::default()),
        None
    );
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
        black_box(crate::patch::pair_sharpness::pair_blur(
            black_box(&ea),
            black_box(&eb),
            1.0,
        ));
    });
    time("whole self-similarity reading of a tile", &mut || {
        black_box(read_tile_ellipse(&b.values, 3, SIDE, &b.data));
    });
    time("one tile's blur assessment (assess_blur)", &mut || {
        black_box(assess_blur(
            &a,
            &ea,
            |v| read_tile_ellipse(v, 3, SIDE, &a.data),
            &mut scratch,
        ));
    });
    let assessment = assessment_of(&a);
    let length = semi_axes(&eb)[1];
    let mut blurred = TilePlanes::default();
    time("a blurred pair, its tile already assessed", &mut || {
        blur_to_length_into(&a, &assessment, length, &mut blurred, &mut scratch);
        black_box(pair_zncc_readings(&blurred, &b, &w));
    });
}
