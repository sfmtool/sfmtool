// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A small deterministic generator, so the random tiles are the same on every
/// run and every platform.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// A `size × size` tile of `channels` planes, each pixel `f(c, x, y)`.
fn tile_of(size: usize, channels: usize, f: impl Fn(usize, f64, f64) -> f64) -> Vec<f32> {
    let mut values = Vec::with_capacity(channels * size * size);
    for c in 0..channels {
        for y in 0..size {
            for x in 0..size {
                values.push(f(c, x as f64, y as f64) as f32);
            }
        }
    }
    values
}

fn tile(values: &[f32], channels: usize, size: usize) -> PatchTile<'_> {
    PatchTile {
        values,
        channels,
        width: size,
        height: size,
    }
}

fn params(r: u32) -> SelfSimilarityParams {
    SelfSimilarityParams {
        max_radius: r,
        ..SelfSimilarityParams::default()
    }
}

/// The reading of the `core × core` template centred in a tile with `r` px
/// around it.
fn centred(values: &[f32], channels: usize, core: usize, r: u32) -> SelfSimilarity {
    let size = core + 2 * r as usize;
    zncc_self_similarity_radius(
        &tile(values, channels, size),
        [r as usize, r as usize, core, core],
        &params(r),
    )
}

/// A straight edge through the tile centre along the direction `angle_deg`
/// (from the `x` axis toward `y`, row-down), 150 grey levels high, with a
/// one-pixel linear ramp across it.
fn edge(size: usize, angle_deg: f64) -> Vec<f32> {
    let (s, c) = angle_deg.to_radians().sin_cos();
    let mid = (size as f64 - 1.0) / 2.0;
    tile_of(size, 1, |_, x, y| {
        let across = -(x - mid) * s + (y - mid) * c;
        50.0 + 150.0 * (0.5 + across).clamp(0.0, 1.0)
    })
}

/// The value of a shift's entry in a surface.
fn surface_at(s: &SelfSimilarity, r: i64, dx: i64, dy: i64) -> f64 {
    let side = 2 * r + 1;
    s.surface[((dy + r) * side + dx + r) as usize]
}

#[test]
fn avx2_matches_scalar() {
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("skipping: not x86_64");
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            eprintln!("skipping: AVX2+FMA not available");
            return;
        }
        let mut rng = Lcg(7);
        let mut cases = 0;
        for r in 1..=3u32 {
            for core in [4usize, 5, 8, 11, 12, 17, 24] {
                for channels in 1..=3 {
                    // A rough tile and a smooth one, so the tolerance test sees
                    // shifts on both sides of it.
                    for smooth in [false, true] {
                        let size = core + 2 * r as usize + 3;
                        let rough: Vec<f32> = (0..channels * size * size)
                            .map(|_| 255.0 * rng.next_f32())
                            .collect();
                        let values = if smooth {
                            let (a, b, p) = (
                                rng.next_f32() as f64,
                                rng.next_f32() as f64,
                                rng.next_f32() as f64,
                            );
                            tile_of(size, channels, |c, x, y| {
                                120.0
                                    + 60.0
                                        * ((0.3 + 0.2 * a) * x
                                            + (0.1 + 0.3 * b) * y
                                            + p * 6.0
                                            + c as f64)
                                            .sin()
                                    + 0.5
                                        * f64::from(
                                            rough[(c * size + y as usize) * size + x as usize]
                                                / 255.0,
                                        )
                            })
                        } else {
                            rough
                        };
                        let t = tile(&values, channels, size);
                        // Off-centre, so the template does not sit flush with
                        // the padding on every side.
                        let template = [r as usize + 1, r as usize + 2, core, core];
                        let simd = radius_with(&t, template, &params(r), Kernel::Dispatch);
                        let scalar = radius_with(&t, template, &params(r), Kernel::Scalar);
                        for (a, b) in simd.surface.iter().zip(&scalar.surface) {
                            assert_eq!(a.is_nan(), b.is_nan());
                            if !a.is_nan() {
                                assert!(
                                    (a - b).abs() < 1e-4,
                                    "r={r} core={core} c={channels}: {a} vs {b}"
                                );
                            }
                        }
                        // The radius is interpolated from the surface, so it
                        // carries the kernels' f32 rounding too, amplified
                        // where the ZNCC either side of a crossing is close;
                        // it is shown to one decimal.
                        assert!(
                            (simd.radius - scalar.radius).abs() < 1e-3,
                            "r={r} core={core} c={channels}: {} against {}",
                            simd.radius,
                            scalar.radius
                        );
                        assert_eq!(simd.slide, scalar.slide, "r={r} core={core} c={channels}");
                        assert_eq!(simd.tolerance, scalar.tolerance);
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 3 * 7 * 3 * 2);
    }
}

#[test]
fn equal_luminance_colour_edge_scores_as_an_edge() {
    // Red falls and green rises across a vertical edge so that Rec.601
    // luminance is the same on both sides.
    let size = 18;
    let values = tile_of(size, 3, |c, x, _| {
        let right = x >= 9.0;
        match (c, right) {
            (0, false) => 200.0,
            (0, true) => 100.0,
            (1, false) => 100.0,
            (1, true) => 100.0 + 100.0 * 0.299 / 0.587,
            _ => 80.0,
        }
    });
    let luma = tile_of(size, 1, |_, x, y| {
        let at = y as usize * size + x as usize;
        let n = size * size;
        0.299 * f64::from(values[at])
            + 0.587 * f64::from(values[n + at])
            + 0.114 * f64::from(values[2 * n + at])
    });
    // Luminance alone is flat.
    let flat = centred(&luma, 1, 12, 3);
    assert_eq!(flat.tolerance, f64::INFINITY);

    let s = centred(&values, 3, 12, 3);
    assert!(s.tolerance.is_finite(), "the colour edge carries texture");
    assert_eq!(s.radius, 3.0, "a vertical edge slides along itself");
    assert!(
        s.slide[1].abs() > 0.99 && s.slide[0].abs() < 1e-9,
        "{:?}",
        s.slide
    );
    // Across the edge, the shift is plainly distinguishable.
    assert!(1.0 - surface_at(&s, 3, 1, 0) > s.tolerance);
}

#[test]
fn a_channel_flat_in_the_template_is_left_out() {
    let size = 18;
    let corner = |x: f64, y: f64| if x >= 9.0 && y >= 9.0 { 200.0 } else { 50.0 };
    let two = tile_of(size, 2, |c, x, y| corner(x, y) + 10.0 * c as f64 * x);
    let three = tile_of(size, 3, |c, x, y| {
        if c == 2 {
            90.0
        } else {
            corner(x, y) + 10.0 * c as f64 * x
        }
    });
    let a = centred(&two, 2, 12, 3);
    let b = centred(&three, 3, 12, 3);
    assert_eq!(a.radius, b.radius);
    assert_eq!(a.slide, b.slide);
    assert!((a.tolerance - b.tolerance).abs() < 1e-12);
    for (x, y) in a.surface.iter().zip(&b.surface) {
        assert!(x.is_nan() && y.is_nan() || (x - y).abs() < 1e-6);
    }
}

#[test]
fn grey_repeated_in_three_channels_scores_as_one() {
    let mut rng = Lcg(3);
    let size = 20;
    let grey: Vec<f32> = (0..size * size)
        .map(|_| 40.0 + 30.0 * rng.next_f32())
        .collect();
    let grey = tile_of(size, 1, |_, x, y| {
        f64::from(grey[y as usize * size + x as usize]) + 20.0 * (0.4 * x).sin()
    });
    let rgb: Vec<f32> = grey.iter().chain(&grey).chain(&grey).copied().collect();
    let one = centred(&grey, 1, 14, 3);
    let three = centred(&rgb, 3, 14, 3);
    assert_eq!(one.radius, three.radius);
    assert_eq!(one.slide, three.slide);
    assert!((one.tolerance - three.tolerance).abs() < 1e-12);
    for (x, y) in one.surface.iter().zip(&three.surface) {
        assert!(x.is_nan() && y.is_nan() || (x - y).abs() < 1e-9);
    }
}

/// A patch that locks reads the fraction of a pixel its peak takes to fall
/// through the level: under one pixel, and more than nothing.
#[test]
fn a_corner_and_a_blob_score_under_a_pixel() {
    let size = 18;
    let corner = tile_of(
        size,
        1,
        |_, x, y| if x >= 9.0 && y >= 9.0 { 200.0 } else { 50.0 },
    );
    let s = centred(&corner, 1, 12, 3);
    assert!(s.radius > 0.0 && s.radius < 1.0, "{s:?}");
    assert_eq!(s.slide, [0.0, 0.0]);

    let blob = tile_of(size, 1, |_, x, y| {
        let (dx, dy) = (x - 8.5, y - 8.5);
        40.0 + 150.0 * (-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5)).exp()
    });
    let s = centred(&blob, 1, 12, 3);
    assert!(s.radius > 0.0 && s.radius < 1.0, "{s:?}");
}

#[test]
fn a_straight_edge_scores_r_in_every_direction_and_slides_along_itself() {
    // From r = 2 up: the r = 1 disk holds only the four axis shifts, so an
    // edge at 45° has no shift along itself to match at.
    for r in 2..=3u32 {
        for angle in [0.0f64, 30.0, 45.0, 90.0] {
            let size = 16 + 2 * r as usize;
            let s = centred(&edge(size, angle), 1, 16, r);
            assert_eq!(s.radius, r as f64, "r={r} angle={angle}: {s:?}");
            let (sin, cos) = angle.to_radians().sin_cos();
            let len = s.slide[0].hypot(s.slide[1]);
            let along = (s.slide[0] * cos + s.slide[1] * sin).abs() / len;
            assert!(len > 0.5, "r={r} angle={angle}: slide {:?}", s.slide);
            assert!(along > 0.95, "r={r} angle={angle}: slide {:?}", s.slide);
        }
    }
}

#[test]
fn flat_noise_and_a_sky_ramp_score_r() {
    let size = 18;
    let flat = tile_of(size, 3, |_, _, _| 128.0);
    let s = centred(&flat, 3, 12, 3);
    assert_eq!(s.radius, 3.0);
    assert_eq!(s.slide, [0.0, 0.0]);
    assert!(s.surface.iter().all(|v| v.is_nan()));

    let mut rng = Lcg(11);
    let noise: Vec<f32> = (0..size * size)
        .map(|_| 100.0 + (rng.next_f32() * 2.0).floor())
        .collect();
    assert_eq!(centred(&noise, 1, 12, 3).radius, 3.0);

    let ramp = tile_of(size, 1, |_, x, y| (100.0 + 0.3 * x + 0.2 * y).round());
    assert_eq!(centred(&ramp, 1, 12, 3).radius, 3.0);
}

#[test]
fn repeats_score_the_repeat_distance() {
    // Two random columns, repeated every 2 px along x.
    let mut rng = Lcg(5);
    let size = 20;
    let columns: Vec<f64> = (0..2 * size)
        .map(|_| 255.0 * rng.next_f32() as f64)
        .collect();
    let axis = tile_of(size, 1, |_, x, y| {
        columns[(x as usize % 2) * size + y as usize]
    });
    let s = centred(&axis, 1, 14, 3);
    // The repeat at 2 px matches, and the surface falls through the level a
    // fraction of a pixel either side of it.
    assert!(s.radius > 2.0 && s.radius < 2.3, "{s:?}");
    assert!(s.slide[0].abs() > 0.99, "{:?}", s.slide);

    // A random texture across the diagonal, constant along (1, 1) but for a
    // slow wave along it: the shift (1, 1) stays within the tolerance and
    // (2, 2) does not. (An exact repeat every (1, 1) also repeats at (2, 2),
    // which lies in the outer ring and saturates.)
    let size = 22;
    let stripe: Vec<f64> = (0..2 * size)
        .map(|_| 255.0 * rng.next_f32() as f64)
        .collect();
    let diagonal = tile_of(size, 1, |_, x, y| {
        let across = (x - y) as i64 + size as i64;
        stripe[across as usize] + 60.0 * (std::f64::consts::TAU * (x + y) / 24.0).sin()
    });
    let s = centred(&diagonal, 1, 16, 3);
    assert!(s.radius > 2f64.sqrt() && s.radius < 2.0, "{s:?}");
}

/// The crossing is where the ZNCC, interpolated linearly along a grid edge,
/// equals the level.
#[test]
fn the_radius_is_where_the_surface_crosses_the_level() {
    // A 3 x 3 surface (r = 1): the centre at 1, the right-hand neighbour at
    // 0.9 and the other three at 0.5, read at the level 0.8.
    let nan = f64::NAN;
    let surface = [nan, 0.5, nan, 0.5, 1.0, 0.9, nan, 0.5, nan];
    // Right: 1 to 0.9 stays above 0.8, so no crossing between them, and the
    // shift (1, 0) is on the disk's rim with no neighbour inside it. Left, up
    // and down: 1 to 0.5 crosses 0.8 two fifths of the way out.
    let radius = super::crossing_radius(&surface, 1, 0.8);
    assert!((radius - 0.4).abs() < 1e-12, "{radius}");
}

#[test]
fn parts_agree_with_separate_calls() {
    let mut rng = Lcg(9);
    let (resolution, r) = (24usize, 3usize);
    let size = resolution + 2 * r;
    let noise: Vec<f32> = (0..3 * size * size).map(|_| rng.next_f32()).collect();
    let values = tile_of(size, 3, |c, x, y| {
        // An edge in one part, texture in another, near-flat elsewhere.
        let edge = if x + 0.3 * y > 14.0 { 60.0 } else { 0.0 };
        let texture = if y < 12.0 {
            40.0 * (1.3 * x + c as f64).sin() * (0.9 * y).cos()
        } else {
            0.0
        };
        80.0 + edge + texture + 3.0 * f64::from(noise[(c * size + y as usize) * size + x as usize])
    });
    let t = tile(&values, 3, size);
    let p = params(r as u32);
    let parts = zncc_self_similarity_parts(&t, resolution, &p);
    let check = |part: &SelfSimilarity, rect: [usize; 4]| {
        let alone = zncc_self_similarity_radius(&t, rect, &p);
        assert!(
            (part.radius - alone.radius).abs() < 1e-3,
            "{rect:?}: {} against {}",
            part.radius,
            alone.radius
        );
        assert!(
            (part.slide[0] - alone.slide[0]).abs() < 1e-9
                && (part.slide[1] - alone.slide[1]).abs() < 1e-9
        );
        assert!((part.tolerance - alone.tolerance).abs() < 1e-9);
        for (a, b) in part.surface.iter().zip(&alone.surface) {
            assert!(
                a.is_nan() && b.is_nan() || (a - b).abs() < 1e-5,
                "{rect:?}: {a} vs {b}"
            );
        }
    };
    check(&parts.whole, [r, r, resolution, resolution]);
    check(&parts.middle, [r + 6, r + 6, 12, 12]);
    for row in 0..3 {
        for col in 0..3 {
            check(&parts.grid[row][col], [r + 8 * col, r + 8 * row, 8, 8]);
        }
    }
}

#[test]
fn the_surface_is_one_at_the_centre_and_nan_outside_the_disk() {
    let s = centred(&edge(18, 30.0), 1, 12, 3);
    assert_eq!(s.surface.len(), 49);
    assert_eq!(surface_at(&s, 3, 0, 0), 1.0);
    for (dx, dy) in [(-3, -3), (3, -2), (-2, 3), (3, 3)] {
        assert!(surface_at(&s, 3, dx, dy).is_nan());
    }
    assert!(surface_at(&s, 3, 3, 0).is_finite() && surface_at(&s, 3, 2, 2).is_finite());
}

#[test]
#[should_panic(expected = "needs 3 px of tile on every side, but the tile is 16×16")]
fn too_little_margin_is_refused() {
    let values = vec![0.0f32; 16 * 16];
    zncc_self_similarity_radius(&tile(&values, 1, 16), [2, 3, 10, 10], &params(3));
}

#[test]
#[should_panic(expected = "needs a 30×30 tile, but the tile is 28×28")]
fn parts_refuse_a_tile_of_the_wrong_size() {
    let values = vec![0.0f32; 28 * 28];
    zncc_self_similarity_parts(&tile(&values, 1, 28), 24, &params(3));
}

#[test]
fn planes_from_interleaved_drops_alpha() {
    let patch: Vec<f32> = (0..2 * 2 * 4).map(|v| v as f32).collect();
    let (planes, channels) = PatchTile::planes_from_interleaved(&patch, 2, 2, 4);
    assert_eq!(channels, 3);
    assert_eq!(
        planes,
        vec![0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0]
    );
}
