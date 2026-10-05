// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::overlap::{parts_with, radius_with, Route};
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
/// around it, so its overlap is the whole template at every shift.
fn centred(values: &[f32], channels: usize, core: usize, r: u32) -> SelfSimilarity {
    let size = core + 2 * r as usize;
    zncc_self_similarity_radius(
        &tile(values, channels, size),
        None,
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

/// The unit direction of a reading's major axis, `[x, y]` in the grid frame.
fn major_direction(s: &SelfSimilarity) -> [f64; 2] {
    let (sin, cos) = s.ellipse.major_angle.sin_cos();
    [cos, sin]
}

/// How long and thin a reading's ellipse is: `1 − (minor / major)²`, 0 for a
/// circle and near 1 for a long thin region.
fn elongation(s: &SelfSimilarity) -> f64 {
    let [a, b] = s.ellipse.axes;
    1.0 - (b / a).powi(2)
}

/// Whether two readings' ellipse axes agree to `tolerance`.
fn axes_close(a: &SelfSimilarity, b: &SelfSimilarity, tolerance: f64) -> bool {
    a.ellipse
        .axes
        .iter()
        .zip(&b.ellipse.axes)
        .all(|(x, y)| x.is_nan() && y.is_nan() || (x - y).abs() < tolerance)
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
                        // the padding on every side; and in the corner, where
                        // the moved windows run off the tile.
                        for template in [
                            [r as usize + 1, r as usize + 2, core, core],
                            [0, 0, core, core],
                        ] {
                            let simd = radius_with(&t, None, template, &params(r), Route::Auto);
                            let scalar =
                                radius_with(&t, None, template, &params(r), Route::DenseScalar);
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
                            assert!(
                                axes_close(&simd, &scalar, 1e-3),
                                "r={r} core={core} c={channels}: {:?} against {:?}",
                                simd.ellipse,
                                scalar.ellipse
                            );
                            assert_eq!(simd.tolerance, scalar.tolerance);
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 3 * 7 * 3 * 2 * 2);
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
        major_direction(&s)[1].abs() > 0.99 && elongation(&s) > 0.9,
        "{:?}",
        s.ellipse
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
    // Interpolated from the surface, so equal to rounding, not to the bit.
    assert!(
        (a.radius - b.radius).abs() < 1e-9,
        "{} vs {}",
        a.radius,
        b.radius
    );
    assert!(
        axes_close(&a, &b, 1e-9),
        "{:?} vs {:?}",
        a.ellipse,
        b.ellipse
    );
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
    // The radius is interpolated from the surface, which the channel mean
    // reproduces to rounding, not to the bit.
    assert!(
        (one.radius - three.radius).abs() < 1e-9,
        "{} vs {}",
        one.radius,
        three.radius
    );
    assert!(axes_close(&one, &three, 1e-9));
    assert!((one.tolerance - three.tolerance).abs() < 1e-12);
    for (x, y) in one.surface.iter().zip(&three.surface) {
        assert!(x.is_nan() && y.is_nan() || (x - y).abs() < 1e-9);
    }
}

/// A patch that locks reads a small region around its peak: an ellipse under
/// one pixel, and more than nothing.
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
    assert!(s.ellipse.axes[1] > 0.0 && !s.radius_is_at_least(), "{s:?}");

    let blob = tile_of(size, 1, |_, x, y| {
        let (dx, dy) = (x - 8.5, y - 8.5);
        40.0 + 150.0 * (-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5)).exp()
    });
    let s = centred(&blob, 1, 12, 3);
    assert!(s.radius > 0.0 && s.radius < 1.0, "{s:?}");
}

#[test]
fn a_straight_edge_scores_r_in_every_direction_with_its_major_axis_along_itself() {
    // From r = 2 up: the r = 1 disk holds only the four axis shifts, so an
    // edge at 45° has no shift along itself to match at.
    for r in 2..=3u32 {
        for angle in [0.0f64, 30.0, 45.0, 90.0] {
            let size = 16 + 2 * r as usize;
            let s = centred(&edge(size, angle), 1, 16, r);
            assert_eq!(s.radius, r as f64, "r={r} angle={angle}: {s:?}");
            let (sin, cos) = angle.to_radians().sin_cos();
            let [x, y] = major_direction(&s);
            let along = (x * cos + y * sin).abs();
            assert!(elongation(&s) > 0.5, "r={r} angle={angle}: {:?}", s.ellipse);
            assert!(along > 0.95, "r={r} angle={angle}: {:?}", s.ellipse);
            assert!(
                s.radius_is_at_least(),
                "r={r} angle={angle}: {:?}",
                s.ellipse
            );
        }
    }
}

#[test]
fn flat_noise_and_a_sky_ramp_score_r() {
    let size = 18;
    let flat = tile_of(size, 3, |_, _, _| 128.0);
    let s = centred(&flat, 3, 12, 3);
    assert_eq!(s.radius, 3.0);
    assert_eq!(s.ellipse.axes, [3.0, 3.0]);
    assert_eq!(s.ellipse.axes_is_at_least, [true, true]);
    assert!(s.ellipse.major_angle.is_nan());
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
fn a_repeat_inside_the_square_lengthens_the_ellipse() {
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
    // The repeats at ±2 px match as well as the centre does, so the region is
    // three equal lobes along x, whose root mean square distance from the
    // centre is 2·√(2/3): the semi-major axis, twice that, is past r.
    assert_eq!(s.radius, 3.0, "{s:?}");
    assert!(s.radius_is_at_least());
    assert!(major_direction(&s)[0].abs() > 0.99, "{:?}", s.ellipse);

    // A random texture across the diagonal, constant along (1, 1) but for a
    // slow wave along it: the shift (1, 1) stays within the tolerance and
    // (2, 2) does not, so the region is a thin one along the diagonal through
    // ±(1, 1), and the major axis runs along it. (An exact repeat every
    // (1, 1) also repeats at (3, 3), on the square's border, and reads 3.)
    let size = 22;
    let stripe: Vec<f64> = (0..2 * size)
        .map(|_| 255.0 * rng.next_f32() as f64)
        .collect();
    let diagonal = tile_of(size, 1, |_, x, y| {
        let across = (x - y) as i64 + size as i64;
        stripe[across as usize] + 60.0 * (std::f64::consts::TAU * (x + y) / 24.0).sin()
    });
    let s = centred(&diagonal, 1, 16, 3);
    assert!(s.radius > 1.0 && s.radius < 2.5, "{s:?}");
    assert!(elongation(&s) > 0.9, "{:?}", s.ellipse);
    let [x, y] = major_direction(&s);
    assert!((x * y - 0.5).abs() < 0.01, "{:?}", s.ellipse);
    assert!(!s.radius_is_at_least(), "{:?}", s.ellipse);
}

#[test]
fn parts_agree_with_separate_calls() {
    let mut rng = Lcg(9);
    let resolution = 24usize;
    let size = resolution;
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
    let p = params(3);
    let parts = zncc_self_similarity_parts(&t, None, &p);
    let check = |part: &SelfSimilarity, rect: [usize; 4]| {
        let alone = zncc_self_similarity_radius(&t, None, rect, &p);
        assert!(
            (part.radius - alone.radius).abs() < 1e-3,
            "{rect:?}: {} against {}",
            part.radius,
            alone.radius
        );
        assert!(axes_close(part, &alone, 1e-3), "{rect:?}");
        assert!((part.tolerance - alone.tolerance).abs() < 1e-9);
        for (a, b) in part.surface.iter().zip(&alone.surface) {
            assert!(
                a.is_nan() && b.is_nan() || (a - b).abs() < 1e-5,
                "{rect:?}: {a} vs {b}"
            );
        }
    };
    check(&parts.whole, [0, 0, resolution, resolution]);
    check(&parts.middle, [6, 6, 12, 12]);
    for row in 0..3 {
        for col in 0..3 {
            check(&parts.grid[row][col], [8 * col, 8 * row, 8, 8]);
        }
    }
}

#[test]
fn the_surface_is_one_at_the_centre_and_covers_the_whole_square() {
    let s = centred(&edge(18, 30.0), 1, 12, 3);
    assert_eq!(s.surface.len(), 49);
    assert_eq!(surface_at(&s, 3, 0, 0), 1.0);
    assert!(s.surface.iter().all(|z| z.is_finite()));
}

#[test]
#[should_panic(expected = "the 10×10 template at (8, 3) does not fit in the 16×16 tile")]
fn a_template_outside_the_tile_is_refused() {
    let values = vec![0.0f32; 16 * 16];
    zncc_self_similarity_radius(&tile(&values, 1, 16), None, [8, 3, 10, 10], &params(3));
}

#[test]
#[should_panic(expected = "the bitmap is 28×24, it needs to be square")]
fn parts_refuse_a_bitmap_that_is_not_square() {
    let values = vec![0.0f32; 28 * 24];
    let bitmap = PatchTile {
        values: &values,
        channels: 1,
        width: 28,
        height: 24,
    };
    zncc_self_similarity_parts(&bitmap, None, &params(3));
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

// ---- The overlap reading -------------------------------------------------

/// The overlap reading of a whole `size × size` bitmap with no ring.
fn overlap_whole(
    values: &[f32],
    channels: usize,
    size: usize,
    data: Option<&[bool]>,
) -> SelfSimilarity {
    zncc_self_similarity_radius(
        &tile(values, channels, size),
        data,
        [0, 0, size, size],
        &params(3),
    )
}

/// Whether two readings are the same bit for bit, a `NaN` matching a `NaN`.
fn same_reading(a: &SelfSimilarity, b: &SelfSimilarity) -> bool {
    let same = |x: f64, y: f64| x.to_bits() == y.to_bits();
    let (e, f) = (&a.ellipse, &b.ellipse);
    same(a.radius, b.radius)
        && same(a.tolerance, b.tolerance)
        && e.axes.iter().zip(&f.axes).all(|(&x, &y)| same(x, y))
        && e.axes_is_at_least == f.axes_is_at_least
        && same(e.major_angle, f.major_angle)
        && e.matrix
            .iter()
            .flatten()
            .zip(f.matrix.iter().flatten())
            .all(|(&x, &y)| same(x, y))
        && a.surface.iter().zip(&b.surface).all(|(&x, &y)| same(x, y))
}

/// A rough three-channel texture over a `size × size` tile: random values
/// smoothed a little, so a shift of a pixel or more decorrelates.
fn rough_texture(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let raw: Vec<f64> = (0..3 * size * size)
        .map(|_| 255.0 * rng.next_f32() as f64)
        .collect();
    let last = size as f64 - 1.0;
    tile_of(size, 3, |c, x, y| {
        let at = |xx: f64, yy: f64| {
            let (xx, yy) = (xx.min(last), yy.min(last));
            raw[(c * size + yy as usize) * size + xx as usize]
        };
        (2.0 * at(x, y) + at(x + 1.0, y) + at(x, y + 1.0)) / 4.0
    })
}

#[test]
fn overlap_a_corner_locks() {
    let size = 24;
    let corner = tile_of(
        size,
        1,
        |_, x, y| if x >= 12.0 && y >= 12.0 { 200.0 } else { 50.0 },
    );
    let s = overlap_whole(&corner, 1, size, None);
    assert!(s.radius > 0.0 && s.radius < 1.0, "{s:?}");
}

#[test]
fn overlap_an_edge_reads_its_major_axis_along_itself() {
    for angle in [0.0f64, 30.0, 45.0, 90.0] {
        let size = 24;
        let s = overlap_whole(&edge(size, angle), 1, size, None);
        assert_eq!(s.radius, 3.0, "angle={angle}: {s:?}");
        let (sin, cos) = angle.to_radians().sin_cos();
        let [x, y] = major_direction(&s);
        let along = (x * cos + y * sin).abs();
        assert!(
            elongation(&s) > 0.5 && along > 0.95,
            "angle={angle}: {:?}",
            s.ellipse
        );
    }
}

#[test]
fn overlap_a_flat_patch_reads_r() {
    let size = 24;
    let flat = tile_of(size, 3, |_, _, _| 128.0);
    let s = overlap_whole(&flat, 3, size, None);
    assert_eq!(s.radius, 3.0);
    assert_eq!(s.ellipse.axes, [3.0, 3.0]);
    assert_eq!(s.tolerance, f64::INFINITY);
    assert!(s.surface.iter().all(|v| v.is_nan()));

    let ramp = tile_of(size, 1, |_, x, y| (100.0 + 0.3 * x + 0.2 * y).round());
    assert_eq!(overlap_whole(&ramp, 1, size, None).radius, 3.0);
}

#[test]
fn overlap_samples_without_data_drop_out() {
    let size = 24;
    let texture = rough_texture(size, 21);
    // The left third carries no data; what it holds must not matter.
    let data: Vec<bool> = (0..size * size).map(|k| k % size >= 8).collect();
    let mut zeroed = texture.clone();
    let mut junk = texture.clone();
    let mut rng = Lcg(3);
    for c in 0..3 {
        for (k, &has) in data.iter().enumerate() {
            if !has {
                zeroed[c * size * size + k] = 0.0;
                junk[c * size * size + k] = 255.0 * rng.next_f32();
            }
        }
    }
    let a = overlap_whole(&zeroed, 3, size, Some(&data));
    let b = overlap_whole(&junk, 3, size, Some(&data));
    assert!(same_reading(&a, &b), "{a:?} vs {b:?}");
    assert!(a.radius < 1.0, "{a:?}");
    // The reading is the one of the covered columns cut out as a bitmap of
    // their own, which takes the dense route: equal to within the kernel's
    // `f32` rounding.
    let covered: Vec<f32> = (0..3 * size)
        .flat_map(|row| texture[row * size + 8..(row + 1) * size].iter().copied())
        .collect();
    let covered_tile = PatchTile {
        values: &covered,
        channels: 3,
        width: size - 8,
        height: size,
    };
    let c = zncc_self_similarity_radius(&covered_tile, None, [0, 0, size - 8, size], &params(3));
    assert!((a.radius - c.radius).abs() < 1e-3, "{a:?} vs {c:?}");
    for (x, y) in a.surface.iter().zip(&c.surface) {
        assert!(x.is_nan() && y.is_nan() || (x - y).abs() < 1e-5);
    }
    let pa = zncc_self_similarity_parts(&tile(&zeroed, 3, size), Some(&data), &params(3));
    let pb = zncc_self_similarity_parts(&tile(&junk, 3, size), Some(&data), &params(3));
    assert!(same_reading(&pa.whole, &pb.whole) && same_reading(&pa.middle, &pb.middle));
    for (row_a, row_b) in pa.grid.iter().zip(&pb.grid) {
        for (a, b) in row_a.iter().zip(row_b) {
            assert!(same_reading(a, b), "{a:?} vs {b:?}");
        }
    }
    // The left column of cells has no data at all: no reading.
    for row in &pa.grid {
        let cell = &row[0];
        assert!(cell.radius.is_nan() && cell.tolerance.is_nan(), "{cell:?}");
        assert!(cell.ellipse.axes.iter().all(|v| v.is_nan()));
        assert!(cell.surface.iter().all(|v| v.is_nan()));
    }
}

#[test]
fn overlap_a_bitmap_with_no_data_has_no_reading() {
    let size = 12;
    let texture = rough_texture(size, 4);
    let data = vec![false; size * size];
    let s = overlap_whole(&texture, 3, size, Some(&data));
    assert!(s.radius.is_nan() && s.tolerance.is_nan());
    assert!(s.ellipse.axes.iter().all(|v| v.is_nan()) && s.ellipse.major_angle.is_nan());
    assert_eq!(s.ellipse.axes_is_at_least, [false, false]);
    let parts = zncc_self_similarity_parts(&tile(&texture, 3, size), Some(&data), &params(3));
    assert!(parts.whole.radius.is_nan() && parts.middle.radius.is_nan());
}

/// On a bitmap cut from a larger textured image, the middle and the centre
/// cell take their shifted windows from the rest of the bitmap and read
/// exactly as the same templates read inside the larger image; the whole
/// bitmap, whose shifted windows lose up to `r` rows and columns, reads close
/// to the same template inside the larger image.
#[test]
fn a_cut_patch_reads_its_middle_as_the_larger_image_does() {
    let (resolution, r) = (24usize, 3usize);
    let wide = resolution + 2 * r;
    for seed in [1u64, 2, 3] {
        // A smooth part and a rough part, so the parts read differently.
        let rough = rough_texture(wide, seed);
        let values = tile_of(wide, 3, |c, x, y| {
            let k = (c * wide + y as usize) * wide + x as usize;
            let smooth = 60.0 * (0.25 * x + 0.1 * y + c as f64).sin();
            let fine = if x > 15.0 { 0.4 } else { 0.05 };
            120.0 + smooth + fine * f64::from(rough[k] - 128.0)
        });
        let wide_tile = tile(&values, 3, wide);
        let inside =
            |rect: [usize; 4]| zncc_self_similarity_radius(&wide_tile, None, rect, &params(3));
        let (inside_whole, inside_middle, inside_centre) = (
            inside([r, r, resolution, resolution]),
            inside([r + 6, r + 6, 12, 12]),
            inside([r + 8, r + 8, 8, 8]),
        );
        let mut core = Vec::with_capacity(3 * resolution * resolution);
        for c in 0..3 {
            for y in 0..resolution {
                let row = (c * wide + y + r) * wide + r;
                core.extend_from_slice(&values[row..row + resolution]);
            }
        }
        let overlap = zncc_self_similarity_parts(&tile(&core, 3, resolution), None, &params(3));
        let same = |a: &SelfSimilarity, b: &SelfSimilarity, what: &str| {
            assert!(
                (a.radius - b.radius).abs() < 1e-4 && (a.tolerance - b.tolerance).abs() < 1e-9,
                "seed {seed} {what}: {} vs {}",
                a.radius,
                b.radius
            );
            for (x, y) in a.surface.iter().zip(&b.surface) {
                assert!(x.is_finite() && y.is_finite(), "{what}: {x} vs {y}");
                assert!(
                    x.is_nan() && y.is_nan() || (x - y).abs() < 1e-5,
                    "{what}: {x} vs {y}"
                );
            }
        };
        same(&inside_middle, &overlap.middle, "middle");
        same(&inside_centre, &overlap.grid[1][1], "centre cell");
        assert!(
            (inside_whole.tolerance - overlap.whole.tolerance).abs() < 1e-9,
            "the tolerance is the whole template's either way"
        );
        assert!(
            (inside_whole.radius - overlap.whole.radius).abs() < 0.25,
            "seed {seed}: whole {} vs {}",
            inside_whole.radius,
            overlap.whole.radius
        );
    }
}

#[test]
fn overlap_parts_agree_with_separate_calls() {
    let size = 24;
    let values = rough_texture(size, 8);
    let data: Vec<bool> = (0..size * size).map(|k| (k * 7) % 11 != 0).collect();
    let t = tile(&values, 3, size);
    let p = params(3);
    let parts = zncc_self_similarity_parts(&t, Some(&data), &p);
    let check = |part: &SelfSimilarity, rect: [usize; 4]| {
        let alone = zncc_self_similarity_radius(&t, Some(&data), rect, &p);
        assert!((part.radius - alone.radius).abs() < 1e-9, "{rect:?}");
        assert!((part.tolerance - alone.tolerance).abs() < 1e-9, "{rect:?}");
        for (a, b) in part.surface.iter().zip(&alone.surface) {
            assert!(a.is_nan() && b.is_nan() || (a - b).abs() < 1e-9, "{rect:?}");
        }
    };
    check(&parts.whole, [0, 0, size, size]);
    check(&parts.middle, [6, 6, 12, 12]);
    for row in 0..3 {
        for col in 0..3 {
            check(&parts.grid[row][col], [8 * col, 8 * row, 8, 8]);
        }
    }
}

/// Where every sample carries data, the dense route reads what the masked
/// route reads, at the bitmap's edges and corners too, for the whole bitmap,
/// its middle and every cell; and data flags that are all set take the dense
/// route.
#[test]
fn the_dense_and_masked_routes_agree() {
    for (size, seed) in [(12usize, 5u64), (24, 6), (25, 7)] {
        let values = rough_texture(size, seed);
        let t = tile(&values, 3, size);
        let p = params(3);
        let close = |a: &SelfSimilarity, b: &SelfSimilarity, what: &str| {
            assert!(
                (a.radius - b.radius).abs() < 1e-3 && (a.tolerance - b.tolerance).abs() < 1e-9,
                "{size} {what}: {a:?} vs {b:?}"
            );
            for (x, y) in a.surface.iter().zip(&b.surface) {
                assert!((x - y).abs() < 1e-5, "{size} {what}: {x} vs {y}");
            }
        };
        let dense = parts_with(&t, None, &p, Route::Auto);
        let masked = parts_with(&t, None, &p, Route::Masked);
        close(&dense.whole, &masked.whole, "whole");
        close(&dense.middle, &masked.middle, "middle");
        for (row_a, row_b) in dense.grid.iter().zip(&masked.grid) {
            for (a, b) in row_a.iter().zip(row_b) {
                close(a, b, "cell");
            }
        }
        for rect in [[0, 0, size, size], [0, 0, 5, 7], [size - 4, 2, 4, 6]] {
            close(
                &radius_with(&t, None, rect, &p, Route::Auto),
                &radius_with(&t, None, rect, &p, Route::Masked),
                "template",
            );
        }
        let all = vec![true; size * size];
        let flagged = zncc_self_similarity_parts(&t, Some(&all), &p);
        assert!(same_reading(&flagged.whole, &dense.whole));
    }
}

/// On a 24×24 bitmap whose halves stand far apart in level, each with a fine
/// texture of its own, the dense route's surface matches the masked route's,
/// which works in `f64` throughout, to `1e-5` for the cells and templates that
/// lie within one half: each template is centred by its own mean before the
/// `f32` kernel runs, so the far level of the other half does not swamp the
/// products of a template's texture. Steps of 1000 grey levels and of 60000 (a
/// float bitmap) are both read. A part that holds both halves is held to
/// `1e-5` at a step of 1000 and to `1e-3` at 60000, since it carries the step
/// in its own `f32` values, which at a level of 30000 keep steps of 0.002, the
/// scale of the bitmap's own `f32` resolution there.
#[test]
fn the_dense_route_keeps_its_precision_across_a_large_step() {
    let size = 24;
    let texture = rough_texture(size, 11);
    for step in [1000.0f64, 60000.0] {
        let values = tile_of(size, 3, |c, x, y| {
            let k = (c * size + y as usize) * size + x as usize;
            let level = if x >= 12.0 { step } else { 0.0 };
            level + 0.1 * f64::from(texture[k])
        });
        let t = tile(&values, 3, size);
        let p = params(3);
        let close = |a: &SelfSimilarity, b: &SelfSimilarity, both_halves: bool, what: &str| {
            let bound = if both_halves && step > 1000.0 {
                1e-3
            } else {
                1e-5
            };
            assert!(
                (a.radius - b.radius).abs() < 10.0 * bound
                    && (a.tolerance - b.tolerance).abs() <= 1e-6 * b.tolerance.abs(),
                "step {step} {what}: {a:?} vs {b:?}"
            );
            for (x, y) in a.surface.iter().zip(&b.surface) {
                assert!(
                    x.is_nan() && y.is_nan() || (x - y).abs() < bound,
                    "step {step} {what}: {x} vs {y}"
                );
            }
        };
        let dense = parts_with(&t, None, &p, Route::Auto);
        let masked = parts_with(&t, None, &p, Route::Masked);
        close(&dense.whole, &masked.whole, true, "whole");
        close(&dense.middle, &masked.middle, true, "middle");
        for (row_a, row_b) in dense.grid.iter().zip(&masked.grid) {
            for (col, (a, b)) in row_a.iter().zip(row_b).enumerate() {
                close(a, b, col == 1, "cell");
            }
        }
        for (rect, both_halves) in [
            ([0, 0, size, size], true),
            ([10, 10, 4, 4], true),
            ([0, 0, 8, 8], false),
            ([16, 4, 8, 8], false),
        ] {
            close(
                &radius_with(&t, None, rect, &p, Route::Auto),
                &radius_with(&t, None, rect, &p, Route::Masked),
                both_halves,
                "template",
            );
        }
        // The cells within one half read their texture: they lock.
        assert!(
            dense.grid[1][0].radius < 1.5,
            "step {step}: {:?}",
            dense.grid[1][0]
        );
        assert!(
            dense.grid[1][2].radius < 1.5,
            "step {step}: {:?}",
            dense.grid[1][2]
        );
    }
}

/// A moved window that is constant at a high level reads as flat by the dense
/// route as it does by the masked route. The dense route takes the window's
/// spread from summed-area tables, whose cancellation at a level of 60000
/// leaves a residue of about `1e-5` that an absolute flat test would take for
/// texture; the test scales with the tables' own size. A float bitmap with a
/// texture of 0 to 25 left of `x = 12` and a constant 60000 right of it is
/// read with a template that straddles the step, whose windows moved right lie
/// wholly on the constant half, and with a 2×2 template just left of the step,
/// whose windows moved 2 or 3 px right do.
#[test]
fn a_window_constant_at_a_high_level_reads_flat_by_both_routes() {
    let size = 24;
    let texture = rough_texture(size, 13);
    let values = tile_of(size, 3, |c, x, y| {
        if x >= 12.0 {
            60000.0
        } else {
            0.1 * f64::from(texture[(c * size + y as usize) * size + x as usize])
        }
    });
    let t = tile(&values, 3, size);
    let p = params(3);
    for rect in [[9, 1, 14, 14], [10, 4, 2, 2]] {
        let dense = radius_with(&t, None, rect, &p, Route::Auto);
        let masked = radius_with(&t, None, rect, &p, Route::Masked);
        assert!(
            (dense.radius - masked.radius).abs() < 1e-4
                && axes_close(&dense, &masked, 1e-4)
                && (dense.tolerance - masked.tolerance).abs() <= 1e-5 * masked.tolerance.abs(),
            "{rect:?}: {dense:?} vs {masked:?}"
        );
        for (x, y) in dense.surface.iter().zip(&masked.surface) {
            assert!(
                x.is_nan() && y.is_nan() || (x - y).abs() < 1e-4,
                "{rect:?}: {x} vs {y}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "a 12×12 tile needs 144 data flags, not 10")]
fn overlap_refuses_the_wrong_number_of_data_flags() {
    let values = vec![0.0f32; 12 * 12];
    overlap_whole(&values, 1, 12, Some(&[true; 10]));
}

#[test]
fn data_from_interleaved_reads_alpha() {
    let patch = [1.0f32, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 9.0];
    assert_eq!(
        PatchTile::data_from_interleaved(&patch, 2, 1, 4),
        Some(vec![false, true])
    );
    assert_eq!(PatchTile::data_from_interleaved(&patch[..6], 2, 1, 3), None);
}

// ---- The ellipse -------------------------------------------------------------

/// A `(2r + 1)²` surface from `z(dx, dy)`, row-major from `(−r, −r)`.
fn surface_of(r: i64, z: impl Fn(i64, i64) -> f64) -> Vec<f64> {
    (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| z(dx, dy))
        .collect()
}

/// An elliptical paraboloid, `z = 1 − 0.05·((along / a)² + (across / b)²)`,
/// its long axis at `angle` from `x` towards `y`, over the square of shifts at
/// `r`. Read at the level 0.95, its region is the ellipse with semi-axes `a`
/// and `b`, as far as the bilinear surface between the shifts follows it.
fn paraboloid(r: i64, a: f64, b: f64, angle: f64) -> SelfSimilarity {
    let (s, c) = angle.sin_cos();
    read_surface(
        surface_of(r, |dx, dy| {
            let (x, y) = (dx as f64, dy as f64);
            let (along, across) = (x * c + y * s, -x * s + y * c);
            1.0 - 0.05 * ((along / a).powi(2) + (across / b).powi(2))
        }),
        r as usize,
        0.05,
    )
}

/// The distance between two angles of an axis, which repeat every π.
fn axis_angle_gap(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::PI);
    d.min(std::f64::consts::PI - d)
}

/// A disc reads a circle whose semi-axis is the disc's radius. The bilinear
/// surface between the shifts lowers a paraboloid by up to half a grid step
/// squared, so the region is a little smaller than the disc, by under 0.05 px
/// at a radius of 12.
#[test]
fn a_disc_reads_a_circle_of_its_radius() {
    let s = paraboloid(20, 12.0, 12.0, 0.0);
    let [a, b] = s.ellipse.axes;
    assert!(
        (a - 12.0).abs() < 0.05 && (b - 12.0).abs() < 0.05,
        "{:?}",
        s.ellipse
    );
    assert!((a - b).abs() < 1e-9, "{:?}", s.ellipse);
    assert_eq!(s.ellipse.axes_is_at_least, [false, false]);
    assert_eq!(s.radius.to_bits(), a.to_bits());
    assert!(!s.radius_is_at_least());
    // E = a²·I.
    let m = s.ellipse.matrix;
    assert!(
        (m[0][0] - a * a).abs() < 1e-6 && m[0][1].abs() < 1e-6,
        "{m:?}"
    );
}

/// A region elongated along `x`, and the same turned by 30° and 120°, read
/// their two semi-axes and the angle of the long one, from `x` towards `y`
/// (row-down). The bilinear surface between the shifts falls a little faster
/// than the paraboloid across each cell, so the axes read up to 1% short.
#[test]
fn an_elongated_region_reads_its_axes_and_angle() {
    for degrees in [0.0f64, 30.0, 120.0] {
        let angle = degrees.to_radians();
        let s = paraboloid(16, 10.0, 4.0, angle);
        let e = s.ellipse;
        assert!(
            (e.axes[0] - 10.0).abs() < 0.1 && (e.axes[1] - 4.0).abs() < 0.04,
            "{degrees}°: {e:?}"
        );
        assert!(
            axis_angle_gap(e.major_angle, angle) < 2e-3,
            "{degrees}°: {e:?}"
        );
        assert!((0.0..std::f64::consts::PI).contains(&e.major_angle));
        assert_eq!(e.axes_is_at_least, [false, false], "{degrees}°");
        // The matrix is the ellipse: its quadratic form is 1 at the end of
        // each semi-axis.
        let (sin, cos) = e.major_angle.sin_cos();
        let m = e.matrix;
        let det = m[0][0] * m[1][1] - m[0][1] * m[0][1];
        let inv = [
            [m[1][1] / det, -m[0][1] / det],
            [-m[0][1] / det, m[0][0] / det],
        ];
        for (len, [ux, uy]) in [(e.axes[0], [cos, sin]), (e.axes[1], [-sin, cos])] {
            let (x, y) = (len * ux, len * uy);
            let q = x * x * inv[0][0] + 2.0 * x * y * inv[0][1] + y * y * inv[1][1];
            assert!((q - 1.0).abs() < 1e-9, "{degrees}°: {q}");
        }
    }
}

/// A separate lobe counts as far from the centre as it lies: the moments are
/// about the true position, not the region's centroid.
#[test]
fn a_separate_lobe_lengthens_the_ellipse_towards_it() {
    let lobe = |with: bool| {
        read_surface(
            surface_of(6, |dx, dy| match (dx, dy) {
                (0, 0) => 1.0,
                (4, 0) if with => 1.0,
                _ => 0.5,
            }),
            6,
            0.05,
        )
    };
    let alone = lobe(false);
    let both = lobe(true);
    // Alone, the peak's diamond reads a small circle.
    assert!(alone.radius < 0.2, "{:?}", alone.ellipse);
    // Two equal lobes 4 apart: a root mean square distance of √8 along x.
    assert!(
        (both.radius - 2.0 * 8f64.sqrt()).abs() < 0.05,
        "{:?}",
        both.ellipse
    );
    assert!(axis_angle_gap(both.ellipse.major_angle, 0.0) < 1e-9);
    assert!(both.ellipse.axes[1] < 0.2);
}

/// The region's area and moments match a dense midpoint sum of the bilinear
/// surface on a real sighting's surface and on a random one, to what the sum
/// resolves: its samples `1/400` px apart straddle the level's curve, which
/// moves the sum by a part in 10⁴ or so. Clipping each cell to the polygon
/// through the crossings on its sides, as marching squares draws it, misses
/// by ten times that.
#[test]
fn the_region_moments_match_a_dense_sum_of_the_bilinear_surface() {
    #[rustfmt::skip]
    let real = vec![
        0.514, 0.594, 0.661, 0.714, 0.748, 0.762, 0.758,
        0.68,  0.755, 0.816, 0.86,  0.879, 0.877, 0.853,
        0.804, 0.873, 0.927, 0.961, 0.963, 0.939, 0.896,
        0.883, 0.94,  0.981, 1.0,   0.98,  0.935, 0.872,
        0.908, 0.946, 0.965, 0.961, 0.923, 0.861, 0.783,
        0.876, 0.891, 0.884, 0.856, 0.803, 0.729, 0.641,
        0.79,  0.783, 0.754, 0.708, 0.641, 0.559, 0.464,
    ];
    let mut rng = Lcg(17);
    let random: Vec<f64> = (0..49)
        .map(|k| {
            if k == 24 {
                1.0
            } else {
                0.85 + 0.15 * f64::from(rng.next_f32())
            }
        })
        .collect();
    for (surface, level) in [(real, 1.0 - 0.05421260937718174), (random, 0.93)] {
        let got = super::ellipse::region_moments(&surface, 3, level);
        let n = 400usize;
        let h = 1.0 / n as f64;
        let mut want = [0.0f64; 4];
        for cy in 0..6 {
            for cx in 0..6 {
                let at = |x: usize, y: usize| surface[y * 7 + x];
                let (v00, v10, v01, v11) = (
                    at(cx, cy),
                    at(cx + 1, cy),
                    at(cx, cy + 1),
                    at(cx + 1, cy + 1),
                );
                for j in 0..n {
                    for i in 0..n {
                        let (u, w) = ((i as f64 + 0.5) * h, (j as f64 + 0.5) * h);
                        let z = v00 * (1.0 - u) * (1.0 - w)
                            + v10 * u * (1.0 - w)
                            + v01 * (1.0 - u) * w
                            + v11 * u * w;
                        if z >= level {
                            let (x, y) = (cx as f64 - 3.0 + u, cy as f64 - 3.0 + w);
                            want[0] += h * h;
                            want[1] += h * h * x * x;
                            want[2] += h * h * x * y;
                            want[3] += h * h * y * y;
                        }
                    }
                }
            }
        }
        let got = [got.area, got.sxx, got.sxy, got.syy];
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() < 2e-4 * (w.abs() + want[0]),
                "{got:?} against {want:?}"
            );
        }
    }
}

/// The quadrature across each cell has converged: on random cells, saddles
/// whose curve bends sharply included, the spans the reading uses, each no
/// longer than its distance from the pole, give the moments that spans an
/// eighth of that give, to `1e-10` of a cell's unit area.
#[test]
fn the_cell_quadrature_has_converged() {
    use super::ellipse::LocalMoments;
    let mut rng = Lcg(29);
    let mut crossed = 0;
    for _ in 0..2000 {
        let values = [0; 4].map(|_| f64::from(rng.next_f32()));
        let level = 0.2 + 0.6 * f64::from(rng.next_f32());
        let above = values.iter().filter(|&&v| v >= level).count();
        if above == 0 || above == 4 {
            continue;
        }
        crossed += 1;
        let used = LocalMoments::bilinear_part_graded(values, level, 1.0);
        let fine = LocalMoments::bilinear_part_graded(values, level, 8.0);
        for (a, b) in [
            (used.area, fine.area),
            (used.sp, fine.sp),
            (used.sq, fine.sq),
            (used.spp, fine.spp),
            (used.spq, fine.spq),
            (used.sqq, fine.sqq),
        ] {
            assert!((a - b).abs() < 1e-10, "{values:?} at {level}: {a} vs {b}");
        }
    }
    assert!(crossed > 1000);
}

/// A strip whose surface falls linearly from its middle line is clipped
/// exactly: its region is `|dx| ≤ 0.5` across the whole square.
#[test]
fn a_strip_integrates_exactly() {
    let surface = surface_of(3, |dx, _| 1.0 - 0.1 * dx.abs() as f64);
    let m = super::ellipse::region_moments(&surface, 3, 0.95);
    assert!((m.area - 6.0).abs() < 1e-12, "{m:?}");
    assert!((m.sxx - 6.0 * 2.0 * 0.125 / 3.0).abs() < 1e-12, "{m:?}");
    assert!(m.sxy.abs() < 1e-12, "{m:?}");
    assert!((m.syy - 2.0 * 27.0 / 3.0).abs() < 1e-12, "{m:?}");
}

/// Readings of the kinds the tests above build, and a few more, of templates
/// with tile around them and of whole bitmaps.
fn fixture_readings() -> Vec<SelfSimilarity> {
    let mut out = Vec::new();
    let corner = tile_of(
        18,
        1,
        |_, x, y| if x >= 9.0 && y >= 9.0 { 200.0 } else { 50.0 },
    );
    out.push(centred(&corner, 1, 12, 3));
    let blob = tile_of(18, 1, |_, x, y| {
        let (dx, dy) = (x - 8.5, y - 8.5);
        40.0 + 150.0 * (-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5)).exp()
    });
    out.push(centred(&blob, 1, 12, 3));
    for r in 1..=3u32 {
        for angle in [0.0f64, 30.0, 45.0, 90.0] {
            out.push(centred(&edge(16 + 2 * r as usize, angle), 1, 16, r));
        }
    }
    out.push(centred(&tile_of(18, 3, |_, _, _| 128.0), 3, 12, 3));
    let ramp = tile_of(18, 1, |_, x, y| (100.0 + 0.3 * x + 0.2 * y).round());
    out.push(centred(&ramp, 1, 12, 3));
    for seed in [3u64, 9, 21] {
        let mut rng = Lcg(seed);
        let (resolution, r) = (24usize, 3usize);
        let size = resolution + 2 * r;
        let raw: Vec<f32> = (0..3 * size * size)
            .map(|_| 255.0 * rng.next_f32())
            .collect();
        // A box blur, so some cells slide and some lock.
        let smooth: Vec<f32> = (0..3 * size * size)
            .map(|i| {
                let (c, p) = (i / (size * size), i % (size * size));
                let (x, y) = (p % size, p / size);
                let (mut sum, mut n) = (0.0f32, 0.0f32);
                for yy in y.saturating_sub(2)..(y + 3).min(size) {
                    for xx in x.saturating_sub(2)..(x + 3).min(size) {
                        sum += raw[c * size * size + yy * size + xx];
                        n += 1.0;
                    }
                }
                sum / n
            })
            .collect();
        let parts = zncc_self_similarity_parts(&tile(&smooth, 3, size), None, &params(3));
        out.push(parts.whole);
        out.push(parts.middle);
        out.extend(parts.grid.into_iter().flatten());
    }
    out
}

/// The radius is the ellipse's semi-major axis to the bit, capped at `r` with
/// the minor axis no longer than it, and its lower bound is the major axis's.
#[test]
fn the_radius_is_the_semi_major_axis() {
    let readings = fixture_readings();
    assert!(readings.len() > 30);
    for s in &readings {
        let e = &s.ellipse;
        assert_eq!(s.radius.to_bits(), e.axes[0].to_bits(), "{s:?}");
        assert_eq!(s.radius_is_at_least(), e.axes_is_at_least[0]);
        let r = ((s.surface.len() as f64).sqrt() as usize - 1) / 2;
        assert!(e.axes[0] <= r as f64 && e.axes[1] <= e.axes[0], "{e:?}");
        if s.radius >= r as f64 {
            assert!(s.radius_is_at_least(), "{e:?}");
        }
    }
}

/// A region that stays inside the square but whose semi-major axis passes `r`,
/// three lobes along the diagonal at `(−2, −2)`, `(0, 0)` and `(2, 2)`, reads
/// its major axis capped at `r` and flagged, and its matrix rebuilt from the
/// capped axes at the same angle.
#[test]
fn an_axis_past_r_is_capped_and_its_matrix_rebuilt() {
    let s = read_surface(
        surface_of(3, |dx, dy| match (dx, dy) {
            (0, 0) | (2, 2) | (-2, -2) => 1.0,
            _ => 0.5,
        }),
        3,
        0.05,
    );
    let e = s.ellipse;
    assert_eq!(e.axes[0], 3.0, "{e:?}");
    assert_eq!(s.radius, 3.0);
    assert!(e.axes[1] < 0.5, "{e:?}");
    assert!(e.axes_is_at_least[0], "{e:?}");
    assert!(
        axis_angle_gap(e.major_angle, std::f64::consts::FRAC_PI_4) < 1e-9,
        "{e:?}"
    );
    // E has the capped axes as its eigenvalues' roots, along the same angle.
    let m = e.matrix;
    let trace = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[0][1];
    assert!((trace - 9.0 - e.axes[1].powi(2)).abs() < 1e-9, "{m:?}");
    assert!((det - 9.0 * e.axes[1].powi(2)).abs() < 1e-9, "{m:?}");
    assert!(m[0][1] > 0.0, "{m:?}");
}

/// A region that runs off the square may go on past it: its major axis and the
/// radius are lower bounds. A ridge along `x` that holds its width over its
/// last two columns is taken to keep it past the border, so its minor axis,
/// across, reads exact; one that widens towards the border may go on widening
/// past it, so its minor axis is a lower bound too.
#[test]
fn running_off_the_square_is_a_lower_bound_on_the_major_axis() {
    let ridge = |widening: f64| {
        read_surface(
            surface_of(3, |dx, dy| {
                let (x, y) = (dx as f64, dy as f64);
                1.0 - 0.5 * y * y * (1.0 - widening * x) - 0.001 * x * x
            }),
            3,
            0.05,
        )
    };
    let even = ridge(0.0);
    assert_eq!(even.radius, 3.0);
    assert_eq!(
        even.ellipse.axes_is_at_least,
        [true, false],
        "{:?}",
        even.ellipse
    );
    assert!(axis_angle_gap(even.ellipse.major_angle, 0.0) < 1e-9);
    // The ridge's half-width, about 0.1, is spread evenly across it.
    let b = even.ellipse.axes[1];
    assert!(b > 0.1 && b < 0.12, "{:?}", even.ellipse);

    // The same ridge along `y` reads the same, turned to π/2.
    let along_y = read_surface(
        surface_of(3, |dx, dy| {
            let (x, y) = (dx as f64, dy as f64);
            1.0 - 0.5 * x * x - 0.001 * y * y
        }),
        3,
        0.05,
    );
    assert_eq!(along_y.ellipse.axes_is_at_least, [true, false]);
    assert!(
        axis_angle_gap(along_y.ellipse.major_angle, std::f64::consts::FRAC_PI_2) < 1e-9,
        "{:?}",
        along_y.ellipse
    );
    assert!((along_y.ellipse.axes[1] - b).abs() < 1e-12);

    let widening = ridge(0.1);
    assert_eq!(
        widening.ellipse.axes_is_at_least,
        [true, true],
        "{:?}",
        widening.ellipse
    );

    // One border shift at the level, with nothing at the level on the column
    // inside it: the region may go on past the border in any direction. The
    // two small lobes read a radius under r, which is only a lower bound.
    let lone = read_surface(
        surface_of(3, |dx, dy| match (dx, dy) {
            (0, 0) => 1.0,
            (3, 1) => 0.97,
            _ => 0.5,
        }),
        3,
        0.05,
    );
    assert!(lone.radius < 3.0, "{:?}", lone.ellipse);
    assert!(lone.radius_is_at_least());
    assert_eq!(lone.ellipse.axes_is_at_least, [true, true]);
    // Its major axis points at the lobe.
    let [x, y] = major_direction(&lone);
    assert!((y / x - 1.0 / 3.0).abs() < 0.05, "{:?}", lone.ellipse);
}

/// A ridge slanted across the square, `z = 1 − 0.5·(dy − 3·dx)²`, runs off the
/// square at `(1, 3)` towards `+y` with nothing at the level on the line inside
/// it, so it may turn either way past the border: both axes are lower bounds.
#[test]
fn a_slanted_ridge_running_off_the_square_is_a_lower_bound_on_both_axes() {
    let s = read_surface(
        surface_of(3, |dx, dy| {
            let off = (dy - 3 * dx) as f64;
            1.0 - 0.5 * off * off
        }),
        3,
        0.05,
    );
    assert_eq!(s.ellipse.axes_is_at_least, [true, true], "{:?}", s.ellipse);
}

/// A tile of uniform stripes under a vertical lighting ramp, read by the
/// kernels rather than given as a hand-built surface: the stripes vary along
/// `x` only and ZNCC ignores the ramp, so the patch matches itself at every
/// vertical shift and the region at the level is the column `dx = 0`, running
/// off the square at `±y`. Its width along `x` is the same on every line in
/// exact arithmetic, but the ramp makes each line's `f32` cross sums round
/// differently, and the crossings on the border line and the line inside it
/// differ by about 1e-7 grid px. The width test allows for that, so the minor
/// axis, across the column, reads exact, while the major axis is a lower bound.
#[test]
fn a_striped_tile_holds_its_width_across_the_run_off_through_f32_rounding() {
    let (core, r) = (24usize, 3u32);
    let size = core + 2 * r as usize;
    let tau = std::f64::consts::TAU;
    for channels in [1usize, 3] {
        let stripes = tile_of(size, channels, |c, x, y| {
            let phase = c as f64 * 0.7;
            let stripe = 50.0 * (tau * x / 7.3 + phase).sin() + 30.0 * (tau * x / 3.1 + 1.0).sin();
            100.0 + 1.5 * y + stripe
        });
        let reading = centred(&stripes, channels, core, r);
        let ri = r as i64;
        // Every vertical shift matches; every other one is under the level.
        let level = 1.0 - reading.tolerance;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let z = surface_at(&reading, ri, dx, dy);
                assert_eq!(z >= level, dx == 0, "({dx}, {dy}): {z} against {level}");
            }
        }
        let e = reading.ellipse;
        assert!(e.axes[1] > 0.0 && e.axes[1] < 1.0, "{e:?}");
        assert_eq!(
            e.axes_is_at_least,
            [true, false],
            "channels {channels}: {e:?}"
        );
        assert_eq!(e.axes[0], 3.0);
    }
}

/// A shift with no reading beside the region at the level may hide more of
/// it, in the cells around it. An axis is a lower bound where the gap's cells
/// reach further from the centre than half that axis, past which added area
/// lengthens it; a gap well inside a large region leaves both axes exact. A
/// gap that reaches the square's border may hide a region running off it, so
/// every axis is a lower bound. A gap diagonal to a shift at the level counts
/// as one beside it does.
#[test]
fn a_gap_beside_the_region_is_a_lower_bound_where_it_reaches_far_enough() {
    let disc_with_gap = |gap: (i64, i64)| {
        let mut surface = surface_of(8, |dx, dy| {
            let d2 = (dx * dx + dy * dy) as f64;
            1.0 - 0.05 * d2 / 36.0
        });
        surface[((gap.1 + 8) * 17 + gap.0 + 8) as usize] = f64::NAN;
        read_surface(surface, 8, 0.05)
    };
    // The gap at (0, 1): its cells reach √5 from the centre, under half the
    // disc's radius of about 6.
    let near = disc_with_gap((0, 1));
    assert!(near.radius > 5.5 && near.radius < 6.1, "{:?}", near.ellipse);
    assert_eq!(
        near.ellipse.axes_is_at_least,
        [false, false],
        "{:?}",
        near.ellipse
    );
    // The gap at (4, 0): its cells reach 5, past half the radius.
    let far = disc_with_gap((4, 0));
    assert!(far.radius_is_at_least(), "{:?}", far.ellipse);

    // A small region with a gap beside it.
    let small = read_surface(
        surface_of(3, |dx, dy| match (dx, dy) {
            (0, 0) => 1.0,
            (1, 0) => 0.97,
            (0, 1) => f64::NAN,
            _ => 0.5,
        }),
        3,
        0.05,
    );
    assert_eq!(
        small.ellipse.axes_is_at_least,
        [true, true],
        "{:?}",
        small.ellipse
    );

    // A gap no shift at the level touches is taken to be below the level.
    let apart = read_surface(
        surface_of(3, |dx, dy| match (dx, dy) {
            (0, 0) => 1.0,
            (3, 3) => f64::NAN,
            _ => 0.5,
        }),
        3,
        0.05,
    );
    assert_eq!(
        apart.ellipse.axes_is_at_least,
        [false, false],
        "{:?}",
        apart.ellipse
    );

    let through = read_surface(
        surface_of(3, |dx, dy| match (dx, dy) {
            (0, 0) => 1.0,
            (1, 0) => 0.97,
            (0, 1..=3) => f64::NAN,
            _ => 0.5,
        }),
        3,
        0.05,
    );
    assert_eq!(through.ellipse.axes_is_at_least, [true, true]);

    // A gap only diagonal to the one shift at the level still shares a cell
    // with it, which the moments leave out, so it counts as a gap.
    let diagonal = read_surface(
        surface_of(3, |dx, dy| match (dx, dy) {
            (0, 0) => 1.0,
            (1, 1) => f64::NAN,
            _ => 0.5,
        }),
        3,
        0.05,
    );
    assert_eq!(
        diagonal.ellipse.axes_is_at_least,
        [true, true],
        "{:?}",
        diagonal.ellipse
    );
}

/// A flat template reads a circle of radius `r`, at least along both axes,
/// with no direction; a reading with no data is `NaN` throughout.
#[test]
fn a_flat_template_and_no_data_read_their_ellipses() {
    let flat = centred(&tile_of(18, 1, |_, _, _| 128.0), 1, 12, 3);
    assert_eq!(flat.ellipse.axes, [3.0, 3.0]);
    assert_eq!(flat.ellipse.axes_is_at_least, [true, true]);
    assert!(flat.ellipse.major_angle.is_nan());
    assert_eq!(flat.ellipse.matrix, [[9.0, 0.0], [0.0, 9.0]]);
    assert!(flat.radius_is_at_least());

    let bitmap = tile_of(12, 1, |_, x, y| 10.0 * x + y);
    let data = vec![false; 144];
    let none = zncc_self_similarity_radius(
        &tile(&bitmap, 1, 12),
        Some(&data),
        [0, 0, 12, 12],
        &params(3),
    );
    assert!(none.radius.is_nan());
    assert!(none.ellipse.axes.iter().all(|v| v.is_nan()));
    assert!(none.ellipse.matrix.iter().flatten().all(|v| v.is_nan()));
    assert!(none.ellipse.major_angle.is_nan());
    assert_eq!(none.ellipse.axes_is_at_least, [false, false]);
    assert!(!none.radius_is_at_least());
    assert_eq!(none.ellipse.mapped([[1.0, 0.0], [0.0, 1.0]]), None);
    assert_eq!(
        SelfSimilarityEllipseUnits::read(&none, None, None, 24),
        None
    );
}

/// Through a linear map the ellipse is `L E Lᵀ`: a stretch along `x` doubles
/// an ellipse long along `x` and leaves one long along `y` as long, and a
/// rotation turns its angle.
#[test]
fn the_ellipse_maps_through_a_linear_map() {
    let s = paraboloid(16, 10.0, 4.0, 0.0);
    let e = s.ellipse;
    let stretched = e.mapped([[2.0, 0.0], [0.0, 1.0]]).expect("a map");
    assert!((stretched.axes[0] - 2.0 * e.axes[0]).abs() < 1e-9);
    assert!((stretched.axes[1] - e.axes[1]).abs() < 1e-9);
    let across = e.mapped([[1.0, 0.0], [0.0, 3.0]]).expect("a map");
    assert!(
        (across.axes[0] - 3.0 * e.axes[1]).abs() < 1e-9,
        "{across:?}"
    );
    assert!((across.axes[1] - e.axes[0]).abs() < 1e-9, "{across:?}");
    assert!(axis_angle_gap(across.major_angle, std::f64::consts::FRAC_PI_2) < 1e-9);
    let turn = 0.4f64;
    let (sin, cos) = turn.sin_cos();
    let rotated = e
        .mapped([[2.0 * cos, -2.0 * sin], [2.0 * sin, 2.0 * cos]])
        .expect("a map");
    assert!((rotated.axes[0] - 2.0 * e.axes[0]).abs() < 1e-9);
    assert!(axis_angle_gap(rotated.major_angle, e.major_angle + turn) < 1e-9);
    assert_eq!(e.mapped([[1.0, 2.0], [0.5, 1.0]]), None);
    assert_eq!(e.mapped([[f64::NAN, 0.0], [0.0, 1.0]]), None);
    // A finite, non-singular map whose image overflows gives none, rather
    // than an infinite axis and a minor axis of 0 read from `inf − inf`.
    assert_eq!(e.mapped([[1e200, 1e200], [1e-200, 0.0]]), None);
    assert_eq!(e.mapped([[1e200, 0.0], [0.0, 1e-200]]), None);
}

/// The lower bounds carry through a map. A ridge along `x` running off the
/// square, at least along its major axis and exact across, stays so under a
/// map that keeps `x` its major direction, and under a shear that does not, its
/// minor axis becomes a lower bound too, since lengthening the ridge along `x`
/// would lengthen it.
#[test]
fn the_lower_bounds_carry_through_a_map() {
    let s = read_surface(
        surface_of(3, |dx, dy| {
            let (x, y) = (dx as f64, dy as f64);
            1.0 - 0.5 * y * y - 0.001 * x * x
        }),
        3,
        0.05,
    );
    assert_eq!(s.ellipse.axes_is_at_least, [true, false]);
    let kept = s.ellipse.mapped([[2.0, 0.0], [0.0, 0.5]]).expect("a map");
    assert_eq!(kept.axes_is_at_least, [true, false], "{kept:?}");
    let sheared = s.ellipse.mapped([[1.0, 0.0], [0.5, 1.0]]).expect("a map");
    assert_eq!(sheared.axes_is_at_least, [true, true], "{sheared:?}");
    // The same in a unit a trillion times smaller: the test has no absolute
    // allowance that would pass a minor axis of 1e-13 as exact.
    for scale in [1e-12, 1e12] {
        let kept = s
            .ellipse
            .mapped([[2.0 * scale, 0.0], [0.0, 0.5 * scale]])
            .expect("a map");
        assert_eq!(kept.axes_is_at_least, [true, false], "{scale}: {kept:?}");
        let sheared = s
            .ellipse
            .mapped([[scale, 0.0], [0.5 * scale, scale]])
            .expect("a map");
        assert_eq!(
            sheared.axes_is_at_least,
            [true, true],
            "{scale}: {sheared:?}"
        );
    }
    // A minor axis at least makes both at least.
    let mut open = s.ellipse;
    open.axes_is_at_least = [false, true];
    let mapped = open.mapped([[2.0, 0.0], [0.0, 0.5]]).expect("a map");
    assert_eq!(mapped.axes_is_at_least, [true, true]);
}

/// Along the patch, one grid px along `x` is `2·half_extent[0]/R` along `u`
/// and one along `y` is `2·half_extent[1]/R` along `−v`; a patch at infinity
/// reads each semi-axis as the angle at the eye, in degrees.
#[test]
fn the_ellipse_on_the_patch_scales_by_each_half_extent() {
    let s = paraboloid(16, 10.0, 4.0, 0.0);
    let e = s.ellipse;
    let mut placement = crate::patch::cloud::OrientedPatch::new(
        nalgebra::Point3::origin(),
        nalgebra::Vector3::x(),
        nalgebra::Vector3::y(),
        [0.5, 2.0],
    );
    let Some(PatchEllipse::Length(on)) = e.on_patch(&placement, 24) else {
        panic!("a finite patch reads a length");
    };
    // 10 × 1/24 along u against 4 × 4/24 along v: the long axis is now v.
    assert!((on.axes[0] - 4.0 * e.axes[1] / 24.0).abs() < 1e-9, "{on:?}");
    assert!((on.axes[1] - e.axes[0] / 24.0).abs() < 1e-9, "{on:?}");
    assert!(axis_angle_gap(on.major_angle, std::f64::consts::FRAC_PI_2) < 1e-9);
    assert_eq!(e.on_patch(&placement, 0), None);

    // A turned ellipse turns the other way along v, which runs up the rows.
    let turned = paraboloid(16, 10.0, 4.0, 0.5).ellipse;
    placement.half_extent = [1.2, 1.2];
    let on = *turned
        .on_patch(&placement, 24)
        .expect("a placement")
        .ellipse();
    assert!(
        axis_angle_gap(on.major_angle, -turned.major_angle) < 1e-9,
        "{on:?}"
    );
    assert!((on.axes[0] - turned.axes[0] * 0.1).abs() < 1e-9);

    placement.w = 0.0;
    let Some(PatchEllipse::Angle(bearing)) = turned.on_patch(&placement, 24) else {
        panic!("a patch at infinity reads an angle");
    };
    let tangent = turned.axes[0] * 0.1;
    assert!((bearing.axes[0] - tangent.atan().to_degrees()).abs() < 1e-9);
    // On a small patch the angle is the offset in radians, to first order.
    placement.half_extent = [0.012, 0.012];
    let small = turned.on_patch(&placement, 24).expect("a placement");
    let tangent = turned.axes[0] * 0.001;
    assert!((small.ellipse().axes[0] - tangent.to_degrees()).abs() < 1e-4 * tangent.to_degrees());
}

/// The units carry what can be computed and nothing else.
#[test]
fn the_units_read_what_they_are_given() {
    let s = paraboloid(16, 10.0, 4.0, 0.0);
    let bare = SelfSimilarityEllipseUnits::read(&s, None, None, 24).expect("a reading");
    assert_eq!(bare.grid_px, s.ellipse);
    assert_eq!(bare.image_px, None);
    assert_eq!(bare.patch, None);
    let placement = crate::patch::cloud::OrientedPatch::new(
        nalgebra::Point3::origin(),
        nalgebra::Vector3::x(),
        nalgebra::Vector3::y(),
        [1.2, 1.2],
    );
    let jacobian = [[2.0, 0.0], [0.0, 1.0]];
    let full = SelfSimilarityEllipseUnits::read(&s, Some(jacobian), Some(&placement), 24)
        .expect("a reading");
    assert_eq!(full.image_px, s.ellipse.mapped(jacobian));
    assert!(matches!(full.patch, Some(PatchEllipse::Length(_))));
}
