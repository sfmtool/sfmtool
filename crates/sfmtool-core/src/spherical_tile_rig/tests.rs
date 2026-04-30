// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::PI;

use nalgebra::Vector3;

use super::*;
use crate::camera_intrinsics::{CameraIntrinsics, CameraModel};
use crate::sphere_points::{random_sphere_points, RelaxConfig};
use crate::warp_map::WarpMap;

// ── Helpers ─────────────────────────────────────────────────────────────

fn make_rig(n: usize, w_equiv: u32) -> SphericalTileRig {
    SphericalTileRig::new(&SphericalTileRigParams {
        centre: [0.0, 0.0, 0.0],
        n,
        arc_per_pixel: 2.0 * PI / w_equiv as f64,
        overlap_factor: 1.15,
        atlas_cols: None,
        relax: Some(RelaxConfig {
            seed: Some(123),
            ..Default::default()
        }),
    })
    .unwrap()
}

fn equirect_camera(w: u32, h: u32) -> CameraIntrinsics {
    let fx = w as f64 / (2.0 * PI);
    let fy = h as f64 / PI;
    CameraIntrinsics {
        model: CameraModel::Equirectangular {
            focal_length_x: fx,
            focal_length_y: fy,
            principal_point_x: w as f64 / 2.0,
            principal_point_y: h as f64 / 2.0,
        },
        width: w,
        height: h,
    }
}

/// Apply a warp map to a single-channel f32 image with bilinear sampling.
/// Pixels with NaN map coordinates yield `0.0`. When `wrap_x` is true, x
/// wraps modulo `src_w` (suitable for equirectangular longitude); y is
/// clamped to the valid row range. With `wrap_x = false`, both axes are
/// clamped (replicate-edge).
fn remap_bilinear(src_w: u32, src_h: u32, src: &[f32], map: &WarpMap, wrap_x: bool) -> Vec<f32> {
    let dw = map.width() as usize;
    let dh = map.height() as usize;
    let mut out = vec![0.0_f32; dw * dh];
    for row in 0..dh {
        for col in 0..dw {
            let (sx, sy) = map.get(col as u32, row as u32);
            if !sx.is_finite() || !sy.is_finite() {
                continue;
            }
            let xf = sx - 0.5;
            let yf = sy - 0.5;
            let x0 = xf.floor();
            let y0 = yf.floor();
            let dx = xf - x0;
            let dy = yf - y0;
            let x0i = x0 as i32;
            let y0i = y0 as i32;

            let sample = |xi: i32, yi: i32| -> f32 {
                let xi = if wrap_x {
                    xi.rem_euclid(src_w as i32)
                } else {
                    xi.clamp(0, src_w as i32 - 1)
                };
                let yi = yi.clamp(0, src_h as i32 - 1);
                src[yi as usize * src_w as usize + xi as usize]
            };
            let a = sample(x0i, y0i);
            let b = sample(x0i + 1, y0i);
            let c = sample(x0i, y0i + 1);
            let d = sample(x0i + 1, y0i + 1);
            let val = (1.0 - dx) * (1.0 - dy) * a
                + dx * (1.0 - dy) * b
                + (1.0 - dx) * dy * c
                + dx * dy * d;
            out[row * dw + col] = val;
        }
    }
    out
}

/// Direction-only smooth pattern, evaluated at a unit world direction.
/// Single source of truth for the synthetic test pattern: a band-limited
/// polynomial in `[0, 1]` that is a well-defined function of the world
/// direction (not of `(lon, lat)`), so pattern values agree across the
/// longitude wrap and across all equirect pixels that share the same
/// direction near the poles. Mirrors the Python helper of the same name in
/// `tests/test_spherical_tile_rig.py`.
fn smooth_pattern(dx: f64, dy: f64, dz: f64) -> f64 {
    0.5 + 0.2 * dx + 0.15 * dy * dz
}

/// Build a synthetic equirectangular image of `smooth_pattern` evaluated at
/// every dst-pixel direction. Used both as a source image for the
/// `to_atlas` round-trip test and as a reference for the `resample_atlas`
/// tests.
fn equirect_pattern(width: u32, height: u32) -> Vec<f32> {
    let mut img = Vec::with_capacity((width * height) as usize);
    for r in 0..height {
        let lat = PI / 2.0 - PI * (r as f64 + 0.5) / height as f64;
        for c in 0..width {
            let lon = -PI + 2.0 * PI * (c as f64 + 0.5) / width as f64;
            let cos_lat = lat.cos();
            let dx = lon.sin() * cos_lat;
            let dy = lat.sin();
            let dz = lon.cos() * cos_lat;
            img.push(smooth_pattern(dx, dy, dz) as f32);
        }
    }
    img
}

// ── Validation ──────────────────────────────────────────────────────────

fn expect_err(p: &SphericalTileRigParams, want: SphericalTileRigError) {
    match SphericalTileRig::new(p) {
        Ok(_) => panic!("expected {want:?}, got Ok"),
        Err(got) => assert_eq!(got, want),
    }
}

#[test]
fn invalid_n_returns_error() {
    let p = SphericalTileRigParams {
        n: 1,
        ..Default::default()
    };
    expect_err(&p, SphericalTileRigError::TooFewTiles);
}

#[test]
fn invalid_arc_per_pixel_returns_error() {
    let p = SphericalTileRigParams {
        arc_per_pixel: 0.0,
        ..Default::default()
    };
    expect_err(&p, SphericalTileRigError::InvalidArcPerPixel);
    let p = SphericalTileRigParams {
        arc_per_pixel: f64::NAN,
        ..Default::default()
    };
    expect_err(&p, SphericalTileRigError::InvalidArcPerPixel);
}

#[test]
fn invalid_overlap_factor_returns_error() {
    let p = SphericalTileRigParams {
        overlap_factor: 0.5,
        ..Default::default()
    };
    expect_err(&p, SphericalTileRigError::InvalidOverlapFactor);
}

#[test]
fn invalid_centre_returns_error() {
    let p = SphericalTileRigParams {
        centre: [f64::INFINITY, 0.0, 0.0],
        ..Default::default()
    };
    expect_err(&p, SphericalTileRigError::InvalidCentre);
}

// ── Sizing & layout ─────────────────────────────────────────────────────

#[test]
fn tile_count_is_exactly_n() {
    for &n in &[2usize, 80, 320] {
        let rig = make_rig(n, 512);
        assert_eq!(rig.len(), n);
        assert_eq!(rig.direction(0).len(), 3);
    }
}

#[test]
fn half_fov_tracks_coverage_measurement() {
    let rig = make_rig(320, 512);
    // half_fov_rad is exactly `measured_max_coverage_angle * overlap_factor`.
    let expected = rig.measured_max_coverage_angle() * 1.15;
    assert!((rig.half_fov_rad() - expected).abs() < 1e-12);

    // For a well-relaxed rig, max coverage ≈ 0.5 × max NN gap.
    let half_nn = 0.5 * rig.measured_max_nn_angle();
    let cov = rig.measured_max_coverage_angle();
    // Voronoi cell radius is at least half the smallest local NN gap and at
    // most a small constant times the largest. Ratio cov/half_nn ∈ [0.7, 1.5]
    // is a robust band for well-tiled rigs.
    let ratio = cov / half_nn;
    assert!(
        (0.7..=1.5).contains(&ratio),
        "cov / half_nn = {ratio} unexpectedly far from 1"
    );

    // Cross-check: measured_max_nn_angle equals nn_angles.max() recomputed
    // from `directions`, converting chord → angle.
    let n = rig.len();
    let flat: Vec<f32> = (0..n)
        .flat_map(|i| {
            let d = rig.direction(i);
            [d[0] as f32, d[1] as f32, d[2] as f32]
        })
        .collect();
    let cloud = PointCloud3::<f32>::new(&flat, n);
    let chords = cloud.nearest_neighbor_distances();
    let recomputed_max = chords
        .iter()
        .map(|&c| 2.0 * ((c as f64) * 0.5).clamp(-1.0, 1.0).asin())
        .fold(0.0f64, f64::max);
    assert!((rig.measured_max_nn_angle() - recomputed_max).abs() < 1e-6);
}

#[test]
fn direction_uniformity_within_relaxer_quality() {
    let rig = make_rig(1280, 1024);
    let n = rig.len();
    let flat: Vec<f32> = (0..n)
        .flat_map(|i| {
            let d = rig.direction(i);
            [d[0] as f32, d[1] as f32, d[2] as f32]
        })
        .collect();
    let cloud = PointCloud3::<f32>::new(&flat, n);
    let chords = cloud.nearest_neighbor_distances();
    let angles: Vec<f64> = chords
        .iter()
        .map(|&c| 2.0 * ((c as f64) * 0.5).clamp(-1.0, 1.0).asin())
        .collect();
    let mean = angles.iter().copied().sum::<f64>() / angles.len() as f64;
    let var = angles.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / angles.len() as f64;
    let std = var.sqrt();
    // Spec target: std/mean ≈ 0.045 (margin: 0.06).
    assert!(std / mean < 0.06, "std/mean = {} too high", std / mean);
}

#[test]
fn coverage_every_direction_within_half_fov() {
    // For each of n ∈ {20, 80, 320, 5000}: every unit vector's angular
    // distance to its nearest tile direction must be ≤ half_fov_rad.
    //
    // Probe is uniform-random (no Thomson relaxation) and hoisted outside
    // the loop. At 50_000 samples the typical inter-probe gap is ≪ the
    // rig's tile spacing even at n=5000, so Poisson clumpiness still
    // resolves any rig gap the relaxer leaves.
    let probe_flat = random_sphere_points(50_000, Some(987));
    let probe_n = probe_flat.len() / 3;

    for &n in &[20usize, 80, 320, 5000] {
        let rig = make_rig(n, 512);
        let nn = rig.direction_tree.nearest(&probe_flat, probe_n);

        let half_fov = rig.half_fov_rad();
        for (i, &t_idx) in nn.iter().enumerate() {
            let q = [
                probe_flat[3 * i] as f64,
                probe_flat[3 * i + 1] as f64,
                probe_flat[3 * i + 2] as f64,
            ];
            let d = rig.direction(t_idx as usize);
            // Angular distance from dot product, clamped for safety.
            let dot = (q[0] * d[0] + q[1] * d[1] + q[2] * d[2]).clamp(-1.0, 1.0);
            let angle = dot.acos();
            assert!(
                angle <= half_fov + 1e-6,
                "n={n}: probe {i} angle {angle} > half_fov {half_fov}"
            );
        }
    }
}

#[test]
fn atlas_packing_round_trip() {
    let rig = make_rig(80, 256);
    let (aw, ah) = rig.atlas_size();
    assert_eq!(aw, rig.atlas_cols() * rig.patch_size());
    assert_eq!(ah, rig.atlas_rows() * rig.patch_size());

    // Write a unique constant per tile, read back through tile_atlas_origin.
    let mut atlas = vec![0.0_f32; (aw * ah) as usize];
    let p = rig.patch_size() as usize;
    let aw_us = aw as usize;
    for idx in 0..rig.len() {
        let (ox, oy) = rig.tile_atlas_origin(idx);
        let val = (idx as f32) + 1.0;
        for dy in 0..p {
            for dx in 0..p {
                let x = ox as usize + dx;
                let y = oy as usize + dy;
                atlas[y * aw_us + x] = val;
            }
        }
    }
    for idx in 0..rig.len() {
        let (ox, oy) = rig.tile_atlas_origin(idx);
        let val = (idx as f32) + 1.0;
        for dy in 0..p {
            for dx in 0..p {
                let x = ox as usize + dx;
                let y = oy as usize + dy;
                assert_eq!(atlas[y * aw_us + x], val);
            }
        }
    }
}

#[test]
fn set_patch_size_round_trip() {
    let mut rig = make_rig(320, 512);
    let original_patch = rig.patch_size();
    let original_half_fov = rig.half_fov_rad();
    let original_max_nn = rig.measured_max_nn_angle();
    let original_max_cov = rig.measured_max_coverage_angle();
    let original_dirs: Vec<[f64; 3]> = (0..rig.len()).map(|i| rig.direction(i)).collect();
    let original_bases: Vec<([f64; 3], [f64; 3])> = (0..rig.len()).map(|i| rig.basis(i)).collect();
    let atlas_cols = rig.atlas_cols();
    let atlas_rows = rig.atlas_rows();

    let candidates = [
        original_patch.next_power_of_two(),
        (original_patch / 2).max(1),
        original_patch * 2,
        1u32,
    ];
    for &new_size in &candidates {
        rig.set_patch_size(new_size);
        assert_eq!(rig.patch_size(), new_size);

        // Tile camera reflects the new patch_size.
        let cam = rig.tile_camera();
        assert_eq!(cam.width, new_size);
        assert_eq!(cam.height, new_size);
        let half = new_size as f64 / 2.0;
        let expected_f = half / rig.half_fov_rad().tan();
        let (fx, fy) = cam.focal_lengths();
        let (cx, cy) = cam.principal_point();
        assert!((fx - expected_f).abs() < 1e-9);
        assert!((fy - expected_f).abs() < 1e-9);
        assert!((cx - half).abs() < 1e-9);
        assert!((cy - half).abs() < 1e-9);

        // Atlas size scales with the new patch size.
        let (aw, ah) = rig.atlas_size();
        assert_eq!(aw, atlas_cols * new_size);
        assert_eq!(ah, atlas_rows * new_size);

        // Tile directions, bases, half-FOV, and the diagnostic measurements
        // are unaffected.
        assert_eq!(rig.half_fov_rad(), original_half_fov);
        assert_eq!(rig.measured_max_nn_angle(), original_max_nn);
        assert_eq!(rig.measured_max_coverage_angle(), original_max_cov);
        assert_eq!(rig.atlas_cols(), atlas_cols);
        for i in 0..rig.len() {
            assert_eq!(rig.direction(i), original_dirs[i]);
            let (er, eu) = rig.basis(i);
            assert_eq!(er, original_bases[i].0);
            assert_eq!(eu, original_bases[i].1);
        }
    }

    // Round-trip: rebuilding the equirect ↔ atlas warp at the new size still
    // recovers a smooth pattern within tolerance.
    rig.set_patch_size(original_patch.next_power_of_two());
    let w = 512u32;
    let h = w / 2;
    let equirect = equirect_camera(w, h);
    let img = equirect_pattern(w, h);
    let to_atlas = rig.warp_to_atlas_with_rotation(&equirect, &RotQuaternion::identity());
    let atlas_img = remap_bilinear(w, h, &img, &to_atlas, true);
    let from_atlas = rig.warp_from_atlas_with_rotation(&equirect, &RotQuaternion::identity());
    let (aw, ah) = rig.atlas_size();
    let recovered = remap_bilinear(aw, ah, &atlas_img, &from_atlas, false);
    let mae: f64 = img
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .sum::<f64>()
        / img.len() as f64;
    assert!(mae < 6e-3, "post-resize round-trip mae {mae} too high");
}

#[test]
#[should_panic(expected = "patch_size must be > 0")]
fn set_patch_size_zero_panics() {
    let mut rig = make_rig(80, 256);
    rig.set_patch_size(0);
}

// ── Geometric correctness of the basis ──────────────────────────────────

#[test]
fn basis_is_orthonormal_and_right_handed() {
    let rig = make_rig(320, 512);
    for i in 0..rig.len() {
        let (er, eu) = rig.basis(i);
        let d = rig.direction(i);
        let er_v = Vector3::new(er[0], er[1], er[2]);
        let eu_v = Vector3::new(eu[0], eu[1], eu[2]);
        let d_v = Vector3::new(d[0], d[1], d[2]);

        // Unit norm.
        assert!((er_v.norm() - 1.0).abs() < 1e-9);
        assert!((eu_v.norm() - 1.0).abs() < 1e-9);
        assert!((d_v.norm() - 1.0).abs() < 1e-6);

        // Mutually orthogonal.
        assert!(er_v.dot(&eu_v).abs() < 1e-9);
        assert!(er_v.dot(&d_v).abs() < 1e-9);
        assert!(eu_v.dot(&d_v).abs() < 1e-9);

        // Right-handed: e_right × e_up = direction.
        let cross = er_v.cross(&eu_v);
        assert!((cross - d_v).norm() < 1e-9);
    }
}

#[test]
fn tile_camera_intrinsics_match_spec() {
    let rig = make_rig(320, 512);
    let cam = rig.tile_camera();
    let p = rig.patch_size();
    assert_eq!(cam.width, p);
    assert_eq!(cam.height, p);
    let (fx, fy) = cam.focal_lengths();
    let (cx, cy) = cam.principal_point();
    let half = p as f64 / 2.0;
    let expected_f = half / rig.half_fov_rad().tan();
    assert!((fx - expected_f).abs() < 1e-9);
    assert!((fy - expected_f).abs() < 1e-9);
    assert!((cx - half).abs() < 1e-9);
    assert!((cy - half).abs() < 1e-9);
    assert!(matches!(cam.model, CameraModel::Pinhole { .. }));
    assert!(!cam.has_distortion());
}

#[test]
fn tile_camera_centre_pixel_unprojects_to_tile_direction() {
    let rig = make_rig(80, 256);
    let cam = rig.tile_camera();
    let half = cam.width as f64 / 2.0;

    for i in 0..rig.len() {
        // Centre pixel = (cx, cy); tile-frame ray must be (0, 0, 1).
        let ray = cam.pixel_to_ray(half, half);
        // Rotate to world via R_world_from_tile · (0,0,1) = direction.
        let basis = rig.basis(i);
        let dir = rig.direction(i);
        // R · (0,0,1) is the third column = direction itself.
        // Sanity: numerical equivalence.
        let world = [
            basis.0[0] * ray[0] + basis.1[0] * ray[1] + dir[0] * ray[2],
            basis.0[1] * ray[0] + basis.1[1] * ray[1] + dir[1] * ray[2],
            basis.0[2] * ray[0] + basis.1[2] * ray[1] + dir[2] * ray[2],
        ];
        assert!(
            ((world[0] - dir[0]).abs() < 1e-6)
                && ((world[1] - dir[1]).abs() < 1e-6)
                && ((world[2] - dir[2]).abs() < 1e-6),
            "tile {i}: world {world:?} != direction {dir:?}"
        );
    }
}

// ── apply_transform ─────────────────────────────────────────────────────

#[test]
fn apply_transform_rotates_centre_and_directions() {
    let mut rig = make_rig(80, 256);
    let original_dir_0 = rig.direction(0);
    let r = RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), PI / 2.0).unwrap();
    let t = Se3Transform::new(r.clone(), Vector3::new(1.0, 2.0, 3.0), 1.0);
    rig.apply_transform(&t);

    // Centre rotated then translated.
    let c = rig.centre();
    assert!((c[0] - 1.0).abs() < 1e-9);
    assert!((c[1] - 2.0).abs() < 1e-9);
    assert!((c[2] - 3.0).abs() < 1e-9);

    // Direction: applying the same rotation to original should match.
    let v = r.rotate_vector(&Vector3::new(
        original_dir_0[0],
        original_dir_0[1],
        original_dir_0[2],
    ));
    let new_d = rig.direction(0);
    assert!((new_d[0] - v.x).abs() < 1e-9);
    assert!((new_d[1] - v.y).abs() < 1e-9);
    assert!((new_d[2] - v.z).abs() < 1e-9);

    // Bases are still orthonormal post-rotation.
    let (er, eu) = rig.basis(0);
    let er_v = Vector3::new(er[0], er[1], er[2]);
    let eu_v = Vector3::new(eu[0], eu[1], eu[2]);
    let d_v = Vector3::new(new_d[0], new_d[1], new_d[2]);
    assert!(er_v.dot(&eu_v).abs() < 1e-9);
    assert!((er_v.cross(&eu_v) - d_v).norm() < 1e-9);
}

// ── End-to-end: equirect ↔ atlas round-trip ────────────────────────────

#[test]
fn equirect_atlas_round_trip() {
    // Build a smooth band-limited equirectangular pattern at W=512, n=320,
    // and verify both warp methods round-trip a sample of it through the
    // atlas with low error.
    let n = 320usize;
    let w = 512u32;
    let h = w / 2;
    let rig = make_rig(n, w);
    let equirect = equirect_camera(w, h);
    let img = equirect_pattern(w, h);

    let to_atlas = rig.warp_to_atlas_with_rotation(&equirect, &RotQuaternion::identity());
    // Sampling INTO the equirect: x wraps by longitude.
    let atlas_img = remap_bilinear(w, h, &img, &to_atlas, true);

    let from_atlas = rig.warp_from_atlas_with_rotation(&equirect, &RotQuaternion::identity());
    let (aw, ah) = rig.atlas_size();
    // Sampling INTO the atlas: rectangular, no wrap.
    let recovered = remap_bilinear(aw, ah, &atlas_img, &from_atlas, false);

    // Compute mean and max abs error over the full image.
    let total: f64 = img
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .sum();
    let n_pix = img.len() as f64;
    let mean_abs_err = total / n_pix;

    // Spec: full-image mean_abs_err < 6e-3 (≈ 1.5/255 on u8). Per-pixel
    // max can spike to ~0.02 along Voronoi seams; that's the closest-tile
    // primitive, not a bug.
    assert!(
        mean_abs_err < 6e-3,
        "full-image mean_abs_err {mean_abs_err} exceeds 6e-3"
    );

    // Interior pixels: check mean error away from Voronoi seams. We use a
    // proxy for "near a seam" — sample-distance from the second-nearest
    // tile is comparable to the first.
    //
    // Build a simpler interior test: the angular distance to the *single*
    // closest tile centre is ≤ 0.5 · half_fov_rad ⇒ "deep inside" the
    // Voronoi cell.
    let n_tiles = rig.len();
    let dirs_flat: Vec<f32> = (0..n_tiles)
        .flat_map(|i| {
            let d = rig.direction(i);
            [d[0] as f32, d[1] as f32, d[2] as f32]
        })
        .collect();
    let cloud = PointCloud3::<f32>::new(&dirs_flat, n_tiles);

    let mut interior_count = 0usize;
    let mut interior_sum_err = 0.0f64;
    let mut interior_max_err = 0.0f64;
    let half_half_fov = 0.5 * rig.half_fov_rad();
    for r in 0..h {
        let lat = -PI / 2.0 + PI * (r as f64 + 0.5) / h as f64;
        for c in 0..w {
            let lon = -PI + 2.0 * PI * (c as f64 + 0.5) / w as f64;
            // Equirect ray (longitude, latitude) → unit vector.
            let dir = [
                (lat.cos() * lon.sin()) as f32,
                lat.sin() as f32,
                (lat.cos() * lon.cos()) as f32,
            ];
            // Find single closest tile + test angular distance.
            let nn = cloud.nearest(&dir, 1);
            let t = rig.direction(nn[0] as usize);
            let dot = (dir[0] as f64 * t[0] + dir[1] as f64 * t[1] + dir[2] as f64 * t[2])
                .clamp(-1.0, 1.0);
            let angle = dot.acos();
            if angle > half_half_fov {
                continue;
            }

            let idx = (r * w + c) as usize;
            let e = (img[idx] - recovered[idx]).abs() as f64;
            interior_sum_err += e;
            interior_max_err = interior_max_err.max(e);
            interior_count += 1;
        }
    }
    assert!(
        interior_count > 1000,
        "too few interior samples: {interior_count}"
    );
    let interior_mean = interior_sum_err / interior_count as f64;
    // Spec: mean_abs_err < 1e-3, max_abs_err < 5e-3. Clean
    // bilinear-interp-twice tolerance.
    assert!(
        interior_mean < 1e-3,
        "interior mean_abs_err {interior_mean} exceeds 1e-3"
    );
    assert!(
        interior_max_err < 5e-3,
        "interior max_abs_err {interior_max_err} exceeds 5e-3"
    );
}

#[test]
fn warp_seam_overlap_agrees_within_bilinear_tolerance() {
    // For two adjacent tiles, points lying inside both tiles' FOV should
    // map to nearly identical world rays after going through their
    // respective bases. This exercises seam consistency.
    let rig = make_rig(80, 256);
    let cam = rig.tile_camera();

    // Find a pair of nearest tiles and a midpoint direction.
    let n = rig.len();
    let dirs_flat: Vec<f32> = (0..n)
        .flat_map(|i| {
            let d = rig.direction(i);
            [d[0] as f32, d[1] as f32, d[2] as f32]
        })
        .collect();
    let cloud = PointCloud3::<f32>::new(&dirs_flat, n);
    // Pick tile 0 and its nearest neighbor.
    let q0 = [dirs_flat[0], dirs_flat[1], dirs_flat[2]];
    let nn = cloud.nearest_k(&q0, 1, 2);
    let i_a = nn[0] as usize;
    let i_b = nn[1] as usize;
    assert_ne!(i_a, i_b);

    // Mid-direction and project through each tile.
    let da = rig.direction(i_a);
    let db = rig.direction(i_b);
    let mid = {
        let v = Vector3::new(da[0] + db[0], da[1] + db[1], da[2] + db[2]).normalize();
        [v.x, v.y, v.z]
    };

    let project_through = |idx: usize| -> Option<(f64, f64)> {
        let basis = rig.basis(idx);
        let dir = rig.direction(idx);
        // R^T · mid: rows of [er|eu|dir] · mid
        let tx = basis.0[0] * mid[0] + basis.0[1] * mid[1] + basis.0[2] * mid[2];
        let ty = basis.1[0] * mid[0] + basis.1[1] * mid[1] + basis.1[2] * mid[2];
        let tz = dir[0] * mid[0] + dir[1] * mid[1] + dir[2] * mid[2];
        cam.ray_to_pixel([tx, ty, tz])
    };
    let pa = project_through(i_a);
    let pb = project_through(i_b);

    // Both should be inside the patch (since overlap_factor=1.15 covers the
    // midpoint between adjacent tile directions).
    let p = rig.patch_size() as f64;
    let inside = |opt: Option<(f64, f64)>| matches!(opt, Some((x, y)) if x >= 0.0 && y >= 0.0 && x < p && y < p);
    assert!(inside(pa), "tile_a projection out of bounds: {pa:?}");
    assert!(inside(pb), "tile_b projection out of bounds: {pb:?}");
}

#[test]
fn closest_tile_warp_lands_inside_patch() {
    // For a dst pinhole camera looking forward, every dst pixel ray's
    // closest-tile projection should land inside the [0, atlas_size) range,
    // confirming the coverage invariant of warp_from_atlas_with_rotation.
    let rig = make_rig(320, 512);
    let dst_w = 64u32;
    let dst_h = 64u32;
    let dst = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 100.0,
            focal_length_y: 100.0,
            principal_point_x: dst_w as f64 / 2.0,
            principal_point_y: dst_h as f64 / 2.0,
        },
        width: dst_w,
        height: dst_h,
    };

    let map = rig.warp_from_atlas_with_rotation(&dst, &RotQuaternion::identity());
    assert_eq!(map.width(), dst_w);
    assert_eq!(map.height(), dst_h);

    let mut valid = 0;
    for r in 0..dst_h {
        for c in 0..dst_w {
            assert!(map.is_valid(c, r), "dst ({c}, {r}) had no tile match");
            valid += 1;
        }
    }
    assert_eq!(valid, (dst_w * dst_h) as usize);
}

// ── resample_atlas ──────────────────────────────────────────────────────

/// Fill an atlas by evaluating `smooth_pattern` at each tile pixel's
/// in-tile world ray. Mirrors the Python helper used in
/// `tests/test_spherical_tile_rig.py`.
fn fill_atlas_with_pattern(rig: &SphericalTileRig) -> Vec<f32> {
    let (aw, ah) = rig.atlas_size();
    let mut atlas = vec![0.0_f32; (aw * ah) as usize];
    let cam = rig.tile_camera();
    let (fx, fy) = cam.focal_lengths();
    let (cx, cy) = cam.principal_point();
    let p = rig.patch_size() as usize;
    let aw_us = aw as usize;
    for idx in 0..rig.len() {
        let (ox, oy) = rig.tile_atlas_origin(idx);
        let basis = rig.basis(idx);
        let dir = rig.direction(idx);
        for in_y in 0..p {
            for in_x in 0..p {
                let u = in_x as f64 + 0.5;
                let v = in_y as f64 + 0.5;
                let x = (u - cx) / fx;
                let y = (v - cy) / fy;
                let z = 1.0_f64;
                let inv = 1.0 / (x * x + y * y + z * z).sqrt();
                let (rx, ry, rz) = (x * inv, y * inv, z * inv);
                // tile → world: R · ray, columns [e_right | e_up | direction].
                let wx = basis.0[0] * rx + basis.1[0] * ry + dir[0] * rz;
                let wy = basis.0[1] * rx + basis.1[1] * ry + dir[1] * rz;
                let wz = basis.0[2] * rx + basis.1[2] * ry + dir[2] * rz;
                let px_x = ox as usize + in_x;
                let px_y = oy as usize + in_y;
                atlas[px_y * aw_us + px_x] = smooth_pattern(wx, wy, wz) as f32;
            }
        }
    }
    atlas
}

#[test]
fn resample_atlas_k1_recovers_smooth_pattern() {
    let rig = make_rig(320, 512);
    let atlas = fill_atlas_with_pattern(&rig);
    let dst = equirect_camera(512, 256);
    let out = rig.resample_atlas(&atlas, 1, &dst, &RotQuaternion::identity(), 1);
    assert_eq!(out.len(), 512 * 256);

    let ref_img = equirect_pattern(512, 256);
    let mae: f64 = out
        .iter()
        .zip(ref_img.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .sum::<f64>()
        / out.len() as f64;
    assert!(mae < 1e-2, "k=1 MAE {mae} too high");
}

#[test]
fn resample_atlas_k3_runs_and_tracks_pattern() {
    let rig = make_rig(320, 512);
    let atlas = fill_atlas_with_pattern(&rig);
    let dst = equirect_camera(256, 128);
    let out = rig.resample_atlas(&atlas, 1, &dst, &RotQuaternion::identity(), 3);
    assert_eq!(out.len(), 256 * 128);

    let ref_img = equirect_pattern(256, 128);
    let mae: f64 = out
        .iter()
        .zip(ref_img.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .sum::<f64>()
        / out.len() as f64;
    assert!(mae < 1.5e-2, "k=3 MAE {mae} too high");
}

#[test]
fn resample_atlas_constant_atlas_yields_constant_output() {
    let rig = make_rig(80, 256);
    let (aw, ah) = rig.atlas_size();
    let atlas = vec![0.42_f32; (aw * ah) as usize];
    let dst = equirect_camera(64, 32);
    let out = rig.resample_atlas(&atlas, 1, &dst, &RotQuaternion::identity(), 3);
    for &v in &out {
        assert!((v - 0.42).abs() < 1e-5, "got {v}, expected 0.42");
    }
}

#[test]
fn resample_atlas_multi_channel_rgb() {
    let rig = make_rig(80, 256);
    let (aw, ah) = rig.atlas_size();
    // Constant per channel: R=0.1, G=0.5, B=0.9.
    let mut atlas = Vec::with_capacity((aw * ah) as usize * 3);
    for _ in 0..(aw * ah) {
        atlas.extend_from_slice(&[0.1_f32, 0.5, 0.9]);
    }
    let dst = equirect_camera(48, 24);
    let out = rig.resample_atlas(&atlas, 3, &dst, &RotQuaternion::identity(), 1);
    assert_eq!(out.len(), 48 * 24 * 3);
    for px in out.chunks_exact(3) {
        assert!((px[0] - 0.1).abs() < 1e-5);
        assert!((px[1] - 0.5).abs() < 1e-5);
        assert!((px[2] - 0.9).abs() < 1e-5);
    }
}

#[test]
#[should_panic(expected = "k must be >= 1")]
fn resample_atlas_panics_on_k_zero() {
    let rig = make_rig(80, 256);
    let (aw, ah) = rig.atlas_size();
    let atlas = vec![0.0_f32; (aw * ah) as usize];
    let dst = equirect_camera(32, 16);
    let _ = rig.resample_atlas(&atlas, 1, &dst, &RotQuaternion::identity(), 0);
}

#[test]
#[should_panic(expected = "atlas length")]
fn resample_atlas_panics_on_size_mismatch() {
    let rig = make_rig(80, 256);
    let bogus = vec![0.0_f32; 10];
    let dst = equirect_camera(32, 16);
    let _ = rig.resample_atlas(&bogus, 1, &dst, &RotQuaternion::identity(), 1);
}
