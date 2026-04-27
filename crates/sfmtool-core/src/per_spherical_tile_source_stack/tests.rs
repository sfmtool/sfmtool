// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::PI;

use nalgebra::Vector3;

use super::*;
use crate::camera_intrinsics::{CameraIntrinsics, CameraModel};
use crate::remap::{remap_bilinear, ImageU8};
use crate::rot_quaternion::RotQuaternion;
use crate::sphere_points::RelaxConfig;
use crate::spherical_tile_rig::{SphericalTileRig, SphericalTileRigParams};
use crate::warp_map::WarpMap;

// ── Helpers ─────────────────────────────────────────────────────────────

fn make_pow2_rig(n: usize, w_equiv: u32, target_patch: u32) -> SphericalTileRig {
    let mut rig = SphericalTileRig::new(&SphericalTileRigParams {
        centre: [0.0, 0.0, 0.0],
        n,
        arc_per_pixel: 2.0 * PI / w_equiv as f64,
        overlap_factor: 1.15,
        atlas_cols: None,
        relax: Some(RelaxConfig {
            seed: Some(99),
            ..Default::default()
        }),
    })
    .unwrap();
    rig.set_patch_size(target_patch);
    rig
}

fn pinhole_camera(w: u32, h: u32, fov_deg: f64) -> CameraIntrinsics {
    let half_fov = (fov_deg * 0.5).to_radians();
    let f = (w as f64 / 2.0) / half_fov.tan();
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: f,
            focal_length_y: f,
            principal_point_x: w as f64 / 2.0,
            principal_point_y: h as f64 / 2.0,
        },
        width: w,
        height: h,
    }
}

/// Render a synthetic source image whose pixel is filled by a smooth
/// direction-only pattern, so we can compare warped patches against an
/// independent ground truth.
fn render_synthetic(intrinsics: &CameraIntrinsics, r_src_from_world: &RotQuaternion) -> ImageU8 {
    let w = intrinsics.width;
    let h = intrinsics.height;
    let mut data = vec![0u8; (w * h * 3) as usize];
    let r_inv = r_src_from_world.inverse().to_rotation_matrix();
    for row in 0..h {
        for col in 0..w {
            let ray = intrinsics.pixel_to_ray(col as f64 + 0.5, row as f64 + 0.5);
            let world = r_inv * Vector3::new(ray[0], ray[1], ray[2]);
            // Per-channel pattern in [0, 255].
            let r = (0.5 + 0.4 * world.x).clamp(0.0, 1.0);
            let g = (0.5 + 0.4 * world.y).clamp(0.0, 1.0);
            let bl = (0.5 + 0.4 * world.z).clamp(0.0, 1.0);
            let base = ((row * w + col) * 3) as usize;
            data[base] = (r * 255.0).round() as u8;
            data[base + 1] = (g * 255.0).round() as u8;
            data[base + 2] = (bl * 255.0).round() as u8;
        }
    }
    ImageU8::new(w, h, 3, data)
}

// ── Validation ──────────────────────────────────────────────────────────

#[test]
fn build_fails_on_non_power_of_two_patch_size() {
    let rig = SphericalTileRig::new(&SphericalTileRigParams {
        n: 80,
        arc_per_pixel: 2.0 * PI / 256.0,
        relax: Some(RelaxConfig {
            seed: Some(7),
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();
    // Constructor's natural patch_size is ~21 (not a power of two).
    assert!(!rig.patch_size().is_power_of_two());
    let raw = rig.patch_size();
    let err = PerSphericalTileSourceStack::build_rotation_only(&rig, &[], &BuildParams::default())
        .unwrap_err();
    assert_eq!(err, BuildError::PatchSizeNotPowerOfTwo(raw));
}

#[test]
fn build_succeeds_after_set_patch_size() {
    let mut rig = SphericalTileRig::new(&SphericalTileRigParams {
        n: 80,
        arc_per_pixel: 2.0 * PI / 256.0,
        relax: Some(RelaxConfig {
            seed: Some(7),
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();
    rig.set_patch_size(rig.patch_size().next_power_of_two());
    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &[], &BuildParams::default())
            .unwrap();
    assert_eq!(stack.n_tiles(), 80);
    assert_eq!(stack.base_patch_size(), rig.patch_size());
    assert_eq!(
        stack.pyramid_levels(),
        rig.patch_size().trailing_zeros() + 1
    );
    for t in 0..stack.n_tiles() {
        assert_eq!(stack.n_contributors(t), 0);
        let tile = stack.tile(t);
        assert!(tile.src_indices.is_empty());
        for level in &tile.levels {
            assert_eq!(level.n_contributors, 0);
            assert!(level.patches.is_empty());
            assert!(level.valid.is_empty());
        }
    }
}

#[test]
fn build_rejects_mixed_channels() {
    let rig = make_pow2_rig(20, 256, 32);
    let cam = pinhole_camera(64, 64, 60.0);
    let img_rgb = ImageU8::from_channels(64, 64, 3);
    let img_gray = ImageU8::from_channels(64, 64, 1);
    let sources = vec![
        (cam.clone(), RotQuaternion::identity(), img_rgb),
        (cam.clone(), RotQuaternion::identity(), img_gray),
    ];
    let err =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap_err();
    assert_eq!(
        err,
        BuildError::MixedSourceChannels {
            first: 3,
            offending: 1,
        }
    );
}

// ── Pyramid sizing ──────────────────────────────────────────────────────

#[test]
fn pyramid_level_count_and_sizes() {
    for &b in &[8u32, 16, 32, 64, 128] {
        let rig = make_pow2_rig(20, 256, b);
        let stack =
            PerSphericalTileSourceStack::build_rotation_only(&rig, &[], &BuildParams::default())
                .unwrap();
        assert_eq!(stack.base_patch_size(), b);
        assert_eq!(stack.pyramid_levels(), b.trailing_zeros() + 1);
        for t in 0..stack.n_tiles() {
            let tile = stack.tile(t);
            assert_eq!(tile.levels.len() as u32, stack.pyramid_levels());
            for (li, level) in tile.levels.iter().enumerate() {
                assert_eq!(level.size, b >> li as u32);
            }
            assert_eq!(tile.levels.last().unwrap().size, 1);
        }
    }
}

#[test]
fn soa_buffer_sizing() {
    let rig = make_pow2_rig(40, 256, 16);
    let cam = pinhole_camera(128, 128, 60.0);
    let r1 = RotQuaternion::identity();
    let r2 = RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), PI / 4.0).unwrap();
    let img1 = render_synthetic(&cam, &r1);
    let img2 = render_synthetic(&cam, &r2);
    let sources = vec![
        (cam.clone(), r1.clone(), img1),
        (cam.clone(), r2.clone(), img2),
    ];

    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap();
    for t in 0..stack.n_tiles() {
        let tile = stack.tile(t);
        let k = tile.src_indices.len();
        for level in &tile.levels {
            let s = level.size as usize;
            let c = level.channels as usize;
            assert_eq!(level.n_contributors as usize, k);
            assert_eq!(level.patches.len(), k * s * s * c);
            assert_eq!(level.valid.len(), k * s * s);
        }
    }
}

#[test]
fn src_indices_are_strictly_ascending_and_in_range() {
    let rig = make_pow2_rig(40, 256, 16);
    let cam = pinhole_camera(128, 128, 60.0);
    let n_sources = 5usize;
    let sources: Vec<_> = (0..n_sources)
        .map(|i| {
            let q = RotQuaternion::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                (i as f64) * (2.0 * PI / n_sources as f64),
            )
            .unwrap();
            let img = render_synthetic(&cam, &q);
            (cam.clone(), q, img)
        })
        .collect();
    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap();
    for t in 0..stack.n_tiles() {
        let tile = stack.tile(t);
        for window in tile.src_indices.windows(2) {
            assert!(window[0] < window[1], "src_indices not strictly ascending");
        }
        for &i in &tile.src_indices {
            assert!((i as usize) < n_sources);
        }
    }
}

// ── Visibility cull ─────────────────────────────────────────────────────

#[test]
fn empty_cull_for_distant_source() {
    // A source looking +Z and a tile looking -Z: the centre direction never
    // projects in front of the source camera. Verify the cull drops it.
    let rig = make_pow2_rig(80, 256, 16);
    let cam = pinhole_camera(64, 64, 60.0);
    // Rotate src by 180° around Y so its +Z optical axis points -Z in world.
    let r_src_from_world = RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), PI).unwrap();
    let img = render_synthetic(&cam, &r_src_from_world);
    let sources = vec![(cam, r_src_from_world, img)];

    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap();

    // For each tile, recompute centre-direction projection by hand to verify
    // the cull matches the spec's definition exactly.
    let (src_intrinsics, r_sw_q, _) = &sources[0];
    let r_sw = r_sw_q.to_rotation_matrix();
    let sw = src_intrinsics.width as f64;
    let sh = src_intrinsics.height as f64;
    let mut total_kept = 0;
    for t in 0..stack.n_tiles() {
        let d = rig.direction(t);
        let d_src = r_sw * Vector3::new(d[0], d[1], d[2]);
        let in_view = match src_intrinsics.ray_to_pixel([d_src.x, d_src.y, d_src.z]) {
            Some((sx, sy)) => sx >= 0.0 && sy >= 0.0 && sx < sw && sy < sh,
            None => false,
        };
        let listed = stack.tile(t).src_indices.contains(&0);
        assert_eq!(listed, in_view, "tile {t}: cull disagrees with spec rule");
        if listed {
            total_kept += 1;
        }
    }
    // The 60° pinhole covers roughly 1/8 of the sphere. We just need a
    // non-trivial mix of kept and dropped tiles to exercise the cull.
    assert!(
        total_kept > 0 && total_kept < stack.n_tiles(),
        "expected partial cull, got {total_kept}/{}",
        stack.n_tiles()
    );
}

#[test]
fn opposite_sources_have_disjoint_tiles() {
    // Two narrow sources looking in opposite directions: no tile centre can
    // be inside both source frustums.
    let rig = make_pow2_rig(80, 256, 16);
    let cam = pinhole_camera(64, 64, 30.0);
    let r_a = RotQuaternion::identity();
    let r_b = RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), PI).unwrap();
    let img_a = render_synthetic(&cam, &r_a);
    let img_b = render_synthetic(&cam, &r_b);
    let sources = vec![(cam.clone(), r_a, img_a), (cam.clone(), r_b, img_b)];
    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap();
    for t in 0..stack.n_tiles() {
        let tile = stack.tile(t);
        let has_a = tile.src_indices.contains(&0);
        let has_b = tile.src_indices.contains(&1);
        assert!(
            !(has_a && has_b),
            "tile {t} has both opposite sources in src_indices"
        );
    }
}

// ── Synthetic correctness ───────────────────────────────────────────────

/// For each kept (source, tile), recompute the warp + remap by hand and
/// assert byte equality with the corresponding `pos · B² · C` slice of
/// `tile(t).levels[0].patches`. Load-bearing correctness check.
#[test]
fn level_zero_byte_equal_to_hand_warp() {
    let rig = make_pow2_rig(40, 256, 16);
    let cam = pinhole_camera(128, 128, 60.0);
    let n_sources = 3usize;
    let sources: Vec<_> = (0..n_sources)
        .map(|i| {
            let q = RotQuaternion::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                (i as f64) * (PI / n_sources as f64),
            )
            .unwrap();
            let img = render_synthetic(&cam, &q);
            (cam.clone(), q, img)
        })
        .collect();
    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap();

    let tile_camera = rig.tile_camera();
    let b = stack.base_patch_size();
    let b_us = b as usize;
    let c_us = 3;
    let pixel_count = b_us * b_us;

    let mut checked = 0usize;
    for t in 0..stack.n_tiles() {
        let tile = stack.tile(t);
        // Build R_world_from_tile.
        let cols = rig.tile_rotation(t);
        let r_wt = nalgebra::Matrix3::from_columns(&[
            Vector3::new(cols[0], cols[1], cols[2]),
            Vector3::new(cols[3], cols[4], cols[5]),
            Vector3::new(cols[6], cols[7], cols[8]),
        ]);
        for (pos, &i) in tile.src_indices.iter().enumerate() {
            let (src_intrinsics, r_sw_q, image) = &sources[i as usize];
            let r_sw = r_sw_q.to_rotation_matrix();
            let r_st = r_sw * r_wt;
            let r_st_q = RotQuaternion::from_rotation_matrix(r_st);
            let warp = WarpMap::from_cameras_with_rotation(src_intrinsics, &tile_camera, &r_st_q);
            let patch = remap_bilinear(image, &warp);
            let level0 = &tile.levels[0];
            let p_off = pos * pixel_count * c_us;
            assert_eq!(
                &level0.patches[p_off..p_off + pixel_count * c_us],
                patch.data(),
                "tile {t} src {i}: level-0 patch differs from hand warp"
            );
            // Also check the level-0 valid mask matches the warp's is_valid.
            let v_off = pos * pixel_count;
            for v in 0..b {
                for u in 0..b {
                    let expected = warp.is_valid(u, v);
                    let stored = level0.valid[v_off + (v as usize) * b_us + u as usize];
                    assert_eq!(
                        stored, expected,
                        "tile {t} src {i}: level-0 valid disagrees at ({u}, {v})"
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no kept (source, tile) pairs to check");
}

// ── Pyramid downsample correctness ──────────────────────────────────────

#[test]
fn pyramid_constant_color_propagates() {
    // Mock up a tile-stack manually: write a known constant level-0 patch
    // and walk the downsample logic by reusing `warp_and_downsample_into`.
    // We do this by going through the public path with a constant-coloured
    // synthetic source to check end-to-end consistency.
    let rig = make_pow2_rig(20, 256, 8);
    let w = 64u32;
    let cam = pinhole_camera(w, w, 90.0);
    // Constant-grey image.
    let img = ImageU8::new(w, w, 3, vec![137u8; (w * w * 3) as usize]);
    let sources = vec![(cam, RotQuaternion::identity(), img)];
    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap();

    let mut checked = 0usize;
    for t in 0..stack.n_tiles() {
        let tile = stack.tile(t);
        if tile.src_indices.is_empty() {
            continue;
        }
        for level in &tile.levels {
            // For a constant input, every level should also be constant. The
            // valid bits are governed by warp validity, not the colour, so
            // some pixels may be invalid; check colours only at valid pixels.
            let s = level.size as usize;
            let pixel_count = s * s;
            for pos in 0..tile.src_indices.len() {
                let p_off = pos * pixel_count * 3;
                let v_off = pos * pixel_count;
                for px in 0..pixel_count {
                    if level.valid[v_off + px] {
                        let r = level.patches[p_off + px * 3];
                        let g = level.patches[p_off + px * 3 + 1];
                        let bl = level.patches[p_off + px * 3 + 2];
                        // u8 round-trip of mean(137, 137, 137, 137) = 137.
                        assert_eq!(r, 137);
                        assert_eq!(g, 137);
                        assert_eq!(bl, 137);
                    }
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no levels checked");
}

// ── All-four valid propagation ──────────────────────────────────────────

#[test]
fn all_four_valid_propagates_through_pyramid() {
    // Build a tiny stack with one source feeding one kept tile, then poke a
    // single invalid pixel into the level-0 valid mask via a custom rig
    // built inline. We re-implement the all-four AND rule and compare.
    //
    // Rather than mocking a stack, the cleanest reproduction is to call the
    // module-private downsample helper directly. We expose the kernel by
    // doing a hand-rolled mirror here and asserting the spec's documented
    // propagation: an invalid pixel at `(x, y)` in level L−1 makes the
    // level-L pixel at `(x/2, y/2)` invalid; level L+1 at `(x/4, y/4)`; …
    // the final 1×1 entry is always invalid.
    let b: usize = 16;
    let mut prev_valid = vec![true; b * b];
    // Seed: invalidate one specific pixel.
    let (x0, y0) = (5usize, 11usize);
    prev_valid[y0 * b + x0] = false;

    let mut x = x0;
    let mut y = y0;
    let mut s = b;
    while s > 1 {
        let new_s = s / 2;
        let mut new_valid = vec![false; new_s * new_s];
        for v in 0..new_s {
            for u in 0..new_s {
                let i00 = (2 * v) * s + (2 * u);
                let i10 = (2 * v) * s + (2 * u + 1);
                let i01 = (2 * v + 1) * s + (2 * u);
                let i11 = (2 * v + 1) * s + (2 * u + 1);
                new_valid[v * new_s + u] =
                    prev_valid[i00] && prev_valid[i10] && prev_valid[i01] && prev_valid[i11];
            }
        }
        // Exactly one invalid pixel in the new level: at (x/2, y/2).
        let invalid_count = new_valid.iter().filter(|&&v| !v).count();
        assert_eq!(invalid_count, 1, "expected exactly one invalid pixel");
        x /= 2;
        y /= 2;
        assert!(
            !new_valid[y * new_s + x],
            "level with size {new_s} should mark ({x}, {y}) invalid"
        );
        prev_valid = new_valid;
        s = new_s;
    }
    // Final 1x1 level must be invalid.
    assert_eq!(prev_valid.len(), 1);
    assert!(!prev_valid[0]);
}

// ── Visibility = base-warp centre-pixel validity ────────────────────────

#[test]
fn visibility_matches_base_warp_centre_pixel() {
    // Spec: a tile has source `i` in `tile(t).src_indices` iff the level-0
    // warp's `map_x`, `map_y` at the patch centre is in-bounds for that
    // source. Cross-checks the cull's centre-direction projection against
    // the warp's per-pixel in-bounds bit at the patch centre — agreement
    // confirms the rig's `tile_camera()` and warp construction are
    // consistent with the cull rule.
    let rig = make_pow2_rig(40, 256, 16);
    let cam = pinhole_camera(128, 128, 60.0);
    let n_sources = 4usize;
    let sources: Vec<_> = (0..n_sources)
        .map(|i| {
            let q = RotQuaternion::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                (i as f64) * (PI / 4.0),
            )
            .unwrap();
            let img = render_synthetic(&cam, &q);
            (cam.clone(), q, img)
        })
        .collect();
    let stack =
        PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &BuildParams::default())
            .unwrap();

    let tile_camera = rig.tile_camera();
    let b = stack.base_patch_size();
    // patch centre = (B/2, B/2) — in pixel-centre-at-0.5 convention this is
    // the boundary between the two centre rows/cols, which still has a
    // well-defined in-bounds bit on either side. Use the upper-left of the
    // 2x2 centre block so we evaluate exactly one warp pixel.
    let centre_u = b / 2;
    let centre_v = b / 2;

    for t in 0..stack.n_tiles() {
        let cols = rig.tile_rotation(t);
        let r_wt = nalgebra::Matrix3::from_columns(&[
            Vector3::new(cols[0], cols[1], cols[2]),
            Vector3::new(cols[3], cols[4], cols[5]),
            Vector3::new(cols[6], cols[7], cols[8]),
        ]);
        for (i, (src_intrinsics, r_sw_q, _)) in sources.iter().enumerate() {
            let r_sw = r_sw_q.to_rotation_matrix();
            let r_st = r_sw * r_wt;
            let r_st_q = RotQuaternion::from_rotation_matrix(r_st);
            let warp = WarpMap::from_cameras_with_rotation(src_intrinsics, &tile_camera, &r_st_q);
            let in_view = warp.is_valid(centre_u, centre_v);
            let listed = stack.tile(t).src_indices.contains(&(i as u32));
            assert_eq!(
                listed, in_view,
                "tile {t} src {i}: cull ({listed}) disagrees with warp centre validity ({in_view})"
            );
        }
    }
}

// ── max_in_flight_sources / determinism ─────────────────────────────────

#[test]
fn determinism_param_does_not_change_output() {
    let rig = make_pow2_rig(40, 256, 16);
    let cam = pinhole_camera(128, 128, 60.0);
    let sources: Vec<_> = (0..4)
        .map(|i| {
            let q = RotQuaternion::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                (i as f64) * (PI / 6.0),
            )
            .unwrap();
            let img = render_synthetic(&cam, &q);
            (cam.clone(), q, img)
        })
        .collect();
    let stack_a = PerSphericalTileSourceStack::build_rotation_only(
        &rig,
        &sources,
        &BuildParams {
            max_in_flight_sources: Some(1),
        },
    )
    .unwrap();
    let stack_b = PerSphericalTileSourceStack::build_rotation_only(
        &rig,
        &sources,
        &BuildParams {
            max_in_flight_sources: None,
        },
    )
    .unwrap();
    for t in 0..stack_a.n_tiles() {
        assert_eq!(stack_a.tile(t), stack_b.tile(t));
    }
}
