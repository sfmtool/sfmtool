// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the tile-batched consensus atlas orchestrator.
//!
//! Covers the spec's validation plan: batch-size invariance (the headline
//! property), the exact tile partition, empty-batch handling, and the
//! up-front rejections.

use std::f64::consts::PI;

use nalgebra::Vector3;

use super::*;
use crate::camera_intrinsics::{CameraIntrinsics, CameraModel};
use crate::per_spherical_tile_source_stack::{BuildParams, PerSphericalTileSourceStack};
use crate::photometric_ransac::refine_photometric_ransac;
use crate::remap::ImageU8;
use crate::rot_quaternion::RotQuaternion;
use crate::sphere_points::RelaxConfig;
use crate::spherical_tile_rig::{SphericalTileRig, SphericalTileRigParams};

type SourceList = Vec<(CameraIntrinsics, RotQuaternion, ImageU8)>;

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

fn pinhole_camera(w: u32, fov_deg: f64) -> CameraIntrinsics {
    let half_fov = (fov_deg * 0.5).to_radians();
    let f = (w as f64 / 2.0) / half_fov.tan();
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: f,
            focal_length_y: f,
            principal_point_x: w as f64 / 2.0,
            principal_point_y: w as f64 / 2.0,
        },
        width: w,
        height: w,
    }
}

/// `n` wide-FOV synthetic sources tightly clustered around the Y axis, so the
/// front tiles each draw on many contributors — enough to push per-tile
/// contributor counts past the RANSAC exhaustive-enumeration cap and exercise
/// the seed-dependent sampling path.
fn synthetic_sources(n: usize) -> SourceList {
    let w = 96u32;
    let cam = pinhole_camera(w, 130.0);
    (0..n)
        .map(|i| {
            let q = RotQuaternion::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                (i as f64 - n as f64 / 2.0) * 0.08,
            )
            .unwrap();
            let r_inv = q.inverse().to_rotation_matrix();
            let mut data = vec![0u8; (w * w * 3) as usize];
            for row in 0..w {
                for col in 0..w {
                    let ray = cam.pixel_to_ray(col as f64 + 0.5, row as f64 + 0.5);
                    let world = r_inv * Vector3::new(ray[0], ray[1], ray[2]);
                    let base = ((row * w + col) * 3) as usize;
                    data[base] = ((0.5 + 0.4 * world.x).clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[base + 1] = ((0.5 + 0.4 * world.y).clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[base + 2] = ((0.5 + 0.4 * world.z).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
            (cam.clone(), q, ImageU8::new(w, w, 3, data))
        })
        .collect()
}

/// Bitwise (NaN-aware) equality on f32 buffers.
fn assert_bits_eq(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: element {i} differs ({x} vs {y})"
        );
    }
}

fn params(batch_size: usize) -> ConsensusAtlasBatchParams {
    ConsensusAtlasBatchParams {
        batch_size,
        ..Default::default()
    }
}

/// Validation plan #1: the `atlas` and per-tile arrays are byte-identical for
/// every `batch_size`, and equal to the monolithic build → RANSAC → consensus
/// trio.
#[test]
fn batch_size_invariance_f32() {
    let rig = make_pow2_rig(20, 256, 4);
    let sources = synthetic_sources(16);

    // The test only bites on `tile_index_base` if some tile has enough
    // contributors to overflow the exhaustive-enumeration cap (C(K,2) > 64,
    // i.e. K >= 12).
    let mono_stack = PerSphericalTileSourceStack::<f32>::build_rotation_only(
        &rig,
        &sources,
        &BuildParams::default(),
    )
    .unwrap();
    assert!(
        (0..mono_stack.n_tiles()).any(|t| mono_stack.n_contributors(t) >= 12),
        "fixture must have an over-cap tile to exercise seed-dependent RANSAC"
    );

    let single = render_consensus_atlas::<f32>(&rig, &sources, &params(rig.len())).unwrap();
    for bs in [1usize, 3, 7] {
        let r = render_consensus_atlas::<f32>(&rig, &sources, &params(bs)).unwrap();
        assert_bits_eq(&r.atlas, &single.atlas, &format!("atlas bs={bs}"));
        assert_eq!(
            r.tile_primary_count, single.tile_primary_count,
            "tile_primary_count bs={bs}"
        );
        assert_eq!(
            r.tile_secondary_count, single.tile_secondary_count,
            "tile_secondary_count bs={bs}"
        );
        assert_bits_eq(
            &r.tile_primary_lum_mad,
            &single.tile_primary_lum_mad,
            &format!("tile_primary_lum_mad bs={bs}"),
        );
        assert_bits_eq(
            &r.tile_secondary_lum_mad,
            &single.tile_secondary_lum_mad,
            &format!("tile_secondary_lum_mad bs={bs}"),
        );
    }

    // ... and identical to the explicit monolithic path.
    let out = refine_photometric_ransac(&mono_stack, &RansacPhotometricParams::default()).unwrap();
    let mono_atlas = mono_stack
        .primary_consensus_atlas(&rig, &out.primary_mask)
        .unwrap();
    assert_bits_eq(&single.atlas, &mono_atlas, "atlas vs monolithic");
    assert_eq!(
        single.tile_primary_count, out.tile_primary_count,
        "tile_primary_count vs monolithic"
    );
    assert_eq!(single.tile_secondary_count, out.tile_secondary_count);
    assert_bits_eq(
        &single.tile_primary_lum_mad,
        &out.tile_primary_lum_mad,
        "tile_primary_lum_mad vs monolithic",
    );
    assert_bits_eq(
        &single.tile_secondary_lum_mad,
        &out.tile_secondary_lum_mad,
        "tile_secondary_lum_mad vs monolithic",
    );
}

/// Validation plan #1, f16 storage: the three batched runs equal each other
/// (not necessarily the f32 run).
#[test]
fn batch_size_invariance_f16() {
    let rig = make_pow2_rig(20, 256, 4);
    let sources = synthetic_sources(16);
    let single = render_consensus_atlas::<half::f16>(&rig, &sources, &params(rig.len())).unwrap();
    for bs in [1usize, 3, 7] {
        let r = render_consensus_atlas::<half::f16>(&rig, &sources, &params(bs)).unwrap();
        assert_bits_eq(&r.atlas, &single.atlas, &format!("f16 atlas bs={bs}"));
        assert_eq!(r.tile_primary_count, single.tile_primary_count);
        assert_eq!(r.tile_secondary_count, single.tile_secondary_count);
    }
}

/// Validation plan #3: every monolithic row belongs to exactly one batch, so
/// the per-batch row counts sum to `total_contrib_rows` for any `batch_size`.
#[test]
fn batch_partition_is_exact() {
    let rig = make_pow2_rig(20, 256, 4);
    let sources = synthetic_sources(8);
    let total = PerSphericalTileSourceStack::<f32>::build_rotation_only(
        &rig,
        &sources,
        &BuildParams::default(),
    )
    .unwrap()
    .total_contrib_rows();
    for batch_size in [1usize, 3, 7, 20, 100] {
        let mut sum = 0usize;
        let n_batches = rig.len().div_ceil(batch_size);
        for b in 0..n_batches {
            let start = b * batch_size;
            let end = ((b + 1) * batch_size).min(rig.len());
            let sub_rig = rig.tiles_subset(start..end);
            let sub = PerSphericalTileSourceStack::<f32>::build_rotation_only(
                &sub_rig,
                &sources,
                &BuildParams::default(),
            )
            .unwrap();
            sum += sub.total_contrib_rows();
        }
        assert_eq!(
            sum, total,
            "batch_size={batch_size}: sum of per-batch R_b != total_contrib_rows"
        );
    }
}

/// Validation plan #6: with no sources every batch is a zero-row batch — the
/// whole atlas is NaN, the count arrays are zero, the MAD arrays are NaN, and
/// the run does not panic at any `batch_size` (including 1).
#[test]
fn empty_sources_yield_nan_atlas_no_panic() {
    let rig = make_pow2_rig(18, 256, 4);
    let empty: SourceList = Vec::new();
    for batch_size in [1usize, 4, 18, 50] {
        let report = render_consensus_atlas::<f32>(&rig, &empty, &params(batch_size)).unwrap();
        assert!(
            report.atlas.iter().all(|v| v.is_nan()),
            "bs={batch_size}: atlas not all NaN"
        );
        assert_eq!(report.tile_primary_count.len(), rig.len());
        assert_eq!(report.tile_secondary_count.len(), rig.len());
        assert!(report.tile_primary_count.iter().all(|&c| c == 0));
        assert!(report.tile_secondary_count.iter().all(|&c| c == 0));
        assert!(report.tile_primary_lum_mad.iter().all(|v| v.is_nan()));
        assert!(report.tile_secondary_lum_mad.iter().all(|v| v.is_nan()));
    }
}

/// Validation plan #7: `batch_size == 0` is rejected up front.
#[test]
fn rejects_zero_batch_size() {
    let rig = make_pow2_rig(20, 256, 4);
    let err = render_consensus_atlas::<f32>(&rig, &[], &params(0)).unwrap_err();
    assert!(matches!(err, ConsensusAtlasBatchError::BatchSizeZero));
}

/// Validation plan #7: a non-power-of-two `rig.patch_size()` is rejected up
/// front (the constructor's `arc_per_pixel`-derived size is not a power of two).
#[test]
fn rejects_non_power_of_two_patch_size() {
    let rig = SphericalTileRig::new(&SphericalTileRigParams {
        n: 20,
        arc_per_pixel: 2.0 * PI / 256.0,
        relax: Some(RelaxConfig {
            seed: Some(7),
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();
    assert!(!rig.patch_size().is_power_of_two());
    let raw = rig.patch_size();
    let err = render_consensus_atlas::<f32>(&rig, &[], &ConsensusAtlasBatchParams::default())
        .unwrap_err();
    assert!(matches!(
        err,
        ConsensusAtlasBatchError::PatchSizeNotPowerOfTwo(p) if p == raw
    ));
}
