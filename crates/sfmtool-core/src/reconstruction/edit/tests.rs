// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Editing a reconstruction that carries per-point patch frames: the frames
//! have to survive the same transforms, filters and merges the points do.

use std::sync::Arc;

use super::*;
use crate::geometry::RotQuaternion;
use crate::Se3Transform;
use nalgebra::{UnitQuaternion, Vector3 as V3};
use ndarray::{Array2, Array4};
use sfmr_format::{
    NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED,
};

/// A demo reconstruction with a per-point patch frame attached: `u` along
/// +x and `v` along +y (so `u × v` is +z), plus distinct-per-cell bitmaps.
fn demo_with_patches() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(4);
    let p = recon.point_set.points.len();
    let mut u = Array2::<f32>::zeros((p, 3));
    let mut v = Array2::<f32>::zeros((p, 3));
    for i in 0..p {
        u[[i, 0]] = 0.1 * (i + 1) as f32;
        v[[i, 1]] = 0.2 * (i + 1) as f32;
    }
    let bitmaps = Array4::<u8>::from_shape_fn((p, 2, 2, 4), |(i, y, x, c)| {
        ((i * 13 + y * 5 + x * 3 + c) % 256) as u8
    });
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmaps));
    recon
}

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-5, "{a} != {b}");
}

#[test]
fn se3_transform_rotates_and_scales_patch_frame_and_normals() {
    let recon = demo_with_patches();
    let u0 = recon.point_set.patch_u_halfvec_xyz.clone().unwrap();
    let v0 = recon.point_set.patch_v_halfvec_xyz.clone().unwrap();
    let bitmaps0 = recon.point_set.patch_bitmaps_y_x_rgba.clone().unwrap();
    let n0: Vec<_> = recon.point_set.points.iter().map(|p| p.normal).collect();

    // 90° about +z, uniform scale 2, arbitrary translation.
    let rot = RotQuaternion::from_nalgebra(UnitQuaternion::from_axis_angle(
        &V3::z_axis(),
        std::f64::consts::FRAC_PI_2,
    ));
    let t = Se3Transform::new(rot.clone(), V3::new(1.0, 2.0, 3.0), 2.0);
    let out = recon.apply_se3_transform(&t);

    // Bitmaps are pose-invariant: carried byte-for-byte.
    assert_eq!(
        out.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap(),
        &bitmaps0
    );

    let u1 = out.point_set.patch_u_halfvec_xyz.as_ref().unwrap();
    let v1 = out.point_set.patch_v_halfvec_xyz.as_ref().unwrap();
    for i in 0..recon.point_set.points.len() {
        // Half-vectors: rotated by R and scaled by s.
        for (a0, a1) in [(&u0, u1), (&v0, v1)] {
            let src = V3::new(a0[[i, 0]] as f64, a0[[i, 1]] as f64, a0[[i, 2]] as f64);
            let want = rot.rotate_vector(&src) * t.scale;
            approx(a1[[i, 0]] as f64, want.x);
            approx(a1[[i, 1]] as f64, want.y);
            approx(a1[[i, 2]] as f64, want.z);
        }
        // Normal: a direction, rotated by R (no scale, stays unit).
        let nn = V3::new(n0[i].x as f64, n0[i].y as f64, n0[i].z as f64);
        let want_n = rot.rotate_vector(&nn);
        approx(out.point_set.points[i].normal.x as f64, want_n.x);
        approx(out.point_set.points[i].normal.y as f64, want_n.y);
        approx(out.point_set.points[i].normal.z as f64, want_n.z);
    }

    // The frame stays rigid: normalize(u × v) just rotates by R. Check pt 0,
    // whose pre-transform u × v is +z.
    let u1v = V3::new(u1[[0, 0]] as f64, u1[[0, 1]] as f64, u1[[0, 2]] as f64);
    let v1v = V3::new(v1[[0, 0]] as f64, v1[[0, 1]] as f64, v1[[0, 2]] as f64);
    let n_patch = u1v.cross(&v1v).normalize();
    let want = rot.rotate_vector(&V3::z());
    approx(n_patch.x, want.x);
    approx(n_patch.y, want.y);
    approx(n_patch.z, want.z);
}

#[test]
fn filter_keeps_patch_rows_for_surviving_points() {
    let recon = demo_with_patches();
    let u0 = recon.point_set.patch_u_halfvec_xyz.clone().unwrap();
    let mask = vec![true, false, true, false];
    let out = recon.filter_points_by_mask(&mask);

    assert_eq!(out.point_count(), 2);
    let u1 = out.point_set.patch_u_halfvec_xyz.as_ref().unwrap();
    assert_eq!(u1.shape(), &[2, 3]);
    // Kept rows are the source rows 0 and 2, unchanged.
    approx(u1[[0, 0]] as f64, u0[[0, 0]] as f64);
    approx(u1[[1, 0]] as f64, u0[[2, 0]] as f64);
    assert_eq!(
        out.point_set
            .patch_bitmaps_y_x_rgba
            .as_ref()
            .unwrap()
            .shape()[0],
        2
    );
}

/// A four-point demo whose points 1 and 3 are constrained: point 1 held, point
/// 3 ranged at 10 m from image 5. The reference is a late image so an image
/// subset that keeps only the early ones drops it.
fn demo_with_constraints() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(4);
    let mut constraints = PointConstraintColumns::all_free(recon.point_set.points.len());
    constraints.point_constraints[1] = POINT_CONSTRAINT_HELD;
    constraints.point_constraints[3] = POINT_CONSTRAINT_RANGED;
    constraints.constraint_distances[3] = 10.0;
    constraints.constraint_reference_images[3] = 5;
    recon.point_set.point_constraints = Some(constraints);
    recon
}

#[test]
fn filter_keeps_constraint_rows_for_surviving_points() {
    let recon = demo_with_constraints();
    let out = recon.filter_points_by_mask(&[false, true, false, true]);

    let c = out.point_set.point_constraints.as_ref().unwrap();
    assert_eq!(c.len(), 2);
    // The source's points 1 and 3, in order and unchanged.
    assert_eq!(
        c.point_constraints,
        vec![POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED]
    );
    assert!(c.constraint_distances[0].is_nan());
    assert_eq!(c.constraint_distances[1], 10.0);
    assert_eq!(c.constraint_reference_images, vec![NO_REFERENCE_IMAGE, 5]);
    out.validate_point_columns().unwrap();
}

#[test]
fn subset_remaps_a_distance_reference_onto_the_kept_images() {
    let recon = demo_with_constraints();
    // Keep images 5, 0 and 2, in that order: the reference image survives at a
    // new index, which is what the distance has to follow.
    let out = recon.subset_by_image_indices(&[5, 0, 2], false).unwrap();

    let c = out.point_set.point_constraints.as_ref().unwrap();
    assert_eq!(c.point_constraints[3], POINT_CONSTRAINT_RANGED);
    assert_eq!(c.constraint_distances[3], 10.0);
    assert_eq!(c.constraint_reference_images[3], 0);
    out.validate_point_columns().unwrap();
}

#[test]
fn subset_frees_a_point_whose_reference_image_is_dropped() {
    let recon = demo_with_constraints();
    // Image 5 is gone, so nothing is left to measure the distance from.
    let out = recon.subset_by_image_indices(&[0, 1, 2], false).unwrap();

    let c = out.point_set.point_constraints.as_ref().unwrap();
    assert_eq!(c.point_constraints[3], POINT_CONSTRAINT_FREE);
    assert!(c.constraint_distances[3].is_nan());
    assert_eq!(c.constraint_reference_images[3], NO_REFERENCE_IMAGE);
    // A held point names no image, so the same subset leaves it held.
    assert_eq!(c.point_constraints[1], POINT_CONSTRAINT_HELD);
    out.validate_point_columns().unwrap();
}

#[test]
fn se3_transform_scales_a_distance_with_the_scene() {
    let recon = demo_with_constraints();
    let rot = RotQuaternion::from_nalgebra(UnitQuaternion::from_axis_angle(
        &V3::z_axis(),
        std::f64::consts::FRAC_PI_2,
    ));
    let out = recon.apply_se3_transform(&Se3Transform::new(rot, V3::new(1.0, 2.0, 3.0), 2.0));

    let c = out.point_set.point_constraints.as_ref().unwrap();
    // The distance is in the solve's own units, which the similarity rescales.
    approx(c.constraint_distances[3], 20.0);
    assert_eq!(c.constraint_reference_images[3], 5);
    assert!(c.constraint_distances[1].is_nan());
}

#[test]
fn subset_keeping_all_images_carries_the_patch_frame() {
    let recon = demo_with_patches();
    let u0 = recon.point_set.patch_u_halfvec_xyz.clone().unwrap();
    let all: Vec<u32> = (0..recon.image_table.images.len() as u32).collect();
    let out = recon.subset_by_image_indices(&all, true).unwrap();

    assert_eq!(out.point_count(), recon.point_count());
    assert_eq!(out.point_set.patch_u_halfvec_xyz.as_ref().unwrap(), &u0);
    assert_eq!(
        out.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap(),
        recon.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap()
    );
}
