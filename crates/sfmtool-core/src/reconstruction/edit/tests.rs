// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Editing a reconstruction that carries per-point patch frames: the frames
//! have to survive the same transforms, filters and merges the points do.

use super::*;
use crate::geometry::RotQuaternion;
use crate::Se3Transform;
use nalgebra::{UnitQuaternion, Vector3 as V3};
use ndarray::{Array2, Array4};

/// A demo reconstruction with a per-point patch frame attached: `u` along
/// +x and `v` along +y (so `u × v` is +z), plus distinct-per-cell bitmaps.
fn demo_with_patches() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(4);
    let p = recon.points.len();
    let mut u = Array2::<f32>::zeros((p, 3));
    let mut v = Array2::<f32>::zeros((p, 3));
    for i in 0..p {
        u[[i, 0]] = 0.1 * (i + 1) as f32;
        v[[i, 1]] = 0.2 * (i + 1) as f32;
    }
    let bitmaps = Array4::<u8>::from_shape_fn((p, 2, 2, 4), |(i, y, x, c)| {
        ((i * 13 + y * 5 + x * 3 + c) % 256) as u8
    });
    recon.patch_u_halfvec_xyz = Some(u);
    recon.patch_v_halfvec_xyz = Some(v);
    recon.patch_bitmaps_y_x_rgba = Some(bitmaps);
    recon
}

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-5, "{a} != {b}");
}

#[test]
fn se3_transform_rotates_and_scales_patch_frame_and_normals() {
    let recon = demo_with_patches();
    let u0 = recon.patch_u_halfvec_xyz.clone().unwrap();
    let v0 = recon.patch_v_halfvec_xyz.clone().unwrap();
    let bitmaps0 = recon.patch_bitmaps_y_x_rgba.clone().unwrap();
    let n0: Vec<_> = recon.points.iter().map(|p| p.normal).collect();

    // 90° about +z, uniform scale 2, arbitrary translation.
    let rot = RotQuaternion::from_nalgebra(UnitQuaternion::from_axis_angle(
        &V3::z_axis(),
        std::f64::consts::FRAC_PI_2,
    ));
    let t = Se3Transform::new(rot.clone(), V3::new(1.0, 2.0, 3.0), 2.0);
    let out = recon.apply_se3_transform(&t);

    // Bitmaps are pose-invariant: carried byte-for-byte.
    assert_eq!(out.patch_bitmaps_y_x_rgba.as_ref().unwrap(), &bitmaps0);

    let u1 = out.patch_u_halfvec_xyz.as_ref().unwrap();
    let v1 = out.patch_v_halfvec_xyz.as_ref().unwrap();
    for i in 0..recon.points.len() {
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
        approx(out.points[i].normal.x as f64, want_n.x);
        approx(out.points[i].normal.y as f64, want_n.y);
        approx(out.points[i].normal.z as f64, want_n.z);
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
    let u0 = recon.patch_u_halfvec_xyz.clone().unwrap();
    let mask = vec![true, false, true, false];
    let out = recon.filter_points_by_mask(&mask);

    assert_eq!(out.point_count(), 2);
    let u1 = out.patch_u_halfvec_xyz.as_ref().unwrap();
    assert_eq!(u1.shape(), &[2, 3]);
    // Kept rows are the source rows 0 and 2, unchanged.
    approx(u1[[0, 0]] as f64, u0[[0, 0]] as f64);
    approx(u1[[1, 0]] as f64, u0[[2, 0]] as f64);
    assert_eq!(out.patch_bitmaps_y_x_rgba.as_ref().unwrap().shape()[0], 2);
}

#[test]
fn subset_keeping_all_images_carries_the_patch_frame() {
    let recon = demo_with_patches();
    let u0 = recon.patch_u_halfvec_xyz.clone().unwrap();
    let all: Vec<u32> = (0..recon.images.len() as u32).collect();
    let out = recon.subset_by_image_indices(&all, true).unwrap();

    assert_eq!(out.point_count(), recon.point_count());
    assert_eq!(out.patch_u_halfvec_xyz.as_ref().unwrap(), &u0);
    assert_eq!(
        out.patch_bitmaps_y_x_rgba.as_ref().unwrap(),
        recon.patch_bitmaps_y_x_rgba.as_ref().unwrap()
    );
}
