// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Arrays a caller hands the writer outside standard layout are written in
//! row-major order, the same bytes a standard-layout copy gives, rather than
//! panicking.

use super::*;
use ndarray::{Array, Axis, Dimension, ShapeBuilder, Slice};

/// `array`'s shape and values in a layout that is not standard: column-major
/// (Fortran) order for an array of two or more axes, and every other element
/// of a buffer twice as long for one axis.
fn non_standard<A: Clone, D: Dimension>(array: &Array<A, D>) -> Array<A, D> {
    let out = if array.ndim() >= 2 {
        let values: Vec<A> = array.t().iter().cloned().collect();
        Array::from_shape_vec(array.raw_dim().f(), values).unwrap()
    } else {
        let mut dim = array.raw_dim();
        dim[0] *= 2;
        let mut out = Array::from_elem(dim, array.first().unwrap().clone());
        out.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
        out.assign(array);
        out
    };
    assert!(
        !out.is_standard_layout(),
        "shape {:?} has no non-standard layout",
        array.shape()
    );
    out
}

/// The rig fixture with every optional column the writer reads from the
/// caller: thumbnails and patch bitmaps with distinct values, normal and
/// observation confidence, point constraints, inline keypoints and the patch
/// frame. Its tracks arrive reversed, so the write sorts them.
fn make_full_test_data() -> SfmrData {
    let mut data = make_rig_test_data();
    let (n, p) = (
        data.metadata.image_count as usize,
        data.metadata.point_count as usize,
    );
    let m = data.metadata.observation_count as usize;
    data.thumbnails_y_x_rgb = Some(Array4::from_shape_fn((n, 128, 128, 3), |(i, y, x, c)| {
        (i * 7 + y * 3 + x + c) as u8
    }));
    data.translations_xyz = Array2::from_shape_fn((n, 3), |(i, k)| 0.01 * (i * 3 + k) as f64);
    data.normal_confidence = Some(Array1::from_vec(vec![255u8, 0, 128, 64, 32]));
    with_point_constraints(&mut data);
    data.patch_u_halfvec_xyz = Some(Array2::from_shape_fn((p, 3), |(i, k)| {
        [0.5, 0.01 * i as f32, 0.0][k]
    }));
    data.patch_v_halfvec_xyz = Some(Array2::from_shape_fn((p, 3), |(i, k)| {
        [0.0, 0.25, 0.02 * i as f32][k]
    }));
    data.patch_bitmaps_y_x_rgba = Some(Array4::from_shape_fn((p, 4, 4, 4), |(i, y, x, c)| {
        (i * 64 + y * 16 + x * 4 + c) as u8
    }));
    data.reference_observations = Some(Array1::from_elem(p, NO_REFERENCE_OBSERVATION));
    // Reverse the tracks, with the per-observation columns beside them.
    let reverse = |a: &Array1<u32>| a.slice(ndarray::s![..;-1]).to_owned();
    data.point_indexes = reverse(&data.point_indexes);
    data.image_indexes = reverse(&data.image_indexes);
    data.feature_indexes = data.feature_indexes.as_ref().map(reverse);
    data.keypoints_xy = Some(Array2::from_shape_fn((m, 2), |(i, k)| {
        [300.5 + i as f32, 400.75 + i as f32][k]
    }));
    data.observation_confidence = Some(Array1::from_iter((0..m).map(|i| (i * 30) as u8)));
    data
}

/// Every array of `data` the writer reads, replaced by [`non_standard`].
fn to_non_standard(data: &mut SfmrData) {
    fn swap<A: Clone, D: Dimension>(array: &mut Array<A, D>) {
        *array = non_standard(array);
    }
    swap(&mut data.camera_indexes);
    swap(&mut data.quaternions_wxyz);
    swap(&mut data.translations_xyz);
    swap(data.thumbnails_y_x_rgb.as_mut().unwrap());
    let readings = data.image_readings.as_mut().unwrap();
    swap(&mut readings.sharpness);
    swap(&mut readings.gradient_structure_tensors_xx_yy_xy);
    swap(&mut data.positions_xyzw);
    swap(&mut data.colors_rgb);
    swap(&mut data.reprojection_errors);
    swap(data.normals_xyz.as_mut().unwrap());
    swap(data.normal_confidence.as_mut().unwrap());
    swap(data.point_constraints.as_mut().unwrap());
    swap(data.constraint_distances.as_mut().unwrap());
    swap(data.constraint_reference_images.as_mut().unwrap());
    swap(data.patch_u_halfvec_xyz.as_mut().unwrap());
    swap(data.patch_v_halfvec_xyz.as_mut().unwrap());
    swap(data.patch_bitmaps_y_x_rgba.as_mut().unwrap());
    swap(&mut data.image_indexes);
    swap(data.feature_indexes.as_mut().unwrap());
    swap(data.keypoints_xy.as_mut().unwrap());
    swap(data.observation_confidence.as_mut().unwrap());
    swap(&mut data.point_indexes);
    swap(&mut data.observation_counts);
    swap(data.reference_observations.as_mut().unwrap());
    swap(&mut data.observed_depth_histogram_counts);
    let rf = data.rig_frame_data.as_mut().unwrap();
    swap(&mut rf.sensor_camera_indexes);
    swap(&mut rf.sensor_quaternions_wxyz);
    swap(&mut rf.sensor_translations_xyz);
    swap(&mut rf.rig_indexes);
    swap(&mut rf.image_sensor_indexes);
    swap(&mut rf.image_frame_indexes);
}

#[test]
fn test_write_arrays_out_of_standard_layout() {
    // Kept statistics, so the caller's histogram counts are what is written.
    let options = WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    let mut standard = make_full_test_data();
    let mut altered = make_full_test_data();
    to_non_standard(&mut altered);

    let dir = std::env::temp_dir().join("sfmr_test_non_standard_layout");
    std::fs::create_dir_all(&dir).unwrap();
    let (path_a, path_b) = (dir.join("standard.sfmr"), dir.join("altered.sfmr"));
    write_sfmr_with_options(&path_a, &mut standard, &options).unwrap();
    write_sfmr_with_options(&path_b, &mut altered, &options).unwrap();

    // Every section hash agrees, so every entry holds the same bytes.
    let a = read_sfmr(&path_a).unwrap();
    let b = read_sfmr(&path_b).unwrap();
    assert_eq!(
        serde_json::to_value(&b.content_hash).unwrap(),
        serde_json::to_value(&a.content_hash).unwrap()
    );
    assert!(b.content_hash.rigs_xxh128.is_some());
    let (valid, errors) = verify_sfmr(&path_b).unwrap();
    assert!(valid, "Verification failed: {errors:?}");

    // And it reads back to what the standard-layout value wrote.
    assert_eq!(b.camera_indexes, a.camera_indexes);
    assert_eq!(b.quaternions_wxyz, a.quaternions_wxyz);
    assert_eq!(b.translations_xyz, a.translations_xyz);
    assert_eq!(b.thumbnails_y_x_rgb, a.thumbnails_y_x_rgb);
    assert_eq!(b.positions_xyzw, a.positions_xyzw);
    assert_eq!(b.colors_rgb, a.colors_rgb);
    assert_eq!(b.reprojection_errors, a.reprojection_errors);
    assert_eq!(b.normals_xyz, a.normals_xyz);
    assert_eq!(b.normal_confidence, a.normal_confidence);
    assert_eq!(b.point_constraints, a.point_constraints);
    assert_eq!(b.constraint_reference_images, a.constraint_reference_images);
    assert_eq!(b.patch_u_halfvec_xyz, a.patch_u_halfvec_xyz);
    assert_eq!(b.patch_v_halfvec_xyz, a.patch_v_halfvec_xyz);
    assert_eq!(b.patch_bitmaps_y_x_rgba, a.patch_bitmaps_y_x_rgba);
    assert_eq!(b.image_indexes, a.image_indexes);
    assert_eq!(b.feature_indexes, a.feature_indexes);
    assert_eq!(b.keypoints_xy, a.keypoints_xy);
    assert_eq!(b.observation_confidence, a.observation_confidence);
    assert_eq!(b.point_indexes, a.point_indexes);
    assert_eq!(b.observation_counts, a.observation_counts);
    assert_eq!(b.reference_observations, a.reference_observations);
    let (rb, ra) = (
        b.rig_frame_data.as_ref().unwrap(),
        a.rig_frame_data.as_ref().unwrap(),
    );
    assert_eq!(rb.sensor_quaternions_wxyz, ra.sensor_quaternions_wxyz);
    assert_eq!(rb.sensor_translations_xyz, ra.sensor_translations_xyz);
    assert_eq!(rb.image_sensor_indexes, ra.image_sensor_indexes);
    assert_eq!(rb.image_frame_indexes, ra.image_frame_indexes);
    // The sort ran: the tracks were handed in reversed.
    assert_eq!(
        a.point_indexes,
        Array1::from_vec(vec![0, 0, 1, 1, 1, 2, 3, 4])
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
