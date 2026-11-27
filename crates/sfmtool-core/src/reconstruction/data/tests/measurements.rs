// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for `ImageSet::measurements` and `SfmrReconstruction` distance
//! measurement round-tripping, hashing, transforms, subsetting and overlay
//! materialization.

use ndarray::{array, Array1, Array2, Array3};

use crate::geometry::{RotQuaternion, Se3Transform};
use crate::progress::Progress;
use crate::reconstruction::EditedReconstruction;
use crate::{
    DistanceMeasurementsData, SfmrReconstruction, LENGTH_UNIT_CM, LENGTH_UNIT_FT, LENGTH_UNIT_M,
    LENGTH_UNIT_MM,
};

fn sample_measurements() -> DistanceMeasurementsData {
    // Demo reconstruction has 8 images with 1920x1080 camera.
    DistanceMeasurementsData {
        distances: array![0.42_f32, 12.5_f32, 3.0_f32],
        units: array![LENGTH_UNIT_M, LENGTH_UNIT_CM, LENGTH_UNIT_FT],
        image_indexes: array![[0_u32, 3_u32], [2_u32, 2_u32], [5_u32, 7_u32]],
        endpoints_xy: array![
            [[10.5_f32, 20.25_f32], [100.0_f32, 200.5_f32]],
            [[300.0_f32, 400.0_f32], [500.5_f32, 600.25_f32]],
            [[50.0_f32, 60.0_f32], [70.0_f32, 80.0_f32]]
        ],
        labels: vec!["ruler edge".into(), "shelf".into(), "door — height".into()],
    }
}

#[test]
fn measurements_round_trip_through_sfmr_data_and_disk() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".sfm-workspace.json"), "{}").unwrap();
    let path = dir.path().join("with_measurements.sfmr");

    let mut recon = SfmrReconstruction::demo(12);
    recon.workspace_dir = dir.path().to_path_buf();
    assert_eq!(recon.distance_measurement_count(), 0);
    assert!(recon.measurements().is_empty());

    let m = sample_measurements();
    recon.image_set.measurements.distance = Some(m.clone());
    assert_eq!(recon.distance_measurement_count(), 3);
    assert_eq!(recon.measurements().distance.as_ref(), Some(&m));

    // SfmrData round-trip preserves measurements and syncs metadata count.
    let sfmr_data = recon.to_sfmr_data();
    assert_eq!(sfmr_data.metadata.distance_measurement_count, Some(3));
    assert_eq!(sfmr_data.measurements.distance.as_ref(), Some(&m));
    let rebuilt = SfmrReconstruction::from_sfmr_data(sfmr_data).unwrap();
    assert_eq!(rebuilt.distance_measurement_count(), 3);
    assert_eq!(rebuilt.measurements().distance.as_ref(), Some(&m));
    assert_eq!(rebuilt.metadata.distance_measurement_count, Some(3));

    // Disk save / load round-trip preserves measurements and metadata count.
    recon.save(&path, &Progress::none()).unwrap();
    let loaded = SfmrReconstruction::load(&path, &Progress::none()).unwrap();
    assert_eq!(loaded.distance_measurement_count(), 3);
    assert_eq!(loaded.measurements().distance.as_ref(), Some(&m));
    assert_eq!(loaded.metadata.distance_measurement_count, Some(3));
    assert!(loaded.content_hash.measurements_xxh128.is_some());
}

#[test]
fn from_sfmr_data_validates_measurements() {
    let base = SfmrReconstruction::demo(8);

    // Out-of-range image index.
    let mut data = base.to_sfmr_data();
    data.measurements.distance = Some(DistanceMeasurementsData {
        distances: array![1.0_f32],
        units: array![LENGTH_UNIT_M],
        image_indexes: array![[0_u32, base.image_count() as u32]],
        endpoints_xy: array![[[10.0_f32, 10.0_f32], [20.0_f32, 20.0_f32]]],
        labels: vec!["bad img".into()],
    });
    assert!(SfmrReconstruction::from_sfmr_data(data).is_err());

    // Endpoint outside camera bounds.
    let mut data = base.to_sfmr_data();
    let w = base.image_set.cameras[0].width as f32;
    data.measurements.distance = Some(DistanceMeasurementsData {
        distances: array![1.0_f32],
        units: array![LENGTH_UNIT_MM],
        image_indexes: array![[0_u32, 1_u32]],
        endpoints_xy: array![[[w, 10.0_f32], [20.0_f32, 20.0_f32]]],
        labels: vec!["oob".into()],
    });
    assert!(SfmrReconstruction::from_sfmr_data(data).is_err());

    // Empty measurements section (must be None instead).
    let mut data = base.to_sfmr_data();
    data.measurements.distance = Some(DistanceMeasurementsData {
        distances: Array1::zeros(0),
        units: Array1::zeros(0),
        image_indexes: Array2::zeros((0, 2)),
        endpoints_xy: Array3::zeros((0, 2, 2)),
        labels: Vec::new(),
    });
    assert!(SfmrReconstruction::from_sfmr_data(data).is_err());
}

#[test]
fn content_xxh128_changes_when_measurements_change() {
    let base = SfmrReconstruction::demo(10);
    let h_none = base.content_xxh128().unwrap();
    assert!(h_none.measurements_xxh128.is_none());

    let mut with_m = base.clone();
    with_m.image_set.measurements.distance = Some(sample_measurements());
    let h_with = with_m.content_xxh128().unwrap();
    assert!(h_with.measurements_xxh128.is_some());
    assert_ne!(h_with.content_xxh128, h_none.content_xxh128);

    let mut edited = with_m.clone();
    edited
        .image_set
        .measurements
        .distance
        .as_mut()
        .unwrap()
        .distances[0] = 0.43;
    let h_edited = edited.content_xxh128().unwrap();
    assert_ne!(h_edited.measurements_xxh128, h_with.measurements_xxh128);
    assert_ne!(h_edited.content_xxh128, h_with.content_xxh128);
}

#[test]
fn apply_se3_transform_preserves_measurements_unchanged() {
    let mut recon = SfmrReconstruction::demo(10);
    let m = sample_measurements();
    recon.image_set.measurements.distance = Some(m.clone());

    let xform = Se3Transform::new(
        RotQuaternion::from_axis_angle(nalgebra::Vector3::z(), 0.7).unwrap(),
        nalgebra::Vector3::new(10.0, -5.0, 2.5),
        3.25,
    );
    let moved = recon.apply_se3_transform(&xform);
    assert_eq!(moved.measurements().distance.as_ref(), Some(&m));
    assert_eq!(moved.distance_measurement_count(), 3);
}

#[test]
fn subset_by_image_indices_filters_and_remaps_measurements() {
    let mut recon = SfmrReconstruction::demo(12);
    recon.image_set.measurements.distance = Some(sample_measurements());
    recon.metadata.distance_measurement_count = Some(3);

    // Keep images [3, 2, 0] (reordered):
    // - row 0: [0, 3] -> [2, 0] (kept)
    // - row 1: [2, 2] -> [1, 1] (kept, same-image pair)
    // - row 2: [5, 7] -> dropped
    let subset = recon.subset_by_image_indices(&[3, 2, 0], false).unwrap();
    assert_eq!(subset.distance_measurement_count(), 2);
    assert_eq!(subset.metadata.distance_measurement_count, Some(2));
    let m = subset
        .measurements()
        .distance
        .as_ref()
        .expect("two measurements survive");
    assert_eq!(m.distances, array![0.42_f32, 12.5_f32]);
    assert_eq!(m.units, array![LENGTH_UNIT_M, LENGTH_UNIT_CM]);
    assert_eq!(m.image_indexes, array![[2_u32, 0_u32], [1_u32, 1_u32]]);
    assert_eq!(
        m.endpoints_xy,
        array![
            [[10.5_f32, 20.25_f32], [100.0_f32, 200.5_f32]],
            [[300.0_f32, 400.0_f32], [500.5_f32, 600.25_f32]]
        ]
    );
    assert_eq!(
        m.labels,
        vec!["ruler edge".to_string(), "shelf".to_string()]
    );

    // Keep only image [1]: no measurement references image 1 -> None.
    let empty_m = recon.subset_by_image_indices(&[1], false).unwrap();
    assert_eq!(empty_m.distance_measurement_count(), 0);
    assert!(empty_m.measurements().is_empty());
    assert_eq!(empty_m.metadata.distance_measurement_count, None);

    // Keep images [0, 5]: row 0 needs 3 (missing), row 2 needs 7 (missing) -> None.
    let partial_miss = recon.subset_by_image_indices(&[0, 5], false).unwrap();
    assert!(partial_miss.measurements().is_empty());
    assert_eq!(partial_miss.metadata.distance_measurement_count, None);
}

#[test]
fn filter_points_and_edited_materialize_preserve_measurements() {
    let mut recon = SfmrReconstruction::demo(10);
    let m = sample_measurements();
    recon.image_set.measurements.distance = Some(m.clone());
    recon.metadata.distance_measurement_count = Some(3);

    // Point filter leaves images and measurements untouched.
    let mut mask = vec![true; recon.point_count()];
    mask[0] = false;
    let filtered = recon.filter_points_by_mask(&mask);
    assert_eq!(filtered.measurements().distance.as_ref(), Some(&m));
    assert_eq!(filtered.distance_measurement_count(), 3);

    // EditedReconstruction::materialize preserves measurements and metadata count.
    let mut edited = EditedReconstruction::new(std::sync::Arc::new(recon));
    edited.delete_point(0).unwrap();
    let (materialized, _) = edited.materialize();
    assert_eq!(materialized.measurements().distance.as_ref(), Some(&m));
    assert_eq!(materialized.distance_measurement_count(), 3);
    assert_eq!(materialized.metadata.distance_measurement_count, Some(3));
}
