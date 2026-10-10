// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Each observation's stored readings travel with the observation through
//! every pass that drops, reorders or moves it, and through the `.sfmr`
//! boundary.

use super::*;
use crate::camera::sampler::SamplerChoice;
use crate::reconstruction::edited::EditedReconstruction;
use crate::reconstruction::{observation_reading_options, ObservationReading, ObservationReadings};

/// A row that names observation `j`: every value derived from `j`, so a row
/// moved to the wrong observation shows up as the wrong value.
fn tagged(j: usize) -> ObservationReading {
    let t = j as f32 + 1.0;
    ObservationReading {
        ellipse_axes: [t, t / 2.0],
        ellipse_axes_is_at_least: [j.is_multiple_of(2), false],
        ellipse_major_angle: 0.01 * t,
        cos_view_angle: 0.5,
        tilt_angle: f32::NAN,
        zoom: [0.25, 0.5],
        plain_bitmap_zncc: 0.001 * t,
        blur_matched_bitmap_zncc: 0.002 * t,
    }
}

/// `demo_embedded` with every observation's row tagged by its index.
fn demo_with_readings(num_points: usize) -> SfmrReconstruction {
    let mut recon = demo_embedded(num_points);
    let m = recon.point_set.tracks.len();
    recon.point_set.observation_readings = Some(ObservationReadings {
        rows: (0..m).map(tagged).collect(),
        options: observation_reading_options(SamplerChoice::per_view()),
    });
    recon.validate_observation_columns().unwrap();
    recon
}

fn rows(recon: &SfmrReconstruction) -> &[ObservationReading] {
    &recon
        .point_set
        .observation_readings
        .as_ref()
        .expect("the column survives")
        .rows
}

#[test]
fn the_readings_round_trip_through_sfmr_data() {
    let recon = demo_with_readings(4);
    let data = recon.to_sfmr_data();
    assert_eq!(
        data.observation_readings.as_ref().expect("carried").len(),
        recon.point_set.tracks.len()
    );
    let back = SfmrReconstruction::from_sfmr_data(data).unwrap();
    assert_eq!(
        back.point_set.observation_readings,
        recon.point_set.observation_readings
    );
}

#[test]
fn a_point_filter_keeps_each_row_with_its_observation() {
    // demo observes each point by 2 cameras, so point i owns rows 2i and 2i+1.
    let recon = demo_with_readings(4);
    let out = recon.filter_points_by_mask(&[true, false, true, false]);
    let expected: Vec<ObservationReading> = [0, 1, 4, 5].into_iter().map(tagged).collect();
    assert_eq!(rows(&out), expected.as_slice());
    out.validate_observation_columns().unwrap();
}

#[test]
fn an_image_subset_keeps_each_row_with_its_observation() {
    let recon = demo_with_readings(4);
    let keep_image = recon.point_set.tracks[0].image_index;
    let expected: Vec<ObservationReading> = recon
        .point_set
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, o)| o.image_index == keep_image)
        .map(|(j, _)| tagged(j))
        .collect();
    let out = recon.subset_by_image_indices(&[keep_image], true).unwrap();
    assert_eq!(rows(&out), expected.as_slice());
}

#[test]
fn a_similarity_carries_the_rows_unchanged() {
    use crate::geometry::RotQuaternion;
    use crate::Se3Transform;
    let recon = demo_with_readings(4);
    let t = Se3Transform::new(
        RotQuaternion::from_nalgebra(nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::z_axis(),
            0.7,
        )),
        nalgebra::Vector3::new(1.0, -2.0, 0.5),
        2.0,
    );
    let out = recon.apply_se3_transform(&t);
    assert_eq!(
        out.point_set.observation_readings,
        recon.point_set.observation_readings
    );
}

#[test]
fn an_edit_that_rewrites_a_point_moves_its_rows_with_it() {
    // Replacing point 0 rewrites its observations through the addition set
    // into the slot it held, and deleting point 1 drops its rows; every other
    // row keeps its observation.
    let recon = demo_with_readings(4);
    let mut edited = EditedReconstruction::new(std::sync::Arc::new(recon.clone()));
    let record = edited.point(0).unwrap().to_record();
    edited.replace_point(0, record).unwrap();
    edited.delete_point(1).unwrap();
    let (out, _) = edited.materialize();
    let expected: Vec<ObservationReading> = [0, 1, 4, 5, 6, 7].into_iter().map(tagged).collect();
    assert_eq!(rows(&out), expected.as_slice());
    out.validate_observation_columns().unwrap();
}

#[test]
fn an_edit_brings_readings_to_a_base_without_them() {
    // A record with a measured row adds the column; the base's rows are not
    // measured.
    let recon = demo_embedded(3);
    let mut edited = EditedReconstruction::new(std::sync::Arc::new(recon));
    let mut record = edited.point(0).unwrap().to_record();
    for (k, o) in record.observations.iter_mut().enumerate() {
        o.reading = Some(tagged(k));
    }
    edited.replace_point(0, record).unwrap();
    let (out, _) = edited.materialize();
    let r = rows(&out);
    assert_eq!(r.len(), out.point_set.tracks.len());
    // The rewritten point keeps its slot, the first.
    assert_eq!(&r[..2], &[tagged(0), tagged(1)]);
    assert!(r[2..].iter().all(|row| !row.is_measured()));
}

#[test]
fn clearing_a_points_scores_keeps_the_rest_of_its_rows() {
    let mut recon = demo_with_readings(3);
    recon.point_set.clear_observation_scores(1);
    let r = rows(&recon);
    for (j, row) in r.iter().enumerate() {
        if (2..4).contains(&j) {
            assert!(row.plain_bitmap_zncc.is_nan() && row.blur_matched_bitmap_zncc.is_nan());
            assert_eq!(row.ellipse_axes, tagged(j).ellipse_axes);
        } else {
            assert_eq!(*row, tagged(j));
        }
    }
}

#[test]
fn a_readings_desync_is_detected() {
    let mut recon = demo_with_readings(3);
    recon
        .point_set
        .observation_readings
        .as_mut()
        .unwrap()
        .rows
        .pop();
    let err = recon.validate_observation_columns().unwrap_err();
    assert!(err.contains("observation_readings"), "{err}");
}
