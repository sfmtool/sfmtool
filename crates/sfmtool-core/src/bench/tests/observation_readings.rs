// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The commit writes each observation's readings from its evaluation, a
//! committed point put on the bench reads them back, and a writer with the
//! photographs reads the same rows again from the committed reconstruction.

use super::*;
use crate::bench::commit::measurement_reading;
use crate::patch::observation_reading::read_cloud_observations;
use crate::patch::PatchCloud;
use crate::reconstruction::ObservationReading;

/// The fixture evaluated and committed: the committed version, the point it
/// wrote and the evaluated track it wrote from.
fn evaluated_and_committed(scene: &Scene) -> (EditedReconstruction, u32, EditableTrack) {
    let edited = edited_with_columns(scene, WORLD);
    let (bench, label) = bench_with_point(&edited, 0);
    let (read, _) = evaluate_over(scene, &edited, &track_of(&bench, &label)).expect("two in");
    let (next, report) = commit(&edited, &read).expect("two in, with a position");
    (next, report.point, read)
}

#[test]
fn a_commit_writes_each_rows_readings_from_its_evaluation() {
    let scene = Scene::new();
    let (next, point, read) = evaluated_and_committed(&scene);
    let written = next.point(point).expect("just written");
    let rows = written
        .observation_readings()
        .expect("a commit brings the readings to a base without them");
    let reference = read.track().and_then(|p| p.reference).expect("a reference");
    // The stored track is in image order, and here row k sees image k.
    for (k, observation) in read.observations.iter().enumerate() {
        let m = observation.track.as_ref().expect("a slot");
        let row = rows[k];
        assert_eq!(row, measurement_reading(m, true));
        let ellipse = m.zncc_self_similarity_ellipse.expect("read").grid_px;
        assert_eq!(row.ellipse_axes, ellipse.axes.map(|a| a as f32));
        assert_eq!(row.ellipse_axes_is_at_least, ellipse.axes_is_at_least);
        assert_eq!(
            row.plain_bitmap_zncc,
            m.plain_zncc.expect("scored") as f32,
            "row {k}"
        );
        assert_eq!(
            row.blur_matched_bitmap_zncc,
            m.blur_matched_zncc.expect("scored") as f32
        );
        let cos = f64::from(row.cos_view_angle);
        assert!(cos > 0.0 && cos <= 1.0, "{cos}");
        assert!((cos - m.viewing_angle_deg.expect("an angle").to_radians().cos()).abs() < 1e-6);
        let zoom = row.zoom;
        assert!(zoom[0] > 0.0 && zoom[0] <= zoom[1], "{zoom:?}");
        let tilt = row.tilt_angle;
        assert!(tilt.is_nan() || (0.0..std::f32::consts::PI).contains(&tilt));
        let angle = row.ellipse_major_angle;
        assert!(angle.is_nan() || (0.0..std::f32::consts::PI).contains(&angle));
    }
    assert_eq!(rows[reference].plain_bitmap_zncc, 1.0);
    assert_eq!(rows[reference].blur_matched_bitmap_zncc, 1.0);
}

#[test]
fn the_readings_are_float32_and_match_a_reading_of_the_committed_reconstruction() {
    // A writer with the photographs renders each observation as the stored
    // bitmap is and scores it with the bitmap scorer the bench uses, so it
    // reads the rows the commit wrote from the evaluation.
    let scene = Scene::new();
    let (next, point, _) = evaluated_and_committed(&scene);
    let (recon, map) = next.materialize();
    let p = map.forward(point).expect("kept") as usize;
    let cloud = PatchCloud::from_stored_frames(&recon).expect("a patch frame");
    let views: Vec<_> = scene.views().into_iter().map(Some).collect();
    let rows = read_cloud_observations(
        &cloud,
        &recon,
        &views,
        BITMAP_R,
        crate::camera::sampler::SamplerChoice::per_view(),
        None,
        &Progress::none(),
    )
    .expect("not cancelled");
    let stored = &recon
        .point_set
        .observation_readings
        .as_ref()
        .expect("stored")
        .rows;
    let run = recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1];
    for j in run {
        // The committed frame is stored in `f32`, so the render differs from
        // the evaluation's by the rounding of the frame.
        let (a, b) = (rows.rows[j], stored[j]);
        assert_eq!(a.ellipse_axes_is_at_least, b.ellipse_axes_is_at_least);
        for (x, y) in [
            (a.ellipse_axes[0], b.ellipse_axes[0]),
            (a.ellipse_axes[1], b.ellipse_axes[1]),
            (a.plain_bitmap_zncc, b.plain_bitmap_zncc),
            (a.blur_matched_bitmap_zncc, b.blur_matched_bitmap_zncc),
            (a.cos_view_angle, b.cos_view_angle),
            (a.zoom[0], b.zoom[0]),
            (a.zoom[1], b.zoom[1]),
        ] {
            assert!((x - y).abs() < 1e-4, "row {j}: {x} against {y}");
        }
    }
}

#[test]
fn a_committed_point_put_on_the_bench_reads_its_stored_readings_back() {
    let scene = Scene::new();
    let (next, point, read) = evaluated_and_committed(&scene);
    let (bench, label) = bench_with_point(&next, point);
    let track = track_of(&bench, &label);
    let written = next.point(point).expect("just written");
    let rows = written.observation_readings().expect("stored");
    for (k, observation) in track.observations.iter().enumerate() {
        let m = observation.track.as_ref().expect("a slot");
        let evaluated = read.observations[k].track.as_ref().expect("a slot");
        assert_eq!(
            m.plain_zncc.map(|z| z as f32),
            Some(rows[k].plain_bitmap_zncc),
            "the float32 score, not the byte"
        );
        assert_eq!(
            m.zncc_self_similarity_radius.map(|r| r as f32),
            Some(rows[k].ellipse_axes[0])
        );
        let angle = m.viewing_angle_deg.expect("read back");
        assert!((angle - evaluated.viewing_angle_deg.expect("evaluated")).abs() < 1e-3);
        // Nothing an evaluation alone reads is carried, so the bars judge no
        // row until the first evaluation.
        assert_eq!(m.seed_shift_px, None);
        // Committing the row as read back writes the row it was read from.
        assert_eq!(measurement_reading(m, true), rows[k]);
    }
    let (again, report) = commit(&next, &track).expect("unchanged");
    assert!(
        !report.changed,
        "the read-back readings commit as they stand"
    );
    let _ = again;
}

/// The options the bench's evaluation reads a row under at `resolution`.
fn bench_options(resolution: usize) -> crate::reconstruction::ObservationReadingOptions {
    crate::reconstruction::observation_reading_options(
        EvaluateOptions::default().localize.sampler,
        resolution,
        crate::patch::member_coherence::MemberCoherenceParams::default().window,
    )
}

#[test]
fn a_commit_records_the_options_its_rows_were_read_under() {
    let scene = Scene::new();
    let (next, point, _) = evaluated_and_committed(&scene);
    let written = next.point(point).expect("just written");
    let options = written.observation_reading_options().expect("stored");
    assert_eq!(options, bench_options(BITMAP_R));
    assert_eq!(options.resolution, BITMAP_R as u32);
    assert_eq!(
        options.sampler,
        crate::reconstruction::ReadingSampler::PerView
    );
}

#[test]
fn a_reading_carries_the_rows_of_a_point_it_does_not_read() {
    // A point with no patch in the cloud keeps the rows the value stores,
    // where they stand under the same options; under other options it is not
    // measured.
    let scene = Scene::new();
    let (next, point, _) = evaluated_and_committed(&scene);
    let (recon, map) = next.materialize();
    let p = map.forward(point).expect("kept") as usize;
    let mut cloud = PatchCloud::from_stored_frames(&recon).expect("a patch frame");
    let at = cloud
        .point_indexes
        .iter()
        .position(|&q| q as usize == p)
        .expect("the point has a patch");
    cloud.patches.remove(at);
    cloud.point_indexes.remove(at);
    let views: Vec<_> = scene.views().into_iter().map(Some).collect();
    let read = |resolution: usize| {
        read_cloud_observations(
            &cloud,
            &recon,
            &views,
            resolution,
            crate::camera::sampler::SamplerChoice::per_view(),
            None,
            &Progress::none(),
        )
        .expect("not cancelled")
    };
    let stored = recon
        .point_set
        .observation_readings
        .as_ref()
        .expect("stored");
    let run = recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1];
    let same = read(BITMAP_R);
    assert_eq!(&same.rows[run.clone()], &stored.rows[run.clone()]);
    let other = read(16);
    assert_eq!(other.options.resolution, 16);
    assert!(other.rows[run].iter().all(|r| !r.is_measured()));
}

#[test]
fn a_base_with_readings_at_another_resolution_gets_unmeasured_rows() {
    let scene = Scene::new();
    let mut recon = fixture_with_columns(&scene, WORLD, BITMAP_R);
    let options = bench_options(BITMAP_R / 2);
    recon.point_set.observation_readings =
        Some(crate::reconstruction::ObservationReadings::not_measured(
            recon.point_set.tracks.len(),
            options,
        ));
    let edited = EditedReconstruction::new(Arc::new(recon));
    let (bench, label) = bench_with_point(&edited, 0);
    let (read, _) = evaluate_over(&scene, &edited, &track_of(&bench, &label)).expect("two in");
    let (next, report) = commit(&edited, &read).expect("two in, with a position");
    let rows = next
        .point(report.point)
        .expect("written")
        .observation_readings()
        .expect("stored");
    assert!(rows.iter().all(|r| *r == ObservationReading::NOT_MEASURED));
}
