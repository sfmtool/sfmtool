// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Each column on small reconstructions built for it, the query's filter and
//! sort rules, cancellation, progress and thread-count independence.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use nalgebra::Point3;

use super::*;
use crate::camera::sampler::{Sampler, SamplerChoice};
use crate::patch::normal_refine::PatchWindow;
use crate::progress::Event;
use crate::reconstruction::{
    observation_reading_options, ImageSet, ObservationReading, ObservationReadings,
    SfmrReconstruction,
};

mod observation_quality;

/// The demo reconstruction of `n` points, each seen by two adjacent cameras
/// of an arc of eight.
fn demo(n: usize) -> SfmrReconstruction {
    SfmrReconstruction::demo(n)
}

fn edited(recon: SfmrReconstruction) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(recon))
}

fn compute(recon: &EditedReconstruction, columns: &[TrackTableColumn]) -> TrackTable {
    TrackTable::compute(recon, columns, &Progress::none()).expect("not cancelled")
}

/// `recon` with point `index` made a point at infinity along its direction
/// from the origin.
fn with_bearing(mut recon: SfmrReconstruction, index: usize) -> SfmrReconstruction {
    let p = &mut recon.point_set.points[index];
    p.position = Point3::from(p.position.coords.normalize());
    p.w = 0.0;
    recon.rebuild_derived_fields();
    recon
}

fn assert_same(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.to_bits(), y.to_bits(), "{x} != {y}");
    }
}

#[test]
fn wire_names_are_unique_and_round_trip() {
    let mut seen = std::collections::HashSet::new();
    for column in TrackTableColumn::ALL {
        assert!(seen.insert(column.wire_name()), "{column:?} repeats a name");
        assert_eq!(
            TrackTableColumn::from_wire_name(column.wire_name()),
            Some(column)
        );
    }
    assert_eq!(TrackTableColumn::from_wire_name("zncc"), None);
    assert_eq!(TrackTableColumn::MinZncc.wire_name(), "min_zncc");
    assert_eq!(
        TrackTableColumn::MaxZnccSelfSimilarityRadius.wire_name(),
        "max_zncc_self_similarity_radius"
    );
}

#[test]
fn the_computed_columns_are_the_ones_that_project_triangulate_or_read() {
    let computed: Vec<TrackTableColumn> = TrackTableColumn::ALL
        .into_iter()
        .filter(|c| c.cost() == ColumnCost::Computed)
        .collect();
    assert_eq!(
        computed,
        [
            TrackTableColumn::BackFacingObservations,
            TrackTableColumn::MaxErrorPx,
            TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
            TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
            TrackTableColumn::MaxRayAngleDeg,
            TrackTableColumn::MaxViewingAngleDeg,
            TrackTableColumn::ConditionNumber,
            TrackTableColumn::InverseDepthZ
        ]
    );
    // The enum's order is ALL's, which the slots are indexed by.
    for (k, column) in TrackTableColumn::ALL.into_iter().enumerate() {
        assert_eq!(column.ordinal(), k, "{column:?}");
    }
    let f64_columns: Vec<TrackTableColumn> = TrackTableColumn::ALL
        .into_iter()
        .filter(|c| c.is_f64())
        .collect();
    assert_eq!(
        f64_columns,
        [
            TrackTableColumn::X,
            TrackTableColumn::Y,
            TrackTableColumn::Z
        ]
    );
}

#[test]
fn stored_columns_read_the_stored_point() {
    let recon = edited(demo(20));
    let table = compute(&recon, &TrackTableColumn::ALL);
    assert_eq!(table.point_indexes(), (0..20).collect::<Vec<u32>>());
    let centroid = recon.base.image_set.camera_centroid();
    for (row, &index) in table.point_indexes().iter().enumerate() {
        let point = recon.point(index).unwrap().point().clone();
        let v = |c| table.values(c).unwrap()[row];
        let v64 = |c| table.values_f64(c).unwrap()[row];
        assert_eq!(v(TrackTableColumn::TrackLength), 2.0);
        assert_eq!(v(TrackTableColumn::RepeatObservations), 0.0);
        assert_eq!(
            v(TrackTableColumn::RmsErrorPx),
            point.rms_reprojection_error
        );
        assert_eq!(v64(TrackTableColumn::X), point.position.x);
        assert_eq!(v64(TrackTableColumn::Y), point.position.y);
        assert_eq!(v64(TrackTableColumn::Z), point.position.z);
        assert_eq!(
            v(TrackTableColumn::CentroidDistance),
            ImageSet::placement_scale_from(&centroid, &point.position) as f32
        );
    }
    // A position column is read as f64 only, and an f32 column as f32 only.
    assert!(table.values(TrackTableColumn::X).is_none());
    assert!(table.values_f64(TrackTableColumn::TrackLength).is_none());
}

#[test]
fn repeat_observations_counts_the_observations_beyond_one_per_image() {
    let mut recon = demo(4);
    // Point 0's two observations both in its first image.
    let first = recon.point_set.tracks[0].image_index;
    recon.point_set.tracks[1].image_index = first;
    recon.rebuild_derived_fields();
    let recon = edited(recon);
    let table = compute(
        &recon,
        &[
            TrackTableColumn::TrackLength,
            TrackTableColumn::RepeatObservations,
        ],
    );
    let repeats = table.values(TrackTableColumn::RepeatObservations).unwrap();
    assert_eq!(table.values(TrackTableColumn::TrackLength).unwrap()[0], 2.0);
    assert_eq!(repeats[0], 1.0);
    assert_eq!(&repeats[1..], [0.0; 3]);
    let worst = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![ColumnBounds {
                column: TrackTableColumn::RepeatObservations,
                min: Some(1.0),
                max: None,
            }],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(worst, [0]);
}

#[test]
fn computed_columns_are_the_track_geometry_definitions() {
    let recon = edited(demo(30));
    let table = compute(
        &recon,
        &[
            TrackTableColumn::MaxRayAngleDeg,
            TrackTableColumn::ConditionNumber,
            TrackTableColumn::InverseDepthZ,
        ],
    );
    let image_set = &recon.base.image_set;
    for (row, &index) in table.point_indexes().iter().enumerate() {
        let view = recon.point(index).unwrap();
        let (cond, z) = compute_point_diagnostics(image_set, &view);
        let angle = table.values(TrackTableColumn::MaxRayAngleDeg).unwrap()[row];
        assert_eq!(angle, compute_max_ray_angle_deg(image_set, &view));
        assert!(angle > 1.0, "two cameras of the arc see point {index}");
        assert_eq!(
            table.values(TrackTableColumn::ConditionNumber).unwrap()[row],
            cond
        );
        assert_eq!(
            table.values(TrackTableColumn::InverseDepthZ).unwrap()[row],
            z
        );
        assert!(cond.is_finite() && cond >= 1.0);
    }
    // The stored columns were not asked for.
    assert!(!table.has(TrackTableColumn::TrackLength));
}

#[test]
fn the_max_ray_angle_is_the_widest_pair_of_rays() {
    let recon = edited(demo(5));
    let view = recon.point(3).unwrap();
    let image_set = &recon.base.image_set;
    let images = view.observations().iter().map(|o| o.image_index as usize);
    let rays = observation_rays(image_set, &view.point().position, false, images);
    assert_eq!(rays.len(), 2);
    let dot: f64 = (0..3).map(|k| rays[0][k] * rays[1][k]).sum();
    let expected = dot.clamp(-1.0, 1.0).acos().to_degrees() as f32;
    assert!((compute_max_ray_angle_deg(image_set, &view) - expected).abs() < 1e-5);
    assert_eq!(compute_max_pairwise_angle(&[]), 0.0);
    assert_eq!(compute_max_pairwise_angle(&[[0.0, 0.0, 1.0]]), 0.0);
}

#[test]
fn a_point_at_infinity_has_no_position_an_infinite_distance_and_no_triangulation() {
    let recon = edited(with_bearing(demo(10), 4));
    let table = compute(&recon, &TrackTableColumn::ALL);
    let v = |c, row: usize| table.values(c).unwrap()[row];
    let v64 = |c, row: usize| table.values_f64(c).unwrap()[row];
    assert_eq!(v(TrackTableColumn::TrackLength, 4), 2.0);
    assert_eq!(v(TrackTableColumn::RepeatObservations, 4), 0.0);
    assert_eq!(
        v(TrackTableColumn::RmsErrorPx, 4),
        recon.point(4).unwrap().point().rms_reprojection_error
    );
    for c in [
        TrackTableColumn::X,
        TrackTableColumn::Y,
        TrackTableColumn::Z,
    ] {
        assert!(v64(c, 4).is_nan(), "{c:?}");
    }
    assert_eq!(v(TrackTableColumn::CentroidDistance, 4), f32::INFINITY);
    // A bearing's rays are parallel.
    assert_eq!(v(TrackTableColumn::MaxRayAngleDeg, 4), 0.0);
    assert!(v(TrackTableColumn::ConditionNumber, 4).is_nan());
    assert!(v(TrackTableColumn::InverseDepthZ, 4).is_nan());
    // Its neighbours are finite and unaffected.
    assert!(v(TrackTableColumn::CentroidDistance, 3).is_finite());
    assert!(v64(TrackTableColumn::X, 3).is_finite());
    assert!(v(TrackTableColumn::ConditionNumber, 5).is_finite());
}

#[test]
fn absent_columns_are_skipped() {
    let recon = edited(demo(6));
    for c in [
        TrackTableColumn::NormalConfidence,
        TrackTableColumn::MinZncc,
        TrackTableColumn::MaxZnccSelfSimilarityRadius,
    ] {
        assert!(!c.is_available(&recon), "{c:?}");
    }
    assert!(TrackTableColumn::CentroidDistance.is_available(&recon));
    let table = compute(&recon, &TrackTableColumn::ALL);
    assert!(!table.has(TrackTableColumn::NormalConfidence));
    assert!(!table.has(TrackTableColumn::MinZncc));
    assert!(!table.has(TrackTableColumn::MaxZnccSelfSimilarityRadius));
    assert!(table.values(TrackTableColumn::MinZncc).is_none());
    assert!(table.has(TrackTableColumn::TrackLength));
}

#[test]
fn centroid_distance_is_absent_with_no_posed_image() {
    let mut recon = demo(6);
    for image in &mut recon.image_set.images {
        image.translation_xyz = nalgebra::Vector3::repeat(f64::NAN);
    }
    let recon = edited(recon);
    assert!(!TrackTableColumn::CentroidDistance.is_available(&recon));
    let table = compute(
        &recon,
        &[TrackTableColumn::CentroidDistance, TrackTableColumn::X],
    );
    assert!(!table.has(TrackTableColumn::CentroidDistance));
    assert!(table.has(TrackTableColumn::X));
}

#[test]
fn normal_confidence_is_the_stored_byte() {
    let mut recon = demo(3);
    recon.point_set.normal_confidence = Some(vec![0, 17, 255]);
    let recon = edited(recon);
    assert!(TrackTableColumn::NormalConfidence.is_available(&recon));
    let table = compute(&recon, &[TrackTableColumn::NormalConfidence]);
    assert_eq!(
        table.values(TrackTableColumn::NormalConfidence).unwrap(),
        [0.0, 17.0, 255.0]
    );
}

#[test]
fn min_zncc_skips_unmeasured_observations() {
    let mut recon = demo(4);
    // Two observations per point, in point order.
    recon.point_set.observation_confidence = Some(vec![0, 128, 0, 0, 200, 100, 255, 255]);
    let recon = edited(with_bearing(recon.clone(), 3));
    assert!(TrackTableColumn::MinZncc.is_available(&recon));
    let table = compute(&recon, &[TrackTableColumn::MinZncc]);
    let v = table.values(TrackTableColumn::MinZncc).unwrap();
    assert_eq!(v[0], (128.0f64 / 255.0) as f32);
    assert!(v[1].is_nan(), "no observation of point 1 was measured");
    assert_eq!(v[2], (100.0f64 / 255.0) as f32);
    // Defined at infinity.
    assert_eq!(v[3], 1.0);
}

#[test]
fn max_self_similarity_radius_skips_unmeasured_readings() {
    let mut recon = demo(3);
    let options = observation_reading_options(
        SamplerChoice::Fixed(Sampler::Bilinear),
        24,
        PatchWindow::Uniform,
    );
    let mut readings = ObservationReadings::not_measured(6, options);
    let measured = |axes: [f32; 2]| ObservationReading {
        ellipse_axes: axes,
        ..ObservationReading::NOT_MEASURED
    };
    readings.rows[1] = measured([1.5, 0.5]);
    readings.rows[4] = measured([2.0, 1.0]);
    readings.rows[5] = measured([3.0, 0.25]);
    recon.point_set.observation_readings = Some(readings);
    let recon = edited(recon);
    assert!(TrackTableColumn::MaxZnccSelfSimilarityRadius.is_available(&recon));
    let table = compute(&recon, &[TrackTableColumn::MaxZnccSelfSimilarityRadius]);
    let v = table
        .values(TrackTableColumn::MaxZnccSelfSimilarityRadius)
        .unwrap();
    assert_eq!(v[0], 1.5);
    assert!(v[1].is_nan(), "every reading of point 1 is unmeasured");
    assert_eq!(v[2], 3.0);
}

#[test]
fn deleted_points_have_no_row() {
    let mut recon = edited(demo(5));
    recon.delete_point(2).unwrap();
    let table = compute(&recon, &[TrackTableColumn::TrackLength]);
    assert_eq!(table.point_indexes(), [0, 1, 3, 4]);
    assert_eq!(
        table.values(TrackTableColumn::TrackLength).unwrap().len(),
        4
    );
}

#[test]
fn an_added_point_is_read_through_its_view() {
    let mut recon = edited(demo(5));
    recon.delete_point(2).unwrap();
    let added = recon
        .add_point(recon.point(0).expect("a live point").to_record())
        .unwrap();
    let table = compute(&recon, &TrackTableColumn::ALL);
    assert_eq!(table.point_indexes(), [0, 1, 3, 4, added]);
    // The added point is a copy of point 0, so every value is point 0's.
    for c in TrackTableColumn::ALL {
        if let Some(v) = table.values(c) {
            assert_eq!(v[4].to_bits(), v[0].to_bits(), "{c:?}");
        }
        if let Some(v) = table.values_f64(c) {
            assert_eq!(v[4].to_bits(), v[0].to_bits(), "{c:?}");
        }
    }
    let image = recon.point(0).unwrap().observations()[0].image_index;
    let seen = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            seen_in_image: Some(image),
            ..TrackTableQueryOptions::default()
        },
    )
    .unwrap();
    assert!(seen.contains(&0) && seen.contains(&added), "{seen:?}");
}

#[test]
fn a_cancelled_compute_returns_cancelled() {
    let recon = edited(demo(10));
    let flag = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    assert!(TrackTable::compute(&recon, &TrackTableColumn::ALL, &progress).is_err());
}

#[test]
fn progress_runs_to_the_end_through_both_passes() {
    let recon = edited(demo(300));
    let fractions = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Fraction { of_whole } = event {
            fractions.lock().unwrap().push(of_whole);
        }
    };
    let progress = Progress::to(&sink);
    TrackTable::compute(
        &recon,
        &[
            TrackTableColumn::TrackLength,
            TrackTableColumn::MaxRayAngleDeg,
        ],
        &progress,
    )
    .unwrap();
    let fractions = fractions.into_inner().unwrap();
    assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "{fractions:?}");
    assert!((fractions.last().unwrap() - 1.0).abs() < 1e-6);
    // The stored pass ends at its share, a tenth, before the computed pass
    // starts.
    assert!(fractions.iter().any(|f| (f - 0.1).abs() < 1e-6));
}

#[test]
fn values_do_not_depend_on_the_thread_count() {
    let recon = edited(with_bearing(demo(300), 7));
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute(&recon, &TrackTableColumn::ALL))
    };
    let (one, four) = (run(1), run(4));
    for c in TrackTableColumn::ALL {
        match (one.values(c), four.values(c)) {
            (Some(a), Some(b)) => assert_same(a, b),
            (None, None) => {}
            _ => panic!("{c:?} filled in one run only"),
        }
        if let (Some(a), Some(b)) = (one.values_f64(c), four.values_f64(c)) {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }
}

// ── Merging ──────────────────────────────────────────────────────────────

#[test]
fn merge_moves_the_missing_columns_without_copying() {
    let recon = edited(with_bearing(demo(20), 3));
    let mut table = compute(&recon, &[TrackTableColumn::TrackLength]);
    let other = compute(
        &recon,
        &[TrackTableColumn::MaxRayAngleDeg, TrackTableColumn::X],
    );
    let angle_ptr = other
        .values(TrackTableColumn::MaxRayAngleDeg)
        .unwrap()
        .as_ptr();
    let x_ptr = other.values_f64(TrackTableColumn::X).unwrap().as_ptr();
    table.merge(other).unwrap();

    let whole = compute(
        &recon,
        &[
            TrackTableColumn::TrackLength,
            TrackTableColumn::MaxRayAngleDeg,
            TrackTableColumn::X,
        ],
    );
    for c in TrackTableColumn::ALL {
        assert_eq!(table.has(c), whole.has(c), "{c:?}");
    }
    assert_same(
        table.values(TrackTableColumn::MaxRayAngleDeg).unwrap(),
        whole.values(TrackTableColumn::MaxRayAngleDeg).unwrap(),
    );
    // The arrays were moved: they are the ones `other` held.
    assert_eq!(
        table
            .values(TrackTableColumn::MaxRayAngleDeg)
            .unwrap()
            .as_ptr(),
        angle_ptr
    );
    assert_eq!(
        table.values_f64(TrackTableColumn::X).unwrap().as_ptr(),
        x_ptr
    );
    assert!(table.values_f64(TrackTableColumn::X).unwrap()[3].is_nan());
}

#[test]
fn merge_keeps_a_column_already_held() {
    let recon = edited(demo(10));
    let mut table = compute(&recon, &[TrackTableColumn::TrackLength]);
    let kept = table
        .values(TrackTableColumn::TrackLength)
        .unwrap()
        .as_ptr();
    let other = compute(
        &recon,
        &[TrackTableColumn::TrackLength, TrackTableColumn::RmsErrorPx],
    );
    table.merge(other).unwrap();
    assert_eq!(
        table
            .values(TrackTableColumn::TrackLength)
            .unwrap()
            .as_ptr(),
        kept
    );
    assert!(table.has(TrackTableColumn::RmsErrorPx));
}

#[test]
fn merge_refuses_a_table_of_another_version() {
    let mut recon = edited(demo(10));
    let mut table = compute(&recon, &[TrackTableColumn::TrackLength]);
    recon.delete_point(4).unwrap();
    let other = compute(&recon, &[TrackTableColumn::RmsErrorPx]);
    let err = table.merge(other).unwrap_err();
    assert_eq!(
        err,
        TrackTableMergeError {
            rows: 10,
            other_rows: 9
        }
    );
    assert!(!table.has(TrackTableColumn::RmsErrorPx));
    assert_eq!(table.point_indexes().len(), 10);
}

// ── The query ────────────────────────────────────────────────────────────

/// Ten points, point 7 at infinity (so no `x` and a `centroid_distance` of
/// `+∞`), with `track_length`, `x` and `centroid_distance` computed.
fn query_fixture() -> (EditedReconstruction, TrackTable) {
    let recon = edited(with_bearing(demo(10), 7));
    let table = compute(
        &recon,
        &[
            TrackTableColumn::TrackLength,
            TrackTableColumn::X,
            TrackTableColumn::CentroidDistance,
        ],
    );
    (recon, table)
}

fn bound(column: TrackTableColumn, min: Option<f64>, max: Option<f64>) -> ColumnBounds {
    ColumnBounds { column, min, max }
}

#[test]
fn no_options_keep_every_point_in_live_order() {
    let (recon, table) = query_fixture();
    let all = query_points(&recon, &table, &TrackTableQueryOptions::default()).unwrap();
    assert_eq!(all, (0..10).collect::<Vec<u32>>());
}

#[test]
fn bounds_are_inclusive_and_all_must_pass() {
    let (recon, table) = query_fixture();
    let xs = table.values_f64(TrackTableColumn::X).unwrap();
    let x2 = xs[2];
    let only_two = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![bound(TrackTableColumn::X, Some(x2), Some(x2))],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(only_two, [2]);
    let finite_with_track_2 = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![
                bound(TrackTableColumn::TrackLength, Some(2.0), None),
                bound(TrackTableColumn::CentroidDistance, None, Some(f64::MAX)),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(finite_with_track_2, [0, 1, 2, 3, 4, 5, 6, 8, 9]);
    let none = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![bound(TrackTableColumn::TrackLength, Some(3.0), None)],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(none.is_empty());
}

#[test]
fn no_value_fails_any_set_bound_and_an_open_bound_filters_nothing() {
    let (recon, table) = query_fixture();
    let wide = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![bound(TrackTableColumn::X, Some(f64::MIN), None)],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!wide.contains(&7), "the bearing has no x");
    assert_eq!(wide.len(), 9);
    let open = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![bound(TrackTableColumn::X, None, None)],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(open.len(), 10);
}

#[test]
fn seen_in_image_keeps_the_points_that_image_observes() {
    let (recon, table) = query_fixture();
    let seen = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            seen_in_image: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    let expected: Vec<u32> = recon
        .live_indexes()
        .filter(|&i| recon.track_image_indices(i).contains(&2))
        .collect();
    assert!(!expected.is_empty());
    assert_eq!(seen, expected);
}

#[test]
fn ties_keep_point_index_order_whichever_way_the_sort_runs() {
    let (recon, table) = query_fixture();
    for order in [SortOrder::Ascending, SortOrder::Descending] {
        let sorted = query_points(
            &recon,
            &table,
            &TrackTableQueryOptions {
                order_by: Some((TrackTableColumn::TrackLength, order)),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sorted, (0..10).collect::<Vec<u32>>(), "{order:?}");
    }
}

#[test]
fn infinity_is_a_value_a_min_keeps_and_a_finite_max_drops() {
    let (recon, table) = query_fixture();
    let query = |min: Option<f64>, max: Option<f64>| {
        query_points(
            &recon,
            &table,
            &TrackTableQueryOptions {
                bounds: vec![bound(TrackTableColumn::CentroidDistance, min, max)],
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(query(Some(f64::INFINITY), None), [7]);
    assert!(query(Some(0.0), None).contains(&7));
    assert!(!query(None, Some(f64::MAX)).contains(&7));
    assert_eq!(query(None, Some(f64::MAX)).len(), 9);
    // An infinite max is no upper end: it keeps the point at infinity.
    assert_eq!(query(None, Some(f64::INFINITY)).len(), 10);
}

#[test]
fn infinity_sorts_as_the_largest_value_and_before_no_value() {
    // Four rows: a distance of 1, `+∞`, no value and 0.5.
    let recon = edited(demo(4));
    let mut table = compute(&recon, &[]);
    table.values_f32[TrackTableColumn::CentroidDistance.ordinal()] =
        Some(vec![1.0, f32::INFINITY, f32::NAN, 0.5]);
    let sorted = |order| {
        query_points(
            &recon,
            &table,
            &TrackTableQueryOptions {
                order_by: Some((TrackTableColumn::CentroidDistance, order)),
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(sorted(SortOrder::Ascending), [3, 0, 1, 2]);
    assert_eq!(sorted(SortOrder::Descending), [1, 0, 3, 2]);
}

#[test]
fn no_value_sorts_last_in_both_orders() {
    let (recon, table) = query_fixture();
    let xs = table.values_f64(TrackTableColumn::X).unwrap();
    for order in [SortOrder::Ascending, SortOrder::Descending] {
        let sorted = query_points(
            &recon,
            &table,
            &TrackTableQueryOptions {
                order_by: Some((TrackTableColumn::X, order)),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sorted.len(), 10);
        assert_eq!(*sorted.last().unwrap(), 7, "{order:?}");
        let values: Vec<f64> = sorted[..9].iter().map(|&i| xs[i as usize]).collect();
        let ordered = values.windows(2).all(|w| match order {
            SortOrder::Ascending => w[0] <= w[1],
            SortOrder::Descending => w[0] >= w[1],
        });
        assert!(ordered, "{order:?}: {values:?}");
    }
}

#[test]
fn a_query_on_a_column_not_computed_is_refused() {
    let (recon, table) = query_fixture();
    let by_bound = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![bound(TrackTableColumn::RmsErrorPx, Some(0.0), None)],
            ..Default::default()
        },
    );
    assert_eq!(
        by_bound,
        Err(TrackTableQueryError::ColumnNotComputed(
            TrackTableColumn::RmsErrorPx
        ))
    );
    let by_sort = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            order_by: Some((TrackTableColumn::MaxRayAngleDeg, SortOrder::Ascending)),
            ..Default::default()
        },
    );
    assert_eq!(
        by_sort,
        Err(TrackTableQueryError::ColumnNotComputed(
            TrackTableColumn::MaxRayAngleDeg
        ))
    );
}

#[test]
fn an_image_past_the_table_is_refused() {
    let (recon, table) = query_fixture();
    let refused = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            seen_in_image: Some(8),
            ..Default::default()
        },
    );
    assert_eq!(
        refused,
        Err(TrackTableQueryError::ImageOutOfRange {
            image: 8,
            image_count: 8
        })
    );
    assert!(refused.unwrap_err().to_string().contains("8 images"));
}
