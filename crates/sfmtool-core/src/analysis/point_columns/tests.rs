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
    observation_reading_options, ImageTable, ObservationReading, ObservationReadings,
    SfmrReconstruction,
};

/// The demo reconstruction of `n` points, each seen by two adjacent cameras
/// of an arc of eight.
fn demo(n: usize) -> SfmrReconstruction {
    SfmrReconstruction::demo(n)
}

fn edited(recon: SfmrReconstruction) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(recon))
}

fn compute(recon: &EditedReconstruction, columns: &[PointColumn]) -> PointColumns {
    PointColumns::compute(recon, columns, &Progress::none()).expect("not cancelled")
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
    for column in PointColumn::ALL {
        assert!(seen.insert(column.wire_name()), "{column:?} repeats a name");
        assert_eq!(
            PointColumn::from_wire_name(column.wire_name()),
            Some(column)
        );
    }
    assert_eq!(PointColumn::from_wire_name("min_zncc"), None);
    assert_eq!(
        PointColumn::LargestZnccSelfSimilarityRadius.wire_name(),
        "largest_zncc_self_similarity_radius"
    );
}

#[test]
fn only_the_ray_angle_and_the_diagnostics_are_computed() {
    let computed: Vec<PointColumn> = PointColumn::ALL
        .into_iter()
        .filter(|c| c.cost() == ColumnCost::Computed)
        .collect();
    assert_eq!(
        computed,
        [
            PointColumn::MaxRayAngleDeg,
            PointColumn::ConditionNumber,
            PointColumn::InverseDepthZ
        ]
    );
    let f64_columns: Vec<PointColumn> = PointColumn::ALL
        .into_iter()
        .filter(|c| c.is_f64())
        .collect();
    assert_eq!(
        f64_columns,
        [PointColumn::X, PointColumn::Y, PointColumn::Z]
    );
}

#[test]
fn stored_columns_read_the_stored_point() {
    let recon = edited(demo(20));
    let columns = compute(&recon, &PointColumn::ALL);
    assert_eq!(columns.point_indexes(), (0..20).collect::<Vec<u32>>());
    let centroid = recon.base.image_table.camera_centroid();
    for (row, &index) in columns.point_indexes().iter().enumerate() {
        let point = recon.point(index).unwrap().point().clone();
        let v = |c| columns.values(c).unwrap()[row];
        let v64 = |c| columns.values_f64(c).unwrap()[row];
        assert_eq!(v(PointColumn::TrackLength), 2.0);
        assert_eq!(v(PointColumn::ImageCount), 2.0);
        assert_eq!(v(PointColumn::RmsErrorPx), point.rms_reprojection_error);
        assert_eq!(v(PointColumn::AtInfinity), 0.0);
        assert_eq!(v64(PointColumn::X), point.position.x);
        assert_eq!(v64(PointColumn::Y), point.position.y);
        assert_eq!(v64(PointColumn::Z), point.position.z);
        assert_eq!(
            v(PointColumn::CentroidDistance),
            ImageTable::placement_scale_from(&centroid, &point.position) as f32
        );
    }
    // A position column is read as f64 only, and an f32 column as f32 only.
    assert!(columns.values(PointColumn::X).is_none());
    assert!(columns.values_f64(PointColumn::TrackLength).is_none());
}

#[test]
fn image_count_counts_an_image_once() {
    let mut recon = demo(4);
    // Point 0's two observations both in its first image.
    let first = recon.point_set.tracks[0].image_index;
    recon.point_set.tracks[1].image_index = first;
    recon.rebuild_derived_fields();
    let recon = edited(recon);
    let columns = compute(&recon, &[PointColumn::TrackLength, PointColumn::ImageCount]);
    assert_eq!(columns.values(PointColumn::TrackLength).unwrap()[0], 2.0);
    assert_eq!(columns.values(PointColumn::ImageCount).unwrap()[0], 1.0);
    assert_eq!(columns.values(PointColumn::ImageCount).unwrap()[1], 2.0);
}

#[test]
fn computed_columns_are_the_track_geometry_definitions() {
    let recon = edited(demo(30));
    let columns = compute(
        &recon,
        &[
            PointColumn::MaxRayAngleDeg,
            PointColumn::ConditionNumber,
            PointColumn::InverseDepthZ,
        ],
    );
    let table = &recon.base.image_table;
    for (row, &index) in columns.point_indexes().iter().enumerate() {
        let view = recon.point(index).unwrap();
        let (cond, z) = compute_point_diagnostics(table, &view);
        let angle = columns.values(PointColumn::MaxRayAngleDeg).unwrap()[row];
        assert_eq!(angle, compute_max_ray_angle_deg(table, &view));
        assert!(angle > 1.0, "two cameras of the arc see point {index}");
        assert_eq!(
            columns.values(PointColumn::ConditionNumber).unwrap()[row],
            cond
        );
        assert_eq!(columns.values(PointColumn::InverseDepthZ).unwrap()[row], z);
        assert!(cond.is_finite() && cond >= 1.0);
    }
    // The stored columns were not asked for.
    assert!(!columns.has(PointColumn::TrackLength));
}

#[test]
fn the_max_ray_angle_is_the_widest_pair_of_rays() {
    let recon = edited(demo(5));
    let view = recon.point(3).unwrap();
    let table = &recon.base.image_table;
    let images = view.observations().iter().map(|o| o.image_index as usize);
    let rays = observation_rays(table, &view.point().position, false, images);
    assert_eq!(rays.len(), 2);
    let dot: f64 = (0..3).map(|k| rays[0][k] * rays[1][k]).sum();
    let expected = dot.clamp(-1.0, 1.0).acos().to_degrees() as f32;
    assert!((compute_max_ray_angle_deg(table, &view) - expected).abs() < 1e-5);
    assert_eq!(compute_max_pairwise_angle(&[]), 0.0);
    assert_eq!(compute_max_pairwise_angle(&[[0.0, 0.0, 1.0]]), 0.0);
}

#[test]
fn a_point_at_infinity_has_no_position_and_no_triangulation() {
    let recon = edited(with_bearing(demo(10), 4));
    let columns = compute(&recon, &PointColumn::ALL);
    let v = |c, row: usize| columns.values(c).unwrap()[row];
    let v64 = |c, row: usize| columns.values_f64(c).unwrap()[row];
    assert_eq!(v(PointColumn::AtInfinity, 4), 1.0);
    assert_eq!(v(PointColumn::TrackLength, 4), 2.0);
    assert_eq!(v(PointColumn::ImageCount, 4), 2.0);
    assert_eq!(
        v(PointColumn::RmsErrorPx, 4),
        recon.point(4).unwrap().point().rms_reprojection_error
    );
    for c in [PointColumn::X, PointColumn::Y, PointColumn::Z] {
        assert!(v64(c, 4).is_nan(), "{c:?}");
    }
    assert!(v(PointColumn::CentroidDistance, 4).is_nan());
    // A bearing's rays are parallel.
    assert_eq!(v(PointColumn::MaxRayAngleDeg, 4), 0.0);
    assert!(v(PointColumn::ConditionNumber, 4).is_nan());
    assert!(v(PointColumn::InverseDepthZ, 4).is_nan());
    // Its neighbours are finite and unaffected.
    assert_eq!(v(PointColumn::AtInfinity, 3), 0.0);
    assert!(v64(PointColumn::X, 3).is_finite());
    assert!(v(PointColumn::ConditionNumber, 5).is_finite());
}

#[test]
fn absent_columns_are_skipped() {
    let recon = edited(demo(6));
    for c in [
        PointColumn::NormalConfidence,
        PointColumn::LowestZncc,
        PointColumn::LargestZnccSelfSimilarityRadius,
    ] {
        assert!(!c.is_available(&recon), "{c:?}");
    }
    assert!(PointColumn::CentroidDistance.is_available(&recon));
    let columns = compute(&recon, &PointColumn::ALL);
    assert!(!columns.has(PointColumn::NormalConfidence));
    assert!(!columns.has(PointColumn::LowestZncc));
    assert!(!columns.has(PointColumn::LargestZnccSelfSimilarityRadius));
    assert!(columns.values(PointColumn::LowestZncc).is_none());
    assert!(columns.has(PointColumn::TrackLength));
}

#[test]
fn centroid_distance_is_absent_with_no_posed_image() {
    let mut recon = demo(6);
    for image in &mut recon.image_table.images {
        image.translation_xyz = nalgebra::Vector3::repeat(f64::NAN);
    }
    let recon = edited(recon);
    assert!(!PointColumn::CentroidDistance.is_available(&recon));
    let columns = compute(&recon, &[PointColumn::CentroidDistance, PointColumn::X]);
    assert!(!columns.has(PointColumn::CentroidDistance));
    assert!(columns.has(PointColumn::X));
}

#[test]
fn normal_confidence_is_the_stored_byte() {
    let mut recon = demo(3);
    recon.point_set.normal_confidence = Some(vec![0, 17, 255]);
    let recon = edited(recon);
    assert!(PointColumn::NormalConfidence.is_available(&recon));
    let columns = compute(&recon, &[PointColumn::NormalConfidence]);
    assert_eq!(
        columns.values(PointColumn::NormalConfidence).unwrap(),
        [0.0, 17.0, 255.0]
    );
}

#[test]
fn lowest_zncc_skips_unmeasured_observations() {
    let mut recon = demo(4);
    // Two observations per point, in point order.
    recon.point_set.observation_confidence = Some(vec![0, 128, 0, 0, 200, 100, 255, 255]);
    let recon = edited(with_bearing(recon.clone(), 3));
    assert!(PointColumn::LowestZncc.is_available(&recon));
    let columns = compute(&recon, &[PointColumn::LowestZncc]);
    let v = columns.values(PointColumn::LowestZncc).unwrap();
    assert_eq!(v[0], (128.0f64 / 255.0) as f32);
    assert!(v[1].is_nan(), "no observation of point 1 was measured");
    assert_eq!(v[2], (100.0f64 / 255.0) as f32);
    // Defined at infinity.
    assert_eq!(v[3], 1.0);
}

#[test]
fn largest_self_similarity_radius_skips_unmeasured_readings() {
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
    assert!(PointColumn::LargestZnccSelfSimilarityRadius.is_available(&recon));
    let columns = compute(&recon, &[PointColumn::LargestZnccSelfSimilarityRadius]);
    let v = columns
        .values(PointColumn::LargestZnccSelfSimilarityRadius)
        .unwrap();
    assert_eq!(v[0], 1.5);
    assert!(v[1].is_nan(), "every reading of point 1 is unmeasured");
    assert_eq!(v[2], 3.0);
}

#[test]
fn deleted_points_have_no_row() {
    let mut recon = edited(demo(5));
    recon.delete_point(2).unwrap();
    let columns = compute(&recon, &[PointColumn::TrackLength]);
    assert_eq!(columns.point_indexes(), [0, 1, 3, 4]);
    assert_eq!(columns.values(PointColumn::TrackLength).unwrap().len(), 4);
}

#[test]
fn an_added_point_is_read_through_its_view() {
    let mut recon = edited(demo(5));
    recon.delete_point(2).unwrap();
    let added = recon
        .add_point(recon.point(0).expect("a live point").to_record())
        .unwrap();
    let columns = compute(&recon, &PointColumn::ALL);
    assert_eq!(columns.point_indexes(), [0, 1, 3, 4, added]);
    // The added point is a copy of point 0, so every value is point 0's.
    for c in PointColumn::ALL {
        if let Some(v) = columns.values(c) {
            assert_eq!(v[4].to_bits(), v[0].to_bits(), "{c:?}");
        }
        if let Some(v) = columns.values_f64(c) {
            assert_eq!(v[4].to_bits(), v[0].to_bits(), "{c:?}");
        }
    }
    let image = recon.point(0).unwrap().observations()[0].image_index;
    let seen = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            seen_in_image: Some(image),
            ..PointQueryOptions::default()
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
    assert!(PointColumns::compute(&recon, &PointColumn::ALL, &progress).is_err());
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
    PointColumns::compute(
        &recon,
        &[PointColumn::TrackLength, PointColumn::MaxRayAngleDeg],
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
            .install(|| compute(&recon, &PointColumn::ALL))
    };
    let (one, four) = (run(1), run(4));
    for c in PointColumn::ALL {
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

// ── The query ────────────────────────────────────────────────────────────

/// Ten points, point 7 at infinity (so no `x`), with `track_length` and `x`
/// computed.
fn query_fixture() -> (EditedReconstruction, PointColumns) {
    let recon = edited(with_bearing(demo(10), 7));
    let columns = compute(
        &recon,
        &[
            PointColumn::TrackLength,
            PointColumn::X,
            PointColumn::AtInfinity,
        ],
    );
    (recon, columns)
}

fn bound(column: PointColumn, min: Option<f64>, max: Option<f64>) -> ColumnBounds {
    ColumnBounds { column, min, max }
}

#[test]
fn no_options_keep_every_point_in_live_order() {
    let (recon, columns) = query_fixture();
    let all = query_points(&recon, &columns, &PointQueryOptions::default()).unwrap();
    assert_eq!(all, (0..10).collect::<Vec<u32>>());
}

#[test]
fn bounds_are_inclusive_and_all_must_pass() {
    let (recon, columns) = query_fixture();
    let xs = columns.values_f64(PointColumn::X).unwrap();
    let x2 = xs[2];
    let only_two = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            bounds: vec![bound(PointColumn::X, Some(x2), Some(x2))],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(only_two, [2]);
    let finite_with_track_2 = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            bounds: vec![
                bound(PointColumn::TrackLength, Some(2.0), None),
                bound(PointColumn::AtInfinity, None, Some(0.0)),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(finite_with_track_2, [0, 1, 2, 3, 4, 5, 6, 8, 9]);
    let none = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            bounds: vec![bound(PointColumn::TrackLength, Some(3.0), None)],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(none.is_empty());
}

#[test]
fn no_value_fails_any_set_bound_and_an_open_bound_filters_nothing() {
    let (recon, columns) = query_fixture();
    let wide = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            bounds: vec![bound(PointColumn::X, Some(f64::MIN), None)],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!wide.contains(&7), "the bearing has no x");
    assert_eq!(wide.len(), 9);
    let open = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            bounds: vec![bound(PointColumn::X, None, None)],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(open.len(), 10);
}

#[test]
fn seen_in_image_keeps_the_points_that_image_observes() {
    let (recon, columns) = query_fixture();
    let seen = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
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
    let (recon, columns) = query_fixture();
    for order in [SortOrder::Ascending, SortOrder::Descending] {
        let sorted = query_points(
            &recon,
            &columns,
            &PointQueryOptions {
                order_by: Some((PointColumn::TrackLength, order)),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sorted, (0..10).collect::<Vec<u32>>(), "{order:?}");
    }
    // The bearing ties with none: it sorts last ascending by at_infinity and
    // first descending, the finite points keeping their order around it.
    let descending = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            order_by: Some((PointColumn::AtInfinity, SortOrder::Descending)),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(descending, [7, 0, 1, 2, 3, 4, 5, 6, 8, 9]);
}

#[test]
fn no_value_sorts_last_in_both_orders() {
    let (recon, columns) = query_fixture();
    let xs = columns.values_f64(PointColumn::X).unwrap();
    for order in [SortOrder::Ascending, SortOrder::Descending] {
        let sorted = query_points(
            &recon,
            &columns,
            &PointQueryOptions {
                order_by: Some((PointColumn::X, order)),
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
    let (recon, columns) = query_fixture();
    let by_bound = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            bounds: vec![bound(PointColumn::RmsErrorPx, Some(0.0), None)],
            ..Default::default()
        },
    );
    assert_eq!(
        by_bound,
        Err(PointQueryError::ColumnNotComputed(PointColumn::RmsErrorPx))
    );
    let by_sort = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            order_by: Some((PointColumn::MaxRayAngleDeg, SortOrder::Ascending)),
            ..Default::default()
        },
    );
    assert_eq!(
        by_sort,
        Err(PointQueryError::ColumnNotComputed(
            PointColumn::MaxRayAngleDeg
        ))
    );
}

#[test]
fn an_image_past_the_table_is_refused() {
    let (recon, columns) = query_fixture();
    let refused = query_points(
        &recon,
        &columns,
        &PointQueryOptions {
            seen_in_image: Some(8),
            ..Default::default()
        },
    );
    assert_eq!(
        refused,
        Err(PointQueryError::ImageOutOfRange {
            image: 8,
            image_count: 8
        })
    );
    assert!(refused.unwrap_err().to_string().contains("8 images"));
}
