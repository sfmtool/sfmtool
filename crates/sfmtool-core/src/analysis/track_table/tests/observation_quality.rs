// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The columns read per observation: the largest reprojection error, the
//! viewing angles and back-facing count, the lowest blur-matched bitmap ZNCC,
//! and the reference view's self-similarity from its two sources.

use std::sync::Arc;

use nalgebra::Vector3;
use ndarray::{Array2, Array4, Axis};

use super::*;
use crate::geometry::RigidTransform;
use crate::patch::normal_refine::viewing_angle;
use crate::patch::observation_reading::observation_reading;
use crate::patch::self_similarity::{zncc_self_similarity_radius, PatchTile, SelfSimilarityParams};
use crate::patch::stored_bitmap::stored_bitmap_planes;
use crate::reconstruction::data::observation_reprojection_residual;
use crate::reconstruction::ObservationSource;
use crate::RotQuaternion;

/// The projection of `point` into `image`, through its bearing at infinity;
/// `None` where it does not project.
fn projection(recon: &SfmrReconstruction, point: usize, image: usize) -> Option<[f64; 2]> {
    let p = &recon.point_set.points[point];
    let img = &recon.image_set.images[image];
    observation_reprojection_residual(
        &img.quaternion_wxyz,
        &img.translation_xyz,
        &recon.image_set.cameras[img.camera_index as usize],
        &p.position,
        p.is_at_infinity(),
        [0.0, 0.0],
    )
}

/// The demo reconstruction of `n` points with point 3 made a point at
/// infinity along the mean viewing direction of its two cameras, so both see
/// it, and that point's index.
fn demo_with_bearing(n: usize) -> (SfmrReconstruction, usize) {
    let p = 3;
    let mut recon = demo(n);
    let forward: Vector3<f64> = recon
        .observations_for_point(p)
        .iter()
        .map(|o| {
            let image = &recon.image_set.images[o.image_index as usize];
            // A camera looks along its -Z.
            image.quaternion_wxyz.inverse() * -Vector3::z()
        })
        .sum();
    let point = &mut recon.point_set.points[p];
    point.position = nalgebra::Point3::from(forward.normalize());
    point.w = 0.0;
    recon.rebuild_derived_fields();
    for o in recon.observations_for_point(p) {
        assert!(projection(&recon, p, o.image_index as usize).is_some());
    }
    (recon, p)
}

/// `recon` with inline keypoints at each observation's projection moved by
/// `offset(row)` pixels.
fn with_keypoints(
    mut recon: SfmrReconstruction,
    offset: impl Fn(usize) -> [f64; 2],
) -> SfmrReconstruction {
    let m = recon.point_set.tracks.len();
    let mut keypoints = Array2::<f32>::zeros((m, 2));
    for (row, obs) in recon.point_set.tracks.iter().enumerate() {
        let xy = projection(&recon, obs.point_index as usize, obs.image_index as usize)
            .expect("each observation projects");
        let d = offset(row);
        keypoints[[row, 0]] = (xy[0] + d[0]) as f32;
        keypoints[[row, 1]] = (xy[1] + d[1]) as f32;
    }
    match &mut recon.point_set.observations {
        ObservationSource::SiftFiles { keypoints_xy, .. } => *keypoints_xy = Some(keypoints),
        ObservationSource::EmbeddedPatches { .. } => unreachable!("the demo reads .sift files"),
    }
    recon
}

/// `recon` with a patch frame on every point, its normal towards the first
/// camera that sees it, or away from it for the points in `flipped`, and a
/// frame of zero half-vectors (no frame) on the points in `frameless`.
fn with_frames(
    mut recon: SfmrReconstruction,
    flipped: &[usize],
    frameless: &[usize],
) -> SfmrReconstruction {
    let n = recon.point_set.points.len();
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    for p in 0..n {
        if frameless.contains(&p) {
            continue;
        }
        let point = &recon.point_set.points[p];
        let first = recon.observations_for_point(p)[0].image_index as usize;
        let mut normal = if point.is_at_infinity() {
            -point.position.coords
        } else {
            recon.image_set.images[first].camera_center() - point.position
        }
        .normalize();
        if flipped.contains(&p) {
            normal = -normal;
        }
        let a = normal.cross(&Vector3::z()).normalize();
        let b = normal.cross(&a);
        // u × v is the outward normal.
        let (pu, pv) = (a * 0.05, b * 0.05);
        assert!(pu.cross(&pv).normalize().dot(&normal) > 0.999);
        for k in 0..3 {
            u[[p, k]] = pu[k] as f32;
            v[[p, k]] = pv[k] as f32;
        }
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.point_set.reference_observations = Some(vec![0; n]);
    recon
}

/// The viewing angle Track View shows for observation `k` of `view`.
fn track_view_angle(recon: &EditedReconstruction, view: &PointView<'_>, k: usize) -> f64 {
    let image_set = &recon.base.image_set;
    let image = &image_set.images[view.observations()[k].image_index as usize];
    let cam_from_world = RigidTransform::new(
        RotQuaternion::from_nalgebra(image.quaternion_wxyz),
        image.translation_xyz,
    );
    let patch = view.patch_frame().unwrap();
    let anchored = view.keypoint_xy(k).and_then(|xy| {
        patch.anchored_at_keypoint(
            &image_set.cameras[image.camera_index as usize],
            &cam_from_world,
            [f64::from(xy[0]), f64::from(xy[1])],
        )
    });
    viewing_angle(anchored.as_ref().unwrap_or(&patch), &cam_from_world)
        .unwrap()
        .angle_deg
}

#[test]
fn the_new_columns_are_absent_without_their_data() {
    let recon = edited(demo(4));
    for c in [
        TrackTableColumn::BackFacingObservations,
        TrackTableColumn::LargestErrorPx,
        TrackTableColumn::LowestBlurMatchedBitmapZncc,
        TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
        TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
        TrackTableColumn::LargestViewingAngleDeg,
    ] {
        assert!(!c.is_available(&recon), "{c:?}");
        assert!(!compute(&recon, &[c]).has(c), "{c:?}");
    }
}

#[test]
fn the_largest_error_is_the_worst_observation_and_defined_at_infinity() {
    // Observation 2k + 1 is moved 3 px, 4 px for point 1's.
    let offset = |row: usize| match row {
        3 => [0.0, 4.0],
        r if r % 2 == 1 => [3.0, 0.0],
        _ => [0.0, 0.0],
    };
    let (recon, bearing) = demo_with_bearing(8);
    assert_ne!(bearing, 1);
    let recon = edited(with_keypoints(recon, offset));
    assert!(TrackTableColumn::LargestErrorPx.is_available(&recon));
    let table = compute(&recon, &[TrackTableColumn::LargestErrorPx]);
    let v = table.values(TrackTableColumn::LargestErrorPx).unwrap();
    assert_eq!(v.len(), 8);
    for (p, &value) in v.iter().enumerate() {
        let expected = if p == 1 { 4.0 } else { 3.0 };
        // The bearing is projected through its direction.
        assert!((value - expected).abs() < 1e-3, "point {p}: {value}");
    }
}

#[test]
fn viewing_angles_are_track_views_and_count_the_back_facing_views() {
    // Point 1 faces away from its cameras, point 2 has no frame, and one
    // point is a bearing.
    let (base, bearing) = demo_with_bearing(8);
    assert!(bearing > 2, "the bearing is {bearing}");
    let base = with_frames(base, &[1], &[2]);
    for keypoints in [false, true] {
        let recon = if keypoints {
            edited(with_keypoints(base.clone(), |row| [row as f64 * 0.5, 1.0]))
        } else {
            edited(base.clone())
        };
        assert!(TrackTableColumn::LargestViewingAngleDeg.is_available(&recon));
        let table = compute(
            &recon,
            &[
                TrackTableColumn::LargestViewingAngleDeg,
                TrackTableColumn::BackFacingObservations,
            ],
        );
        let largest = table
            .values(TrackTableColumn::LargestViewingAngleDeg)
            .unwrap();
        let back = table
            .values(TrackTableColumn::BackFacingObservations)
            .unwrap();
        for p in (0..8).filter(|&p| p != 2) {
            let view = recon.point(p as u32).unwrap();
            let angles: Vec<f64> = (0..view.observations().len())
                .map(|k| track_view_angle(&recon, &view, k))
                .collect();
            let expected = angles.iter().copied().fold(f64::NAN, f64::max);
            assert_eq!(
                largest[p], expected as f32,
                "point {p}, keypoints {keypoints}"
            );
            let count = angles.iter().filter(|&&a| a >= 90.0).count();
            assert_eq!(back[p], count as f32, "point {p}");
        }
        // Facing its first camera, the second one of the arc sees point 0
        // obliquely but from the front.
        assert!(largest[0] > 1.0 && largest[0] < 90.0, "{}", largest[0]);
        assert_eq!(back[0], 0.0);
        // Seen from behind by both cameras.
        assert!(largest[1] > 90.0);
        assert_eq!(back[1], 2.0);
        // No frame, no value.
        assert!(largest[2].is_nan() && back[2].is_nan());
        // A bearing's patch faces every camera along the same direction, up
        // to the keypoint's own ray where it is re-anchored there.
        assert!(largest[bearing] < 1.0, "{}", largest[bearing]);
        assert_eq!(back[bearing], 0.0);
    }
}

#[test]
fn lowest_blur_matched_bitmap_zncc_skips_unread_scores() {
    let mut recon = demo(4);
    let options = observation_reading_options(
        SamplerChoice::Fixed(Sampler::Bilinear),
        24,
        PatchWindow::Uniform,
    );
    let mut readings = ObservationReadings::not_measured(8, options);
    let scored = |z: f32| ObservationReading {
        ellipse_axes: [1.0, 0.5],
        blur_matched_bitmap_zncc: z,
        plain_bitmap_zncc: z - 0.1,
        ..ObservationReading::NOT_MEASURED
    };
    readings.rows[0] = scored(1.0);
    readings.rows[1] = scored(0.62);
    // Point 1: measured, but no score read.
    readings.rows[2] = scored(f32::NAN);
    readings.rows[4] = scored(-0.25);
    readings.rows[5] = scored(0.9);
    readings.rows[6] = scored(0.5);
    readings.rows[7] = scored(0.75);
    recon.point_set.observation_readings = Some(readings);
    let recon = edited(with_bearing(recon, 3));
    let table = compute(&recon, &[TrackTableColumn::LowestBlurMatchedBitmapZncc]);
    let v = table
        .values(TrackTableColumn::LowestBlurMatchedBitmapZncc)
        .unwrap();
    assert_eq!(v[0], 0.62);
    assert!(v[1].is_nan());
    assert_eq!(v[2], -0.25);
    // Defined at infinity.
    assert_eq!(v[3], 0.5);
}

/// A `(n, 24, 24, 4)` bitmap column: point 0 a two-way texture, point 1
/// vertical stripes (an edge pattern, which slides along the stripes), point
/// 2 a zero row (no bitmap), the rest a flat grey.
fn bitmaps(n: usize) -> Array4<u8> {
    let r = 24;
    let mut b = Array4::<u8>::zeros((n, r, r, 4));
    for p in 0..n {
        if p == 2 {
            continue;
        }
        for y in 0..r {
            for x in 0..r {
                let value = match p {
                    0 => ((x * 37 + y * 91 + x * y * 13) % 199) as u8,
                    1 => {
                        if (x / 3) % 2 == 0 {
                            40
                        } else {
                            210
                        }
                    }
                    _ => 128,
                };
                for c in 0..3 {
                    b[[p, y, x, c]] = value;
                }
                b[[p, y, x, 3]] = 255;
            }
        }
    }
    b
}

/// The reading of point `p`'s bitmap in `b`, as the bitmap fallback reads it.
fn bitmap_axes(b: &Array4<u8>, p: usize) -> [f64; 2] {
    let planes = stored_bitmap_planes(b.index_axis(Axis(0), p)).unwrap();
    let side = planes.side;
    zncc_self_similarity_radius(
        &PatchTile {
            values: &planes.values,
            channels: planes.channels.min(3),
            width: side,
            height: side,
        },
        Some(&planes.data),
        [0, 0, side, side],
        &SelfSimilarityParams::default(),
    )
    .ellipse
    .axes
}

#[test]
fn the_reference_reading_falls_back_to_the_bitmap() {
    let mut recon = demo(5);
    let b = bitmaps(5);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(b.clone()));
    let recon = edited(with_bearing(recon, 4));
    for c in [
        TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
        TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
    ] {
        assert!(c.is_available(&recon), "{c:?}");
    }
    let table = compute(
        &recon,
        &[
            TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
            TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
        ],
    );
    let radius = table
        .values(TrackTableColumn::ReferenceZnccSelfSimilarityRadius)
        .unwrap();
    let elongation = table
        .values(TrackTableColumn::ReferenceZnccSelfSimilarityElongation)
        .unwrap();
    for p in [0usize, 1, 3, 4] {
        let [major, minor] = bitmap_axes(&b, p);
        assert_eq!(radius[p], major as f32, "point {p}");
        assert_eq!(elongation[p], (major / minor) as f32, "point {p}");
    }
    // The texture fixes the patch both ways, the stripes only across them,
    // and a flat patch in no direction: its ellipse is the cap's circle.
    assert!(radius[0] < 1.5, "{}", radius[0]);
    assert!(elongation[1] > 2.0 * elongation[0], "{elongation:?}");
    assert_eq!(radius[3], 3.0);
    assert_eq!(elongation[3], 1.0);
    // A zero row has no reading.
    assert!(radius[2].is_nan() && elongation[2].is_nan());
    // A point at infinity reads its bitmap as any other.
    assert_eq!(radius[4], radius[3]);
}

#[test]
fn the_stored_reading_at_the_reference_observation_comes_first() {
    let mut recon = demo(4);
    let b = bitmaps(4);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(b.clone()));
    let options = observation_reading_options(
        SamplerChoice::Fixed(Sampler::Bilinear),
        24,
        PatchWindow::Uniform,
    );
    let mut readings = ObservationReadings::not_measured(8, options);
    let measured = |axes: [f32; 2]| ObservationReading {
        ellipse_axes: axes,
        ..ObservationReading::NOT_MEASURED
    };
    // Point 0's reference is its second observation; its first is read too.
    readings.rows[0] = measured([2.5, 2.0]);
    readings.rows[1] = measured([1.25, 0.5]);
    // Point 1's reference row is not measured, so its bitmap is read.
    readings.rows[3] = measured([2.75, 0.25]);
    // Point 3 stores no reference observation (-1), so its bitmap is read.
    readings.rows[6] = measured([0.5, 0.5]);
    readings.rows[7] = measured([0.5, 0.5]);
    recon.point_set.observation_readings = Some(readings);
    recon.point_set.reference_observations = Some(vec![1, 0, 0, -1]);
    let recon = edited(recon);
    let table = compute(
        &recon,
        &[
            TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
            TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
        ],
    );
    let radius = table
        .values(TrackTableColumn::ReferenceZnccSelfSimilarityRadius)
        .unwrap();
    let elongation = table
        .values(TrackTableColumn::ReferenceZnccSelfSimilarityElongation)
        .unwrap();
    assert_eq!(radius[0], 1.25);
    assert_eq!(elongation[0], 2.5);
    for p in [1usize, 3] {
        let [major, minor] = bitmap_axes(&b, p);
        assert_eq!(radius[p], major as f32, "point {p}");
        assert_eq!(elongation[p], (major / minor) as f32, "point {p}");
    }
    // Point 2's reference row is unmeasured and its bitmap a zero row.
    assert!(radius[2].is_nan());
}

#[test]
fn readings_alone_supply_the_reference_columns() {
    let mut recon = demo(2);
    let options = observation_reading_options(
        SamplerChoice::Fixed(Sampler::Bilinear),
        24,
        PatchWindow::Uniform,
    );
    // A reading the bench commit writes for the render the bitmap holds:
    // the two sources agree to the stored precision.
    let b = bitmaps(2);
    let [major, minor] = bitmap_axes(&b, 0);
    let ellipse = crate::patch::self_similarity::SelfSimilarityEllipse::from_axes(
        [major, minor],
        [false, false],
        0.0,
    );
    let mut readings = ObservationReadings::not_measured(4, options);
    readings.rows[0] = observation_reading(Some(&ellipse), None, None, None, Some((1.0, 1.0)));
    recon.point_set.observation_readings = Some(readings);
    recon.point_set.reference_observations = Some(vec![0, 0]);
    let readings_only = edited(recon.clone());
    assert!(!readings_only.has_patch_bitmaps());
    assert!(TrackTableColumn::ReferenceZnccSelfSimilarityRadius.is_available(&readings_only));
    let stored = compute(
        &readings_only,
        &[TrackTableColumn::ReferenceZnccSelfSimilarityRadius],
    );
    recon.point_set.observation_readings = None;
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(b));
    let from_bitmap = compute(
        &edited(recon),
        &[TrackTableColumn::ReferenceZnccSelfSimilarityRadius],
    );
    let a = stored
        .values(TrackTableColumn::ReferenceZnccSelfSimilarityRadius)
        .unwrap();
    let b = from_bitmap
        .values(TrackTableColumn::ReferenceZnccSelfSimilarityRadius)
        .unwrap();
    assert_eq!(a[0], b[0]);
    // Point 1 has no measured reading and, here, no bitmap.
    assert!(a[1].is_nan());
}

#[test]
fn a_back_facing_sort_puts_the_frameless_points_last() {
    let recon = edited(with_frames(demo(6), &[1, 4], &[2]));
    let table = compute(&recon, &[TrackTableColumn::BackFacingObservations]);
    let sorted = |order| {
        query_points(
            &recon,
            &table,
            &TrackTableQueryOptions {
                order_by: Some((TrackTableColumn::BackFacingObservations, order)),
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(sorted(SortOrder::Descending), [1, 4, 0, 3, 5, 2]);
    assert_eq!(sorted(SortOrder::Ascending), [0, 3, 5, 1, 4, 2]);
    let kept = query_points(
        &recon,
        &table,
        &TrackTableQueryOptions {
            bounds: vec![bound(
                TrackTableColumn::BackFacingObservations,
                Some(1.0),
                None,
            )],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(kept, [1, 4]);
}
