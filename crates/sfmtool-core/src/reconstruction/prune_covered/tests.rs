// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pruning a whole reconstruction: what the value that comes back holds, what
//! it leaves alone, and every refusal.
//!
//! The fixture is a synthetic scene of three cameras on a short arc and six
//! points in three pairs, every observation the exact projection and every
//! patch frame a square of a chosen extent, so each observation's projected
//! radius is a known multiple of its neighbour's.

use std::sync::atomic::AtomicBool;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use sfmtool_sfmr_format::{POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED};

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::progress::Progress;
use crate::reconstruction::data::{
    ObservationSource, Point3D, PointConstraintColumns, SfmrImage, SfmrReconstruction,
    TrackObservation,
};
use crate::reconstruction::edited::{EditedReconstruction, PointRecord};

use super::*;

const IMG_W: u32 = 640;
const IMG_H: u32 = 480;
const IMAGES: usize = 3;

/// `(x, y, z, patch half-extent)` per point.
///
/// Three pairs, one per answer the rule has to give. Points 0 and 1 are a
/// coarse feature with a ten-times finer one two pixels away, which is the
/// hand-over. Points 2 and 3 are the same pair pulled apart until the fine one
/// sits outside the coarse one's footprint. Points 4 and 5 sit on top of one
/// another at the same extent, so no band separates them.
const SCENE: [(f64, f64, f64, f64); 6] = [
    (0.0, 0.0, 0.0, 0.20),
    (0.03, 0.0, 0.0, 0.02),
    (1.0, 0.5, 0.0, 0.20),
    (1.5, 0.5, 0.0, 0.02),
    (-1.0, -0.5, 0.0, 0.20),
    (-1.03, -0.5, 0.0, 0.20),
];

const POINTS: usize = SCENE.len();

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// Where `world` lands in image `i`, or `None` when it is outside the frame.
fn project(recon: &SfmrReconstruction, i: usize, world: Point3<f64>) -> Option<[f32; 2]> {
    let image = &recon.image_table.images[i];
    let camera = &recon.image_table.cameras[image.camera_index as usize];
    let local = image.quaternion_wxyz * world.coords + image.translation_xyz;
    let (u, v) = camera.ray_to_pixel([local.x, local.y, local.z])?;
    if u < 0.0 || v < 0.0 || u >= camera.width as f64 || v >= camera.height as f64 {
        return None;
    }
    Some([u as f32, v as f32])
}

/// The fixture: cameras on an arc at radius 8 around the origin, [`SCENE`]'s
/// points inside it, every observation the exact projection, and one square
/// patch frame per point at that point's own extent.
fn scene() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..IMAGES)
        .map(|i| {
            let angle = 0.2 * (i as f64 - (IMAGES as f64 - 1.0) / 2.0);
            let centre = Vector3::new(8.0 * angle.sin(), 0.0, 8.0 * angle.cos());
            let rotation = UnitQuaternion::face_towards(&centre, &Vector3::y()).inverse();
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: 0,
                quaternion_wxyz: rotation,
                translation_xyz: -(rotation * centre),
            }
        })
        .collect();
    recon.image_table.thumbnails_y_x_rgb = Arc::new(Array4::zeros((
        IMAGES,
        sfmtool_sfmr_format::THUMBNAIL_SIZE,
        sfmtool_sfmr_format::THUMBNAIL_SIZE,
        3,
    )));
    let stats = recon.image_table.depth_statistics.images[0].clone();
    recon.image_table.depth_statistics.images = vec![stats; IMAGES];
    recon.image_table.depth_histogram_counts =
        vec![recon.image_table.depth_histogram_counts[0].clone(); IMAGES];

    recon.point_set.points = SCENE
        .iter()
        .map(|&(x, y, z, _)| Point3D {
            position: Point3::new(x, y, z),
            w: 1.0,
            color: [10, 20, 30],
            error: 0.25,
            normal: Vector3::new(0.0, 0.0, 1.0),
        })
        .collect();

    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for p in 0..POINTS {
        let mut count = 0u32;
        for i in 0..IMAGES {
            let pixel = project(&recon, i, recon.point_set.points[p].position)
                .expect("the fixture's points are in frame");
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push(pixel);
            count += 1;
        }
        counts.push(count);
    }
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    let set = &mut recon.point_set;
    set.tracks = tracks;
    set.observation_counts = counts;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; IMAGES],
    };
    let mut u = Array2::<f32>::zeros((POINTS, 3));
    let mut v = Array2::<f32>::zeros((POINTS, 3));
    for (p, &(_, _, _, extent)) in SCENE.iter().enumerate() {
        u[[p, 0]] = extent as f32;
        v[[p, 1]] = extent as f32;
    }
    set.patch_u_halfvec_xyz = Some(u);
    set.patch_v_halfvec_xyz = Some(v);
    set.point_constraints = Some(PointConstraintColumns::all_free(POINTS));
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

fn edited(recon: SfmrReconstruction) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(recon))
}

/// The prune at its defaults, which is what every case below varies from.
fn prune(
    value: &EditedReconstruction,
) -> Result<(EditedReconstruction, PointMap, PruneCoveredReport), PruneCoveredError> {
    prune_covered_observations(value, &PruneCoveredOptions::default(), &Progress::none())
}

/// Set point `p`'s constraint triple.
fn constrain(recon: &mut SfmrReconstruction, p: usize, kind: u8) {
    let columns = recon
        .point_set
        .point_constraints
        .as_mut()
        .expect("the fixture carries the columns");
    columns.point_constraints[p] = kind;
    columns.constraint_distances[p] = 8.0;
    columns.constraint_reference_images[p] = 0;
}

// ─── the rule over a value ────────────────────────────────────────────────

/// The coarse point's rows go, on every image that sees the fine one, and the
/// point goes with them.
#[test]
fn the_covered_coarse_point_is_retired_whole() {
    let start = edited(scene());
    let (next, map, report) = prune(&start).expect("the fixture prunes");

    assert!(report.changed);
    assert_eq!(report.census.rows, POINTS * IMAGES);
    assert_eq!(report.census.rows_flagged, IMAGES);
    assert_eq!(report.census.rows_removed, IMAGES);
    assert_eq!(report.census.owners_dropped_all_covered, 1);
    assert_eq!(report.census.owners_dropped_by_sweep, 0);
    assert_eq!(report.census.owners_kept, POINTS - 1);
    assert_eq!(report.degenerate_rows, 0);
    assert_eq!(report.protected_rows, 0);
    assert_eq!(report.points_before, POINTS);
    assert_eq!(report.points_after, POINTS - 1);
    assert_eq!(report.observations_before, POINTS * IMAGES);
    assert_eq!(report.observations_after, (POINTS - 1) * IMAGES);

    // Point 0 is gone; every other point kept its place, one lower.
    assert_eq!(map.forward(0), None);
    for p in 1..POINTS as u32 {
        assert_eq!(map.forward(p), Some(p - 1));
        assert_eq!(map.inverse(p - 1), Some(p));
    }
    assert_eq!(next.point_count(), POINTS - 1);
}

/// A surviving point keeps everything but the observations that went: its
/// position, its frame, its bitmap, its colour and its constraint.
#[test]
fn a_surviving_point_keeps_everything_but_the_rows() {
    let mut start = scene();
    // A bitmap column, so the write has one more thing to carry.
    start.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(Array4::from_shape_fn(
        (POINTS, 3, 3, 4),
        |(p, y, x, c)| (p * 64 + y * 16 + x * 4 + c) as u8,
    )));
    constrain(&mut start, 4, POINT_CONSTRAINT_RANGED);
    let start = edited(start);
    let (next, _map, _report) = prune(&start).expect("the fixture prunes");

    // Point 4 of the input is point 3 of the answer.
    let before = start.point(4).expect("a live point");
    let after = next.point(3).expect("a live point");
    assert_eq!(before.point(), after.point());
    assert_eq!(before.patch_u_halfvec(), after.patch_u_halfvec());
    assert_eq!(before.patch_v_halfvec(), after.patch_v_halfvec());
    assert_eq!(before.patch_bitmap(), after.patch_bitmap());
    assert_eq!(before.constraint(), after.constraint());
    assert_eq!(before.observations().len(), after.observations().len());
}

/// The pair the footprint does not reach is left alone, and the same-scale
/// pair is spared by the scale test rather than by the distance.
#[test]
fn the_pairs_the_rule_does_not_fire_on_are_left_alone() {
    let start = edited(scene());
    let (_next, _map, report) = prune(&start).expect("the fixture prunes");
    // Points 4 and 5 sit inside one another's footprints, so the pair is
    // counted contained on every image, and the ratio is what spares them.
    assert!(report.census.pairs_contained > report.census.pairs_finer);
    assert_eq!(report.census.pairs_finer, IMAGES);
}

/// The input is not touched, whatever the answer says.
#[test]
fn the_input_is_left_as_it_was() {
    let start = edited(scene());
    let before = start.clone();
    let _ = prune(&start).expect("the fixture prunes");
    assert_eq!(start, before);
}

/// Two prunes of one value give one answer.
#[test]
fn two_prunes_agree() {
    let start = edited(scene());
    let (a, _, ra) = prune(&start).expect("the fixture prunes");
    let (b, _, rb) = prune(&start).expect("the fixture prunes");
    assert_eq!(a, b);
    assert_eq!(ra, rb);
}

// ─── protection ───────────────────────────────────────────────────────────

/// A point the value holds or ranges keeps every observation, however well
/// covered, and the report says how many rows that bought.
#[test]
fn a_constrained_point_is_never_retired() {
    for kind in [POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED] {
        let mut recon = scene();
        constrain(&mut recon, 0, kind);
        let start = edited(recon);
        let (next, map, report) = prune(&start).expect("the fixture prunes");
        assert!(!report.changed, "constraint {kind} did not spare the point");
        assert_eq!(report.protected_rows, IMAGES);
        assert_eq!(report.protected_rows_spared, IMAGES);
        assert_eq!(report.census.rows_flagged, 0);
        assert_eq!(report.census.rows_removed, 0);
        // Nothing was written, so the value comes back as it stands under an
        // empty map.
        assert_eq!(next, start);
        assert_eq!(map, PointMap::Chain(Vec::new()));
        for p in 0..POINTS as u32 {
            assert_eq!(map.forward(p), Some(p));
        }
    }
}

/// A protected row still covers: the fine point is the one constrained here,
/// and the coarse point it covers is retired as before.
#[test]
fn a_protected_row_still_covers() {
    let mut recon = scene();
    constrain(&mut recon, 1, POINT_CONSTRAINT_HELD);
    let (_next, _map, report) = prune(&edited(recon)).expect("the fixture prunes");
    assert_eq!(report.protected_rows, IMAGES);
    assert_eq!(report.protected_rows_spared, 0);
    assert_eq!(report.census.rows_flagged, IMAGES);
}

// ─── the options ──────────────────────────────────────────────────────────

/// The footprint fraction is what decides how far a row reaches: at a fraction
/// small enough the hand-over stops firing, and at a large one the pair the
/// default does not reach is retired too.
#[test]
fn the_footprint_fraction_decides_the_reach() {
    let start = edited(scene());
    let narrow = prune_covered_observations(
        &start,
        &PruneCoveredOptions {
            footprint_fraction: 0.05,
            ..PruneCoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture prunes");
    assert!(!narrow.2.changed);

    let wide = prune_covered_observations(
        &start,
        &PruneCoveredOptions {
            footprint_fraction: 4.0,
            ..PruneCoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture prunes");
    // Point 2's fine neighbour is 31 px away, which only a much wider reach
    // finds.
    assert_eq!(wide.2.census.rows_flagged, 2 * IMAGES);
    assert_eq!(wide.2.points_after, POINTS - 2);
}

/// The fine-radius floor takes a collapsed projection out of the rule, and
/// the default floor is a pixel.
#[test]
fn the_fine_radius_floor_refuses_a_collapsed_projection() {
    let mut recon = scene();
    // Point 1's patch is a fiftieth of what it was, so it projects under a
    // pixel while staying a whole octave finer than point 0.
    for column in [
        recon.point_set.patch_u_halfvec_xyz.as_mut(),
        recon.point_set.patch_v_halfvec_xyz.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        column[[1, 0]] *= 0.02;
        column[[1, 1]] *= 0.02;
    }
    let start = edited(recon);

    let (_next, _map, floored) = prune(&start).expect("the fixture prunes");
    assert!(!floored.changed);

    let off = prune_covered_observations(
        &start,
        &PruneCoveredOptions {
            min_fine_radius_px: 0.0,
            ..PruneCoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture prunes");
    assert_eq!(off.2.census.rows_flagged, IMAGES);
}

/// The scale ratio is what "finer" means: at a ratio the same-scale pair
/// clears, both of its rows' coarse sides go.
#[test]
fn the_ratio_decides_what_counts_as_finer() {
    let start = edited(scene());
    let (_next, _map, report) = prune_covered_observations(
        &start,
        &PruneCoveredOptions {
            ratio: 1.0,
            ..PruneCoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture prunes");
    // Every contained pair is now a finer one, so the same-scale pair is
    // retired wherever the two do not project to exactly the same radius.
    assert_eq!(report.census.pairs_finer, report.census.pairs_contained);
    assert!(report.census.rows_flagged > IMAGES);
}

/// The observation bar is what an owner has to clear, and raising it takes
/// points the rule never touched.
#[test]
fn the_observation_bar_takes_the_thinly_seen() {
    let start = edited(scene());
    let (_next, _map, report) = prune_covered_observations(
        &start,
        &PruneCoveredOptions {
            min_observations: 4,
            ..PruneCoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture prunes");
    assert_eq!(report.points_after, 0);
    assert_eq!(report.census.owners_dropped_by_sweep, POINTS - 1);
    assert_eq!(report.census.owners_dropped_all_covered, 1);
}

// ─── the band table ───────────────────────────────────────────────────────

/// The bands are anchored at the widest radius measured, and they account for
/// every row that projected.
#[test]
fn the_bands_say_where_the_prune_bit() {
    let start = edited(scene());
    let (_next, _map, report) = prune(&start).expect("the fixture prunes");
    assert!(!report.bands.is_empty());
    let rows: usize = report.bands.iter().map(|b| b.rows).sum();
    assert_eq!(rows, POINTS * IMAGES - report.degenerate_rows);
    let retired: usize = report.bands.iter().map(|b| b.rows_retired).sum();
    assert_eq!(retired, report.census.rows_flagged);
    let dropped: usize = report.bands.iter().map(|b| b.points_dropped).sum();
    assert_eq!(dropped, report.points_before - report.points_after);
    // Band 0 is the coarsest octave and holds the widest row measured.
    assert_eq!(report.bands[0].band, 0);
    assert!(report.bands[0].upper_px > report.bands[0].lower_px);
    // The retired rows are the coarse ones, so they are all in band 0.
    assert_eq!(report.bands[0].rows_retired, IMAGES);
}

// ─── the overlay ──────────────────────────────────────────────────────────

/// A value with an overlay is folded in first, and the map carries an index
/// through both steps.
#[test]
fn an_overlay_is_folded_in_first() {
    let mut start = edited(scene());
    // Point 3 is re-added with a shorter track, which makes it an addition and
    // leaves index 3 deleted.
    let mut record: PointRecord = start.point(3).expect("a live point").to_record();
    record.observations.truncate(2);
    let moved = start.replace_point(3, record).expect("a live point");
    assert_eq!(moved, POINTS as u32);

    let (next, map, report) = prune(&start).expect("the fixture prunes");
    assert!(report.changed);
    assert_eq!(report.points_before, POINTS);
    assert_eq!(report.points_after, POINTS - 1);
    // The addition went back into point 3's slot at materialisation, and then
    // point 0 was dropped, so it is point 2 of the answer.
    assert_eq!(map.forward(moved), Some(2));
    assert_eq!(next.point(2).expect("a live point").observations().len(), 2);
}

// ─── refusals ─────────────────────────────────────────────────────────────

/// A value with no patch frame states no footprint.
#[test]
fn a_value_without_patch_frames_is_refused() {
    let mut recon = scene();
    recon.point_set.patch_u_halfvec_xyz = None;
    recon.point_set.patch_v_halfvec_xyz = None;
    let error = prune(&edited(recon)).expect_err("no frame, no footprint");
    assert_eq!(error, PruneCoveredError::NoPatchFrames);
    assert!(error.to_string().contains("patch frame"));
}

/// A value whose observations are `.sift` feature indexes carries no pixel for
/// a footprint to sit at.
#[test]
fn a_value_without_inline_keypoints_is_refused() {
    let mut recon = scene();
    recon.point_set.observations = ObservationSource::SiftFiles {
        feature_indexes: vec![0; recon.point_set.tracks.len()],
        keypoints_xy: None,
        feature_tool_hashes: vec![[0u8; 16]; IMAGES],
        sift_content_hashes: vec![[0u8; 16]; IMAGES],
    };
    recon.rebuild_derived_fields();
    let error = prune(&edited(recon)).expect_err("no pixel, no footprint");
    assert_eq!(error, PruneCoveredError::NoKeypoints);
}

/// A value no image of which carries a pose projects nothing.
#[test]
fn a_value_without_poses_is_refused() {
    let mut recon = scene();
    for image in recon.image_table.images.iter_mut() {
        image.translation_xyz = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
    }
    let error = prune(&edited(recon)).expect_err("no pose, no projection");
    assert_eq!(error, PruneCoveredError::NoPosedImages);
}

/// A footprint fraction that names no disk is refused rather than read as an
/// empty one.
#[test]
fn an_unusable_footprint_fraction_is_refused() {
    let start = edited(scene());
    let error = prune_covered_observations(
        &start,
        &PruneCoveredOptions {
            footprint_fraction: 0.0,
            ..PruneCoveredOptions::default()
        },
        &Progress::none(),
    )
    .expect_err("a fraction of nothing");
    assert_eq!(error, PruneCoveredError::BadFootprintFraction(0.0));
}

/// A prune asked to stop writes nothing.
#[test]
fn a_cancelled_prune_writes_nothing() {
    let start = edited(scene());
    let flag = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    let error = prune_covered_observations(&start, &PruneCoveredOptions::default(), &progress)
        .expect_err("it was asked to stop");
    assert_eq!(error, PruneCoveredError::Cancelled);
}

// ─── degenerate rows ──────────────────────────────────────────────────────

/// A point with no patch frame of its own is neither retired nor able to
/// retire anything, and the report counts its rows.
#[test]
fn a_point_with_no_frame_of_its_own_is_counted_and_read_past() {
    let mut recon = scene();
    for column in [
        recon.point_set.patch_u_halfvec_xyz.as_mut(),
        recon.point_set.patch_v_halfvec_xyz.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        for c in 0..3 {
            column[[1, c]] = 0.0;
        }
    }
    let (_next, _map, report) = prune(&edited(recon)).expect("the fixture prunes");
    assert_eq!(report.degenerate_rows, IMAGES);
    // Point 1 was the only cover point 0 had, so nothing is retired now, and
    // no pair the degenerate rows are on passes the scale test either way.
    assert!(!report.changed);
    assert_eq!(report.census.pairs_finer, 0);
}
