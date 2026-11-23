// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench steps' photograph reads, their own and those of the patch
//! kernels beneath them, give the same result whichever path a read takes:
//! every read copying its regions into the pool, every read going through the
//! photograph's tiles, and a pool too small for all of a call's regions,
//! where the two paths mix within one call. Each pool holds no more than its
//! capacity and has every byte back once the call returns. A photograph that
//! lacks the pixels a read needs fails the step.

use std::fmt::Debug;

use super::grown_plane_normal::{camera, textured_plane};
use super::scene::{fixture_of, Scene, WORLD};
use super::*;
use crate::bench::{
    grid_plane_normal, grown_plane_normal, pairwise_flow_normal, GridPlaneNormalOptions,
    GrownPlaneNormalOptions, PairwiseFlowNormalOptions,
};
use crate::camera::TileKey;
use crate::camera::{PhotographReadError, TiledPhotograph};

/// The tile side the scenes are cut into for the reads through tiles: a
/// quarter of the photographs' 128 px, so a read crosses several tiles.
const TILE: u32 = 32;

/// Run `step` three ways and assert it returns the same track and report each
/// time:
///
/// - (a) over the scene as built, through a pool that holds any read, so
///   every read copies its regions;
/// - (b) over the scene cut into [`TILE`] px tiles, through a pool of no
///   bytes, so every read goes through the tiles;
/// - (c) over the tiled scene, through a pool of half the most bytes (a)
///   held at once, so some reads copy their regions and others go through
///   the tiles.
///
/// Each pool holds no more than its capacity and holds nothing once the call
/// returns.
fn assert_the_same_on_every_path<R: Debug>(
    build: impl Fn() -> Scene,
    step: impl Fn(&Scene, &RegionPool) -> (EditableTrack, R),
) {
    let whole = build();
    let tiled = build().with_tile_size(TILE);

    let copying = RegionPool::with_capacity(u64::MAX);
    let a = step(&whole, &copying);
    let a_stats = copying.stats();
    assert!(
        a_stats.copied_reads + a_stats.reused_reads > 0 && a_stats.tiled_reads == 0,
        "every read copies its regions: {a_stats:?}"
    );
    assert_eq!(copying.held_bytes(), 0, "{a_stats:?}");

    let on_tiles = RegionPool::with_capacity(0);
    let b = step(&tiled, &on_tiles);
    let b_stats = on_tiles.stats();
    assert!(
        b_stats.tiled_reads > 0 && b_stats.copied_reads + b_stats.reused_reads == 0,
        "every read goes through the tiles: {b_stats:?}"
    );
    assert_eq!(b_stats.peak_bytes, 0);
    assert_eq!(on_tiles.held_bytes(), 0);
    assert_same("copied regions", &a, "tiles", &b);

    let small = RegionPool::with_capacity(a_stats.peak_bytes / 2);
    let c = step(&tiled, &small);
    let c_stats = small.stats();
    assert!(
        c_stats.copied_reads > 0 && c_stats.tiled_reads > 0,
        "a pool of {} bytes copies some reads and takes others through the tiles: {c_stats:?}",
        small.capacity()
    );
    assert!(
        c_stats.peak_bytes <= small.capacity(),
        "{c_stats:?} over {}",
        small.capacity()
    );
    assert_eq!(small.held_bytes(), 0, "{c_stats:?}");
    assert_same("copied regions", &a, "a small pool", &c);
}

/// The two runs returned the same track and report, value for value.
///
/// The tracks are compared with `==`, which reads every element of their
/// arrays, and by their `Debug` text, which tells `-0.0` from `0.0`. The
/// reports are compared by their `Debug` text alone, since a report holds
/// `NaN` for a reading that was not taken and `NaN != NaN`; the text shows
/// every float in the shortest form that reads back to the same bits, and the
/// check that it elides nothing keeps an array's middle from going unread.
fn assert_same<R: Debug>(
    a_name: &str,
    a: &(EditableTrack, R),
    b_name: &str,
    b: &(EditableTrack, R),
) {
    assert_same_text(a_name, &format!("{:?}", a.0), b_name, &format!("{:?}", b.0));
    assert!(a.0 == b.0, "the tracks from {a_name} and {b_name} differ");
    let (a_report, b_report) = (format!("{:?}", a.1), format!("{:?}", b.1));
    assert!(
        !a_report.contains("..."),
        "a report's text elides: {a_report}"
    );
    assert_same_text(a_name, &a_report, b_name, &b_report);
}

/// `a == b`, and on a difference, the first place the texts part.
fn assert_same_text(a_name: &str, a: &str, b_name: &str, b: &str) {
    if a != b {
        let at = a
            .bytes()
            .zip(b.bytes())
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        let from = at.saturating_sub(200);
        panic!(
            "the results differ from byte {at}:\n{a_name}: ...{}\n{b_name}: ...{}",
            &a[from..(at + 200).min(a.len())],
            &b[from..(at + 200).min(b.len())],
        );
    }
}

/// Four views 20 degrees off the textured plane of the grown plane normal's
/// tests, 90 degrees apart in azimuth.
fn four_views_of_the_plane() -> Scene {
    let centres: Vec<[f64; 3]> = [0.0f64, 90.0, 180.0, 270.0]
        .iter()
        .map(|azimuth| {
            let (t, p) = (20f64.to_radians(), azimuth.to_radians());
            [
                WORLD.x + 4.0 * t.sin() * p.cos(),
                WORLD.y + 4.0 * t.sin() * p.sin(),
                WORLD.z - 4.0 * t.cos(),
            ]
        })
        .collect();
    Scene::looking_at(&centres, WORLD, camera(), textured_plane)
}

/// A version holding one point at [`WORLD`], seen by every image of `scene`,
/// on a patch of half-length `half` facing the cameras.
fn edited_of(scene: &Scene, half: f64) -> EditedReconstruction {
    let patch = OrientedPatch::from_center_normal(
        WORLD,
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [half, half],
    );
    let observing: Vec<u32> = (0..scene.len() as u32).collect();
    EditedReconstruction::new(Arc::new(fixture_of(scene, WORLD, 1.0, &observing, &patch)))
}

/// The version's track, its normal turned 12 degrees off the plane about
/// `x + y`, so a normal step has a turn to make.
fn tilted_track(edited: &EditedReconstruction) -> EditableTrack {
    let (bench, label) = bench_with_point(edited, 0);
    let track = track_of(&bench, &label);
    let turn = nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0)),
        12f64.to_radians(),
    );
    let (tilted, _) =
        tilt_patch(&track, edited, turn * Vector3::new(0.0, 0.0, -1.0)).expect("a finite track");
    tilted
}

#[test]
fn the_grown_plane_normal_is_the_same_on_every_read_path() {
    let edited = edited_of(&four_views_of_the_plane(), 0.4);
    let track = tilted_track(&edited);
    assert_the_same_on_every_path(four_views_of_the_plane, |scene, pool| {
        grown_plane_normal(
            &track,
            &edited,
            &scene.views(),
            pool,
            &GrownPlaneNormalOptions::default(),
            &Progress::none(),
        )
        .expect("the step runs on a finite track")
    });
}

#[test]
fn the_grid_plane_normal_is_the_same_on_every_read_path() {
    let edited = edited_of(&Scene::new(), 0.4);
    let track = tilted_track(&edited);
    assert_the_same_on_every_path(Scene::new, |scene, pool| {
        grid_plane_normal(
            &track,
            &edited,
            &scene.views(),
            pool,
            &GridPlaneNormalOptions::default(),
            &Progress::none(),
        )
        .expect("the step runs on a finite track with three views")
    });
}

/// The pairwise flow normal over one round and two radii rather than the
/// defaults' three and ten: one round reads every view's crop, which is all
/// the comparison needs, and the flow between them costs most of the time.
#[test]
fn the_pairwise_flow_normal_is_the_same_on_every_read_path() {
    let edited = edited_of(&four_views_of_the_plane(), 0.4);
    let track = tilted_track(&edited);
    let options = PairwiseFlowNormalOptions {
        rounds: 1,
        radii: vec![1.0, 2.0],
        ..Default::default()
    };
    assert_the_same_on_every_path(four_views_of_the_plane, |scene, pool| {
        pairwise_flow_normal(
            &track,
            &edited,
            &scene.views(),
            pool,
            &options,
            &Progress::none(),
        )
        .expect("the step runs on a finite track")
    });
}

/// A cluster of two sightings of the scene's plane, pointed at by hand, and
/// the version it is evaluated against.
fn cluster_of_two() -> (EditedReconstruction, EditableTrack) {
    let scene = Scene::new();
    let edited = edited_with_columns(&scene, WORLD);
    let seed = ClusterSeed::from_pixel(0, "image_0", scene.project(0, WORLD), PIXEL_SEED_RADIUS_PX);
    let (bench, created) = create_cluster(&Bench::new(), &seed).expect("a usable seed");
    let track = track_of(&bench, &created.label);
    let (track, added) = add_observation(
        &track,
        &ObservationSeed::at_pixel(1, scene.project(1, WORLD)),
    )
    .expect("a finite pixel");
    let (track, _) = set_verdict(&track, added.observation, Verdict::In).expect("a live row");
    (edited, track)
}

#[test]
fn a_cluster_evaluation_is_the_same_on_every_read_path() {
    let (edited, track) = cluster_of_two();
    assert_the_same_on_every_path(Scene::new, |scene, pool| {
        let (refined, report) = evaluate(
            &track,
            &edited,
            &scene.views(),
            pool,
            &EvaluateOptions::default(),
            &Progress::none(),
        )
        .expect("a cluster of two");
        assert_eq!(report.stage, StageKind::Cluster);
        assert_eq!(report.measured, 2);
        (refined, report)
    });
}

/// The fixture's point put on the bench as a track-stage track; the third
/// image shows the same plane patch, so a geometry search from its first
/// observation scores and admits it.
fn track_to_search() -> EditableTrack {
    let edited = edited_fixture(&Scene::new(), WORLD);
    let (bench, report) = create_track(&Bench::new(), &edited, 0, &CreateTrackOptions::default())
        .expect("the point goes on the bench");
    track_of(&bench, &report.label)
}

#[test]
fn a_geometry_search_is_the_same_on_every_read_path() {
    let track = track_to_search();
    assert_the_same_on_every_path(Scene::new, |scene, pool| {
        let (grown, report) = search_geometry(
            &track,
            0,
            &scene.views(),
            pool,
            &GeometrySearchOptions::default(),
            &Progress::none(),
        )
        .expect("the third image shows the same plane patch");
        assert_eq!(report.added(), 1, "{report}");
        (grown, report)
    });
}

/// `scene`'s photograph 0 made of only the [`TILE`] px tile in the corner of
/// its finest level, and none of its coarser levels, so that a read anywhere
/// near the point asks for tiles it does not hold.
fn without_pixels(scene: &Scene) -> TiledPhotograph {
    let whole = TiledPhotograph::cut(scene.pyramid(0), TILE);
    let corner = whole
        .tile(TileKey {
            level: 0,
            col: 0,
            row: 0,
        })
        .expect("a tile");
    TiledPhotograph::from_tiles(whole.layout().clone(), [corner]).expect("the layout's tile")
}

/// A photograph that lacks the pixels a read needs fails the step with the
/// read's error, on either path, rather than reading as a refusal.
#[test]
fn a_read_of_pixels_a_photograph_lacks_fails_the_step() {
    let scene = four_views_of_the_plane();
    let edited = edited_of(&scene, 0.4);
    let track = tilted_track(&edited);
    let broken = without_pixels(&scene);
    let mut views = scene.views();
    views[1].source = &broken;
    for capacity in [u64::MAX, 0] {
        let pool = RegionPool::with_capacity(capacity);
        let error = grown_plane_normal(
            &track,
            &edited,
            &views,
            &pool,
            &GrownPlaneNormalOptions::default(),
            &Progress::none(),
        )
        .expect_err("image 1 cannot be read");
        assert!(
            matches!(
                error,
                NormalError::Photograph(PhotographReadError::NeedsTiles { .. })
            ),
            "pool of {capacity}: {error}"
        );
        assert_eq!(pool.held_bytes(), 0);
    }

    let scene = Scene::new();
    let (edited, track) = cluster_of_two();
    let broken = without_pixels(&scene);
    let mut views = scene.views();
    views[1].source = &broken;
    for capacity in [u64::MAX, 0] {
        let pool = RegionPool::with_capacity(capacity);
        let error = evaluate(
            &track,
            &edited,
            &views,
            &pool,
            &EvaluateOptions::default(),
            &Progress::none(),
        )
        .expect_err("image 1 cannot be read");
        assert!(
            matches!(
                error,
                EvaluateError::Photograph(PhotographReadError::NeedsTiles { .. })
            ),
            "pool of {capacity}: {error}"
        );
        assert_eq!(pool.held_bytes(), 0);
    }
    let scene = Scene::new();
    let track = track_to_search();
    let broken = without_pixels(&scene);
    let mut views = scene.views();
    views[1].source = &broken;
    for capacity in [u64::MAX, 0] {
        let pool = RegionPool::with_capacity(capacity);
        let error = search_geometry(
            &track,
            0,
            &views,
            &pool,
            &GeometrySearchOptions::default(),
            &Progress::none(),
        )
        .expect_err("image 1 cannot be read");
        assert!(
            matches!(
                error,
                GeometrySearchError::Photograph(PhotographReadError::NeedsTiles { .. })
            ),
            "pool of {capacity}: {error}"
        );
        assert_eq!(pool.held_bytes(), 0);
    }
}

/// The fixture's track-stage track and the version it is read against.
fn track_stage() -> (EditedReconstruction, EditableTrack) {
    let edited = edited_fixture(&Scene::new(), WORLD);
    let (bench, report) = create_track(&Bench::new(), &edited, 0, &CreateTrackOptions::default())
        .expect("the point goes on the bench");
    let track = track_of(&bench, &report.label);
    (edited, track)
}

/// A track-stage reading renders each row's tile, runs the localizer and the
/// reference-view rule over them, renders the bitmap and scores every row
/// against it: the reference view's reads and everything beneath them.
#[test]
fn a_track_stage_reading_is_the_same_on_every_read_path() {
    let (edited, track) = track_stage();
    assert_the_same_on_every_path(Scene::new, |scene, pool| {
        let (read, report) = evaluate_rendering_bitmap(
            &track,
            &edited,
            &scene.views(),
            pool,
            &EvaluateOptions::default(),
            &FitOptions::default(),
            &Progress::none(),
        )
        .expect("the track stage reads");
        assert!(
            read.track().is_some_and(|p| p.bitmap.is_some()),
            "the reading renders a bitmap, so the stored bitmap's reads are taken"
        );
        (read, report)
    });
}

/// A track-stage fit runs the localizer and the sub-pixel refiner over the
/// rows and renders the bitmap at the fitted keypoints; rendering the bitmap
/// in place and scoring the rows against it are their own calls.
#[test]
fn a_track_stage_fit_and_bitmap_are_the_same_on_every_read_path() {
    let (edited, track) = track_stage();
    assert_the_same_on_every_path(Scene::new, |scene, pool| {
        let views = scene.views();
        let (fitted, report) = fit(
            &track,
            &edited,
            &views,
            pool,
            &FitOptions::default(),
            &Progress::none(),
        )
        .expect("the track stage fits");
        let rendered =
            render_bitmap_in_place(&fitted, &edited, &views, pool, &FitOptions::default())
                .expect("every row names a view");
        let scored = score_bitmap(
            &rendered,
            &edited,
            &views,
            pool,
            &EvaluateOptions::default(),
            &Progress::none(),
        )
        .expect("every row names a view");
        assert!(scored.track().is_some_and(|p| p.bitmap.is_some()));
        (scored, report)
    });
}

/// A photograph that lacks the pixels a read needs fails each track-stage
/// step with the read's error, on either path: the reading, whose tiles,
/// localizer and reference-view rule read it, the fit, and the bitmap's render
/// and scores.
#[test]
fn a_read_of_pixels_a_photograph_lacks_fails_each_track_stage_step() {
    let (edited, track) = track_stage();
    let scene = Scene::new();
    let broken = without_pixels(&scene);
    let mut views = scene.views();
    // The fit's bitmap, rendered from the readable views, for the steps that
    // read a track that holds one.
    let fitted = fit(
        &track,
        &edited,
        &scene.views(),
        &RegionPool::for_kernel_call(),
        &FitOptions::default(),
        &Progress::none(),
    )
    .expect("the track stage fits")
    .0;
    views[1].source = &broken;
    let is_read_error = |e: &EvaluateError| {
        matches!(
            e,
            EvaluateError::Photograph(PhotographReadError::NeedsTiles { .. })
        )
    };
    for capacity in [u64::MAX, 0] {
        let pool = RegionPool::with_capacity(capacity);
        let error = evaluate(
            &track,
            &edited,
            &views,
            &pool,
            &EvaluateOptions::default(),
            &Progress::none(),
        )
        .expect_err("image 1 cannot be read");
        assert!(is_read_error(&error), "pool of {capacity}: {error}");
        let error = fit(
            &track,
            &edited,
            &views,
            &pool,
            &FitOptions::default(),
            &Progress::none(),
        )
        .expect_err("image 1 cannot be read");
        assert!(
            matches!(
                error,
                FitError::Photograph(PhotographReadError::NeedsTiles { .. })
            ),
            "pool of {capacity}: {error}"
        );
        let error = render_bitmap_in_place(&fitted, &edited, &views, &pool, &FitOptions::default())
            .expect_err("image 1 cannot be read");
        assert!(is_read_error(&error), "pool of {capacity}: {error}");
        let error = score_bitmap(
            &fitted,
            &edited,
            &views,
            &pool,
            &EvaluateOptions::default(),
            &Progress::none(),
        )
        .expect_err("image 1 cannot be read");
        assert!(is_read_error(&error), "pool of {capacity}: {error}");
        assert_eq!(pool.held_bytes(), 0);
    }
}
