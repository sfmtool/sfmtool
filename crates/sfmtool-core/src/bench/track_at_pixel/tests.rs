// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Building a track at a pixel, decided against the bench's synthetic capture:
//! pinhole cameras looking down world `+z` at a textured plane, with a grid of
//! points on the plane that every camera sees at its exact projection. One
//! point is deleted from the version and the query is made at its pixel, the
//! way the harness holds a point out.

use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use ndarray::{Array1, Array2, Array3};
use sfmtool_matches_format::{
    ClusterPatchData, ClustersData, MatchesContentHash, MatchesData, MatchesMetadata,
    WorkspaceContents, WorkspaceMetadata, MATCHES_FORMAT_VERSION,
};

use crate::bench::tests::scene::{fixture_points, Scene, PLANE_Z};
use crate::bench::track::Verdict;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::members::{depth_modes, fit_affine};
use super::neighbourhood::{ObservationIndex, ViewCamera};
use super::*;

/// The grid's point at the middle, which the queries hold out.
const HELD_OUT: u32 = 12;

/// A five-by-five grid of points on the plane, 0.3 apart, which all three of
/// the scene's cameras see.
fn grid() -> Vec<Point3<f64>> {
    (-2..=2)
        .flat_map(|j| (-2..=2).map(move |i| Point3::new(0.3 * i as f64, 0.3 * j as f64, PLANE_Z)))
        .collect()
}

/// The grid as a version with the middle point deleted.
fn held_out(scene: &Scene) -> EditedReconstruction {
    let mut edited = EditedReconstruction::new(Arc::new(fixture_points(scene, &grid())));
    edited.delete_point(HELD_OUT).expect("the point is live");
    edited
}

/// Where the held-out point sits in image 0.
fn held_out_pixel(scene: &Scene) -> [f64; 2] {
    scene.project(0, grid()[HELD_OUT as usize])
}

fn observation(depth: Option<f64>, distance_px: f64) -> NearbyObservation {
    NearbyObservation {
        point: 0,
        keypoint: [0.0, 0.0],
        distance_px,
        depth,
        normal: Vector3::new(0.0, 0.0, 1.0),
        half_extent: 0.1,
        half_px: 5.0,
        at_infinity: depth.is_none(),
    }
}

// ---- Arithmetic --------------------------------------------------------------

#[test]
fn the_weighted_affine_recovers_an_exact_map() {
    let a = [[1.1, 0.2], [-0.1, 0.9]];
    let b = [3.0, -4.0];
    let src = [[0.0, 0.0], [5.0, 1.0], [-3.0, 4.0], [2.0, -6.0], [7.0, 7.0]];
    let dst: Vec<[f64; 2]> = src
        .iter()
        .map(|s| {
            [
                a[0][0] * s[0] + a[0][1] * s[1] + b[0],
                a[1][0] * s[0] + a[1][1] * s[1] + b[1],
            ]
        })
        .collect();
    let w = [1.0, 0.5, 0.25, 0.2, 0.1];
    let (fa, fb) = fit_affine(&src, &dst, &w, &[true; 5]).expect("finite");
    for r in 0..2 {
        assert!((fb[r] - b[r]).abs() < 1e-9);
        for c in 0..2 {
            assert!((fa[r][c] - a[r][c]).abs() < 1e-9);
        }
    }
}

#[test]
fn the_weighted_affine_over_collinear_pairs_is_the_minimum_norm_fit() {
    // Every source on the line y = x: the map across it is not determined, and
    // the fit is still finite and still carries each source to its match.
    let src = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let dst: Vec<[f64; 2]> = src.iter().map(|s| [2.0 * s[0] + 1.0, s[1] - 1.0]).collect();
    let (a, b) = fit_affine(&src, &dst, &[1.0; 4], &[true; 4]).expect("finite");
    for (s, d) in src.iter().zip(&dst) {
        let x = a[0][0] * s[0] + a[0][1] * s[1] + b[0];
        let y = a[1][0] * s[0] + a[1][1] * s[1] + b[1];
        assert!((x - d[0]).abs() < 1e-9 && (y - d[1]).abs() < 1e-9);
    }
    // Minimum norm splits the slope evenly between the two equal columns.
    assert!((a[0][0] - a[0][1]).abs() < 1e-9);
}

#[test]
fn depth_modes_split_where_one_depth_jumps_past_the_gap() {
    let near = [
        observation(Some(2.1), 1.0),
        observation(Some(1.0), 2.0),
        observation(None, 3.0),
        observation(Some(1.1), 4.0),
        observation(Some(-1.0), 5.0),
        observation(Some(2.0), 6.0),
        observation(Some(1.05), 7.0),
    ];
    let modes = depth_modes(&near, 1.15);
    let depths: Vec<Vec<f64>> = modes
        .iter()
        .map(|m| m.iter().map(|o| o.depth.unwrap()).collect())
        .collect();
    // The bearing and the point behind the camera state no depth in front of it.
    assert_eq!(depths, vec![vec![1.0, 1.05, 1.1], vec![2.0, 2.1]]);
}

// ---- Neighbourhoods ----------------------------------------------------------

#[test]
fn the_observation_index_does_not_see_a_deleted_point() {
    let scene = Scene::new();
    let edited = held_out(&scene);
    let views = scene.views();
    let camera = ViewCamera::new(&views[0]);
    let index = ObservationIndex::new(&edited);
    let near = index.near(0, held_out_pixel(&scene), 20.0, &camera);

    assert!(!near.is_empty());
    assert!(near.iter().all(|o| o.point != HELD_OUT));
    assert!(near
        .windows(2)
        .all(|w| w[0].distance_px <= w[1].distance_px));
    // The four grid neighbours at 0.3 world units, 12 px away, come first.
    assert_eq!(near.len(), 8);
    for o in &near[..4] {
        assert!((o.distance_px - 12.0).abs() < 1e-3);
        assert!((o.depth.unwrap() - PLANE_Z).abs() < 1e-9);
        // A 0.12 half-extent at depth 4 through a focal of 160 is 4.8 px.
        assert!((o.half_px - 4.8).abs() < 1e-5);
        assert!((o.normal.z.abs() - 1.0).abs() < 1e-9);
    }
}

/// A cluster-patches `.matches` file over images named `names`, whose members
/// are `(image, position, status)`, one cluster per entry of `clusters`, each
/// member with the shape `scale` times the identity.
fn matches_file(
    names: &[&str],
    clusters: &[Vec<(u32, [f32; 2], u8)>],
    scale: f32,
    with_patches: bool,
) -> MatchesData {
    let members: Vec<&(u32, [f32; 2], u8)> = clusters.iter().flatten().collect();
    let m = members.len();
    let mut starts = vec![0u32];
    for c in clusters {
        starts.push(starts.last().unwrap() + c.len() as u32);
    }
    MatchesData {
        metadata: MatchesMetadata {
            version: MATCHES_FORMAT_VERSION,
            matching_method: "test".into(),
            matching_tool: "test".into(),
            matching_tool_version: "0".into(),
            matching_options: std::collections::BTreeMap::new(),
            workspace: WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: WorkspaceContents {
                    feature_tool: "none".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::json!({}),
                    feature_prefix_dir: String::new(),
                },
            },
            timestamp: String::new(),
            image_count: names.len() as u32,
            image_pair_count: None,
            match_count: None,
            cluster_count: Some(clusters.len() as u32),
            cluster_member_count: Some(m as u32),
            has_two_view_geometries: false,
            has_clusters: true,
            has_cluster_patches: with_patches,
        },
        content_hash: MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: None,
            clusters_xxh128: None,
            cluster_patches_xxh128: None,
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names: names.iter().map(|n| n.to_string()).collect(),
        feature_tool_hashes: vec![[0u8; 16]; names.len()],
        sift_content_hashes: vec![[1u8; 16]; names.len()],
        feature_counts: Array1::from_vec(vec![m as u32; names.len()]),
        image_dims: Some(
            Array2::from_shape_vec((names.len(), 2), [128u32, 128].repeat(names.len()))
                .expect("(N, 2)"),
        ),
        image_pairs: None,
        clusters: Some(ClustersData {
            cluster_starts: Array1::from_vec(starts),
            member_images: Array1::from_vec(members.iter().map(|m| m.0).collect()),
            member_features: Array1::from_vec((0..m as u32).collect()),
            member_positions: Some(
                Array2::from_shape_vec((m, 2), members.iter().flat_map(|m| m.1).collect())
                    .expect("(M, 2)"),
            ),
            member_affine_shapes: Some(
                Array3::from_shape_vec((m, 2, 2), [scale, 0.0, 0.0, scale].repeat(m))
                    .expect("(M, 2, 2)"),
            ),
            matcher_options: serde_json::json!({}),
        }),
        cluster_patches: with_patches.then(|| ClusterPatchData {
            reference_members: Array1::from_vec(starts_of(clusters)),
            member_status: Array1::from_vec(members.iter().map(|m| m.2).collect()),
            member_zncc: Array1::from_vec(vec![0.9f32; m]),
            member_shift_px: Array1::from_vec(vec![0.0f32; m]),
            member_consistency_residual: Array1::from_vec(vec![f32::NAN; m]),
            refine_options: serde_json::json!({ "patch_size": 16.0 }),
        }),
        two_view_geometries: None,
    }
}

/// Each cluster's first member index.
fn starts_of(clusters: &[Vec<(u32, [f32; 2], u8)>]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut at = 0u32;
    for c in clusters {
        out.push(at);
        at += c.len() as u32;
    }
    out
}

#[test]
fn clusters_are_found_by_their_nearest_member_and_matched_to_images_by_name() {
    // The file lists an image the reconstruction does not hold, first, so its
    // image indexes are not the reconstruction's.
    let names = ["other.jpg", "image_1.jpg", "image_0.jpg"];
    let file = matches_file(
        &names,
        &[
            vec![
                (2, [50.0, 50.0], 0),
                (1, [60.0, 50.0], 1),
                (0, [51.0, 50.0], 1),
            ],
            vec![
                (2, [55.0, 50.0], 0),
                (2, [52.0, 50.0], 1),
                (1, [40.0, 40.0], 1),
            ],
            vec![(2, [90.0, 90.0], 0), (1, [90.0, 90.0], 1)],
        ],
        1.0,
        true,
    );
    let clusters =
        MatchesClusters::new(&file, &["image_0.jpg", "image_1.jpg"]).expect("a cluster file");
    assert_eq!(clusters.cluster_count(), 3);

    // Image 0 of the reconstruction is the file's image 2.
    let near = clusters.near(0, [53.0, 50.0], 10.0);
    let found: Vec<(u32, f64)> = near.iter().map(|c| (c.cluster, c.distance_px)).collect();
    assert_eq!(found, vec![(1, 1.0), (0, 3.0)]);
    // Cluster 1's nearest member in the image is the second of its two there.
    assert_eq!(near[0].member, 4);
    assert_eq!(near[0].members, 3..6);
    // The member in the image the reconstruction does not hold has no image
    // and is never found.
    assert_eq!(clusters.member(2).image, None);
    assert!(clusters.near(0, [51.0, 50.0], 0.5).is_empty());
    assert!(clusters.member(1).is_kept_or_reference());
}

#[test]
fn a_file_with_no_patch_section_has_no_kept_members() {
    let file = matches_file(&["image_0.jpg"], &[vec![(0, [10.0, 10.0], 1)]], 1.0, false);
    let clusters = MatchesClusters::new(&file, &["image_0.jpg"]).expect("a cluster file");
    assert!(!clusters.member(0).is_kept_or_reference());
    assert!(clusters.member(0).zncc.is_nan());
}

// ---- The query ---------------------------------------------------------------

#[test]
fn a_query_that_names_no_place_is_refused_before_any_member_runs() {
    let scene = Scene::new();
    let edited = held_out(&scene);
    let views = scene.views();
    let sources = TrackAtPixelSources::default();
    let options = TrackAtPixelOptions::default();
    let run = |views: &[crate::patch::normal_refine::ProjectedImage<'_>], image, pixel| {
        build_track_at_pixel(
            &edited,
            views,
            &sources,
            image,
            pixel,
            &options,
            &Progress::none(),
        )
    };

    let err = run(&views, 7, [10.0, 10.0]).unwrap_err();
    assert_eq!(
        err,
        TrackAtPixelError::NoSuchImage {
            image: 7,
            image_count: 3
        }
    );
    assert_eq!(err.stage(), "query");
    assert!(matches!(
        run(&views, 0, [128.0, 10.0]),
        Err(TrackAtPixelError::PixelOffImage { .. })
    ));
    assert!(matches!(
        run(&views, 0, [f64::NAN, 10.0]),
        Err(TrackAtPixelError::PixelOffImage { .. })
    ));
    assert!(matches!(
        run(&views[..2], 0, [10.0, 10.0]),
        Err(TrackAtPixelError::InputMismatch { input: "views", .. })
    ));
}

#[test]
fn every_member_refusing_reports_each_refusal_in_order() {
    let scene = Scene::new();
    let edited = held_out(&scene);
    let views = scene.views();
    let options = TrackAtPixelOptions {
        members: vec![CascadeMember::Clusters, CascadeMember::Constellation],
        ..TrackAtPixelOptions::default()
    };
    let err = build_track_at_pixel(
        &edited,
        &views,
        &TrackAtPixelSources::default(),
        0,
        held_out_pixel(&scene),
        &options,
        &Progress::none(),
    )
    .unwrap_err();
    let TrackAtPixelError::Refused { refusals } = &err else {
        panic!("expected every member to refuse, got {err:?}");
    };
    let got: Vec<(CascadeMember, RefusalStage)> =
        refusals.iter().map(|r| (r.member, r.stage)).collect();
    assert_eq!(
        got,
        vec![
            (CascadeMember::Clusters, RefusalStage::Clusters),
            (CascadeMember::Constellation, RefusalStage::Constellation),
        ]
    );
    assert_eq!(err.stage(), "cascade");
    assert!(err
        .to_string()
        .starts_with("every member refused; the last, constellation, at constellation: "));
}

/// Check a returned track the way the harness's good-track bar does: the
/// queried sighting `in` and on the pixel, every camera `in`, and the point
/// within a patch half-extent of the held-out one.
fn assert_rebuilt(scene: &Scene, track: &EditableTrack, report: &TrackAtPixelReport) {
    let pixel = held_out_pixel(scene);
    let q = &track.observations[report.query_observation];
    assert_eq!(q.image, 0);
    assert_eq!(q.verdict, Verdict::In);
    let kp = q.track.as_ref().and_then(|m| m.keypoint).expect("placed");
    let offset =
        ((f64::from(kp[0]) - pixel[0]).powi(2) + (f64::from(kp[1]) - pixel[1]).powi(2)).sqrt();
    assert!(offset <= 2.0, "the queried sighting sits {offset} px off");
    assert_eq!(track.verdict_counts().0, 3);
    let position = track
        .track()
        .and_then(|p| p.position)
        .expect("a finite point");
    let err = (position - grid()[HELD_OUT as usize]).norm();
    assert!(err < 0.12, "the point is {err} from the held-out one");
    assert!(matches!(
        report.stages.last(),
        Some(StageRecord::Final { .. })
    ));
}

#[test]
fn with_no_clusters_the_cascade_falls_through_to_the_neighbours_transfer() {
    let scene = Scene::new();
    let edited = held_out(&scene);
    let views = scene.views();
    let (track, report) = build_track_at_pixel(
        &edited,
        &views,
        &TrackAtPixelSources::default(),
        0,
        held_out_pixel(&scene),
        &TrackAtPixelOptions::default(),
        &Progress::none(),
    )
    .expect("the transfer builds a track");
    assert_eq!(report.member, CascadeMember::Transfer);
    assert_eq!(report.refusals.len(), 1);
    assert_eq!(report.refusals[0].member, CascadeMember::Clusters);
    assert_rebuilt(&scene, &track, &report);
}

#[test]
fn the_sweep_builds_the_track_from_the_neighbours_plane() {
    let scene = Scene::new();
    let edited = held_out(&scene);
    let views = scene.views();
    let options = TrackAtPixelOptions {
        members: vec![CascadeMember::Sweep],
        ..TrackAtPixelOptions::default()
    };
    let (track, report) = build_track_at_pixel(
        &edited,
        &views,
        &TrackAtPixelSources::default(),
        0,
        held_out_pixel(&scene),
        &options,
        &Progress::none(),
    )
    .expect("the sweep builds a track");
    assert_eq!(report.member, CascadeMember::Sweep);
    assert_rebuilt(&scene, &track, &report);
}

#[test]
fn a_cluster_near_the_pixel_carries_it_into_the_other_photographs() {
    let scene = Scene::new();
    let edited = held_out(&scene);
    let views = scene.views();
    let world = grid()[HELD_OUT as usize];
    // A cluster whose members sit two px to the right of the held-out point's
    // sightings. The cameras all face the plane square-on at one depth, so the
    // same step carries the pixel into every image.
    let members: Vec<(u32, [f32; 2], u8)> = (0..3u32)
        .map(|i| {
            let p = scene.project(i as usize, world);
            (i, [p[0] as f32 + 2.0, p[1] as f32], u8::from(i != 0))
        })
        .collect();
    let file = matches_file(
        &["image_0.jpg", "image_1.jpg", "image_2.jpg"],
        &[members],
        2.0,
        true,
    );
    let clusters = MatchesClusters::new(&file, &["image_0.jpg", "image_1.jpg", "image_2.jpg"])
        .expect("a cluster file");
    let sources = TrackAtPixelSources {
        sift_index: None,
        clusters: Some(&clusters),
    };
    let (track, report) = build_track_at_pixel(
        &edited,
        &views,
        &sources,
        0,
        held_out_pixel(&scene),
        &TrackAtPixelOptions::default(),
        &Progress::none(),
    )
    .expect("the cluster builds a track");
    assert_eq!(report.member, CascadeMember::Clusters);
    assert!(report.refusals.is_empty());
    assert_rebuilt(&scene, &track, &report);
}

/// The returned track carries its consensus bitmap, fused where the track
/// stands after the final slide onto the pixel, on the reconstruction's own
/// bitmap grid, and the colour at its centre; fusing it again moves nothing.
#[test]
fn the_returned_track_carries_a_bitmap_on_the_reconstructions_grid() {
    let scene = Scene::new();
    let mut edited = EditedReconstruction::new(Arc::new(crate::bench::tests::scene::with_columns(
        fixture_points(&scene, &grid()),
        6,
    )));
    edited.delete_point(HELD_OUT).expect("the point is live");
    let views = scene.views();
    let (track, _) = build_track_at_pixel(
        &edited,
        &views,
        &TrackAtPixelSources::default(),
        0,
        held_out_pixel(&scene),
        &TrackAtPixelOptions::default(),
        &Progress::none(),
    )
    .expect("the transfer builds a track");
    let payload = track.track().expect("the track stage");
    let bitmap = payload
        .bitmap
        .as_ref()
        .expect("the returned track has a bitmap");
    assert_eq!(bitmap.shape(), &[6, 6, 4], "not the reconstruction's grid");
    assert_eq!(
        payload.color,
        [bitmap[[3, 3, 0]], bitmap[[3, 3, 1]], bitmap[[3, 3, 2]]],
        "the colour is not the tile's centre"
    );

    let again = crate::bench::fit::fuse_where_it_stands(
        &track,
        &edited,
        &views,
        &crate::bench::FitOptions::default(),
    );
    let after = again.track().expect("the track stage");
    assert_eq!(after.position, payload.position);
    assert_eq!(after.placement, payload.placement);
    for (a, b) in again.observations.iter().zip(&track.observations) {
        assert_eq!(a.verdict, b.verdict);
        assert_eq!(
            a.track.as_ref().and_then(|m| m.keypoint),
            b.track.as_ref().and_then(|m| m.keypoint)
        );
    }
}
