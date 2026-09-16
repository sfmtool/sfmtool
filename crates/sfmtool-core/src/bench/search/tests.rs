// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A patch planted in three images, a `.kdf` written over them, and one track
//! searched from.
//!
//! The corpus is built here rather than read from a fixture so the warp between
//! the searched image and each other image is a number the assertions can name:
//! a candidate's seed is the searched observation's pixel and shape under that
//! warp, and a test that could not state the warp could only check that
//! something was added.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::bench::track::{
    ClusterMeasurement, ClusterPayload, Observation, Stage, Thresholds, Verdict,
};
use crate::features::kdforest::{
    FeatureGeometry, FeatureOrigin, KdForestParams, KdForestU8, KdfSiftSources,
    KdfWorkspaceContents, KdfWorkspaceMetadata, KdfWriteOptions, LazyKdForestOptions,
};

const DIM: usize = 128;
/// Features of the planted patch, and so the inlier count every found image
/// reaches.
const PLANTED: usize = 24;
/// Features of the searched image that sit outside the search radius.
const OUTSIDE: usize = 12;
/// Features of the unrelated image.
const UNRELATED: usize = 40;
/// Where the searched observation sits, and the centre the patch is planted
/// around.
const CENTER: [f64; 2] = [600.0, 400.0];
/// Half-width of the box the patch is planted in, comfortably inside
/// [`RADIUS_PX`].
const SPREAD: f32 = 120.0;
/// The radius the search takes its constellation from.
const RADIUS_PX: f32 = 200.0;
/// The searched observation's own keypoint-frame shape.
const SHAPE: [[f64; 2]; 2] = [[2.0, 0.5], [-0.5, 2.0]];

/// The warp planted between the searched image and image 1.
fn warp_to_1(p: [f64; 2]) -> [f64; 2] {
    [
        0.8 * p[0] - 0.2 * p[1] + 130.0,
        0.2 * p[0] + 0.8 * p[1] - 55.0,
    ]
}

/// The warp planted between the searched image and image 2.
fn warp_to_2(p: [f64; 2]) -> [f64; 2] {
    [
        1.2 * p[0] + 0.1 * p[1] - 240.0,
        -0.1 * p[0] + 1.2 * p[1] + 60.0,
    ]
}

/// The 2x2 linear part of a planted warp, read off it by differencing, so the
/// expected shape is derived from the same function the positions are.
fn linear_of(warp: fn([f64; 2]) -> [f64; 2]) -> [[f64; 2]; 2] {
    let origin = warp([0.0, 0.0]);
    let x = warp([1.0, 0.0]);
    let y = warp([0.0, 1.0]);
    [
        [x[0] - origin[0], y[0] - origin[0]],
        [x[1] - origin[1], y[1] - origin[1]],
    ]
}

struct Corpus {
    descriptors: Vec<u8>,
    count: usize,
    sources: KdfSiftSources,
    /// The searched image's keypoints, in its own `.sift` row order.
    keypoints: ImageKeypoints,
}

/// Image 0 is searched: the planted patch around [`CENTER`], then features far
/// outside the radius. Images 1 and 2 hold the whole patch under their own
/// warps. Image 3 is unrelated.
fn corpus() -> Corpus {
    let mut rng = StdRng::seed_from_u64(7);
    let mut descriptors: Vec<u8> = Vec::new();
    let mut origins: Vec<FeatureOrigin> = Vec::new();
    let mut geometry: Vec<FeatureGeometry> = Vec::new();
    let shape = [[1.5_f32, 0.0], [0.0, 1.5_f32]];
    let push = |descriptors: &mut Vec<u8>,
                origins: &mut Vec<FeatureOrigin>,
                geometry: &mut Vec<FeatureGeometry>,
                vector: &[u8],
                image: u32,
                row: usize,
                at: [f64; 2]| {
        descriptors.extend_from_slice(vector);
        origins.push(FeatureOrigin {
            image_index: image,
            image_feature_index: row as u32,
        });
        geometry.push([[at[0] as f32, at[1] as f32], shape[0], shape[1]]);
    };

    let mut patch: Vec<Vec<u8>> = Vec::new();
    let mut patch_at: Vec<[f64; 2]> = Vec::new();
    for row in 0..PLANTED + OUTSIDE {
        let vector: Vec<u8> = (0..DIM).map(|_| rng.random_range(0..=u8::MAX)).collect();
        let at = if row < PLANTED {
            [
                CENTER[0] + f64::from(rng.random_range(-SPREAD..SPREAD)),
                CENTER[1] + f64::from(rng.random_range(-SPREAD..SPREAD)),
            ]
        } else {
            // Far outside the radius, so the constellation is exactly the patch.
            [
                CENTER[0] + f64::from(rng.random_range(2000.0..3000.0_f32)),
                CENTER[1] + f64::from(rng.random_range(2000.0..3000.0_f32)),
            ]
        };
        if row < PLANTED {
            patch.push(vector.clone());
            patch_at.push(at);
        }
        push(
            &mut descriptors,
            &mut origins,
            &mut geometry,
            &vector,
            0,
            row,
            at,
        );
    }

    for (image, warp) in [
        (1u32, warp_to_1 as fn([f64; 2]) -> [f64; 2]),
        (2, warp_to_2),
    ] {
        for (row, vector) in patch.iter().enumerate() {
            push(
                &mut descriptors,
                &mut origins,
                &mut geometry,
                vector,
                image,
                row,
                warp(patch_at[row]),
            );
        }
    }

    for row in 0..UNRELATED {
        let vector: Vec<u8> = (0..DIM).map(|_| rng.random_range(0..=u8::MAX)).collect();
        let at = [
            f64::from(rng.random_range(0.0..4000.0_f32)),
            f64::from(rng.random_range(0.0..4000.0_f32)),
        ];
        push(
            &mut descriptors,
            &mut origins,
            &mut geometry,
            &vector,
            3,
            row,
            at,
        );
    }

    // The searched image's keypoints are its own corpus rows, in row order,
    // which is the order a `.sift` file holds them in.
    let keypoints = ImageKeypoints {
        positions: geometry[..PLANTED + OUTSIDE]
            .iter()
            .map(|row| row[0])
            .collect(),
        affine_shapes: geometry[..PLANTED + OUTSIDE]
            .iter()
            .map(|row| [row[1], row[2]])
            .collect(),
    };

    let images = 4u32;
    let count = origins.len();
    Corpus {
        descriptors,
        count,
        sources: KdfSiftSources {
            workspace: KdfWorkspaceMetadata {
                absolute_path: "/ws".into(),
                relative_path: ".".into(),
                contents: KdfWorkspaceContents {
                    feature_tool: "test".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::json!({}),
                    feature_prefix_dir: "features/sift".into(),
                },
            },
            image_names: (0..images).map(|i| format!("img{i}.jpg")).collect(),
            feature_tool_hashes: (0..images).map(|i| [i as u8; 16]).collect(),
            sift_content_hashes: (0..images).map(|i| [100 + i as u8; 16]).collect(),
            origins,
            geometry,
        },
        keypoints,
    }
}

/// The corpus written to a `.kdf` and opened, with the directory kept alive for
/// as long as the forest is.
fn index(corpus: &Corpus) -> (tempfile::TempDir, LazyKdForestU8) {
    let forest = KdForestU8::build(
        &corpus.descriptors,
        corpus.count,
        DIM,
        KdForestParams {
            num_trees: 4,
            leaf_size: 8,
            seed: 3,
            ..KdForestParams::balanced()
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.kdf");
    forest
        .write_kdf(&path, Some(&corpus.sources), &KdfWriteOptions::default())
        .unwrap();
    let lazy = LazyKdForestU8::open(&path, LazyKdForestOptions::default()).unwrap();
    (dir, lazy)
}

/// A cluster-stage track whose observation 0 sits at [`CENTER`] in image 0,
/// plus one observation per extra image named, each `out`, so the search meets
/// an image the track already holds and one it does not.
fn track(also_in: &[u32]) -> EditableTrack {
    let mut observations = vec![Observation {
        image: 0,
        provenance: Provenance::Pixel,
        verdict: Verdict::In,
        pinned: false,
        cluster: Some(ClusterMeasurement::from_seed(CENTER, SHAPE)),
        track: None,
    }];
    for &image in also_in {
        observations.push(Observation {
            image,
            provenance: Provenance::Pixel,
            // `out` on purpose: a refusal is a decision, and the search must
            // not propose the image again over the top of it.
            verdict: Verdict::Out,
            pinned: true,
            cluster: Some(ClusterMeasurement::from_seed([10.0, 10.0], SHAPE)),
            track: None,
        });
    }
    EditableTrack {
        observations,
        stage: Stage::Cluster(ClusterPayload::default()),
        origin: None,
        thresholds: Thresholds::default(),
    }
}

fn options(min_inliers: usize) -> SearchOptions {
    SearchOptions {
        radius_px: RADIUS_PX,
        min_inliers,
        ..SearchOptions::default()
    }
}

#[test]
fn the_planted_images_are_found_and_seeded_at_the_warped_pixel_and_shape() {
    let corpus = corpus();
    let (_dir, forest) = index(&corpus);
    // Image 2 is already on the track; image 1 is not.
    let track = track(&[2]);

    let (grown, report) = search_descriptors(
        &track,
        0,
        &corpus.keypoints,
        &forest,
        &options(8),
        &Progress::none(),
    )
    .unwrap();

    // The whole patch and nothing else: the outside features are far beyond the
    // radius, so the constellation is exactly what was planted.
    assert_eq!(report.constellation, PLANTED);
    assert_eq!(report.observation, 0);
    assert_eq!(report.observation_count, 2);
    assert_eq!(report.image, 0);

    let found: Vec<u32> = report.matches.iter().map(|m| m.image).collect();
    assert_eq!(found, vec![1, 2], "the two planted images, in index order");
    assert!(
        !found.contains(&0),
        "the searched image is never a candidate of its own search"
    );
    for m in &report.matches {
        assert_eq!(m.inliers, PLANTED);
    }

    // Image 1 became a candidate; image 2 was left alone.
    assert_eq!(report.added(), 1);
    assert_eq!(report.already_in_track(), 1);
    assert_eq!(
        report.matches[1].found,
        Found::AlreadyInTrack { observation: 1 }
    );
    assert_eq!(grown.observations.len(), 3);
    assert_eq!(
        grown.observations[1], track.observations[1],
        "an image the track already names is untouched, verdict and pin included"
    );

    let Found::Added { observation } = report.matches[0].found else {
        panic!("image 1 was not added");
    };
    assert_eq!(observation, 2);
    let added = &grown.observations[observation];
    assert_eq!(added.image, 1);
    assert_eq!(added.verdict, Verdict::Candidate);
    assert!(!added.pinned);
    assert_eq!(
        added.provenance,
        Provenance::Search {
            inliers: PLANTED as u32
        }
    );

    // The seed is the observation's own pixel and shape under the planted warp.
    // The corpus positions round trip through f32, so the recovered affine is
    // the planted one to a hundredth of a pixel rather than exactly.
    let seed = added
        .cluster
        .as_ref()
        .expect("a searched seed is a cluster seed");
    let want_pixel = warp_to_1(CENTER);
    assert!(
        (seed.seed_position[0] - want_pixel[0]).abs() < 1e-2
            && (seed.seed_position[1] - want_pixel[1]).abs() < 1e-2,
        "{:?} against {want_pixel:?}",
        seed.seed_position
    );
    assert_eq!(report.matches[0].pixel, seed.seed_position);

    let linear = linear_of(warp_to_1);
    let want_shape = [
        [
            linear[0][0] * SHAPE[0][0] + linear[0][1] * SHAPE[1][0],
            linear[0][0] * SHAPE[0][1] + linear[0][1] * SHAPE[1][1],
        ],
        [
            linear[1][0] * SHAPE[0][0] + linear[1][1] * SHAPE[1][0],
            linear[1][0] * SHAPE[0][1] + linear[1][1] * SHAPE[1][1],
        ],
    ];
    for (row, want) in seed.seed_shape.iter().zip(want_shape.iter()) {
        for (got, want) in row.iter().zip(want.iter()) {
            assert!((got - want).abs() < 1e-3, "{got} against {want}");
        }
    }
    assert!(seed.shape.is_none(), "a seed is not a measurement");

    let sentence = report.to_string();
    assert_eq!(
        sentence,
        "Searched from observation 0 of 2: 2 images matched, 1 candidates added, \
         1 already in the track"
    );
}

#[test]
fn a_bar_no_image_reaches_adds_nothing() {
    let corpus = corpus();
    let (_dir, forest) = index(&corpus);
    let track = track(&[]);

    let (grown, report) = search_descriptors(
        &track,
        0,
        &corpus.keypoints,
        &forest,
        &options(PLANTED + 1),
        &Progress::none(),
    )
    .unwrap();

    assert!(report.matches.is_empty());
    assert_eq!(report.added(), 0);
    assert_eq!(grown, track, "a search that finds nothing moves nothing");
    assert_eq!(
        report.to_string(),
        format!("Searched from observation 0 of 1: no image matched the {PLANTED} keypoints of it")
    );
}

#[test]
fn a_search_from_an_observation_with_no_place_is_refused() {
    let corpus = corpus();
    let (_dir, forest) = index(&corpus);
    let mut track = track(&[]);
    track.observations[0].cluster = None;

    let error = search_descriptors(
        &track,
        0,
        &corpus.keypoints,
        &forest,
        &options(8),
        &Progress::none(),
    )
    .unwrap_err();
    assert_eq!(error, SearchError::NoPlace { observation: 0 });
    assert!(error.to_string().contains("no patch to search from"));

    let error = search_descriptors(
        &track,
        5,
        &corpus.keypoints,
        &forest,
        &options(8),
        &Progress::none(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        SearchError::NoSuchObservation {
            observation: 5,
            observation_count: 1,
        }
    );
}

#[test]
fn a_radius_that_holds_no_keypoint_is_refused_naming_the_radius() {
    let corpus = corpus();
    let (_dir, forest) = index(&corpus);
    let track = track(&[]);

    let error = search_descriptors(
        &track,
        0,
        &corpus.keypoints,
        &forest,
        &SearchOptions {
            radius_px: 0.5,
            ..options(8)
        },
        &Progress::none(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        SearchError::NoConstellation {
            radius_px: 0.5,
            keypoint_count: PLANTED + OUTSIDE,
        }
    );
    assert!(error.to_string().contains("within 0 px of the observation"));
}
