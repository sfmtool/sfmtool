// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A planted patch, four images, and the two forests that must agree about it.

use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::features::kdforest::{
    KdForestParams, KdForestU8, KdfSiftSources, KdfWorkspaceContents, KdfWorkspaceMetadata,
    KdfWriteOptions, LazyKdForestOptions, LazyKdForestU8,
};

const DIM: usize = 128;
/// Features in the planted patch, and so the inlier count the fit must reach.
const PLANTED: usize = 40;
/// Features of the query image that are not in the patch.
const BACKGROUND: usize = 20;
/// Features each distractor image holds.
const DISTRACTOR: usize = 60;
/// How many of the patch's features the near-miss image copies: enough to be
/// fitted, too few to be reported.
const NEAR_MISS: usize = 4;
/// A wide canvas keeps a wrong correspondence from landing inside the inlier
/// threshold by luck, which is what makes an exact inlier count assertable.
const CANVAS: f32 = 4000.0;

/// A warp planted between the query image and one further corpus image.
type PlantedWarp = fn([f32; 2]) -> [f32; 2];

/// The warp planted between the query image and image 1.
fn planted_affine(p: [f32; 2]) -> [f32; 2] {
    let (x, y) = (p[0] as f64, p[1] as f64);
    [
        (0.9 * x - 0.3 * y + 120.0) as f32,
        (0.3 * x + 0.9 * y - 45.0) as f32,
    ]
}

/// A doubling: a change of viewpoint the guard has to let through.
fn doubled_affine(p: [f32; 2]) -> [f32; 2] {
    [2.0 * p[0] + 70.0, 2.0 * p[1] - 40.0]
}

/// A reflection, scale 0.95: no pair of cameras can mirror one surface.
fn mirrored_affine(p: [f32; 2]) -> [f32; 2] {
    let (x, y) = (p[0] as f64, p[1] as f64);
    [
        (-0.9 * x + 0.3 * y + 3000.0) as f32,
        (0.3 * x + 0.9 * y - 45.0) as f32,
    ]
}

/// A tenfold blow-up, well outside the default scale bound.
fn inflated_affine(p: [f32; 2]) -> [f32; 2] {
    [10.0 * p[0] + 5.0, 10.0 * p[1] + 9.0]
}

struct Corpus {
    descriptors: Vec<u8>,
    count: usize,
    sources: KdfSiftSources,
    /// Positions of the query image's features, in constellation order.
    query_positions: Vec<[f32; 2]>,
    /// Corpus feature IDs of those same features.
    query_ids: Vec<u32>,
}

/// [`corpus_with`] and nothing else planted.
fn corpus() -> Corpus {
    corpus_with(&[])
}

/// Image 0 is the query image. Image 1 holds every one of its patch features
/// under [`planted_affine`]. Images 2 and 3 are unrelated. Image 4 copies four
/// patch features under a different warp. Then one further image per entry of
/// `extra`, holding the whole patch under that warp, so a test about which
/// warps are refused adds images without disturbing the ones above it: the
/// extras draw nothing from the generator, so with none of them the corpus is
/// the same bytes it always was.
fn corpus_with(extra: &[PlantedWarp]) -> Corpus {
    let mut rng = StdRng::seed_from_u64(11);
    let mut descriptors: Vec<u8> = Vec::new();
    let mut origins = Vec::new();
    let mut geometry = Vec::new();
    let mut query_positions = Vec::new();
    let mut query_ids = Vec::new();

    let shape = |scale: f32| [[scale, 0.0], [0.0, scale]];
    let push = |descriptors: &mut Vec<u8>,
                origins: &mut Vec<FeatureOrigin>,
                geometry: &mut Vec<FeatureGeometry>,
                vector: &[u8],
                image: u32,
                row: u32,
                at: [f32; 2],
                scale: f32| {
        descriptors.extend_from_slice(vector);
        origins.push(FeatureOrigin {
            image_index: image,
            image_feature_index: row,
        });
        let s = shape(scale);
        geometry.push([at, s[0], s[1]]);
    };

    // Image 0: the patch, then background features that belong to no other image.
    let mut patch_vectors: Vec<Vec<u8>> = Vec::new();
    let mut patch_positions: Vec<[f32; 2]> = Vec::new();
    for row in 0..PLANTED + BACKGROUND {
        let vector: Vec<u8> = (0..DIM).map(|_| rng.random_range(0..=u8::MAX)).collect();
        let at = [rng.random_range(0.0..CANVAS), rng.random_range(0.0..CANVAS)];
        if row < PLANTED {
            patch_vectors.push(vector.clone());
            patch_positions.push(at);
            query_positions.push(at);
            query_ids.push(descriptors.len() as u32 / DIM as u32);
        }
        push(
            &mut descriptors,
            &mut origins,
            &mut geometry,
            &vector,
            0,
            row as u32,
            at,
            2.0,
        );
    }

    // Image 1: the same patch descriptors, warped.
    for (row, vector) in patch_vectors.iter().enumerate() {
        push(
            &mut descriptors,
            &mut origins,
            &mut geometry,
            vector,
            1,
            row as u32,
            planted_affine(patch_positions[row]),
            3.0,
        );
    }

    // Images 2 and 3: unrelated features.
    for image in 2..4u32 {
        for row in 0..DISTRACTOR {
            let vector: Vec<u8> = (0..DIM).map(|_| rng.random_range(0..=u8::MAX)).collect();
            let at = [rng.random_range(0.0..CANVAS), rng.random_range(0.0..CANVAS)];
            push(
                &mut descriptors,
                &mut origins,
                &mut geometry,
                &vector,
                image,
                row as u32,
                at,
                1.5,
            );
        }
    }

    // Image 4: a few of the patch's features, enough to fit and not enough to report.
    for row in 0..NEAR_MISS {
        let at = patch_positions[row];
        push(
            &mut descriptors,
            &mut origins,
            &mut geometry,
            &patch_vectors[row],
            4,
            row as u32,
            [at[1] * 0.5 + 30.0, at[0] * 0.5 - 12.0],
            1.0,
        );
    }

    // One image per extra warp, each holding the whole patch under it.
    for (offset, warp) in extra.iter().enumerate() {
        for (row, vector) in patch_vectors.iter().enumerate() {
            push(
                &mut descriptors,
                &mut origins,
                &mut geometry,
                vector,
                5 + offset as u32,
                row as u32,
                warp(patch_positions[row]),
                2.5,
            );
        }
    }

    let images = 5 + extra.len() as u32;
    let count = origins.len();
    let sources = KdfSiftSources {
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
    };
    Corpus {
        descriptors,
        count,
        sources,
        query_positions,
        query_ids,
    }
}

fn forest(corpus: &Corpus) -> KdForestU8 {
    KdForestU8::build(
        &corpus.descriptors,
        corpus.count,
        DIM,
        KdForestParams {
            num_trees: 4,
            leaf_size: 8,
            seed: 3,
            ..KdForestParams::balanced()
        },
    )
}

fn resident(corpus: &Corpus) -> ResidentSources {
    ResidentSources::new(
        corpus.sources.origins.clone(),
        corpus.sources.geometry.clone(),
    )
    .unwrap()
}

fn params() -> ConstellationParams {
    ConstellationParams {
        max_leaf_checks: 512,
        iterations: 2000,
        ..ConstellationParams::default()
    }
}

fn query<'a>(corpus: &'a Corpus, ids: &'a [u32]) -> Constellation<'a, u8> {
    Constellation {
        positions: &corpus.query_positions,
        descriptors: ConstellationDescriptors::FeatureIds(ids),
        image_index: Some(0),
    }
}

#[test]
fn the_planted_image_wins_with_the_planted_warp() {
    let corpus = corpus();
    let forest = forest(&corpus);
    let sources = resident(&corpus);
    let found = constellation_query(
        &forest,
        &sources,
        &query(&corpus, &corpus.query_ids),
        &params(),
    )
    .unwrap();

    assert!(!found.is_empty(), "the planted image was not found");
    let best = &found[0];
    assert_eq!(best.image_index, 1);
    assert_eq!(best.inliers, PLANTED);
    assert_eq!(best.inlier_correspondences.len(), PLANTED);

    // Every inlier is its own feature's twin, and the recovered warp is the
    // planted one to within the floating-point round trip through f32 pixels.
    for correspondence in &best.inlier_correspondences {
        let at = corpus.query_positions[correspondence.query_index as usize];
        let want = planted_affine(at);
        let x =
            best.affine[0][0] * at[0] as f64 + best.affine[0][1] * at[1] as f64 + best.affine[0][2];
        let y =
            best.affine[1][0] * at[0] as f64 + best.affine[1][1] * at[1] as f64 + best.affine[1][2];
        assert!(
            (x - want[0] as f64).abs() < 1e-2,
            "x {x} against {}",
            want[0]
        );
        assert!(
            (y - want[1] as f64).abs() < 1e-2,
            "y {y} against {}",
            want[1]
        );
        assert_eq!(correspondence.position, want);
        assert_eq!(correspondence.affine_shape, [[3.0, 0.0], [0.0, 3.0]]);
    }
}

#[test]
fn the_query_image_and_the_thin_candidate_are_absent() {
    let corpus = corpus();
    let forest = forest(&corpus);
    let sources = resident(&corpus);
    let found = constellation_query(
        &forest,
        &sources,
        &query(&corpus, &corpus.query_ids),
        &params(),
    )
    .unwrap();

    assert!(
        found.iter().all(|m| m.image_index != 0),
        "the query image matched itself"
    );
    // Image 4 holds four of the patch's features: enough correspondences to be
    // fitted, fewer inliers than `min_inliers`.
    assert!(
        found.iter().all(|m| m.image_index != 4),
        "an image under min_inliers was reported"
    );
    assert!(NEAR_MISS < ConstellationParams::default().min_inliers);

    // Without the exclusion the query image is the strongest candidate of all,
    // which is what makes excluding it worth doing.
    let mut including = query(&corpus, &corpus.query_ids);
    including.image_index = None;
    let found = constellation_query(&forest, &sources, &including, &params()).unwrap();
    assert_eq!(found[0].image_index, 0);
    assert_eq!(found[0].inliers, PLANTED);
}

#[test]
fn a_mirrored_or_an_inflated_candidate_is_refused_and_a_doubled_one_is_not() {
    let corpus = corpus_with(&[doubled_affine, mirrored_affine, inflated_affine]);
    let forest = forest(&corpus);
    let sources = resident(&corpus);
    let found = constellation_query(
        &forest,
        &sources,
        &query(&corpus, &corpus.query_ids),
        &params(),
    )
    .unwrap();

    // Twice the size is a viewpoint a caller wants back.
    let doubled = found
        .iter()
        .find(|m| m.image_index == 5)
        .expect("the doubled image");
    assert_eq!(doubled.inliers, PLANTED);
    assert!(
        found.iter().all(|m| m.image_index != 6),
        "a mirrored model was reported"
    );
    assert!(
        found.iter().all(|m| m.image_index != 7),
        "a tenfold model was reported"
    );

    // Both images hold every one of the patch's descriptors, so their absence
    // is the guard's doing and not a missing correspondence: lift the scale
    // bound and the tenfold image comes back with every feature as an inlier.
    // The reflection has no bound to lift, which is the point of testing it
    // against the sign rather than against a number.
    let unbounded = ConstellationParams {
        max_scale: f64::INFINITY,
        ..params()
    };
    let found = constellation_query(
        &forest,
        &sources,
        &query(&corpus, &corpus.query_ids),
        &unbounded,
    )
    .unwrap();
    let inflated = found
        .iter()
        .find(|m| m.image_index == 7)
        .expect("the tenfold image, with the bound lifted");
    assert_eq!(inflated.inliers, PLANTED);
    assert!(
        found.iter().all(|m| m.image_index != 6),
        "a mirrored model survived the sign test"
    );
}

#[test]
fn the_model_solver_refuses_a_reflection_and_a_scale_far_from_unity() {
    let src = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let default = ConstellationParams::default().max_scale;

    // Swapping the axes mirrors the patch: determinant -1, scale 1.
    let mirrored = [[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]];
    assert!(solve_affine(src, mirrored, default).is_none());
    assert!(solve_affine(src, mirrored, f64::INFINITY).is_none());

    let tenfold = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]];
    assert!(solve_affine(src, tenfold, default).is_none());
    assert!(solve_affine(src, tenfold, 20.0).is_some());

    let tenth = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    assert!(solve_affine(src, tenth, default).is_none());
    assert!(solve_affine(src, tenth, 20.0).is_some());

    let doubled = [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]];
    assert!(solve_affine(src, doubled, default).is_some());
}

#[test]
fn the_radius_rule_holds_the_features_it_promises() {
    // A uniform scattering: the disc of the returned radius covers
    // `target / keypoints` of the frame, so it holds `target` of them.
    let radius = radius_for_feature_count(2160, 3840, 8112, 50);
    let covered = std::f64::consts::PI * (radius as f64) * (radius as f64);
    let expected = 50.0 / 8112.0 * (2160.0 * 3840.0);
    assert!((covered - expected).abs() < 1e-3 * expected);
    // The five measured captures, to the pixel the report quotes.
    assert_eq!(radius.round() as i32, 128);
    assert_eq!(
        radius_for_feature_count(270, 480, 2186, 50).round() as i32,
        31
    );
    // No keypoints, no radius that holds any.
    assert_eq!(radius_for_feature_count(640, 480, 0, 50), 0.0);
}

#[test]
fn the_two_forests_answer_identically() {
    let corpus = corpus();
    let forest = forest(&corpus);
    let sources = resident(&corpus);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("constellation.kdf");
    forest
        .write_kdf(
            &path,
            Some(&corpus.sources),
            &KdfWriteOptions {
                target_descriptor_block_bytes: 256,
                target_chunk_bytes: 512,
                ..Default::default()
            },
        )
        .unwrap();
    let lazy = LazyKdForestU8::open(&path, LazyKdForestOptions::default()).unwrap();

    let eager = constellation_query(
        &forest,
        &sources,
        &query(&corpus, &corpus.query_ids),
        &params(),
    )
    .unwrap();
    // The file carries its own origins and geometry, so the lazy arm supplies
    // neither: agreement here covers the sources as well as the search.
    let file_backed =
        constellation_query(&lazy, &lazy, &query(&corpus, &corpus.query_ids), &params()).unwrap();
    assert_eq!(eager, file_backed);

    // Same again with the descriptors handed over directly rather than read
    // back by ID, which is the other way a caller reaches this function.
    let vectors = forest.resolve_descriptors(&corpus.query_ids).unwrap();
    let direct = constellation_query(
        &lazy,
        &lazy,
        &Constellation {
            positions: &corpus.query_positions,
            descriptors: ConstellationDescriptors::Vectors(&vectors),
            image_index: Some(0),
        },
        &params(),
    )
    .unwrap();
    assert_eq!(eager, direct);
}

#[test]
fn a_pixel_and_a_radius_find_the_patch_through_the_sift_file() {
    let corpus = corpus();
    let forest = forest(&corpus);
    let sources = resident(&corpus);
    let dir = tempfile::tempdir().unwrap();
    let sift_path = dir.path().join("img0.jpg.sift");
    write_query_sift(&sift_path, &corpus);

    // A radius around the first patch feature, wide enough to hold a handful of
    // its neighbours and far short of the canvas.
    let center = corpus.query_positions[0];
    let radius = CANVAS / 3.0;
    let expected = expected_rows(&corpus, center, radius);
    // Only the patch features have a twin in image 1; background features
    // inside the radius join the constellation and stay outliers there.
    let planted_inside = expected.iter().filter(|&&r| (r as usize) < PLANTED).count();
    assert!(planted_inside >= ConstellationParams::default().min_inliers);

    let indexed = constellation_at_pixel(
        &forest,
        &sources,
        &QueryImage {
            sift_path: &sift_path,
            keypoints: None,
            image_index: Some(0),
        },
        center,
        radius,
        &params(),
    )
    .unwrap();
    assert_eq!(indexed.feature_rows, expected);
    // Image 0's features are the first rows of the corpus, so its `.sift` rows
    // and its corpus feature IDs coincide here.
    assert_eq!(indexed.feature_ids, expected);
    assert_eq!(indexed.matches[0].image_index, 1);
    assert_eq!(indexed.matches[0].inliers, planted_inside);

    // With no image index the descriptors come from the `.sift` file's own rows
    // instead of the corpus, and nothing is excluded, so the query image itself
    // is a candidate. The planted image still matches with the same inliers.
    let unindexed = constellation_at_pixel(
        &forest,
        &sources,
        &QueryImage {
            sift_path: &sift_path,
            keypoints: None,
            image_index: None,
        },
        center,
        radius,
        &params(),
    )
    .unwrap();
    assert_eq!(unindexed.feature_rows, expected);
    assert!(unindexed.feature_ids.is_empty());
    let planted = unindexed
        .matches
        .iter()
        .find(|m| m.image_index == 1)
        .expect("the planted image");
    assert_eq!(planted.inliers, planted_inside);
    assert!(unindexed.matches.iter().any(|m| m.image_index == 0));

    // Keypoints already in hand answer the same, without reading them again.
    let keypoints = ImageKeypoints::read(&sift_path).unwrap();
    let reused = constellation_at_pixel(
        &forest,
        &sources,
        &QueryImage {
            sift_path: &sift_path,
            keypoints: Some(&keypoints),
            image_index: Some(0),
        },
        center,
        radius,
        &params(),
    )
    .unwrap();
    assert_eq!(reused.matches, indexed.matches);
}

/// Rows of image 0 inside the radius, which for this corpus are exactly the
/// patch features (the background ones share no descriptor with any image).
fn expected_rows(corpus: &Corpus, center: [f32; 2], radius: f32) -> Vec<u32> {
    (0..PLANTED + BACKGROUND)
        .filter(|&row| {
            let at = corpus.sources.geometry[row][0];
            let dx = (at[0] - center[0]) as f64;
            let dy = (at[1] - center[1]) as f64;
            dx * dx + dy * dy <= (radius as f64) * (radius as f64)
        })
        .map(|row| row as u32)
        .collect()
}

/// Write image 0's features as a real `.sift` file.
fn write_query_sift(path: &std::path::Path, corpus: &Corpus) {
    let count = PLANTED + BACKGROUND;
    let mut positions = Array2::<f32>::zeros((count, 2));
    let mut affine = Array3::<f32>::zeros((count, 2, 2));
    let mut descriptors = Array2::<u8>::zeros((count, DIM));
    for row in 0..count {
        let geometry = corpus.sources.geometry[row];
        positions[[row, 0]] = geometry[0][0];
        positions[[row, 1]] = geometry[0][1];
        affine[[row, 0, 0]] = geometry[1][0];
        affine[[row, 0, 1]] = geometry[1][1];
        affine[[row, 1, 0]] = geometry[2][0];
        affine[[row, 1, 1]] = geometry[2][1];
        for column in 0..DIM {
            descriptors[[row, column]] = corpus.descriptors[row * DIM + column];
        }
    }
    let data = sfmtool_sift_format::SiftData {
        feature_tool_metadata: sfmtool_sift_format::FeatureToolMetadata {
            feature_tool: "test".into(),
            feature_type: "sift".into(),
            feature_options: serde_json::json!({}),
        },
        metadata: sfmtool_sift_format::SiftMetadata {
            version: sfmtool_sift_format::SIFT_FORMAT_VERSION,
            image_name: "img0.jpg".into(),
            image_file_xxh128: "0".repeat(32),
            image_file_size: 1,
            image_width: CANVAS as u32,
            image_height: CANVAS as u32,
            feature_count: count as u32,
        },
        content_hash: sfmtool_sift_format::SiftContentHash::default(),
        positions_xy: positions,
        affine_shapes: affine,
        descriptors,
        thumbnail_y_x_rgb: Array3::<u8>::zeros((128, 128, 3)),
    };
    sfmtool_sift_format::write_sift(path, &data, 3).unwrap();
}
