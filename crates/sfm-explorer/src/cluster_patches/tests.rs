// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cluster-patches file of a node: where it goes, what a build writes into
//! it, and what says it is out of date.
//!
//! Over the SIFT index's workspace fixture ([`crate::sift_index::tests`]), which
//! writes a `.sift` file and a photograph per image and builds both search
//! files. The test that matters most re-derives the file the way the two CLI
//! steps make one, from the same index and the same photographs, and compares
//! every array.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::features::cluster_match::{
    background_floor_clusters_from_neighbors, BackgroundFloorParams, NeighborTable,
};
use sfmtool_core::features::kdforest::{KdForestParams, LazyKdForestOptions, LazyKdForestU8};
use sfmtool_core::patch::cluster_refine::{
    refine_cluster_patches, warp_consistency_residuals, ClusterRefineParams, FeatureGeometry,
    MemberStatus,
};
use sfmtool_core::patch::normal_refine::PatchWindow;

use super::{cluster_patches_path, CLUSTER_PATCHES_FILE_SUFFIX, INDEX_HASH_OPTION};
use crate::index_files::IndexFileState;
use crate::scene::ImageRef;
use crate::sift_index::tests::{searchable, state_in};
use crate::state::AppState;

/// Where the fixture's cluster-patches file goes.
fn patches_of(dir: &Path) -> PathBuf {
    dir.join(format!("demo{CLUSTER_PATCHES_FILE_SUFFIX}"))
}

/// The sentence the node's cluster patches are stale with.
fn stale_reason(state: &AppState, id: crate::scene::ReconId) -> String {
    state
        .cluster_patches(id)
        .expect("open")
        .stale_reason()
        .expect("stale")
        .to_string()
}

// ── Where the file goes ─────────────────────────────────────────────────

/// The file takes its name from the `.sfmr`'s stem and sits beside it and
/// beside the index, spelled in one convention.
#[test]
fn the_cluster_patches_path_is_the_sfmr_s_stem_beside_the_sfmr() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    assert_eq!(
        cluster_patches_path(state.node(id).expect("loaded")),
        Some(patches_of(dir.path()))
    );
    assert_eq!(state.cluster_patches_path(id), Some(patches_of(dir.path())));
    {
        let node = state.scene.first_mut().expect("one node");
        node.path = Some(PathBuf::from(format!(
            "{}/runs/demo.sfmr",
            dir.path().display()
        )));
    }
    let shown = state
        .cluster_patches_path(id)
        .expect("a saved node has a path")
        .display()
        .to_string();
    let backslash = char::from(0x5c_u8);
    let foreign = if std::path::MAIN_SEPARATOR == '/' {
        backslash
    } else {
        '/'
    };
    assert!(!shown.contains(foreign), "{shown}");
    assert!(shown.ends_with(CLUSTER_PATCHES_FILE_SUFFIX), "{shown}");
}

/// A node with no path on disk has nowhere to put one.
#[test]
fn an_unsaved_node_has_no_cluster_patches_path() {
    let mut state = AppState::new();
    state.append_node(crate::scene::SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let id = state.scene[0].id;
    assert_eq!(state.cluster_patches_path(id), None);
    state.refresh_index_files(id);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::None);
}

// ── What a build makes ──────────────────────────────────────────────────

/// A build writes the file beside the `.sfmr`, opens it, and it reads current:
/// the clusters and the cluster-patches sections, the node's images in its
/// order, and the content hash of the index it was made from.
#[test]
fn a_build_writes_cluster_patches_that_read_current() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Current);
    let file = state.cluster_patches(id).expect("the build opened it");
    assert_eq!(file.path, patches_of(dir.path()));
    assert!(file.clusters > 0, "the planted patch clusters");

    let data = sfmtool_matches_format::read_matches(&file.path).expect("a readable file");
    assert!(data.metadata.has_clusters && data.metadata.has_cluster_patches);
    assert!(data.clusters.is_some() && data.cluster_patches.is_some());
    let recon = state.node(id).expect("loaded").recon();
    let names: Vec<String> = recon
        .image_table
        .images
        .iter()
        .map(|image| image.name.clone())
        .collect();
    assert_eq!(data.image_names, names);
    let index = state.sift_index(id).expect("built");
    assert_eq!(
        data.metadata.matching_options[INDEX_HASH_OPTION],
        serde_json::json!(index.forest.content_xxh128())
    );
    assert_eq!(data.metadata.cluster_count, Some(file.clusters as u32));
}

/// The file the build writes holds what `sfm match --cluster` over the index
/// and then `sfm cluster-patches` write, array for array.
///
/// The expectation is derived here the way the two CLI steps derive it, not by
/// calling the build's own functions: the self-join and the clustering as the
/// `background_floor_clusters_kdf` binding ran them, the members' detections
/// read from the `.sift` files as `sfm match --cluster` reads them, the
/// photographs in OpenCV's channel order with the binding's pyramid depth, and
/// the whole refinement in one call with the options `sfm cluster-patches`
/// passes, rather than in batches.
#[test]
fn the_file_holds_what_the_two_cli_steps_make_from_the_same_index() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    let file = state.cluster_patches(id).expect("built").path.clone();
    let built = sfmtool_matches_format::read_matches(&file).expect("a readable file");
    let recon = state.node(id).expect("loaded").recon();
    let images = recon.image_table.images.len();

    // `sfm match --cluster`, over the index: per-image counts off the `.sift`
    // metadata, a self-join at the binding's budget, the clustering.
    let sift: Vec<Option<PathBuf>> = (0..images)
        .map(|image| Some(recon.sift_path_for_image(image)).filter(|p| p.is_file()))
        .collect();
    let counts: Vec<u32> = sift
        .iter()
        .map(|path| match path {
            None => 0,
            Some(path) => {
                sfmtool_sift_format::read_sift_metadata(path)
                    .expect("readable")
                    .1
                    .feature_count
            }
        })
        .collect();
    let mut starts = vec![0u32];
    for count in &counts {
        starts.push(starts.last().unwrap() + count);
    }
    let index_path = state.sift_index(id).expect("built").path.clone();
    let forest = LazyKdForestU8::open(&index_path, LazyKdForestOptions::default()).unwrap();
    let (indexes, distances_sq) = forest.self_join_with_distances(11, 128, None).unwrap();
    let params = BackgroundFloorParams {
        d: 10,
        alpha: 0.8,
        min_size: 2,
        forest: KdForestParams {
            max_leaf_checks: 128,
            ..KdForestParams::accurate()
        },
    };
    let clusters = background_floor_clusters_from_neighbors(
        forest.len(),
        &starts,
        &params,
        &NeighborTable {
            indexes,
            distances_sq,
            width: 11,
        },
    )
    .unwrap();
    let members = clusters.member_images.len();
    let mut detected_positions = Array2::<f32>::zeros((members, 2));
    let mut detected_shapes = Array3::<f32>::zeros((members, 2, 2));
    for (k, (&image, &feature)) in clusters
        .member_images
        .iter()
        .zip(clusters.member_features.iter())
        .enumerate()
    {
        let path = sift[image as usize]
            .as_ref()
            .expect("a member has features");
        let data = sfmtool_sift_format::read_sift_partial(path, feature as usize + 1).unwrap();
        let row = feature as usize;
        detected_positions[[k, 0]] = data.positions_xy[[row, 0]];
        detected_positions[[k, 1]] = data.positions_xy[[row, 1]];
        for r in 0..2 {
            for c in 0..2 {
                detected_shapes[[k, r, c]] = data.affine_shapes[[row, r, c]];
            }
        }
    }
    let got = built.clusters.as_ref().expect("clusters");
    assert_eq!(got.cluster_starts, clusters.cluster_starts);
    assert_eq!(got.member_images, clusters.member_images);
    assert_eq!(got.member_features, clusters.member_features);
    assert_eq!(built.feature_counts.to_vec(), counts);

    // `sfm cluster-patches`: the photographs as `cv2.imread` hands them back,
    // the detections scattered to their rows, one refinement call.
    let pyramids: Vec<ImageU8Pyramid> = recon
        .image_table
        .images
        .iter()
        .map(|image| {
            let mut bgr = image::open(recon.workspace_dir.join(&image.name))
                .unwrap()
                .to_rgb8();
            for pixel in bgr.pixels_mut() {
                pixel.0.swap(0, 2);
            }
            let (w, h) = bgr.dimensions();
            let src = ImageU8::new(w, h, 3, bgr.into_raw());
            ImageU8Pyramid::build(&src, ImageU8Pyramid::full_levels(w, h))
        })
        .collect();
    let mut positions: Vec<Array2<f32>> = counts
        .iter()
        .map(|&n| Array2::zeros((n as usize, 2)))
        .collect();
    let mut shapes: Vec<Array3<f32>> = counts
        .iter()
        .map(|&n| Array3::zeros((n as usize, 2, 2)))
        .collect();
    for k in 0..members {
        let (image, row) = (
            clusters.member_images[k] as usize,
            clusters.member_features[k] as usize,
        );
        positions[image][[row, 0]] = detected_positions[[k, 0]];
        positions[image][[row, 1]] = detected_positions[[k, 1]];
        for r in 0..2 {
            for c in 0..2 {
                shapes[image][[row, r, c]] = detected_shapes[[k, r, c]];
            }
        }
    }
    let features: Vec<FeatureGeometry<'_>> = positions
        .iter()
        .zip(&shapes)
        .map(|(p, a)| FeatureGeometry {
            positions_xy: ArrayView2::from(p),
            affine_shapes: ArrayView3::from(a),
        })
        .collect();
    let refine_params = ClusterRefineParams {
        radius: 12.0 / 2.0,
        resolution: 25,
        window: PatchWindow::GaussianDisk { sigma: 0.5 },
        min_zncc: 0.85,
        max_shift_px: 3.0,
        max_keypoint_uncertainty: 0.35,
        max_iters: 120,
        ..ClusterRefineParams::default()
    };
    let result = refine_cluster_patches(
        &pyramids,
        &features,
        clusters.cluster_starts.as_slice().unwrap(),
        clusters.member_images.as_slice().unwrap(),
        clusters.member_features.as_slice().unwrap(),
        &refine_params,
        None,
    );
    let consistency = warp_consistency_residuals(
        clusters.cluster_starts.as_slice().unwrap(),
        clusters.member_images.as_slice().unwrap(),
        &result.member_status,
        &result.reference_members,
        result.member_affine_shapes.view(),
        images,
    );

    let patches = built.cluster_patches.as_ref().expect("cluster patches");
    assert_eq!(patches.reference_members.to_vec(), result.reference_members);
    let status: Vec<u8> = result.member_status.iter().map(|&s| s as u8).collect();
    assert_eq!(patches.member_status.to_vec(), status);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&patches.member_zncc.to_vec()),
        bits(&result.member_zncc)
    );
    assert_eq!(
        bits(&patches.member_shift_px.to_vec()),
        bits(&result.member_shift_px)
    );
    assert_eq!(
        bits(&patches.member_consistency_residual.to_vec()),
        bits(&consistency)
    );
    // The backbone's geometry is the refinement's where the cascade measured
    // the member, and the detection everywhere else.
    for k in 0..members {
        let measured = matches!(
            result.member_status[k],
            MemberStatus::Reference
                | MemberStatus::Kept
                | MemberStatus::RejectedLowZncc
                | MemberStatus::RejectedShift
        );
        let positions = got.member_positions.as_ref().expect("positions");
        let shapes = got.member_affine_shapes.as_ref().expect("shapes");
        for r in 0..2 {
            let want = match measured {
                true => result.member_positions[[k, r]] as f32,
                false => detected_positions[[k, r]],
            };
            assert_eq!(positions[[k, r]].to_bits(), want.to_bits(), "member {k}");
            for c in 0..2 {
                let want = match measured {
                    true => result.member_affine_shapes[[k, r, c]] as f32,
                    false => detected_shapes[[k, r, c]],
                };
                assert_eq!(shapes[[k, r, c]].to_bits(), want.to_bits(), "member {k}");
            }
        }
    }
    assert_eq!(
        patches.refine_options,
        serde_json::json!({
            "patch_size": 12.0,
            "resolution": 25,
            "min_zncc": 0.85,
            "max_shift_px": 3.0,
            "max_keypoint_uncertainty": 0.35,
        })
    );
}

// ── Current or stale ────────────────────────────────────────────────────

/// Features extracted again make the index stale, and a cluster-patches file
/// made from a stale index is stale too, naming the index.
#[test]
fn cluster_patches_from_a_stale_index_are_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    {
        let recon = state.node(id).expect("loaded").recon();
        crate::sift_index::tests::write_sift(
            &recon.sift_path_for_image(2),
            &recon.image_table.images[2].name,
            &vec![vec![7u8; 128]; 4],
            &[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
        );
    }
    let path = state.sift_index(id).expect("built").path.clone();
    state.open_sift_index(id, path).expect("it opens");
    assert_eq!(state.sift_index_state(id), IndexFileState::Stale);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Stale);
    let why = stale_reason(&state, id);
    assert!(why.contains("which is out of date"), "{why}");
}

/// A file made from an index other than the one open beside the node is stale,
/// even when that index is current.
#[test]
fn cluster_patches_from_another_index_are_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    // The image the fixture left without features gets some, and the index is
    // built again over them: a current index, and not the one the cluster
    // patches were made from.
    {
        let recon = state.node(id).expect("loaded").recon();
        let image = crate::sift_index::tests::UNINDEXED_IMAGE as usize;
        let sift = recon.sift_path_for_image(image);
        std::fs::create_dir_all(sift.parent().expect("a feature directory")).unwrap();
        crate::sift_index::tests::write_sift(
            &sift,
            &recon.image_table.images[image].name,
            &vec![vec![3u8; 128]; 2],
            &[[1.0, 2.0], [3.0, 4.0]],
        );
    }
    let plan = crate::sift_index::BuildPlan::of(state.node(id).expect("loaded")).unwrap();
    let built = crate::sift_index::build_index(plan, &sfmtool_core::progress::Progress::none());
    let index = built.ok().expect("the index is built again").path;
    state
        .open_index_files(id, Some(index.clone()), None)
        .expect("the new index opens");
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Stale);
    let why = stale_reason(&state, id);
    assert!(why.contains("other than"), "{why}");
    assert!(why.contains(&index.display().to_string()), "{why}");
}

/// With no index open there is nothing to say the file was made from the
/// node's features.
#[test]
fn cluster_patches_with_no_index_open_are_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    std::fs::remove_file(crate::sift_index::tests::index_of(dir.path())).unwrap();
    state.close_index_files(id).expect("both are open");
    state
        .open_index_files(id, None, None)
        .expect("the cluster patches are still there");
    assert_eq!(state.sift_index_state(id), IndexFileState::None);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Stale);
    let why = stale_reason(&state, id);
    assert!(why.contains("No SIFT index is open"), "{why}");
}

/// A version that moves the image table re-derives the verdict, and one that
/// does not leaves it alone.
#[test]
fn a_version_that_moves_the_image_table_makes_the_file_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, label) = searchable(dir.path());
    state
        .set_bench_verdict(id, &label, 0, sfmtool_core::bench::Verdict::Out)
        .expect("a verdict is a version");
    state.refresh_index_files(id);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Current);

    state
        .delete_image(ImageRef::new(id, 5))
        .expect("an image the node can lose");
    state.refresh_index_files(id);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Stale);
    let why = stale_reason(&state, id);
    assert!(why.contains("covers 8 images"), "{why}");
}

/// A file over other images opens all the same and says which image it first
/// disagrees on.
#[test]
fn cluster_patches_over_other_images_are_stale_naming_the_first() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    let path = state.cluster_patches(id).expect("built").path.clone();

    let other = tempfile::tempdir().unwrap();
    let (mut renamed, other_id) = state_in(other.path());
    {
        let node = renamed.scene.first_mut().expect("one node");
        let recon = Arc::make_mut(&mut node.history.current_mut().base);
        recon.image_table.images[0].name = "somewhere_else.jpg".into();
    }
    renamed
        .open_index_files(other_id, None, Some(path.clone()))
        .expect("a file that opens is adopted whether or not it fits");
    assert_eq!(
        renamed.cluster_patches_state(other_id),
        IndexFileState::Stale
    );
    let why = stale_reason(&renamed, other_id);
    assert!(why.contains("somewhere_else.jpg"), "{why}");
    assert_eq!(
        renamed.cluster_patches(other_id).expect("open").path,
        path,
        "a stale file stays named"
    );
}

/// A `.matches` with clusters and no cluster-patches section holds clusters
/// that were never refined, and is stale for that.
#[test]
fn a_clusters_file_with_no_patches_section_is_stale() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    let path = state.cluster_patches(id).expect("built").path.clone();
    let mut data = sfmtool_matches_format::read_matches(&path).unwrap();
    data.cluster_patches = None;
    data.metadata.has_cluster_patches = false;
    let clusters_only = dir.path().join("clusters.matches");
    sfmtool_matches_format::write_matches(&clusters_only, &data, 3).unwrap();

    state
        .open_index_files(id, None, Some(clusters_only))
        .expect("it opens");
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Stale);
    let why = stale_reason(&state, id);
    assert!(why.contains("no cluster patches section"), "{why}");
}

// ── Opening on sight ────────────────────────────────────────────────────

/// A file that is not there is the ordinary state, not a refusal, and the look
/// is remembered.
#[test]
fn looking_for_a_file_that_is_not_there_is_silent_and_remembered() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    state.refresh_index_files(id);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::None);
    assert!(
        !state
            .action_log
            .entries()
            .any(|entry| entry.kind == crate::action_log::Kind::Bench),
        "a file that is not there is the ordinary state"
    );
    assert!(state.cluster_patches.contains_key(&id));
}

/// A second session finds the files the first one built, and opens both on
/// sight with rows of the viewer's own.
#[test]
fn a_later_session_opens_both_files_on_sight() {
    let dir = tempfile::tempdir().unwrap();
    let (_built, _, _) = searchable(dir.path());
    let (mut state, id) = state_in(dir.path());
    state.refresh_index_files(id);
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Current);
    let opened = state
        .action_log
        .entries()
        .find(|entry| entry.text.starts_with("Opened the cluster patches"))
        .expect("the open wrote a row");
    assert_eq!(opened.actor, crate::action_log::Actor::Viewer);
}
