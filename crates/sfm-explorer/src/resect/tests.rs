// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `Resect Image`'s need of a current cluster-patches file: the sentence that
//! says why a missing or stale one will not do, and the step that refuses with
//! it. Also the fixture the other resection tests give a node its file with.

use std::path::{Path, PathBuf};

use sfmtool_core::SfmrReconstruction;
use sfmtool_matches_format::{
    ClusterMemberStatus, ClusterPatchData, ClustersData, MatchesContentHash, MatchesData,
    MatchesMetadata, WorkspaceContents, WorkspaceMetadata,
};

use crate::scene::{ImageRef, ReconId};
use crate::state::AppState;

/// A cluster-patches file over `recon` with one cluster per point, its kept
/// members at the point's observations: clusters that say what the tracks say.
///
/// Needs inline keypoints, which every resectable fixture carries.
pub(crate) fn write_cluster_patches_like_tracks(recon: &SfmrReconstruction, path: &Path) {
    let keypoints = recon.keypoints_xy().expect("an embedded fixture");
    let n = recon.image_table.images.len();
    let mut starts = vec![0u32];
    let mut member_images = Vec::new();
    let mut member_features = Vec::new();
    let mut positions = Vec::new();
    let mut status = Vec::new();
    let mut per_image = vec![0u32; n];
    for p in 0..recon.point_set.points.len() {
        let rows =
            recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1];
        for (k, row) in rows.enumerate() {
            let image = recon.point_set.tracks[row].image_index as usize;
            member_images.push(image as u32);
            member_features.push(per_image[image]);
            per_image[image] += 1;
            positions.extend([keypoints[[row, 0]], keypoints[[row, 1]]]);
            status.push(if k == 0 {
                ClusterMemberStatus::Reference
            } else {
                ClusterMemberStatus::Kept
            } as u8);
        }
        starts.push(member_images.len() as u32);
    }
    let m = member_images.len();
    let clusters = starts.len() - 1;
    let mut shapes = ndarray::Array3::<f32>::zeros((m, 2, 2));
    for k in 0..m {
        shapes[[k, 0, 0]] = 1.0;
        shapes[[k, 1, 1]] = 1.0;
    }
    let data = MatchesData {
        metadata: MatchesMetadata {
            version: sfmtool_matches_format::MATCHES_FORMAT_VERSION,
            matching_method: "cluster".into(),
            matching_tool: "test".into(),
            matching_tool_version: "0".into(),
            matching_options: Default::default(),
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
            image_count: n as u32,
            image_pair_count: None,
            match_count: None,
            cluster_count: Some(clusters as u32),
            cluster_member_count: Some(m as u32),
            has_two_view_geometries: false,
            has_clusters: true,
            has_cluster_patches: true,
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
        image_names: recon
            .image_table
            .images
            .iter()
            .map(|image| image.name.clone())
            .collect(),
        feature_tool_hashes: vec![[0u8; 16]; n],
        sift_content_hashes: vec![[0u8; 16]; n],
        feature_counts: per_image.iter().copied().collect(),
        image_dims: Some(
            ndarray::Array2::from_shape_vec(
                (n, 2),
                recon
                    .image_table
                    .images
                    .iter()
                    .flat_map(|image| {
                        let camera = &recon.image_table.cameras[image.camera_index as usize];
                        [camera.width, camera.height]
                    })
                    .collect(),
            )
            .expect("two per image"),
        ),
        image_pairs: None,
        clusters: Some(ClustersData {
            cluster_starts: starts.iter().copied().collect(),
            member_images: member_images.into_iter().collect(),
            member_features: member_features.into_iter().collect(),
            member_positions: Some(
                ndarray::Array2::from_shape_vec((m, 2), positions).expect("two per member"),
            ),
            member_affine_shapes: Some(shapes),
            matcher_options: serde_json::json!({}),
        }),
        cluster_patches: Some(ClusterPatchData {
            reference_members: starts[..clusters].iter().copied().collect(),
            member_status: status.into_iter().collect(),
            member_zncc: ndarray::Array1::from_elem(m, 1.0),
            member_shift_px: ndarray::Array1::zeros(m),
            member_consistency_residual: ndarray::Array1::zeros(m),
            refine_options: serde_json::json!({}),
        }),
        two_view_geometries: None,
    };
    sfmtool_matches_format::write_matches(path, &data, 3).expect("the fixture writes");
}

/// Give node `id` a current cluster-patches file whose clusters say what its
/// tracks say, written to a directory that outlives the test's state.
pub(crate) fn give_cluster_patches(state: &mut AppState, id: ReconId) -> PathBuf {
    let dir = tempfile::tempdir().expect("a temp dir").keep();
    let path = dir.join("run-cluster-patches.matches");
    write_cluster_patches_like_tracks(state.node(id).expect("loaded").recon(), &path);
    state.adopt_current_cluster_patches(id, path.clone());
    path
}

/// A resectable node, image 1 moved off its pose, with no cluster-patches file.
fn resectable() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    state.scene[0].recon_mut().image_table.images[1].translation_xyz +=
        nalgebra::Vector3::new(0.30, -0.20, 0.15);
    (state, id)
}

#[test]
fn with_no_file_the_step_refuses_naming_the_state_and_the_build() {
    let (mut state, id) = resectable();
    let refusal = state
        .resect_image_refusal(ImageRef::new(id, 1))
        .expect("no file is open");
    assert!(refusal.contains("none is open"), "{refusal}");
    assert!(refusal.contains("Build Index Files"), "{refusal}");

    let why = state.resect_image(id, 1).expect_err("the file is missing");
    assert!(why.ends_with(&refusal), "{why}");
    assert!(
        why.starts_with("Resect image_001.jpg in run_a refused: "),
        "{why}"
    );
    assert_eq!(state.scene[0].history.versions().len(), 1);
    let last = state.action_log.entries().last().expect("a row");
    assert!(last.failed);
    assert_eq!(last.text, why);
}

#[test]
fn with_a_stale_file_the_step_refuses_naming_the_state_and_the_reason() {
    let (mut state, id) = resectable();
    give_cluster_patches(&mut state, id);
    state.mark_cluster_patches_stale(id, "It covers other images.");
    let refusal = state
        .resect_image_refusal(ImageRef::new(id, 1))
        .expect("the file is stale");
    assert!(refusal.contains("out of date"), "{refusal}");
    assert!(refusal.contains("It covers other images."), "{refusal}");
    assert!(refusal.contains("Rebuild Index Files"), "{refusal}");

    let why = state.resect_image(id, 1).expect_err("the file is stale");
    assert!(why.ends_with(&refusal), "{why}");
    assert_eq!(state.scene[0].history.versions().len(), 1);
}

#[test]
fn an_unsaved_node_is_told_to_save_first() {
    let mut state = AppState::new();
    state.append_node(crate::scene::SceneNode::demo(
        crate::scene_graph::tests::resectable_node("/runs/run_a.sfmr")
            .recon()
            .clone(),
    ));
    let id = state.scene[0].id;
    let refusal = state
        .resect_image_refusal(ImageRef::new(id, 1))
        .expect("no file is open");
    assert!(refusal.contains("Save"), "{refusal}");
}

#[test]
fn the_image_s_own_reasons_come_before_the_file_s() {
    let (mut state, id) = resectable();
    state.scene[0].recon_mut().image_table.images[2].translation_xyz =
        nalgebra::Vector3::new(f64::NAN, 0.0, 0.0);
    assert_eq!(
        state.resect_image_refusal(ImageRef::new(id, 2)).as_deref(),
        Some(crate::image_menu::NOT_POSED_HINT)
    );
}

#[test]
fn with_a_current_file_the_log_line_splits_the_correspondences_by_source() {
    let (mut state, id) = resectable();
    give_cluster_patches(&mut state, id);
    assert_eq!(state.resect_image_refusal(ImageRef::new(id, 1)), None);

    state
        .resect_image(id, 1)
        .expect("the ring corroborates image 1");
    let last = state.action_log.entries().last().expect("a row");
    assert!(!last.failed, "{}", last.text);
    assert!(
        last.text
            .starts_with("Resected image_001.jpg (run_a): 240 pts (120 tracks, 120 clusters), "),
        "{}",
        last.text
    );
    assert!(
        last.text.contains(
            "; clusters 120 considered, 0 skipped, 0 failed to triangulate, 0 inconsistent"
        ),
        "{}",
        last.text
    );
    // The file is still current after the version: a resection moves no image.
    assert_eq!(state.resect_image_refusal(ImageRef::new(id, 1)), None);
}
