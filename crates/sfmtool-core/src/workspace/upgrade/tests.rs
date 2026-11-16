// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array1, Array2, Array3};
use sfmtool_archive_io::{parse_hash_bytes, WorkspaceContents, WorkspaceMetadata};
use sfmtool_matches_format::{
    ClustersData, MatchesContentHash, MatchesData, MatchesMetadata, PairsData,
};
use sfmtool_sift_format::{
    FeatureToolMetadata, SiftContentHash, SiftData, SiftMetadata, SIFT_ZSTD_LEVEL, THUMBNAIL_SIZE,
};

use super::*;
use crate::camera::image::ImageU8;
use crate::features::sift::xxh128_of_file;
use crate::progress::Event;
use crate::workspace::{feature_prefix_dir, upgrade_feature_options};

const SEOUL_BULL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture"
);

/// The images of the fixture, one for each of the demo reconstruction's.
const IMAGES: usize = 8;
/// How many features each synthetic `.sift` file holds.
const FEATURES: usize = 16;
/// How many points the reconstructions hold; each point's two observations
/// name feature `point index`, so this is at most [`FEATURES`].
const POINTS: usize = 12;

fn image_name(i: usize) -> String {
    format!("seoul_bull_sculpture_{:02}.jpg", i + 1)
}

fn v1_record() -> WorkspaceContents {
    let options = serde_json::json!({
        "max_num_features": FEATURES,
        "peak_ratio": 0.8,
        "gray_formula": "0.2126*R + 0.7152*G + 0.0722*B",
    });
    WorkspaceContents {
        feature_prefix_dir: feature_prefix_dir("sfmtool", "sift-sfmtool", &options),
        feature_tool: "sfmtool".into(),
        feature_type: "sift-sfmtool".into(),
        feature_options: options,
    }
}

/// The record the upgrade must convert [`v1_record`] to, stated from the two
/// public functions rather than read back from the upgrade.
fn v2_record() -> WorkspaceContents {
    let old = v1_record();
    let options = upgrade_feature_options(&old.feature_tool, &old.feature_options).unwrap();
    WorkspaceContents {
        feature_prefix_dir: feature_prefix_dir(&old.feature_tool, &old.feature_type, &options),
        feature_tool: old.feature_tool,
        feature_type: old.feature_type,
        feature_options: options,
    }
}

/// Write a version 1 `.sift` file for `image` into `folder`, with features
/// that vary with `seed` and a thumbnail; the image's hash, size and decoded
/// dimensions are its own, so the conversion accepts it.
fn write_version_1(image: &Path, folder: &Path, seed: usize) -> PathBuf {
    let decoded = ImageU8::read_rgb(image).unwrap();
    let (image_file_xxh128, image_file_size) = xxh128_of_file(image).unwrap();
    let name = image.file_name().unwrap().to_string_lossy().into_owned();
    std::fs::create_dir_all(folder).unwrap();
    let path = folder.join(format!("{name}.sift"));
    let record = v1_record();
    let data = SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: record.feature_tool,
            feature_type: record.feature_type,
            feature_options: record.feature_options,
        },
        metadata: SiftMetadata {
            version: 1,
            image_name: name,
            image_file_xxh128,
            image_file_size,
            image_width: decoded.width(),
            image_height: decoded.height(),
            feature_count: FEATURES as u32,
            sharpness: None,
            gradient_structure_tensor_xx_yy_xy: None,
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::from_shape_fn((FEATURES, 2), |(i, j)| {
            10.5 + 13.25 * i as f32 + 0.125 * j as f32 + seed as f32
        }),
        affine_shapes: Array3::from_shape_fn((FEATURES, 2, 2), |(i, j, k)| {
            1.5 + i as f32 - 0.25 * j as f32 + 0.0625 * k as f32
        }),
        descriptors: Array2::from_shape_fn((FEATURES, 128), |(i, j)| (i * 31 + j * 7 + seed) as u8),
        thumbnail_y_x_rgb: Array3::from_shape_fn(
            (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
            |(y, x, c)| (y + 2 * x + 3 * c) as u8,
        ),
    };
    sfmtool_sift_format::write_sift(&path, &data, SIFT_ZSTD_LEVEL).unwrap();
    path
}

fn hash_bytes(hex: &str) -> [u8; 16] {
    parse_hash_bytes(hex).unwrap()
}

/// A workspace in a temporary directory, as sfmtool 0.3 would leave it.
struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn images(&self) -> PathBuf {
        self.root.join("images")
    }
    fn old_folder(&self) -> PathBuf {
        self.images().join(v1_record().feature_prefix_dir)
    }
    fn new_folder(&self) -> PathBuf {
        self.images().join(v2_record().feature_prefix_dir)
    }
    fn orphan_folder(&self) -> PathBuf {
        self.images().join("features/sift-orphan")
    }
    fn nested(&self) -> PathBuf {
        self.root.join("nested")
    }
    fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

/// A reconstruction of the fixture's images in its workspace, recording
/// `contents` and the `.sift` files in `folder`: `embedded` keeps each
/// observation's keypoint inline and the images' hashes, otherwise it names
/// the `.sift` files by content hash.
fn reconstruction(
    root: &Path,
    folder: &Path,
    contents: WorkspaceContents,
    embedded: bool,
) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(POINTS);
    let sift: Vec<SiftData> = (0..IMAGES)
        .map(|i| {
            sfmtool_sift_format::read_sift(&folder.join(format!("{}.sift", image_name(i)))).unwrap()
        })
        .collect();
    recon.image_set.cameras[0].width = sift[0].metadata.image_width;
    recon.image_set.cameras[0].height = sift[0].metadata.image_height;
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("images/{}", image_name(i));
    }
    recon.metadata.workspace = WorkspaceMetadata {
        absolute_path: String::new(),
        relative_path: "..".into(),
        contents,
    };
    recon.workspace_dir = root.to_path_buf();
    let ObservationSource::SiftFiles {
        feature_indexes, ..
    } = &recon.point_set.observations
    else {
        unreachable!("the demo names .sift features")
    };
    let feature_indexes = feature_indexes.clone();
    if embedded {
        // An `embedded_patches` file anchors each observation's patch, which
        // needs a patch frame for every point.
        recon.point_set.patch_u_halfvec_xyz = Some(Array2::from_shape_fn((POINTS, 3), |(_, j)| {
            [0.01, 0.0, 0.0][j]
        }));
        recon.point_set.patch_v_halfvec_xyz = Some(Array2::from_shape_fn((POINTS, 3), |(_, j)| {
            [0.0, 0.01, 0.0][j]
        }));
    }
    recon.point_set.observations = if embedded {
        let keypoints = Array2::from_shape_fn((feature_indexes.len(), 2), |(o, j)| {
            let image = recon.point_set.tracks[o].image_index as usize;
            sift[image].positions_xy[[feature_indexes[o] as usize, j]]
        });
        ObservationSource::EmbeddedPatches {
            keypoints_xy: keypoints,
            image_file_hashes: sift
                .iter()
                .map(|s| hash_bytes(&s.metadata.image_file_xxh128))
                .collect(),
        }
    } else {
        ObservationSource::SiftFiles {
            feature_indexes,
            keypoints_xy: None,
            feature_tool_hashes: sift
                .iter()
                .map(|s| hash_bytes(&s.content_hash.feature_tool_xxh128))
                .collect(),
            sift_content_hashes: sift
                .iter()
                .map(|s| hash_bytes(&s.content_hash.content_xxh128))
                .collect(),
        }
    };
    recon
}

/// A pairwise `.matches` file of the first three images, recording
/// `contents` and the `.sift` files in `folder`.
fn pairwise_matches(folder: &Path, contents: WorkspaceContents) -> MatchesData {
    let sift: Vec<_> = (0..3)
        .map(|i| {
            sfmtool_sift_format::read_sift_metadata(&folder.join(format!("{}.sift", image_name(i))))
                .unwrap()
        })
        .collect();
    MatchesData {
        metadata: MatchesMetadata {
            version: sfmtool_matches_format::MATCHES_FORMAT_VERSION,
            matching_method: "exhaustive".into(),
            matching_tool: "sfmtool".into(),
            matching_tool_version: "0".into(),
            matching_options: BTreeMap::new(),
            workspace: WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: "..".into(),
                contents,
            },
            timestamp: "2026-11-16T00:00:00Z".into(),
            image_count: 3,
            image_pair_count: Some(2),
            match_count: Some(3),
            cluster_count: None,
            cluster_member_count: None,
            has_two_view_geometries: false,
            has_clusters: false,
            has_cluster_patches: false,
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
        image_names: (0..3)
            .map(|i| format!("images/{}", image_name(i)))
            .collect(),
        feature_tool_hashes: sift
            .iter()
            .map(|(_, _, h)| hash_bytes(&h.feature_tool_xxh128))
            .collect(),
        sift_content_hashes: sift
            .iter()
            .map(|(_, _, h)| hash_bytes(&h.content_xxh128))
            .collect(),
        feature_counts: Array1::from_elem(3, FEATURES as u32),
        image_dims: Some(Array2::from_shape_fn((3, 2), |(i, j)| {
            let m = &sift[i].1;
            [m.image_width, m.image_height][j]
        })),
        image_pairs: Some(PairsData {
            image_index_pairs: Array2::from_shape_vec((2, 2), vec![0, 1, 1, 2]).unwrap(),
            match_counts: Array1::from_vec(vec![2, 1]),
            match_feature_indexes: Array2::from_shape_vec((3, 2), vec![0, 1, 2, 3, 4, 5]).unwrap(),
            match_descriptor_distances: Array1::from_vec(vec![10.0, 20.0, 30.0]),
        }),
        clusters: None,
        cluster_patches: None,
        two_view_geometries: None,
    }
}

/// [`pairwise_matches`] with one cluster of the three images instead of the
/// pairs.
fn cluster_matches(folder: &Path, contents: WorkspaceContents) -> MatchesData {
    let mut data = pairwise_matches(folder, contents);
    data.metadata.image_pair_count = None;
    data.metadata.match_count = None;
    data.metadata.cluster_count = Some(1);
    data.metadata.cluster_member_count = Some(3);
    data.metadata.has_clusters = true;
    data.image_pairs = None;
    data.clusters = Some(ClustersData {
        cluster_starts: Array1::from_vec(vec![0, 3]),
        member_images: Array1::from_vec(vec![0, 1, 2]),
        member_features: Array1::from_vec(vec![0, 1, 2]),
        member_positions: Some(Array2::from_elem((3, 2), 12.5)),
        member_affine_shapes: Some(Array3::from_shape_fn((3, 2, 2), |(_, j, k)| {
            if j == k {
                2.0
            } else {
                0.0
            }
        })),
        matcher_options: serde_json::json!({}),
    });
    data
}

/// Rewrite the archive at `path` in place: the JSON entry `entry` through
/// `edit`, and without the entries `drop` names. Neither `.matches` nor
/// `.camrig` readers check the content hashes, so an older writer's bytes
/// are produced this way.
fn rewrite_archive(
    path: &Path,
    entry: &str,
    edit: impl Fn(&mut serde_json::Value),
    drop: impl Fn(&str) -> bool,
) {
    let bytes = std::fs::read(path).unwrap();
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    let names: Vec<String> = archive
        .file_names()
        .map(|s| s.unwrap().into_owned())
        .collect();
    let mut out = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for name in names {
        if drop(&name) {
            continue;
        }
        let mut raw = Vec::new();
        archive
            .by_name(&name)
            .unwrap()
            .read_to_end(&mut raw)
            .unwrap();
        if name == entry {
            let mut json: serde_json::Value =
                serde_json::from_slice(&zstd::stream::decode_all(&raw[..]).unwrap()).unwrap();
            edit(&mut json);
            raw = zstd::bulk::compress(&serde_json::to_vec(&json).unwrap(), 3).unwrap();
        }
        out.start_file(name, stored).unwrap();
        out.write_all(&raw).unwrap();
    }
    out.finish().unwrap();
}

/// The top-level metadata entry of every archive format.
const METADATA: &str = "metadata.json.zst";

fn set_version(version: u32) -> impl Fn(&mut serde_json::Value) {
    move |json| {
        let map = json.as_object_mut().unwrap();
        map.insert("version".into(), version.into());
        if version <= 2 {
            for key in [
                "has_clusters",
                "has_cluster_patches",
                "cluster_count",
                "cluster_member_count",
            ] {
                map.remove(key);
            }
        }
    }
}

/// A `.kdf` index of three features of the first image's `.sift` file,
/// recording `contents`.
fn write_index(path: &Path, folder: &Path, contents: WorkspaceContents) {
    use sfmtool_kdf_format::*;
    let sift =
        sfmtool_sift_format::read_sift(&folder.join(format!("{}.sift", image_name(0)))).unwrap();
    let vectors: Vec<u8> = sift.descriptors.iter().take(3 * 128).copied().collect();
    let data = KdfForestData {
        vectors: &vectors,
        feature_count: 3,
        dimension: 128,
        trees: vec![KdfTree {
            nodes: vec![
                KdfNode::Internal {
                    split_dimension: 0,
                    split: vectors[128],
                    left: 1,
                    right: 2,
                },
                KdfNode::Leaf { start: 0, len: 2 },
                KdfNode::Leaf { start: 2, len: 1 },
            ],
            feature_ids: vec![0, 1, 2],
        }],
        provenance: None,
        descriptor_order: None,
    };
    let sources = KdfSiftSources {
        workspace: KdfWorkspaceMetadata {
            absolute_path: String::new(),
            relative_path: "..".into(),
            contents: KdfWorkspaceContents {
                feature_tool: contents.feature_tool,
                feature_type: contents.feature_type,
                feature_options: contents.feature_options,
                feature_prefix_dir: contents.feature_prefix_dir,
            },
        },
        image_names: vec![format!("images/{}", image_name(0))],
        feature_tool_hashes: vec![hash_bytes(&sift.content_hash.feature_tool_xxh128)],
        sift_content_hashes: vec![hash_bytes(&sift.content_hash.content_xxh128)],
        origins: (0..3)
            .map(|f| FeatureOrigin {
                image_index: 0,
                image_feature_index: f,
            })
            .collect(),
        geometry: (0..3)
            .map(|f| [[f as f32, f as f32], [1.0, 0.0], [0.0, 1.0]])
            .collect(),
    };
    sfmtool_kdf_format::write_kdf(
        path,
        &data,
        Some(&sources),
        &KdfWriteOptions::default(),
        &Progress::none(),
    )
    .unwrap();
}

fn write_workspace_file(dir: &Path) {
    let record = v1_record();
    let config = serde_json::json!({
        "version": 1,
        "feature_tool": record.feature_tool,
        "feature_type": record.feature_type,
        "feature_options": record.feature_options,
        "feature_prefix_dir": record.feature_prefix_dir,
        "custom_note": "kept as it is",
    });
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join(WORKSPACE_FILE_NAME),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

/// The fixture: a version 1 workspace file; [`IMAGES`] images with version 1
/// `.sift` files in the record's folder; an orphan version 1 folder; a
/// `sift_files` and an `embedded_patches` `.sfmr`; a `.sfmr` whose recorded
/// `.sift` hash is not its file's (`stale`); a `.sfmr` of the nested
/// workspace (`other`); a version 1 pairwise, a version 10 cluster and a
/// version 5 cluster `.matches`; a version 1 `.camrig`; a `.kdf` of the old
/// folder; and a nested workspace with a version 1 folder of its own.
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("ws");
    let fixture = Fixture {
        _dir: dir,
        root: root.clone(),
    };
    write_workspace_file(&root);
    std::fs::create_dir_all(fixture.images()).unwrap();
    for i in 0..IMAGES {
        let image = fixture.images().join(image_name(i));
        std::fs::copy(Path::new(SEOUL_BULL).join(image_name(i)), &image).unwrap();
        write_version_1(&image, &fixture.old_folder(), 0);
    }
    for i in 0..2 {
        write_version_1(
            &fixture.images().join(image_name(i)),
            &fixture.orphan_folder(),
            7,
        );
    }

    let old = fixture.old_folder();
    std::fs::create_dir_all(root.join("sfmr")).unwrap();
    reconstruction(&root, &old, v1_record(), false)
        .save(&fixture.path("sfmr/sift.sfmr"), &Progress::none())
        .unwrap();
    reconstruction(&root, &old, v1_record(), true)
        .save(&fixture.path("sfmr/embedded.sfmr"), &Progress::none())
        .unwrap();
    let mut stale = reconstruction(&root, &old, v1_record(), false);
    if let ObservationSource::SiftFiles {
        sift_content_hashes,
        ..
    } = &mut stale.point_set.observations
    {
        sift_content_hashes[3] = [9; 16];
    }
    stale
        .save(&fixture.path("sfmr/stale.sfmr"), &Progress::none())
        .unwrap();

    write_workspace_file(&fixture.nested());
    let nested_sift = fixture
        .nested()
        .join("images")
        .join(v1_record().feature_prefix_dir);
    std::fs::create_dir_all(&nested_sift).unwrap();
    let name = format!("{}.sift", image_name(0));
    std::fs::copy(old.join(&name), nested_sift.join(&name)).unwrap();
    let mut other = reconstruction(&root, &old, v1_record(), false);
    other.metadata.workspace.relative_path = "../nested".into();
    other
        .save(&fixture.path("sfmr/other.sfmr"), &Progress::none())
        .unwrap();

    let pairs = fixture.path("matches/pairs.matches");
    sfmtool_matches_format::write_matches(&pairs, &pairwise_matches(&old, v1_record()), 3).unwrap();
    rewrite_archive(&pairs, METADATA, set_version(1), |n| {
        n.starts_with("images/image_dims.")
    });
    let clusters = fixture.path("matches/clusters.matches");
    sfmtool_matches_format::write_matches(&clusters, &cluster_matches(&old, v1_record()), 3)
        .unwrap();
    let old_clusters = fixture.path("matches/old-clusters.matches");
    std::fs::copy(&clusters, &old_clusters).unwrap();
    rewrite_archive(&old_clusters, METADATA, set_version(5), |n| {
        n.starts_with("clusters/member_positions.")
            || n.starts_with("clusters/member_affine_shapes.")
    });

    let rig = fixture.path("rig.camrig");
    write_rig(&rig);
    rewrite_archive(&rig, METADATA, set_version(1), |_| false);

    write_index(&fixture.path("sfmr/sift-sift-index.kdf"), &old, v1_record());
    fixture
}

fn write_rig(path: &Path) {
    use sfmtool_camrig_format::*;
    let mut parameters = BTreeMap::new();
    for (key, value) in [
        ("focal_length_x", 300.0),
        ("focal_length_y", 300.0),
        ("principal_point_x", 135.0),
        ("principal_point_y", 240.0),
    ] {
        parameters.insert(key.to_string(), value);
    }
    let data = CamRigData {
        metadata: CamRigMetadata {
            version: CAMRIG_FORMAT_VERSION,
            name: "pair".into(),
            sensor_count: 2,
            camera_count: 1,
            rig_type: "stereo_pair".into(),
            rig_attributes: serde_json::json!({}),
        },
        content_hash: CamRigContentHash::default(),
        cameras: vec![CamRigCamera {
            model: "PINHOLE".into(),
            width: 270,
            height: 480,
            parameters,
        }],
        sensor_image_patterns: vec!["left/%04d.jpg".into(), "right/%04d.jpg".into()],
        camera_indexes: vec![0, 0],
        quaternions_wxyz: Array2::from_shape_vec(
            (2, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap(),
        translations_xyz: Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75])
            .unwrap(),
    };
    write_camrig(path, &data, 3).unwrap();
}

/// Every file under `dir`, by path, with its bytes.
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Bytes> {
    let mut files = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.insert(path.clone(), Bytes(std::fs::read(&path).unwrap()));
            }
        }
    }
    files
}

fn outcome_of<'a>(report: &'a UpgradeWorkspaceReport, path: &Path) -> &'a UpgradeOutcome {
    &report
        .files
        .iter()
        .find(|f| f.path == path)
        .unwrap_or_else(|| panic!("{} is not in the report", path.display()))
        .outcome
}

fn not_converted(outcome: &UpgradeOutcome) -> bool {
    matches!(outcome, UpgradeOutcome::NotConverted(_))
}

/// The state a finished upgrade of the fixture leaves, whatever the run
/// that left it: `before` is the fixture as made.
fn assert_upgraded(fixture: &Fixture, before: &BTreeMap<PathBuf, Bytes>) {
    let new_record = v2_record();
    let new_folder = fixture.new_folder();

    // The workspace file is version 2, names the new folder, and keeps every
    // other key.
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.root.join(WORKSPACE_FILE_NAME)).unwrap())
            .unwrap();
    assert_eq!(config["version"], 2);
    assert_eq!(
        config["feature_prefix_dir"],
        new_record.feature_prefix_dir.as_str()
    );
    assert_eq!(config["feature_options"], new_record.feature_options);
    assert_eq!(config["custom_note"], "kept as it is");

    // Every image has a version 2 file in the new folder, and the old and
    // orphan folders are gone with their `features` parent kept for the new.
    let mut new_sift = Vec::new();
    for i in 0..IMAGES {
        let data =
            sfmtool_sift_format::read_sift(&new_folder.join(format!("{}.sift", image_name(i))))
                .unwrap();
        assert_eq!(data.metadata.version, 2);
        new_sift.push(data);
    }
    assert!(!fixture.old_folder().exists());
    assert!(!fixture.orphan_folder().exists());
    assert!(fixture.images().join("features").is_dir());

    // The `sift_files` reconstruction names the new folder and files, and its
    // feature indexes give the keypoints the old files held; its points and
    // observations are unchanged.
    let original = load_without_workspace(&before[&fixture.path("sfmr/sift.sfmr")].0);
    let recon =
        SfmrReconstruction::load(&fixture.path("sfmr/sift.sfmr"), &Progress::none()).unwrap();
    assert_eq!(
        sfmtool_sfmr_format::read_sfmr_metadata(&fixture.path("sfmr/sift.sfmr"))
            .unwrap()
            .version,
        sfmtool_sfmr_format::SFMR_FORMAT_VERSION
    );
    assert_eq!(
        recon.metadata.workspace.contents.feature_prefix_dir,
        new_record.feature_prefix_dir
    );
    assert_eq!(
        recon.metadata.workspace.contents.feature_options,
        new_record.feature_options
    );
    let ObservationSource::SiftFiles {
        feature_indexes,
        keypoints_xy,
        feature_tool_hashes,
        sift_content_hashes,
    } = &recon.point_set.observations
    else {
        panic!("the reconstruction is no longer sift_files")
    };
    for (i, data) in new_sift.iter().enumerate() {
        assert_eq!(
            sift_content_hashes[i],
            hash_bytes(&data.content_hash.content_xxh128)
        );
        assert_eq!(
            feature_tool_hashes[i],
            hash_bytes(&data.content_hash.feature_tool_xxh128)
        );
    }
    let keypoints = keypoints_xy.as_ref().unwrap();
    for (o, track) in recon.point_set.tracks.iter().enumerate() {
        let sift = &new_sift[track.image_index as usize];
        let feature = feature_indexes[o] as usize;
        let expected = 10.5 + 13.25 * feature as f32;
        assert_eq!(sift.positions_xy[[feature, 0]], expected);
        assert_eq!(keypoints[[o, 0]], expected);
        assert_eq!(keypoints[[o, 1]], expected + 0.125);
    }
    assert_eq!(recon.point_set.points, original.0);
    assert_eq!(recon.point_set.tracks, original.1);
    assert_eq!(feature_indexes, &original.2);

    // The `embedded_patches` reconstruction names the new folder, its
    // keypoints unchanged.
    let embedded =
        SfmrReconstruction::load(&fixture.path("sfmr/embedded.sfmr"), &Progress::none()).unwrap();
    assert_eq!(
        embedded.metadata.workspace.contents.feature_prefix_dir,
        new_record.feature_prefix_dir
    );
    let original_embedded = load_without_workspace(&before[&fixture.path("sfmr/embedded.sfmr")].0);
    assert_eq!(embedded.keypoints_xy(), original_embedded.3.as_ref());

    // The version 1 pairwise `.matches` is version 10, names the new files,
    // and takes the image sizes from them.
    let pairs =
        sfmtool_matches_format::read_matches_metadata(&fixture.path("matches/pairs.matches"))
            .unwrap();
    assert_eq!(
        pairs.version,
        sfmtool_matches_format::MATCHES_FORMAT_VERSION
    );
    let pairs =
        sfmtool_matches_format::read_matches(&fixture.path("matches/pairs.matches")).unwrap();
    assert_eq!(
        pairs.metadata.workspace.contents.feature_prefix_dir,
        new_record.feature_prefix_dir
    );
    for i in 0..3 {
        let m = &new_sift[i].metadata;
        assert_eq!(
            pairs.sift_content_hashes[i],
            hash_bytes(&new_sift[i].content_hash.content_xxh128)
        );
        let dims = pairs.image_dims.as_ref().unwrap();
        assert_eq!(
            (dims[[i, 0]], dims[[i, 1]]),
            (m.image_width, m.image_height)
        );
    }
    let clusters =
        sfmtool_matches_format::read_matches(&fixture.path("matches/clusters.matches")).unwrap();
    assert_eq!(
        clusters.sift_content_hashes[2],
        hash_bytes(&new_sift[2].content_hash.content_xxh128)
    );

    // The `.camrig` is version 2, and the old folder's index is gone.
    assert_eq!(
        sfmtool_camrig_format::read_camrig_metadata(&fixture.path("rig.camrig"))
            .unwrap()
            .0
            .version,
        2
    );
    assert!(!fixture.path("sfmr/sift-sift-index.kdf").exists());

    // The files not converted, and the nested workspace, are as they were.
    let after = snapshot(&fixture.root);
    for unchanged in [
        "sfmr/stale.sfmr",
        "sfmr/other.sfmr",
        "matches/old-clusters.matches",
    ] {
        let path = fixture.path(unchanged);
        assert_eq!(after[&path], before[&path], "{unchanged} changed");
    }
    for (path, bytes) in before {
        if path.starts_with(fixture.nested()) {
            assert_eq!(&after[path], bytes, "{} changed", path.display());
        }
    }
}

/// The points, tracks, feature indexes and inline keypoints of the `.sfmr`
/// file `bytes`, read without its workspace.
fn load_without_workspace(
    bytes: &[u8],
) -> (
    Vec<crate::Point3D>,
    Vec<crate::TrackObservation>,
    Vec<u32>,
    Option<Array2<f32>>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("copy.sfmr");
    std::fs::write(&path, bytes).unwrap();
    let data = sfmtool_sfmr_format::read_sfmr(&path).unwrap();
    let recon = SfmrReconstruction::from_sfmr_data(data).unwrap();
    (
        recon.point_set.points.clone(),
        recon.point_set.tracks.clone(),
        recon
            .feature_indexes()
            .map(<[u32]>::to_vec)
            .unwrap_or_default(),
        recon.keypoints_xy().cloned(),
    )
}

fn run(fixture: &Fixture, dry_run: bool) -> UpgradeWorkspaceReport {
    upgrade_workspace(
        &fixture.images(),
        &UpgradeWorkspaceOptions { dry_run },
        &Progress::none(),
    )
    .unwrap()
}

#[test]
fn a_version_1_workspace_is_upgraded_and_reported() {
    let fixture = fixture();
    let before = snapshot(&fixture.root);
    let report = run(&fixture, false);
    assert_upgraded(&fixture, &before);

    let path = |p: &str| fixture.path(p);
    assert_eq!(report.workspace_dir, fixture.root);
    assert!(!report.dry_run);
    assert_eq!(
        outcome_of(&report, &fixture.root.join(WORKSPACE_FILE_NAME)),
        &UpgradeOutcome::Converted
    );
    for converted in [
        "sfmr/sift.sfmr",
        "sfmr/embedded.sfmr",
        "matches/pairs.matches",
        "matches/clusters.matches",
        "rig.camrig",
    ] {
        assert_eq!(
            outcome_of(&report, &path(converted)),
            &UpgradeOutcome::Converted,
            "{converted}"
        );
    }
    assert_eq!(
        outcome_of(&report, &path("sfmr/sift-sift-index.kdf")),
        &UpgradeOutcome::Deleted
    );
    for left in [
        "sfmr/stale.sfmr",
        "sfmr/other.sfmr",
        "matches/old-clusters.matches",
    ] {
        assert!(not_converted(outcome_of(&report, &path(left))), "{left}");
    }
    assert_eq!(report.nested_workspaces, vec![fixture.nested()]);
    assert_eq!(report.converted_sift_file_count, IMAGES);
    assert!(report.old_folders_deleted);
    let folders: BTreeSet<_> = report
        .folders
        .iter()
        .map(|f| {
            (
                f.old_folder.clone(),
                f.new_folder.clone(),
                f.sift_file_count,
                f.deleted,
            )
        })
        .collect();
    assert_eq!(
        folders,
        BTreeSet::from([
            (
                fixture.old_folder(),
                Some(fixture.new_folder()),
                IMAGES,
                true
            ),
            (fixture.orphan_folder(), None, 2, true),
        ])
    );
    assert!(report.notes.iter().any(|n| n.contains("pt3d_")));
    assert!(!report.nothing_to_do());

    // A second run finds nothing to do and writes nothing.
    let settled = snapshot(&fixture.root);
    let again = run(&fixture, false);
    assert!(again.nothing_to_do(), "{again:?}");
    assert_eq!(snapshot(&fixture.root), settled);
}

#[test]
fn a_dry_run_reports_the_plan_and_writes_nothing() {
    let fixture = fixture();
    let before = snapshot(&fixture.root);
    let report = run(&fixture, true);
    assert_eq!(snapshot(&fixture.root), before);
    assert!(report.dry_run);
    assert_eq!(
        outcome_of(&report, &fixture.path("sfmr/sift.sfmr")),
        &UpgradeOutcome::Converted
    );
    assert_eq!(
        outcome_of(&report, &fixture.path("sfmr/sift-sift-index.kdf")),
        &UpgradeOutcome::Deleted
    );
    // The dry run finds the reconstruction whose `.sift` file has changed.
    assert!(not_converted(outcome_of(
        &report,
        &fixture.path("sfmr/stale.sfmr")
    )));
    assert_eq!(report.converted_sift_file_count, IMAGES);
    assert!(report.old_folders_deleted);
    assert_eq!(report.folders.len(), 2);
}

#[test]
fn a_missing_image_fails_the_run_before_any_file_is_rewritten() {
    let fixture = fixture();
    let missing = fixture.images().join(image_name(5));
    std::fs::remove_file(&missing).unwrap();
    let before = snapshot(&fixture.root);
    let error = upgrade_workspace(
        &fixture.root,
        &UpgradeWorkspaceOptions::default(),
        &Progress::none(),
    )
    .unwrap_err();
    let UpgradeWorkspaceError::SiftFiles(failures) = &error else {
        panic!("{error}")
    };
    assert_eq!(failures.len(), 1);
    assert_eq!(
        failures[0].sift_path,
        fixture.old_folder().join(format!("{}.sift", image_name(5)))
    );
    // Only new `.sift` files were written: every file that was there is as
    // it was, the old folders included.
    let after = snapshot(&fixture.root);
    for (path, bytes) in &before {
        assert_eq!(after.get(path), Some(bytes), "{} changed", path.display());
    }
    assert!(after
        .keys()
        .filter(|p| !before.contains_key(*p))
        .all(|p| p.starts_with(fixture.new_folder())));
}

#[test]
fn two_old_folders_with_different_files_for_one_new_file_are_refused() {
    let fixture = fixture();
    // A hand-named folder with the same record, named by a reconstruction,
    // holding another file for the first image.
    let mut record = v1_record();
    record.feature_prefix_dir = "features/sift-hand-named".into();
    let hand = fixture.images().join(&record.feature_prefix_dir);
    for i in 0..IMAGES {
        let seed = usize::from(i == 0);
        write_version_1(&fixture.images().join(image_name(i)), &hand, seed);
    }
    reconstruction(&fixture.root, &hand, record, true)
        .save(&fixture.path("sfmr/hand.sfmr"), &Progress::none())
        .unwrap();
    let before = snapshot(&fixture.root);
    let error = run_error(&fixture, true);
    let UpgradeWorkspaceError::Conflicts(conflicts) = &error else {
        panic!("{error}")
    };
    assert_eq!(conflicts.len(), 1, "{error}");
    let new_file = fixture.new_folder().join(format!("{}.sift", image_name(0)));
    assert_eq!(conflicts[0].new_file, new_file);
    let pair = BTreeSet::from([conflicts[0].first.clone(), conflicts[0].second.clone()]);
    assert_eq!(
        pair,
        BTreeSet::from([
            hand.join(format!("{}.sift", image_name(0))),
            fixture.old_folder().join(format!("{}.sift", image_name(0))),
        ])
    );
    assert_eq!(snapshot(&fixture.root), before);
}

fn run_error(fixture: &Fixture, dry_run: bool) -> UpgradeWorkspaceError {
    upgrade_workspace(
        &fixture.root,
        &UpgradeWorkspaceOptions { dry_run },
        &Progress::none(),
    )
    .unwrap_err()
}

/// Cancel a run when `stop` says so for an event, check that it stopped, then
/// run again and check the workspace is upgraded.
fn stop_then_finish(stop: impl Fn(&Event<'_>, &mut bool) -> bool + Sync) {
    let fixture = fixture();
    let before = snapshot(&fixture.root);
    let flag = AtomicBool::new(false);
    let state = std::sync::Mutex::new(false);
    let sink = |event: Event<'_>| {
        if stop(&event, &mut state.lock().unwrap()) {
            flag.store(true, Ordering::Relaxed);
        }
    };
    let progress = Progress::to(&sink).cancelled_by(&flag);
    let error = upgrade_workspace(
        &fixture.root,
        &UpgradeWorkspaceOptions::default(),
        &progress,
    )
    .unwrap_err();
    assert!(matches!(error, UpgradeWorkspaceError::Cancelled), "{error}");
    assert!(flag.load(Ordering::Relaxed));
    run(&fixture, false);
    assert_upgraded(&fixture, &before);
}

fn at_phase(name: &'static str) -> impl Fn(&Event<'_>, &mut bool) -> bool + Sync {
    move |event, _| matches!(event, Event::Enter { phase, .. } if *phase == name)
}

#[test]
fn a_run_stopped_before_converting_finishes_when_run_again() {
    stop_then_finish(at_phase("convert .sift files"));
}

#[test]
fn a_run_stopped_before_rewriting_finishes_when_run_again() {
    stop_then_finish(at_phase("rewrite files"));
}

#[test]
fn a_run_stopped_part_way_through_rewriting_finishes_when_run_again() {
    // Stopped after the first file of phase 4 is written.
    stop_then_finish(|event, rewriting| {
        match event {
            Event::Enter { phase, .. } if *phase == "rewrite files" => *rewriting = true,
            Event::Count { unit: "file", .. } if *rewriting => return true,
            _ => {}
        }
        false
    });
}

#[test]
fn a_run_stopped_before_writing_the_workspace_file_finishes_when_run_again() {
    stop_then_finish(at_phase("workspace file"));
}

#[test]
fn a_run_stopped_before_deleting_finishes_when_run_again() {
    stop_then_finish(at_phase("delete old folders"));
}

#[test]
fn an_undistorted_record_stops_the_run_before_anything_is_written() {
    let fixture = fixture();
    let mut record = v1_record();
    record.feature_tool = "sfmtool-undistort".into();
    reconstruction(&fixture.root, &fixture.old_folder(), record, true)
        .save(&fixture.path("sfmr/undistorted.sfmr"), &Progress::none())
        .unwrap();
    let before = snapshot(&fixture.root);
    let error = run_error(&fixture, false);
    let UpgradeWorkspaceError::FeatureOptions { files, error } = &error else {
        panic!("{error}")
    };
    assert_eq!(files, &vec![fixture.path("sfmr/undistorted.sfmr")]);
    assert_eq!(error, &FeatureOptionsError::Undistort);
    assert_eq!(snapshot(&fixture.root), before);
}

/// A file's bytes, which a failed comparison prints as their length and hash
/// rather than in full.
#[derive(PartialEq, Eq)]
struct Bytes(Vec<u8>);

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hash = xxhash_rust::xxh3::xxh3_128(&self.0);
        write!(f, "{} bytes, xxh3 {hash:032x}", self.0.len())
    }
}

/// Write a `.sfmr` naming the fixture's old folder whose camera lacks a
/// parameter, as a file from before a camera model gained one: the format
/// reader reads it, the reconstruction does not.
fn write_refused_sfmr(fixture: &Fixture, path: &Path) {
    reconstruction(&fixture.root, &fixture.old_folder(), v1_record(), false)
        .save(path, &Progress::none())
        .unwrap();
    rewrite_archive(
        path,
        "cameras/metadata.json.zst",
        |cameras| {
            let camera = &mut cameras.as_array_mut().unwrap()[0];
            camera["parameters"]
                .as_object_mut()
                .unwrap()
                .remove("focal_length_y");
        },
        |_| false,
    );
}

#[test]
fn a_folder_a_refused_reconstruction_names_is_kept_and_then_settled() {
    let fixture = fixture();
    let path = fixture.path("sfmr/refused.sfmr");
    write_refused_sfmr(&fixture, &path);
    let refused = std::fs::read(&path).unwrap();
    let is_refusal = |outcome: &UpgradeOutcome| matches!(outcome, UpgradeOutcome::NotConverted(r) if r.contains("reader refuses"));
    let kept = |report: &UpgradeWorkspaceReport| {
        report
            .folders
            .iter()
            .find(|f| f.old_folder == fixture.old_folder())
            .cloned()
            .unwrap()
    };
    let plan = run(&fixture, true);
    assert!(is_refusal(outcome_of(&plan, &path)), "{plan:?}");
    assert!(!kept(&plan).deleted);

    let report = run(&fixture, false);
    assert!(is_refusal(outcome_of(&report, &path)), "{report:?}");
    // The old folder is converted for the files that can be, and kept for
    // the one that cannot; the orphan is still deleted.
    let folder = kept(&report);
    assert_eq!(folder.new_folder, Some(fixture.new_folder()));
    assert_eq!(
        folder.kept_because.as_deref(),
        Some("kept: named by 1 file(s) the current reader refuses")
    );
    assert!(!folder.deleted);
    assert!(report.old_folders_deleted);
    assert_eq!(
        std::fs::read_dir(fixture.old_folder()).unwrap().count(),
        IMAGES
    );
    assert!(!fixture.orphan_folder().exists());
    assert_eq!(std::fs::read(&path).unwrap(), refused);
    assert_eq!(
        outcome_of(&report, &fixture.path("sfmr/sift.sfmr")),
        &UpgradeOutcome::Converted
    );

    // A rerun finds the kept folder settled: nothing to convert or delete.
    let settled = snapshot(&fixture.root);
    let again = run(&fixture, false);
    assert!(again.nothing_to_do(), "{again:?}");
    assert_eq!(again.converted_sift_file_count, 0);
    assert!(kept(&again).is_settled());
    assert_eq!(snapshot(&fixture.root), settled);
}

#[test]
fn a_version_1_workspace_file_with_version_2_options_is_refused() {
    let fixture = fixture();
    let mut record = v2_record();
    record.feature_prefix_dir = v1_record().feature_prefix_dir;
    let config = serde_json::json!({
        "version": 1,
        "feature_tool": record.feature_tool,
        "feature_type": record.feature_type,
        "feature_options": record.feature_options,
        "feature_prefix_dir": record.feature_prefix_dir,
    });
    let workspace_file = fixture.root.join(WORKSPACE_FILE_NAME);
    std::fs::write(
        &workspace_file,
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    let before = snapshot(&fixture.root);
    for dry_run in [true, false] {
        let error = run_error(&fixture, dry_run);
        let UpgradeWorkspaceError::FeatureOptions { files, error } = &error else {
            panic!("{error}")
        };
        assert_eq!(files, &vec![workspace_file.clone()]);
        assert_eq!(error, &FeatureOptionsError::AlreadyVersion2);
    }
    assert_eq!(snapshot(&fixture.root), before);
}

#[test]
fn a_feature_prefix_dir_that_leaves_the_image_directory_is_refused() {
    let fixture = fixture();
    let mut record = v1_record();
    record.feature_prefix_dir = "../features/elsewhere".into();
    reconstruction(&fixture.root, &fixture.old_folder(), record, true)
        .save(&fixture.path("sfmr/escape.sfmr"), &Progress::none())
        .unwrap();
    let before = snapshot(&fixture.root);
    let error = run_error(&fixture, false);
    let UpgradeWorkspaceError::FeaturePrefixDir {
        files,
        feature_prefix_dir,
    } = &error
    else {
        panic!("{error}")
    };
    assert_eq!(files, &vec![fixture.path("sfmr/escape.sfmr")]);
    assert_eq!(feature_prefix_dir, "../features/elsewhere");
    assert_eq!(snapshot(&fixture.root), before);
}

/// The fixture with an inner workspace at `inner/` holding images and their
/// version 1 `.sift` files, and an outer reconstruction naming them.
fn fixture_with_inner_workspace() -> (Fixture, PathBuf, PathBuf) {
    let fixture = fixture();
    let inner = fixture.root.join("inner");
    write_workspace_file(&inner);
    let inner_images = inner.join("images");
    std::fs::create_dir_all(&inner_images).unwrap();
    let old = inner_images.join(v1_record().feature_prefix_dir);
    for i in 0..IMAGES {
        let image = inner_images.join(image_name(i));
        std::fs::copy(fixture.images().join(image_name(i)), &image).unwrap();
        write_version_1(&image, &old, 0);
    }
    let mut recon = reconstruction(&fixture.root, &old, v1_record(), false);
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("inner/images/{}", image_name(i));
    }
    let outer_sfmr = fixture.path("sfmr/outer-on-inner.sfmr");
    recon.save(&outer_sfmr, &Progress::none()).unwrap();
    (fixture, inner, outer_sfmr)
}

/// Check that every `.sift` file the `sift_files` reconstruction at `path`
/// records is on disk, version 2, with the content hash it records.
fn assert_sift_files_resolve(path: &Path) {
    let recon = SfmrReconstruction::load(path, &Progress::none()).unwrap();
    let ObservationSource::SiftFiles {
        sift_content_hashes,
        ..
    } = &recon.point_set.observations
    else {
        panic!("not sift_files")
    };
    for (i, hash) in sift_content_hashes.iter().enumerate() {
        let sift = recon.sift_path_for_image(i);
        let (_, metadata, recorded) = sfmtool_sift_format::read_sift_metadata(&sift)
            .unwrap_or_else(|e| panic!("{}: {e}", sift.display()));
        assert_eq!(metadata.version, 2, "{}", sift.display());
        assert_eq!(&hash_bytes(&recorded.content_xxh128), hash);
    }
}

#[test]
fn an_outer_file_naming_a_nested_workspace_survives_both_upgrades() {
    let (fixture, inner, outer_sfmr) = fixture_with_inner_workspace();
    let inner_old = inner.join("images").join(v1_record().feature_prefix_dir);
    let inner_new = inner.join("images").join(v2_record().feature_prefix_dir);

    let outer = run(&fixture, false);
    assert_eq!(outcome_of(&outer, &outer_sfmr), &UpgradeOutcome::Converted);
    let folder = outer
        .folders
        .iter()
        .find(|f| f.old_folder == inner_old)
        .unwrap_or_else(|| panic!("{outer:?}"));
    assert_eq!(folder.new_folder, Some(inner_new.clone()));
    assert!(folder.kept_because.as_deref().unwrap().contains("nested"));
    assert!(!folder.deleted);
    assert!(inner_old.is_dir());
    assert_sift_files_resolve(&outer_sfmr);

    // The inner workspace's own run takes the conversions already there and
    // deletes its old folder; the outer file still resolves.
    let inner_report = upgrade_workspace(
        &inner,
        &UpgradeWorkspaceOptions::default(),
        &Progress::none(),
    )
    .unwrap();
    assert!(inner_report.old_folders_deleted, "{inner_report:?}");
    assert!(!inner_old.exists());
    assert_sift_files_resolve(&outer_sfmr);
}

#[test]
fn a_nested_workspace_is_refused_while_its_enclosing_one_is_version_1() {
    let (fixture, inner, _) = fixture_with_inner_workspace();
    let before = snapshot(&fixture.root);
    for dry_run in [true, false] {
        let error = upgrade_workspace(
            &inner,
            &UpgradeWorkspaceOptions { dry_run },
            &Progress::none(),
        )
        .unwrap_err();
        let UpgradeWorkspaceError::EnclosingWorkspace { workspace_file } = &error else {
            panic!("{error}")
        };
        assert_eq!(workspace_file, &fixture.root.join(WORKSPACE_FILE_NAME));
    }
    assert_eq!(snapshot(&fixture.root), before);
}

/// Make `link` a link to the directory `target`: a junction on Windows, which
/// needs no privilege, and a symbolic link elsewhere.
fn link_directory(target: &Path, link: &Path) {
    #[cfg(windows)]
    {
        let output = std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
    }
    #[cfg(not(windows))]
    std::os::unix::fs::symlink(target, link).unwrap();
}

#[test]
fn a_linked_directory_is_listed_and_not_entered() {
    let fixture = fixture();
    let outside = fixture._dir.path().join("outside");
    let elsewhere = outside.join("features").join("sift-elsewhere");
    for i in 0..2 {
        write_version_1(&fixture.images().join(image_name(i)), &elsewhere, 3);
    }
    let link = fixture.root.join("linked");
    link_directory(&outside, &link);
    let before = snapshot(&outside);
    let report = run(&fixture, false);
    assert_eq!(report.linked_directories, vec![link.clone()]);
    assert!(report
        .folders
        .iter()
        .all(|f| !f.old_folder.starts_with(&link)));
    assert_eq!(snapshot(&outside), before);
    // A junction is removed as a directory, which leaves its target.
    #[cfg(windows)]
    std::fs::remove_dir(&link).unwrap();
}

#[cfg(windows)]
#[test]
fn a_reconstruction_locked_during_the_rewrite_fails_and_stops_the_deletion() {
    use std::os::windows::fs::OpenOptionsExt;
    let fixture = fixture();
    let target = fixture.path("sfmr/sift.sfmr");
    let lock = std::sync::Mutex::new(None);
    let sink = |event: Event<'_>| {
        if matches!(event, Event::Enter { phase, .. } if phase == "rewrite files") {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .share_mode(0)
                .open(&target)
                .unwrap();
            *lock.lock().unwrap() = Some(file);
        }
    };
    let report = upgrade_workspace(
        &fixture.root,
        &UpgradeWorkspaceOptions::default(),
        &Progress::to(&sink),
    )
    .unwrap();
    drop(lock.lock().unwrap().take());
    assert!(
        matches!(outcome_of(&report, &target), UpgradeOutcome::Failed(_)),
        "{report:?}"
    );
    assert!(!report.old_folders_deleted);
    // The file still finds its features in the old folder.
    assert_eq!(
        std::fs::read_dir(fixture.old_folder()).unwrap().count(),
        IMAGES
    );
    SfmrReconstruction::load(&target, &Progress::none()).unwrap();
}
