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
    write_version_1_of(image, folder, seed, v1_record())
}

/// [`write_version_1`] recording `record`'s tool, type and options.
fn write_version_1_of(
    image: &Path,
    folder: &Path,
    seed: usize,
    record: WorkspaceContents,
) -> PathBuf {
    let decoded = ImageU8::read_rgb(image).unwrap();
    let (image_file_xxh128, image_file_size) = xxh128_of_file(image).unwrap();
    let name = image.file_name().unwrap().to_string_lossy().into_owned();
    std::fs::create_dir_all(folder).unwrap();
    let path = folder.join(format!("{name}.sift"));
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
/// folder; and a nested workspace with the images and a version 1 folder of
/// its own.
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

    // The nested workspace holds copies of the images and their version 1
    // `.sift` files, which `other` names.
    write_workspace_file(&fixture.nested());
    let nested_sift = fixture
        .nested()
        .join("images")
        .join(v1_record().feature_prefix_dir);
    std::fs::create_dir_all(&nested_sift).unwrap();
    for i in 0..IMAGES {
        let name = image_name(i);
        std::fs::copy(
            fixture.images().join(&name),
            fixture.nested().join("images").join(&name),
        )
        .unwrap();
        let sift = format!("{name}.sift");
        std::fs::copy(old.join(&sift), nested_sift.join(&sift)).unwrap();
    }
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

    // The nested workspace is upgraded with the rest: its workspace file is
    // version 2, its old folder gone, and the file of its own that `other`
    // is resolves in its new folder.
    let nested_config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.nested().join(WORKSPACE_FILE_NAME)).unwrap())
            .unwrap();
    assert_eq!(nested_config["version"], 2);
    assert!(!fixture
        .nested()
        .join("images")
        .join(v1_record().feature_prefix_dir)
        .exists());
    assert_sift_files_resolve(&fixture.path("sfmr/other.sfmr"));
    // The reader resolves the workspace to its real path, which differs from
    // the fixture's spelling where the temporary directory is reached
    // through a link (`/var` on macOS) or a short name, so the two are
    // compared by their real paths.
    let sift = SfmrReconstruction::load(&fixture.path("sfmr/other.sfmr"), &Progress::none())
        .unwrap()
        .sift_path_for_image(0);
    assert!(std::fs::canonicalize(&sift)
        .unwrap()
        .starts_with(std::fs::canonicalize(fixture.nested()).unwrap()));

    // The files not converted are as they were.
    let after = snapshot(&fixture.root);
    for unchanged in ["sfmr/stale.sfmr", "matches/old-clusters.matches"] {
        let path = fixture.path(unchanged);
        assert_eq!(after[&path], before[&path], "{unchanged} changed");
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
        report.workspaces,
        vec![fixture.root.clone(), fixture.nested()]
    );
    for workspace in &report.workspaces {
        assert_eq!(
            outcome_of(&report, &workspace.join(WORKSPACE_FILE_NAME)),
            &UpgradeOutcome::Converted
        );
    }
    for converted in [
        "sfmr/sift.sfmr",
        "sfmr/other.sfmr",
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
    for left in ["sfmr/stale.sfmr", "matches/old-clusters.matches"] {
        assert!(not_converted(outcome_of(&report, &path(left))), "{left}");
    }
    assert_eq!(report.converted_sift_file_count, 2 * IMAGES);
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
            (
                fixture
                    .nested()
                    .join("images")
                    .join(v1_record().feature_prefix_dir),
                Some(
                    fixture
                        .nested()
                        .join("images")
                        .join(v2_record().feature_prefix_dir)
                ),
                IMAGES,
                true
            ),
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
    assert_eq!(report.converted_sift_file_count, 2 * IMAGES);
    assert!(report.old_folders_deleted);
    assert_eq!(report.folders.len(), 3);
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
    assert!(after.keys().filter(|p| !before.contains_key(*p)).all(|p| p
        .starts_with(fixture.new_folder())
        || p.starts_with(
            fixture
                .nested()
                .join("images")
                .join(v2_record().feature_prefix_dir)
        )));
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
    make_refused(path);
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
fn one_run_upgrades_an_outer_file_naming_a_nested_workspace_and_that_workspace() {
    let (fixture, inner, outer_sfmr) = fixture_with_inner_workspace();
    let inner_old = inner.join("images").join(v1_record().feature_prefix_dir);
    let inner_new = inner.join("images").join(v2_record().feature_prefix_dir);

    let report = run(&fixture, false);
    assert!(report.workspaces.contains(&inner), "{report:?}");
    assert_eq!(outcome_of(&report, &outer_sfmr), &UpgradeOutcome::Converted);
    assert_eq!(
        outcome_of(&report, &inner.join(WORKSPACE_FILE_NAME)),
        &UpgradeOutcome::Converted
    );
    let folder = report
        .folders
        .iter()
        .find(|f| f.old_folder == inner_old)
        .unwrap_or_else(|| panic!("{report:?}"));
    assert_eq!(folder.new_folder, Some(inner_new));
    assert!(folder.deleted);
    assert!(!inner_old.exists());
    assert_sift_files_resolve(&outer_sfmr);
}

#[test]
fn a_run_on_a_nested_workspace_is_refused_naming_the_root() {
    let (fixture, inner, _) = fixture_with_inner_workspace();
    let before = snapshot(&fixture.root);
    for dry_run in [true, false] {
        let error = upgrade_workspace(
            &inner.join("images"),
            &UpgradeWorkspaceOptions { dry_run },
            &Progress::none(),
        )
        .unwrap_err();
        let UpgradeWorkspaceError::NotTreeRoot { root } = &error else {
            panic!("{error}")
        };
        assert_eq!(root, &fixture.root);
    }
    assert_eq!(snapshot(&fixture.root), before);
}

/// Write a version 2 workspace file naming the converted record into `dir`,
/// as `sfm ws upgrade` before this run left one.
fn write_v2_workspace_file(dir: &Path) {
    let record = v2_record();
    let config = serde_json::json!({
        "version": 2,
        "feature_tool": record.feature_tool,
        "feature_type": record.feature_type,
        "feature_options": record.feature_options,
        "feature_prefix_dir": record.feature_prefix_dir,
    });
    std::fs::write(
        dir.join(WORKSPACE_FILE_NAME),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

/// A1: the enclosing workspace file is already version 2, as the shipped
/// `sfm ws upgrade` leaves it, but a version 1 file of the enclosing
/// workspace still names the inner workspace's `.sift` files. A run on the
/// inner workspace is refused, and the run on the root keeps them readable.
#[test]
fn a_version_2_enclosing_workspace_with_version_1_files_loses_nothing() {
    let (fixture, inner, outer_sfmr) = fixture_with_inner_workspace();
    write_v2_workspace_file(&fixture.root);
    let error = upgrade_workspace(
        &inner,
        &UpgradeWorkspaceOptions::default(),
        &Progress::none(),
    )
    .unwrap_err();
    assert!(
        matches!(error, UpgradeWorkspaceError::NotTreeRoot { .. }),
        "{error}"
    );
    let report = run(&fixture, false);
    assert_eq!(outcome_of(&report, &outer_sfmr), &UpgradeOutcome::Converted);
    assert_sift_files_resolve(&outer_sfmr);
}

/// A2: a file inside a nested workspace records the enclosing workspace; the
/// run finds it and converts it before deleting the folder it names.
#[test]
fn a_nested_file_recording_the_enclosing_workspace_loses_nothing() {
    let fixture = fixture();
    let mut recon = reconstruction(&fixture.root, &fixture.old_folder(), v1_record(), false);
    recon.metadata.workspace.relative_path = "..".into();
    let path = fixture.nested().join("inner-on-outer.sfmr");
    recon.save(&path, &Progress::none()).unwrap();
    let report = run(&fixture, false);
    assert_eq!(outcome_of(&report, &path), &UpgradeOutcome::Converted);
    assert!(!fixture.old_folder().exists());
    assert_sift_files_resolve(&path);
}

/// A folder of the tree that a file of a workspace outside the tree names is
/// kept: the run cannot see that workspace's other files. An image name is
/// a path inside its workspace, so the outside workspace reaches the tree's
/// images through a link of its own, `linked`, to the tree's `images`.
#[test]
fn a_folder_a_workspace_outside_the_tree_names_is_kept() {
    let fixture = fixture();
    let outside = fixture._dir.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    write_v2_workspace_file(&outside);
    let link = outside.join("linked");
    link_directory(&fixture.images(), &link);
    let mut recon = reconstruction(&outside, &fixture.old_folder(), v1_record(), false);
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("linked/{}", image_name(i));
    }
    recon.metadata.workspace.relative_path = "../../outside".into();
    let path = fixture.path("sfmr/from-outside.sfmr");
    recon.save(&path, &Progress::none()).unwrap();
    assert_eq!(missing_sift(&path), (0, 0));
    let report = run(&fixture, false);
    // A junction is removed as a directory, which leaves its target.
    #[cfg(windows)]
    std::fs::remove_dir(&link).unwrap();
    assert!(not_converted(outcome_of(&report, &path)), "{report:?}");
    let folder = report
        .folders
        .iter()
        .find(|f| f.old_folder == fixture.old_folder())
        .unwrap();
    assert!(
        folder
            .kept_because
            .as_deref()
            .unwrap()
            .contains("outside this tree"),
        "{folder:?}"
    );
    assert_eq!(
        std::fs::read_dir(fixture.old_folder()).unwrap().count(),
        IMAGES
    );
}

/// A7: a version 1 `.sift` file locked with no sharing for the whole run
/// stops the run before anything is written.
#[cfg(windows)]
#[test]
fn a_locked_sift_file_stops_the_run_before_anything_is_written() {
    use std::os::windows::fs::OpenOptionsExt;
    let fixture = fixture();
    let before = snapshot(&fixture.root);
    let locked = fixture.old_folder().join(format!("{}.sift", image_name(3)));
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(&locked)
        .unwrap();
    let error = upgrade_workspace(
        &fixture.root,
        &UpgradeWorkspaceOptions::default(),
        &Progress::none(),
    )
    .unwrap_err();
    drop(lock);
    let UpgradeWorkspaceError::UnreadableSiftFiles(failures) = &error else {
        panic!("{error}")
    };
    assert_eq!(failures.len(), 1, "{error}");
    assert_eq!(failures[0].sift_path, locked);
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

/// How many of the `.sift` files the `sift_files` reconstruction at `path`
/// records are not on disk, and how many are there with another content
/// hash than it records.
fn missing_sift(path: &Path) -> (usize, usize) {
    let recon = SfmrReconstruction::load(path, &Progress::none()).unwrap();
    let ObservationSource::SiftFiles {
        sift_content_hashes,
        ..
    } = &recon.point_set.observations
    else {
        panic!("not sift_files")
    };
    let mut missing = 0;
    let mut wrong = 0;
    for (i, hash) in sift_content_hashes.iter().enumerate() {
        match sfmtool_sift_format::read_sift_metadata(&recon.sift_path_for_image(i)) {
            Ok((_, _, recorded)) if &hash_bytes(&recorded.content_xxh128) != hash => wrong += 1,
            Ok(_) => {}
            Err(_) => missing += 1,
        }
    }
    (missing, wrong)
}

/// The record the `.sfmr` at `path` holds.
fn record_of(path: &Path) -> WorkspaceContents {
    sfmtool_sfmr_format::read_sfmr_metadata(path)
        .unwrap()
        .workspace
        .contents
}

/// A file behind a junction or symbolic link is not examined, so a tree
/// with a link deletes no old folder.
#[cfg(windows)]
#[test]
fn a_file_behind_a_link_keeps_its_folder() {
    let fixture = fixture();
    let results = fixture._dir.path().join("results");
    std::fs::create_dir_all(&results).unwrap();
    let mut recon = reconstruction(&fixture.root, &fixture.old_folder(), v1_record(), false);
    recon.metadata.workspace.relative_path = "../ws".into();
    recon
        .save(&results.join("behind.sfmr"), &Progress::none())
        .unwrap();
    let link = fixture.path("results");
    link_directory(&results, &link);
    let via_link = link.join("behind.sfmr");
    assert_eq!(missing_sift(&via_link), (0, 0));
    let report = run(&fixture, false);
    let after = missing_sift(&via_link);
    std::fs::remove_dir(&link).unwrap();
    assert_eq!(after, (0, 0), "a file behind a link lost its features");
    assert_eq!(report.linked_directories, vec![link]);
    assert!(!report.old_folders_deleted);
    assert!(report
        .notes
        .iter()
        .any(|n| n.contains("linked directories")));
    assert_eq!(
        outcome_of(&report, &fixture.path("sfmr/sift.sfmr")),
        &UpgradeOutcome::Converted
    );
}

/// The SeoulBull mix: the workspace file records `sift-sfmtool`, a `.sfmr`
/// records `sift`, and both name the same folder. The folder converts into
/// the one new folder the workspace file names, and the `.sfmr` is pointed
/// there with the workspace file's converted record.
#[test]
fn files_whose_records_differ_in_feature_type_follow_the_workspace_file() {
    let fixture = fixture();
    let mut record = v1_record();
    record.feature_type = "sift".into();
    let path = fixture.path("sfmr/typed-sift.sfmr");
    reconstruction(&fixture.root, &fixture.old_folder(), record, false)
        .save(&path, &Progress::none())
        .unwrap();
    let report = run(&fixture, false);
    assert_eq!(outcome_of(&report, &path), &UpgradeOutcome::Converted);
    let new = v2_record();
    let written = record_of(&path);
    assert_eq!(written.feature_type, new.feature_type);
    assert_eq!(written.feature_prefix_dir, new.feature_prefix_dir);
    assert_eq!(written.feature_options, new.feature_options);
    assert_eq!(missing_sift(&path), (0, 0));
    assert!(!fixture.old_folder().exists());
    // Only the one new folder was made beside the images.
    let made: Vec<_> = std::fs::read_dir(fixture.images().join("features"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(made, vec![fixture.new_folder()]);
}

/// Two records that differ in their options name one hand-named folder that
/// no workspace file names: the folder's own `.sift` files decide its one
/// new folder, and both files are pointed there with the same record.
#[test]
fn records_that_differ_in_options_follow_the_folders_sift_files() {
    let fixture = fixture();
    // The folder's `.sift` files record 15 features; one record says 16.
    let hand = fixture.images().join("features/sift-hand");
    let mut sift_record = v1_record();
    sift_record.feature_options["max_num_features"] = serde_json::json!(15);
    for i in 0..IMAGES {
        write_version_1_of(
            &fixture.images().join(image_name(i)),
            &hand,
            5,
            sift_record.clone(),
        );
    }
    let mut a_record = v1_record();
    a_record.feature_prefix_dir = "features/sift-hand".into();
    let mut b_record = sift_record.clone();
    b_record.feature_prefix_dir = "features/sift-hand".into();
    let a = fixture.path("sfmr/hand-a.sfmr");
    let b = fixture.path("sfmr/hand-b.sfmr");
    reconstruction(&fixture.root, &hand, a_record, false)
        .save(&a, &Progress::none())
        .unwrap();
    reconstruction(&fixture.root, &hand, b_record, false)
        .save(&b, &Progress::none())
        .unwrap();
    let report = run(&fixture, false);
    for path in [&a, &b] {
        assert_eq!(outcome_of(&report, path), &UpgradeOutcome::Converted);
        assert_eq!(missing_sift(path), (0, 0), "{}", path.display());
    }
    // The folder's one new folder is named from the `.sift` files' options,
    // converted, under the tool's workspace feature type; both records take
    // them, `a`'s 16 features replaced.
    let options = upgrade_feature_options("sfmtool", &sift_record.feature_options).unwrap();
    let expected = feature_prefix_dir("sfmtool", "sift-sfmtool", &options);
    for path in [&a, &b] {
        let written = record_of(path);
        assert_eq!(written.feature_prefix_dir, expected);
        assert_eq!(written.feature_options, options);
        assert_eq!(written.feature_type, "sift-sfmtool");
    }
    assert!(!hand.exists());
}

#[test]
fn a_folder_whose_sift_files_record_different_options_is_refused() {
    let fixture = fixture();
    let hand = fixture.images().join("features/sift-hand");
    let mut other = v1_record();
    other.feature_options["max_num_features"] = serde_json::json!(15);
    for i in 0..IMAGES {
        let record = if i == 2 { other.clone() } else { v1_record() };
        write_version_1_of(&fixture.images().join(image_name(i)), &hand, 5, record);
    }
    let mut record = v1_record();
    record.feature_prefix_dir = "features/sift-hand".into();
    reconstruction(&fixture.root, &hand, record, true)
        .save(&fixture.path("sfmr/hand.sfmr"), &Progress::none())
        .unwrap();
    let before = snapshot(&fixture.root);
    let error = run_error(&fixture, true);
    let UpgradeWorkspaceError::FolderFeatureOptions {
        folder,
        first,
        second,
    } = &error
    else {
        panic!("{error}")
    };
    assert_eq!(folder, &hand);
    let pair = BTreeSet::from([first.clone(), second.clone()]);
    assert!(
        pair.contains(&hand.join(format!("{}.sift", image_name(2)))),
        "{error}"
    );
    assert_eq!(snapshot(&fixture.root), before);
}

/// Write a version 1 `sift_files` `.sfmr` or pairwise `.matches` file of the
/// fixture's images at `path`, by its extension, recording the workspace at
/// `relative_path` from the file's directory, and then put each image name
/// through `rename` (the name's index, the name). The writers refuse a name
/// the image-name rule refuses, but a file another tool wrote or one edited
/// by hand can hold one, so such a file is made by editing the names entry
/// of one written with valid names; the reader refuses it.
fn write_with_image_names(
    fixture: &Fixture,
    path: &Path,
    relative_path: &str,
    rename: impl Fn(usize, &str) -> String,
) {
    write_with_names_entry(fixture, path, relative_path, |names| {
        for (i, name) in names.iter_mut().enumerate() {
            *name = rename(i, name.as_str().unwrap()).into();
        }
    });
}

/// [`write_with_image_names`] with the names entry, a JSON list, put through
/// `edit` as a whole.
fn write_with_names_entry(
    fixture: &Fixture,
    path: &Path,
    relative_path: &str,
    edit: impl Fn(&mut Vec<serde_json::Value>),
) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let old = fixture.old_folder();
    let sfmr = path.extension().unwrap() == "sfmr";
    if sfmr {
        let mut recon = reconstruction(&fixture.root, &old, v1_record(), false);
        recon.metadata.workspace.relative_path = relative_path.into();
        recon.save(path, &Progress::none()).unwrap();
    } else {
        let mut data = pairwise_matches(&old, v1_record());
        data.metadata.workspace.relative_path = relative_path.into();
        sfmtool_matches_format::write_matches(path, &data, 3).unwrap();
    }
    rewrite_archive(
        path,
        "images/names.json.zst",
        |names| edit(names.as_array_mut().unwrap()),
        |_| false,
    );
    let error = if sfmr {
        let Err(error) = sfmtool_sfmr_format::read_sfmr(path) else {
            panic!("the reader accepts the names")
        };
        error.to_string()
    } else {
        let Err(error) = sfmtool_matches_format::read_matches(path) else {
            panic!("the reader accepts the names")
        };
        error.to_string()
    };
    assert!(error.contains("image name"), "{error}");
}

/// A `.sfmr` or `.matches` file of a workspace outside the tree whose image
/// names reach the tree's images through `..`, which the reader refuses,
/// still keeps the folder it names in the tree.
#[test]
fn a_refused_file_outside_the_tree_keeps_its_folder() {
    for name in ["sfmr/from-outside.sfmr", "matches/from-outside.matches"] {
        let fixture = fixture();
        let outside = fixture._dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        write_v2_workspace_file(&outside);
        let path = fixture.path(name);
        write_with_image_names(&fixture, &path, "../../outside", |_, n| {
            format!("../ws/{n}")
        });
        let report = run(&fixture, false);
        assert!(not_converted(outcome_of(&report, &path)), "{report:?}");
        let folder = report
            .folders
            .iter()
            .find(|f| f.old_folder == fixture.old_folder())
            .unwrap();
        assert!(
            folder
                .kept_because
                .as_deref()
                .is_some_and(|k| k.contains("outside this tree")),
            "{name}: {folder:?}"
        );
        assert_eq!(old_folder_count(&fixture), IMAGES, "{name}");
    }
}

/// A `.sift` file whose content the reader rejects is reported as not
/// readable as the file a reconstruction records, not as changed.
#[test]
fn a_sift_file_the_reader_rejects_is_reported_as_unreadable_not_changed() {
    let fixture = fixture();
    let broken = fixture.old_folder().join(format!("{}.sift", image_name(4)));
    std::fs::write(&broken, b"not a zip archive").unwrap();
    let report = run(&fixture, true);
    let UpgradeOutcome::NotConverted(reason) = outcome_of(&report, &fixture.path("sfmr/sift.sfmr"))
    else {
        panic!("{report:?}")
    };
    assert!(
        reason.contains("cannot be read as the version 1 file"),
        "{reason}"
    );
    assert!(!reason.contains("changed"), "{reason}");
}

/// A `.sfmr` or `.matches` file of the nested workspace whose image names
/// reach the root's images through `..`, which the reader refuses, is not
/// converted, and the root's old folder that holds its features is kept,
/// though that folder is outside the file's own workspace.
#[test]
fn a_refused_file_naming_images_through_a_parent_directory_keeps_their_folder() {
    for name in ["parent-dir.sfmr", "parent-dir.matches"] {
        let fixture = fixture();
        let path = fixture.nested().join(name);
        write_with_image_names(&fixture, &path, ".", |_, n| format!("../{n}"));
        let before = Bytes(std::fs::read(&path).unwrap());
        let report = run(&fixture, false);
        let UpgradeOutcome::NotConverted(reason) = outcome_of(&report, &path) else {
            panic!("{report:?}")
        };
        assert!(reason.contains("reader refuses"), "{name}: {reason}");
        let folder = report
            .folders
            .iter()
            .find(|f| f.old_folder == fixture.old_folder())
            .unwrap();
        assert_eq!(
            folder.kept_because.as_deref(),
            Some("kept: named by 1 file(s) the current reader refuses"),
            "{name}: {folder:?}"
        );
        assert_eq!(old_folder_count(&fixture), IMAGES, "{name}");
        assert_eq!(Bytes(std::fs::read(&path).unwrap()), before, "{name}");
    }
}

/// A `.sfmr` or `.matches` file of the tree one of whose image names leaves
/// the tree's root through `..`, which the reader refuses, is not converted;
/// the old folder it names in the tree is kept, and nothing outside the tree
/// is touched.
#[test]
fn a_file_naming_images_outside_the_tree_keeps_its_folder() {
    for name in ["sfmr/partly-outside.sfmr", "matches/partly-outside.matches"] {
        let fixture = fixture();
        let elsewhere = fixture._dir.path().join("elsewhere");
        let first = image_name(0);
        let elsewhere_sift = elsewhere.join(v1_record().feature_prefix_dir);
        std::fs::create_dir_all(&elsewhere_sift).unwrap();
        std::fs::copy(fixture.images().join(&first), elsewhere.join(&first)).unwrap();
        let sift_name = format!("{first}.sift");
        std::fs::copy(
            fixture.old_folder().join(&sift_name),
            elsewhere_sift.join(&sift_name),
        )
        .unwrap();
        let path = fixture.path(name);
        write_with_image_names(&fixture, &path, "..", |i, n| {
            if i == 0 {
                format!("../elsewhere/{first}")
            } else {
                n.to_string()
            }
        });
        let before_elsewhere = snapshot(&elsewhere);

        let report = run(&fixture, false);
        let UpgradeOutcome::NotConverted(reason) = outcome_of(&report, &path) else {
            panic!("{report:?}")
        };
        assert!(reason.contains("reader refuses"), "{name}: {reason}");
        let folder = report
            .folders
            .iter()
            .find(|f| f.old_folder == fixture.old_folder())
            .unwrap();
        assert_eq!(
            folder.kept_because.as_deref(),
            Some("kept: named by 1 file(s) the current reader refuses"),
            "{name}: {folder:?}"
        );
        assert_eq!(old_folder_count(&fixture), IMAGES, "{name}");
        assert_eq!(snapshot(&elsewhere), before_elsewhere, "{name}");
    }
}

/// The workspace file's options differ from those its folder's `.sift` files
/// record (KerryPark360's workspace file records another `feature_type`
/// than the tool's), so the folder converts to the workspace file's new
/// folder, not to the one the `.sift` files alone would name. A file locked
/// during the rewrite fails, so the workspace file is left at version 1;
/// the rerun then converts the failed file into the same new folder.
#[cfg(windows)]
#[test]
fn a_file_that_fails_then_converts_into_the_workspace_files_new_folder() {
    use std::os::windows::fs::OpenOptionsExt;
    let fixture = fixture();
    let workspace_file = fixture.root.join(WORKSPACE_FILE_NAME);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&workspace_file).unwrap()).unwrap();
    config["feature_options"]["max_num_features"] = serde_json::json!(15);
    let v1_config = serde_json::to_string_pretty(&config).unwrap();
    std::fs::write(&workspace_file, &v1_config).unwrap();

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
    let first = upgrade_workspace(
        &fixture.root,
        &UpgradeWorkspaceOptions::default(),
        &Progress::to(&sink),
    )
    .unwrap();
    drop(lock.lock().unwrap().take());
    assert!(
        matches!(outcome_of(&first, &target), UpgradeOutcome::Failed(_)),
        "{first:?}"
    );
    assert!(not_converted(outcome_of(&first, &workspace_file)));
    assert_eq!(std::fs::read_to_string(&workspace_file).unwrap(), v1_config);
    assert!(first.notes.iter().any(|n| n.contains("workspace files")));

    let again = run(&fixture, false);
    assert!(again.old_folders_deleted, "{again:?}");
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&workspace_file).unwrap()).unwrap();
    assert_eq!(written["version"], 2);
    let prefix = written["feature_prefix_dir"].as_str().unwrap();
    for file in ["sfmr/sift.sfmr", "sfmr/embedded.sfmr"] {
        assert_eq!(
            record_of(&fixture.path(file)).feature_prefix_dir,
            prefix,
            "{file}"
        );
    }
    assert_sift_files_resolve(&target);
    // The one new folder is the only one beside the images.
    let made: Vec<_> = std::fs::read_dir(fixture.images().join("features"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(made, vec![fixture.images().join(prefix)]);
}

/// A version 1 record carries no `double_image_below_long_side`, so its
/// converted options carry none either and every image is doubled as it was
/// when extracted. For the standard version 1 sfmtool options, the new
/// folder is the one options without the key keep.
#[test]
fn a_version_1_record_keeps_doubling_every_image() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("ws");
    std::fs::create_dir_all(&root).unwrap();
    let options = serde_json::json!({
        "octave_layers": 3,
        "sigma": 1.6,
        "blur_radius_factor": 2.25,
        "input_sigma": 0.5,
        "double_image": true,
        "contrast_threshold": 0.0067,
        "edge_threshold": 10.0,
        "max_num_features": 8192,
        "orientation_bins": 36,
        "peak_ratio": 0.8,
        "descriptor_width": 4,
        "descriptor_bins": 8,
        "descriptor_magnification": 3.0,
        "descriptor_clamp": 0.2,
        "gray_formula": "0.2126*R + 0.7152*G + 0.0722*B",
    });
    let config = serde_json::json!({
        "version": 1,
        "feature_tool": "sfmtool",
        "feature_type": "sift-sfmtool",
        "feature_options": options,
        "feature_prefix_dir": feature_prefix_dir("sfmtool", "sift-sfmtool", &options),
    });
    assert_eq!(
        config["feature_prefix_dir"],
        "features/sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9"
    );
    let workspace_file = root.join(WORKSPACE_FILE_NAME);
    std::fs::write(&workspace_file, serde_json::to_string(&config).unwrap()).unwrap();
    upgrade_workspace(
        &root,
        &UpgradeWorkspaceOptions::default(),
        &Progress::none(),
    )
    .unwrap();
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&workspace_file).unwrap()).unwrap();
    assert_eq!(written["version"], 2);
    assert!(written["feature_options"]
        .get("double_image_below_long_side")
        .is_none());
    assert_eq!(written["feature_options"]["double_image"], true);
    assert_eq!(
        written["feature_prefix_dir"],
        "features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d"
    );
}

/// A workspace file names only folders of its own workspace, not those of a
/// workspace nested in it: a version 1 folder of a version 2 nested
/// workspace that no file names is an orphan, even where its path ends in
/// the enclosing workspace file's prefix.
#[test]
fn a_workspace_file_does_not_name_a_nested_workspaces_folder() {
    let fixture = fixture();
    write_v2_workspace_file(&fixture.nested());
    std::fs::remove_file(fixture.path("sfmr/other.sfmr")).unwrap();
    let nested_old = fixture
        .nested()
        .join("images")
        .join(v1_record().feature_prefix_dir);
    let report = run(&fixture, true);
    let folder = report
        .folders
        .iter()
        .find(|f| f.old_folder == nested_old)
        .unwrap_or_else(|| panic!("{report:?}"));
    assert_eq!(folder.new_folder, None, "{folder:?}");
    assert!(folder.deleted, "{folder:?}");
}

/// Save a `sift_files` reconstruction of the fixture's old folder at `path`
/// whose image names spell the image directory `Images`, where the disk has
/// `images`: on Windows, which ignores case, the same directory.
#[cfg(windows)]
fn save_with_upper_case_image_dir(fixture: &Fixture, path: &Path) {
    let mut recon = reconstruction(&fixture.root, &fixture.old_folder(), v1_record(), false);
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("Images/{}", image_name(i));
    }
    recon.save(path, &Progress::none()).unwrap();
}

/// Make the `.sfmr` at `path` one the current reader refuses: its camera
/// lacks a parameter.
fn make_refused(path: &Path) {
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

/// How many version 1 `.sift` files the fixture's old folder holds.
fn old_folder_count(fixture: &Fixture) -> usize {
    std::fs::read_dir(fixture.old_folder())
        .map(|d| d.count())
        .unwrap_or(0)
}

/// A refused file that spells the image directory in another case than the
/// disk keeps the one folder it names.
#[cfg(windows)]
#[test]
fn a_refused_file_spelling_the_image_directory_in_another_case_keeps_its_folder() {
    let fixture = fixture();
    let path = fixture.path("sfmr/refused.sfmr");
    save_with_upper_case_image_dir(&fixture, &path);
    make_refused(&path);
    let report = run(&fixture, false);
    assert!(not_converted(outcome_of(&report, &path)), "{report:?}");
    assert_eq!(old_folder_count(&fixture), IMAGES, "{report:?}");
}

/// The same with a version 2 workspace file, so only files name the folder:
/// one rewritten and one refused, both spelling `Images`.
#[cfg(windows)]
#[test]
fn a_refused_file_in_another_case_keeps_its_folder_under_a_version_2_workspace_file() {
    let fixture = fixture();
    write_v2_workspace_file(&fixture.root);
    let refused = fixture.path("sfmr/refused.sfmr");
    save_with_upper_case_image_dir(&fixture, &refused);
    make_refused(&refused);
    let readable = fixture.path("sfmr/upper.sfmr");
    save_with_upper_case_image_dir(&fixture, &readable);
    let report = run(&fixture, false);
    assert_eq!(outcome_of(&report, &readable), &UpgradeOutcome::Converted);
    assert_sift_files_resolve(&readable);
    assert_eq!(old_folder_count(&fixture), IMAGES, "{report:?}");
}

/// A readable file that spells the image directory in another case than the
/// disk is converted with the others into the one new folder: each folder
/// is listed once, each `.sift` file converted once, and the old folders
/// deleted.
#[cfg(windows)]
#[test]
fn a_file_spelling_the_image_directory_in_another_case_converts_with_the_others() {
    let fixture = fixture();
    let path = fixture.path("sfmr/upper.sfmr");
    save_with_upper_case_image_dir(&fixture, &path);
    let report = run(&fixture, false);
    assert_eq!(outcome_of(&report, &path), &UpgradeOutcome::Converted);
    let folders: BTreeSet<_> = report
        .folders
        .iter()
        .map(|f| f.old_folder.to_string_lossy().to_lowercase())
        .collect();
    assert_eq!(folders.len(), report.folders.len(), "{report:?}");
    assert_eq!(report.folders.len(), 3, "{report:?}");
    assert_eq!(report.converted_sift_file_count, 2 * IMAGES);
    assert!(report.old_folders_deleted, "{report:?}");
    assert_eq!(
        record_of(&path).feature_prefix_dir,
        v2_record().feature_prefix_dir
    );
    assert_sift_files_resolve(&path);
    assert!(run(&fixture, false).nothing_to_do());
}

/// A file of a workspace outside the tree that names the root's images as
/// `../ws/Images/...`, which the reader refuses, keeps the folder it names.
#[cfg(windows)]
#[test]
fn an_outside_file_spelling_the_image_directory_in_another_case_keeps_its_folder() {
    let fixture = fixture();
    let outside = fixture._dir.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    write_v2_workspace_file(&outside);
    let path = fixture.path("sfmr/from-outside.sfmr");
    write_with_image_names(&fixture, &path, "../../outside", |i, _| {
        format!("../ws/Images/{}", image_name(i))
    });
    let report = run(&fixture, false);
    assert!(not_converted(outcome_of(&report, &path)), "{report:?}");
    assert_eq!(old_folder_count(&fixture), IMAGES, "{report:?}");
}

/// A root given with `..` in it, absolute or relative to the current
/// directory, is the workspace it names: the run plans and upgrades it as it
/// does that workspace given plainly.
#[test]
fn a_root_given_with_a_parent_directory_upgrades_like_the_root() {
    let fixture = fixture();
    let plain = run(&fixture, true);
    let dotted = upgrade_workspace(
        &fixture.images().join(".."),
        &UpgradeWorkspaceOptions { dry_run: true },
        &Progress::none(),
    )
    .unwrap();
    assert_eq!(dotted, plain);

    // A relative path: `cargo test` runs a library's tests in its package
    // directory, so a temporary directory under the target directory has a
    // relative path from there whatever drive the system's temporary
    // directory is on.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(
        std::fs::canonicalize(std::env::current_dir().unwrap()).unwrap(),
        std::fs::canonicalize(manifest).unwrap()
    );
    let parent = manifest.join("../../target/upgrade-tests");
    std::fs::create_dir_all(&parent).unwrap();
    let dir = tempfile::tempdir_in(&parent).unwrap();
    let root = dir.path().join("ws");
    write_workspace_file(&root);
    std::fs::create_dir_all(root.join("images")).unwrap();
    let relative = Path::new("../../target/upgrade-tests")
        .join(dir.path().file_name().unwrap())
        .join("ws/images/..");
    let report = upgrade_workspace(
        &relative,
        &UpgradeWorkspaceOptions::default(),
        &Progress::none(),
    )
    .unwrap();
    assert_eq!(
        outcome_of(&report, &report.workspace_dir.join(WORKSPACE_FILE_NAME)),
        &UpgradeOutcome::Converted
    );
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(WORKSPACE_FILE_NAME)).unwrap()).unwrap();
    assert_eq!(written["version"], 2);
}

/// A path in a nested workspace given with `..` in it is still refused,
/// naming the tree's root.
#[test]
fn a_nested_workspace_given_with_a_parent_directory_is_refused_naming_the_root() {
    let (fixture, inner, _) = fixture_with_inner_workspace();
    let error = upgrade_workspace(
        &inner.join("images").join(".."),
        &UpgradeWorkspaceOptions { dry_run: true },
        &Progress::none(),
    )
    .unwrap_err();
    let UpgradeWorkspaceError::NotTreeRoot { root } = &error else {
        panic!("{error}")
    };
    assert_eq!(root, &fixture.root);
}

/// A root given with `..` after a linked directory is the workspace the path
/// names on disk, which on a system that follows the link before `..` (not
/// Windows) is not the one its spelling names with each `..` resolved by
/// name: the run upgrades the workspace the path names, never the other.
#[test]
fn a_root_given_with_a_parent_directory_after_a_link_upgrades_the_workspace_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let spelled = dir.path().join("spelled");
    let linked = dir.path().join("linked");
    write_workspace_file(&spelled);
    write_workspace_file(&linked);
    std::fs::create_dir_all(linked.join("images")).unwrap();
    link_directory(&linked.join("images"), &spelled.join("images"));
    let path = spelled.join("images").join("..");
    let named = std::fs::canonicalize(&path).unwrap();
    let report = upgrade_workspace(
        &path,
        &UpgradeWorkspaceOptions { dry_run: true },
        &Progress::none(),
    );
    // A junction is removed as a directory, which leaves its target.
    #[cfg(windows)]
    std::fs::remove_dir(spelled.join("images")).unwrap();
    let report = report.unwrap();
    assert_eq!(
        std::fs::canonicalize(&report.workspace_dir).unwrap(),
        named,
        "{report:?}"
    );
    assert_eq!(report.files.len(), 1, "{report:?}");
    assert_eq!(
        report.files[0].path,
        report.workspace_dir.join(WORKSPACE_FILE_NAME)
    );
}

/// Each error says what stopped the run and what to do, and a list longer
/// than the items it names gives the count of the rest.
#[test]
fn each_error_says_what_stopped_the_run() {
    let p = |s: &str| PathBuf::from(s);
    let many: Vec<PathBuf> = (0..NAMED_ITEMS + 2)
        .map(|i| p(&format!("f{i}.sfmr")))
        .collect();
    let failure = SiftFileFailure {
        sift_path: p("a.sift"),
        problem: "locked".into(),
    };
    let conflict_text = format!(
        "{} and {} would both become {}",
        p("one/a.sift").display(),
        p("two/a.sift").display(),
        p("new/a.sift").display()
    );
    let cases = [
        (
            UpgradeWorkspaceError::WorkspaceFile {
                path: p("ws.json"),
                problem: "has no integer 'version'".into(),
            },
            vec!["the workspace file ws.json has no integer 'version'"],
        ),
        (
            UpgradeWorkspaceError::ListFolder {
                folder: p("dir"),
                problem: "denied".into(),
            },
            vec!["failed to list dir: denied"],
        ),
        (
            UpgradeWorkspaceError::FeatureOptions {
                files: many.clone(),
                error: FeatureOptionsError::Undistort,
            },
            vec!["cannot be converted to version 2", "f0.sfmr", "and 2 more"],
        ),
        (
            UpgradeWorkspaceError::FeaturePrefixDir {
                files: vec![p("x.sfmr")],
                feature_prefix_dir: "../up".into(),
            },
            vec!["the feature_prefix_dir '../up'", "x.sfmr"],
        ),
        (
            UpgradeWorkspaceError::NotTreeRoot { root: p("outer") },
            vec!["nested in the workspace outer; run the upgrade there"],
        ),
        (
            UpgradeWorkspaceError::UnreadableSiftFiles(vec![failure.clone()]),
            vec!["1 .sift file(s) could not be read", "a.sift: locked"],
        ),
        (
            UpgradeWorkspaceError::FolderFeatureOptions {
                folder: p("old"),
                first: p("old/a.sift"),
                second: p("old/b.sift"),
            },
            vec!["the features folder old holds .sift files that record different"],
        ),
        (
            UpgradeWorkspaceError::FolderTargets {
                file: p("x.sfmr"),
                folders: vec![p("one"), p("two")],
            },
            vec![
                "x.sfmr names old features folders that convert to different",
                "two",
            ],
        ),
        (
            UpgradeWorkspaceError::Conflicts(vec![SiftConflict {
                new_file: p("new/a.sift"),
                first: p("one/a.sift"),
                second: p("two/a.sift"),
            }]),
            vec![conflict_text.as_str()],
        ),
        (
            UpgradeWorkspaceError::SiftFiles(vec![failure]),
            vec![
                "1 version 1 .sift file(s) could not be converted",
                "a.sift: locked",
            ],
        ),
        (
            UpgradeWorkspaceError::Cancelled,
            vec!["cancelled; run it again"],
        ),
    ];
    for (error, expected) in cases {
        let text = error.to_string();
        for part in expected {
            assert!(text.contains(part), "{text:?} lacks {part:?}");
        }
    }
}

/// A file whose content no reader accepts fails, wherever it is in the
/// plan: the workspace files are left at version 1 and nothing is deleted,
/// in the dry run's report and in the run's. A `.camrig` that cannot be
/// read fails too, and a `.kdf` that cannot be read is left.
#[test]
fn a_file_that_cannot_be_read_stops_the_deletion_and_the_workspace_files() {
    let fixture = fixture();
    for name in [
        "sfmr/garbage.sfmr",
        "matches/garbage.matches",
        "bad.camrig",
        "bad.kdf",
    ] {
        std::fs::write(fixture.path(name), b"not an archive").unwrap();
    }
    let workspace_file = fixture.root.join(WORKSPACE_FILE_NAME);
    let before = std::fs::read(&workspace_file).unwrap();
    for dry_run in [true, false] {
        let report = run(&fixture, dry_run);
        for name in ["sfmr/garbage.sfmr", "matches/garbage.matches", "bad.camrig"] {
            let outcome = outcome_of(&report, &fixture.path(name));
            assert!(
                matches!(outcome, UpgradeOutcome::Failed(r) if r.contains("cannot be read")),
                "{name}: {outcome:?}"
            );
        }
        assert!(not_converted(outcome_of(&report, &fixture.path("bad.kdf"))));
        let UpgradeOutcome::NotConverted(reason) = outcome_of(&report, &workspace_file) else {
            panic!("{report:?}")
        };
        assert!(reason.contains("left at version 1"), "{reason}");
        assert!(!report.old_folders_deleted);
        assert!(report.folders.iter().all(|f| !f.deleted));
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("workspace files were not written")
                    && n.contains("move them out of the tree")),
            "{:?}",
            report.notes
        );
    }
    assert_eq!(std::fs::read(&workspace_file).unwrap(), before);
    assert_eq!(old_folder_count(&fixture), IMAGES);
}

/// A `.matches` file of a workspace outside the tree is left; a `.kdf` of a
/// version 2 record is already current.
#[test]
fn a_matches_file_of_another_workspace_and_a_current_index_are_left() {
    let fixture = fixture();
    let outside = fixture._dir.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    write_v2_workspace_file(&outside);
    let mut elsewhere = pairwise_matches(&fixture.old_folder(), v1_record());
    elsewhere.metadata.workspace.relative_path = "../../outside".into();
    sfmtool_matches_format::write_matches(
        &fixture.path("matches/elsewhere.matches"),
        &elsewhere,
        3,
    )
    .unwrap();
    let mut current = v2_record();
    current.feature_prefix_dir = v1_record().feature_prefix_dir;
    write_index(&fixture.path("current.kdf"), &fixture.old_folder(), current);

    let report = run(&fixture, true);
    let UpgradeOutcome::NotConverted(reason) =
        outcome_of(&report, &fixture.path("matches/elsewhere.matches"))
    else {
        panic!("{report:?}")
    };
    assert!(reason.contains("outside this tree"), "{reason}");
    assert_eq!(
        outcome_of(&report, &fixture.path("current.kdf")),
        &UpgradeOutcome::AlreadyCurrent
    );
}

/// A refused `.sfmr` or `.matches` file whose image names cannot be read
/// still names the folders its record's prefix names below its workspace,
/// which are kept.
#[test]
fn a_refused_file_whose_image_names_cannot_be_read_keeps_its_folder() {
    let fixture = fixture();
    let names_entry = |name: &str| name == "images/names.json.zst";
    let sfmr = fixture.path("sfmr/no-names.sfmr");
    std::fs::copy(fixture.path("sfmr/sift.sfmr"), &sfmr).unwrap();
    rewrite_archive(&sfmr, METADATA, |_| {}, names_entry);
    let matches = fixture.path("matches/no-names.matches");
    std::fs::copy(fixture.path("matches/clusters.matches"), &matches).unwrap();
    rewrite_archive(&matches, METADATA, |_| {}, names_entry);

    let report = run(&fixture, true);
    for path in [&sfmr, &matches] {
        assert!(not_converted(outcome_of(&report, path)), "{report:?}");
    }
    let folder = report
        .folders
        .iter()
        .find(|f| f.old_folder == fixture.old_folder())
        .unwrap();
    assert_eq!(
        folder.kept_because.as_deref(),
        Some("kept: named by 2 file(s) the current reader refuses"),
        "{folder:?}"
    );
    assert!(!folder.deleted);
}

/// A `.sfmr` or `.matches` file whose names entry holds fewer names than it
/// has images, which the reader refuses, may name images its names do not:
/// it names the folders its record's prefix names below its workspace as
/// well as those its names name, each once, so a folder only the prefix
/// reaches is kept too.
#[test]
fn a_refused_file_with_too_few_image_names_keeps_the_folders_of_its_prefix() {
    for name in ["sfmr/few-names.sfmr", "matches/few-names.matches"] {
        let fixture = fixture();
        // A second image directory, whose version 1 folder the file's names
        // do not reach.
        let images2 = fixture.root.join("images2");
        let folder2 = images2.join(v1_record().feature_prefix_dir);
        std::fs::create_dir_all(&folder2).unwrap();
        for i in 0..IMAGES {
            let image = image_name(i);
            let sift = format!("{image}.sift");
            std::fs::copy(fixture.images().join(&image), images2.join(&image)).unwrap();
            std::fs::copy(fixture.old_folder().join(&sift), folder2.join(&sift)).unwrap();
        }
        let path = fixture.path(name);
        write_with_names_entry(&fixture, &path, "..", |names| {
            names.pop();
        });
        let report = run(&fixture, false);
        assert!(not_converted(outcome_of(&report, &path)), "{report:?}");
        for folder in [fixture.old_folder(), folder2.clone()] {
            let upgraded = report
                .folders
                .iter()
                .find(|f| f.old_folder == folder)
                .unwrap();
            assert_eq!(
                upgraded.kept_because.as_deref(),
                Some("kept: named by 1 file(s) the current reader refuses"),
                "{name}: {upgraded:?}"
            );
        }
        assert_eq!(std::fs::read_dir(&folder2).unwrap().count(), IMAGES);
    }
}

/// A file whose record names no folder on disk takes its workspace file's
/// converted record when the prefixes agree, and otherwise its own record
/// converted under the tool's workspace `feature_type`.
#[test]
fn a_file_naming_no_folder_on_disk_takes_a_converted_record() {
    let fixture = fixture();
    let bare = fixture.root.join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    // Options other than the workspace file's under its prefix: the file takes
    // the workspace file's converted record, not its own.
    let mut same_record = v1_record();
    same_record.feature_options["max_num_features"] = serde_json::json!(15);
    let mut same = reconstruction(&fixture.root, &fixture.old_folder(), same_record, true);
    for (i, image) in same.image_set.images.iter_mut().enumerate() {
        image.name = format!("bare/{}", image_name(i));
    }
    let same_path = fixture.path("sfmr/same-prefix.sfmr");
    same.save(&same_path, &Progress::none()).unwrap();
    let mut gone = v1_record();
    gone.feature_prefix_dir = "features/sift-gone".into();
    gone.feature_type = "sift".into();
    let gone_path = fixture.path("sfmr/gone-prefix.sfmr");
    reconstruction(&fixture.root, &fixture.old_folder(), gone.clone(), true)
        .save(&gone_path, &Progress::none())
        .unwrap();

    let report = run(&fixture, false);
    for path in [&same_path, &gone_path] {
        assert_eq!(outcome_of(&report, path), &UpgradeOutcome::Converted);
    }
    assert_eq!(fields(&record_of(&same_path)), fields(&v2_record()));
    let options = upgrade_feature_options("sfmtool", &gone.feature_options).unwrap();
    let feature_type = crate::workspace::feature_type_for_tool("sfmtool", &options);
    assert_eq!(
        fields(&record_of(&gone_path)),
        fields(&WorkspaceContents {
            feature_prefix_dir: feature_prefix_dir("sfmtool", &feature_type, &options),
            feature_tool: "sfmtool".into(),
            feature_type,
            feature_options: options,
        }),
    );
}

/// A folder no workspace file names whose `.sift` files record options that
/// cannot be converted stops the run, naming a file that holds them.
#[test]
fn a_folder_whose_sift_files_hold_options_that_cannot_convert_is_refused() {
    let fixture = fixture();
    let hand = fixture.images().join("features/sift-hand");
    let mut bad = v1_record();
    bad.feature_options["gray_formula"] = serde_json::json!("not a formula");
    for i in 0..2 {
        write_version_1_of(&fixture.images().join(image_name(i)), &hand, 5, bad.clone());
    }
    let mut record = v1_record();
    record.feature_prefix_dir = "features/sift-hand".into();
    reconstruction(&fixture.root, &fixture.old_folder(), record, true)
        .save(&fixture.path("sfmr/hand.sfmr"), &Progress::none())
        .unwrap();
    let error = run_error(&fixture, true);
    let UpgradeWorkspaceError::FeatureOptions { files, .. } = &error else {
        panic!("{error}")
    };
    assert_eq!(files.len(), 1, "{error}");
    assert!(files[0].starts_with(&hand), "{error}");
}

/// A file whose images lie in two directories whose same-named folders hold
/// files of different options names folders that convert to two new
/// folders; the run stops before writing, naming the file.
#[test]
fn a_file_naming_folders_that_convert_apart_is_refused() {
    let fixture = fixture();
    let images2 = fixture.root.join("images2");
    std::fs::create_dir_all(&images2).unwrap();
    let mut other = v1_record();
    other.feature_options["max_num_features"] = serde_json::json!(15);
    // Neither folder's options are the workspace file's, so neither converts
    // into its new folder.
    let mut first = v1_record();
    first.feature_options["max_num_features"] = serde_json::json!(14);
    let hand = fixture.images().join("features/sift-hand");
    let hand2 = images2.join("features/sift-hand");
    for i in 0..IMAGES {
        let name = image_name(i);
        if i < 4 {
            write_version_1_of(&fixture.images().join(&name), &hand, 5, first.clone());
        } else {
            std::fs::copy(fixture.images().join(&name), images2.join(&name)).unwrap();
            write_version_1_of(&images2.join(&name), &hand2, 5, other.clone());
        }
    }
    let mut record = v1_record();
    record.feature_prefix_dir = "features/sift-hand".into();
    let mut recon = reconstruction(&fixture.root, &fixture.old_folder(), record, true);
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        if i >= 4 {
            image.name = format!("images2/{}", image_name(i));
        }
    }
    let path = fixture.path("sfmr/span.sfmr");
    recon.save(&path, &Progress::none()).unwrap();
    let before = snapshot(&fixture.root);
    for dry_run in [true, false] {
        let error = run_error(&fixture, dry_run);
        let UpgradeWorkspaceError::FolderTargets { file, folders } = &error else {
            panic!("{error}")
        };
        assert_eq!(file, &path);
        assert_eq!(folders.len(), 2, "{error}");
    }
    assert_eq!(snapshot(&fixture.root), before);
}

/// The same `.sift` file in two old folders that convert to one new folder
/// is converted once.
#[test]
fn the_same_file_in_two_old_folders_is_converted_once() {
    let fixture = fixture();
    let copy = fixture.images().join("features/sift-copy");
    std::fs::create_dir_all(&copy).unwrap();
    for i in 0..IMAGES {
        let name = format!("{}.sift", image_name(i));
        std::fs::copy(fixture.old_folder().join(&name), copy.join(&name)).unwrap();
    }
    let mut record = v1_record();
    record.feature_prefix_dir = "features/sift-copy".into();
    reconstruction(&fixture.root, &copy, record, true)
        .save(&fixture.path("sfmr/copy.sfmr"), &Progress::none())
        .unwrap();
    let report = run(&fixture, true);
    let copy_row = report
        .folders
        .iter()
        .find(|f| f.old_folder == copy)
        .unwrap();
    assert_eq!(copy_row.new_folder, Some(fixture.new_folder()));
    assert_eq!(report.converted_sift_file_count, 2 * IMAGES);
}

/// A new folder that cannot be made fails the conversion, naming it, before
/// any other file is written.
#[test]
fn a_new_folder_that_cannot_be_made_fails_the_run() {
    let fixture = fixture();
    std::fs::write(fixture.new_folder(), b"a file where the folder goes").unwrap();
    let error = run_error(&fixture, false);
    let UpgradeWorkspaceError::SiftFiles(failures) = &error else {
        panic!("{error}")
    };
    assert!(
        failures.iter().any(|f| f.sift_path == fixture.new_folder()),
        "{error}"
    );
    assert_eq!(
        record_of(&fixture.path("sfmr/sift.sfmr")).feature_prefix_dir,
        v1_record().feature_prefix_dir
    );
}

/// A file in the new folder that is not the conversion of the old one is
/// not replaced: the run fails before rewriting any file.
#[test]
fn a_different_file_in_the_new_folder_is_not_replaced() {
    let fixture = fixture();
    let staging = fixture._dir.path().join("staging");
    let image = fixture.images().join(image_name(0));
    let other = write_version_1(&image, &staging, 3);
    let converted = Version2SiftFolder::create(&fixture.new_folder())
        .unwrap()
        .convert_version_1_sift_file(&other, &image, &Progress::none())
        .unwrap();
    let present = std::fs::read(&converted.path).unwrap();
    let error = run_error(&fixture, false);
    let UpgradeWorkspaceError::SiftFiles(failures) = &error else {
        panic!("{error}")
    };
    assert!(
        failures
            .iter()
            .any(|f| f.problem.contains("not this file's conversion")),
        "{error}"
    );
    assert_eq!(std::fs::read(&converted.path).unwrap(), present);
}

/// A file in an old folder that is not a version 1 `.sift` file keeps the
/// folder and is listed; an orphan folder that is not in a `features` folder
/// is removed up to its parent.
#[test]
fn a_file_left_in_an_old_folder_keeps_it_and_a_loose_orphan_is_removed() {
    let fixture = fixture();
    let note = fixture.old_folder().join("notes.txt");
    std::fs::write(&note, b"mine").unwrap();
    let loose = fixture.images().join("sift-loose");
    for i in 0..2 {
        write_version_1(&fixture.images().join(image_name(i)), &loose, 9);
    }
    let report = run(&fixture, false);
    let row = report
        .folders
        .iter()
        .find(|f| f.old_folder == fixture.old_folder())
        .unwrap();
    assert_eq!(row.kept_files, vec![note.clone()]);
    assert!(row.deleted);
    assert_eq!(std::fs::read_dir(fixture.old_folder()).unwrap().count(), 1);
    let loose_row = report
        .folders
        .iter()
        .find(|f| f.old_folder == loose)
        .unwrap();
    assert_eq!(loose_row.new_folder, None);
    assert!(loose_row.deleted);
    assert!(!loose.exists());
    assert!(fixture.images().is_dir());
}

/// A file naming images through a linked directory names a folder whose
/// real path is outside the tree: the folder is kept, and the run deletes
/// no old folder.
#[test]
fn a_folder_reached_through_a_link_is_kept_and_nothing_is_deleted() {
    let fixture = fixture();
    let outside = fixture._dir.path().join("outside-images");
    let outside_old = outside.join(v1_record().feature_prefix_dir);
    for i in 0..IMAGES {
        let name = image_name(i);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::copy(fixture.images().join(&name), outside.join(&name)).unwrap();
        write_version_1(&outside.join(&name), &outside_old, 0);
    }
    let link = fixture.root.join("linked-images");
    link_directory(&outside, &link);
    let mut recon = reconstruction(&fixture.root, &outside_old, v1_record(), true);
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("linked-images/{}", image_name(i));
    }
    recon
        .save(&fixture.path("sfmr/linked.sfmr"), &Progress::none())
        .unwrap();
    let report = run(&fixture, true);
    #[cfg(windows)]
    std::fs::remove_dir(&link).unwrap();
    let row = report
        .folders
        .iter()
        .find(|f| f.old_folder.starts_with(&link))
        .unwrap_or_else(|| panic!("{report:?}"));
    assert!(
        row.kept_because
            .as_deref()
            .is_some_and(|k| k.starts_with("it lies outside the tree")),
        "{row:?}"
    );
    assert!(!report.old_folders_deleted);
    assert!(report.folders.iter().all(|f| !f.deleted));
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.starts_with("No version 1 features folder was deleted")),
        "{:?}",
        report.notes
    );
}

/// Set the Unix permission bits of `path`.
#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// Sets the permission bits of a path back when dropped, so a test that
/// fails while they are restricted still leaves a tree its temporary
/// directory can remove.
#[cfg(unix)]
struct RestoreMode(PathBuf, u32);

#[cfg(unix)]
impl Drop for RestoreMode {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(self.1));
    }
}

/// Whether permission bits stop this process: false when it runs as root,
/// which reads and writes whatever they say.
#[cfg(unix)]
fn permissions_apply(dir: &Path) -> bool {
    let probe = dir.join("permission-probe");
    std::fs::create_dir_all(&probe).unwrap();
    set_mode(&probe, 0o555);
    let applies = std::fs::write(probe.join("x"), b"").is_err();
    set_mode(&probe, 0o755);
    std::fs::remove_dir_all(&probe).unwrap();
    applies
}

/// A `.sift` file the file system will not give up stops the run before
/// anything is written, and so does a directory that cannot be listed.
#[cfg(unix)]
#[test]
fn an_unreadable_sift_file_or_folder_stops_the_run() {
    let fixture = fixture();
    if !permissions_apply(fixture._dir.path()) {
        return;
    }
    let before = snapshot(&fixture.root);
    let locked = fixture.old_folder().join(format!("{}.sift", image_name(3)));
    let error = {
        let _restore = RestoreMode(locked.clone(), 0o644);
        set_mode(&locked, 0o000);
        run_error(&fixture, false)
    };
    let UpgradeWorkspaceError::UnreadableSiftFiles(failures) = &error else {
        panic!("{error}")
    };
    assert_eq!(failures.len(), 1, "{error}");
    assert_eq!(failures[0].sift_path, locked);

    let matches = fixture.path("matches");
    let error = {
        let _restore = RestoreMode(matches.clone(), 0o755);
        set_mode(&matches, 0o000);
        run_error(&fixture, false)
    };
    let UpgradeWorkspaceError::ListFolder { folder, .. } = &error else {
        panic!("{error}")
    };
    assert_eq!(folder, &matches);
    assert_eq!(snapshot(&fixture.root), before);
}

/// Run the upgrade of `fixture`, setting `mode` on each of `paths` when
/// phase `phase_name` opens and restoring `restore` after the run.
#[cfg(unix)]
fn run_with_modes_from(
    fixture: &Fixture,
    phase_name: &'static str,
    paths: &[(PathBuf, u32, u32)],
) -> UpgradeWorkspaceReport {
    let _restore: Vec<_> = paths
        .iter()
        .map(|(path, _, restore)| RestoreMode(path.clone(), *restore))
        .collect();
    let sink = |event: Event<'_>| {
        if matches!(event, Event::Enter { phase, .. } if phase == phase_name) {
            for (path, mode, _) in paths {
                set_mode(path, *mode);
            }
        }
    };
    upgrade_workspace(
        &fixture.root,
        &UpgradeWorkspaceOptions::default(),
        &Progress::to(&sink),
    )
    .unwrap()
}

/// A file that cannot be read during the rewrite fails, as does one whose
/// folder cannot be written and a `.kdf` that cannot be deleted; the old
/// folders are kept.
#[cfg(unix)]
#[test]
fn files_that_cannot_be_read_or_written_during_the_rewrite_fail() {
    let reading = fixture();
    if !permissions_apply(reading._dir.path()) {
        return;
    }
    let read = [
        reading.path("sfmr/sift.sfmr"),
        reading.path("matches/pairs.matches"),
        reading.path("rig.camrig"),
    ];
    let modes: Vec<_> = read.iter().map(|p| (p.clone(), 0o000, 0o644)).collect();
    let report = run_with_modes_from(&reading, "rewrite files", &modes);
    for path in &read {
        let outcome = outcome_of(&report, path);
        assert!(
            matches!(outcome, UpgradeOutcome::Failed(r) if r.starts_with("it cannot be read")),
            "{}: {outcome:?}",
            path.display()
        );
    }
    assert!(!report.old_folders_deleted);

    let writing = fixture();
    let fixture = writing;
    let dirs = [
        fixture.root.clone(),
        fixture.path("sfmr"),
        fixture.path("matches"),
    ];
    let modes: Vec<_> = dirs.iter().map(|p| (p.clone(), 0o555, 0o755)).collect();
    let report = run_with_modes_from(&fixture, "rewrite files", &modes);
    for name in ["sfmr/sift.sfmr", "matches/pairs.matches", "rig.camrig"] {
        let outcome = outcome_of(&report, &fixture.path(name));
        assert!(
            matches!(outcome, UpgradeOutcome::Failed(r) if r.starts_with("it could not be written")),
            "{name}: {outcome:?}"
        );
    }
    let outcome = outcome_of(&report, &fixture.path("sfmr/sift-sift-index.kdf"));
    assert!(
        matches!(outcome, UpgradeOutcome::Failed(r) if r.starts_with("it could not be deleted")),
        "{outcome:?}"
    );
    assert!(!report.old_folders_deleted);
}

/// A workspace file that cannot be written fails and keeps the old folders.
#[cfg(unix)]
#[test]
fn a_workspace_file_that_cannot_be_written_fails() {
    let fixture = fixture();
    if !permissions_apply(fixture._dir.path()) {
        return;
    }
    let modes = [(fixture.root.clone(), 0o555, 0o755)];
    let report = run_with_modes_from(&fixture, "workspace file", &modes);
    let outcome = outcome_of(&report, &fixture.root.join(WORKSPACE_FILE_NAME));
    assert!(
        matches!(outcome, UpgradeOutcome::Failed(r) if r.starts_with("it could not be written")),
        "{outcome:?}"
    );
    assert!(!report.old_folders_deleted);
    assert_eq!(old_folder_count(&fixture), IMAGES);
}

/// A version 1 `.sift` file that cannot be deleted keeps its folder and is
/// listed.
#[cfg(unix)]
#[test]
fn a_sift_file_that_cannot_be_deleted_keeps_its_folder() {
    let fixture = fixture();
    if !permissions_apply(fixture._dir.path()) {
        return;
    }
    let modes = [(fixture.old_folder(), 0o555, 0o755)];
    let report = run_with_modes_from(&fixture, "delete old folders", &modes);
    let row = report
        .folders
        .iter()
        .find(|f| f.old_folder == fixture.old_folder())
        .unwrap();
    assert!(!row.deleted, "{row:?}");
    assert_eq!(row.kept_files.len(), IMAGES, "{row:?}");
    assert!(!report.old_folders_deleted);
}

/// The four fields of a record, which `WorkspaceContents` does not compare.
fn fields(contents: &WorkspaceContents) -> (String, String, String, serde_json::Value) {
    (
        contents.feature_tool.clone(),
        contents.feature_type.clone(),
        contents.feature_prefix_dir.clone(),
        contents.feature_options.clone(),
    )
}
