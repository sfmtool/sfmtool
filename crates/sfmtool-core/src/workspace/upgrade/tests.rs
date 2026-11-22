// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array1, Array2};
use sfmtool_archive_io::{WorkspaceContents, WorkspaceMetadata};
use sfmtool_matches_format::{MatchesContentHash, MatchesData, MatchesMetadata, PairsData};

use super::*;
use crate::progress::Event;

/// How many images the `.matches` files name.
const IMAGES: usize = 3;
/// The size every image the `.matches` files name records.
const IMAGE_DIMS: [u32; 2] = [270, 480];

fn image_name(i: usize) -> String {
    format!("image_{i:02}.jpg")
}

/// The version 2 record the fixture's files hold.
fn record() -> WorkspaceContents {
    let feature_options = serde_json::json!({"image_to_gray": {"formula": "G"}});
    WorkspaceContents {
        feature_prefix_dir: crate::workspace::feature_prefix_dir(
            "sfmtool",
            "sift-sfmtool",
            &feature_options,
        ),
        feature_tool: "sfmtool".into(),
        feature_type: "sift-sfmtool".into(),
        feature_options,
    }
}

/// A workspace in a temporary directory.
struct Fixture {
    dir: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

fn write_workspace_file(dir: &Path, version: u32) {
    let record = record();
    let config = serde_json::json!({
        "version": version,
        "feature_tool": record.feature_tool,
        "feature_type": record.feature_type,
        "feature_options": record.feature_options,
        "feature_prefix_dir": record.feature_prefix_dir,
    });
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join(WORKSPACE_FILE_NAME),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

/// A pairwise `.matches` file of the [`IMAGES`] images, recording the
/// fixture's record and image sizes.
fn pairwise_matches() -> MatchesData {
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
                contents: record(),
            },
            timestamp: "2026-11-16T00:00:00Z".into(),
            image_count: IMAGES as u32,
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
        image_names: (0..IMAGES)
            .map(|i| format!("images/{}", image_name(i)))
            .collect(),
        feature_tool_hashes: vec![[1; 16]; IMAGES],
        sift_content_hashes: vec![[2; 16]; IMAGES],
        feature_counts: Array1::from_elem(IMAGES, 8),
        image_dims: Array2::from_shape_fn((IMAGES, 2), |(_, j)| IMAGE_DIMS[j]),
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

/// Rewrite the archive at `path` in place: each entry through `edit`, which
/// is given its name and uncompressed bytes and returns the name and bytes to
/// write, or `None` to drop it. None of the `.sfmr`, `.matches` and `.camrig`
/// readers checks the content hashes, so an older writer's file is produced
/// this way.
fn rewrite_archive(path: &Path, edit: impl Fn(&str, Vec<u8>) -> Option<(String, Vec<u8>)>) {
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
        let mut raw = Vec::new();
        archive
            .by_name(&name)
            .unwrap()
            .read_to_end(&mut raw)
            .unwrap();
        let Some((name, bytes)) = edit(&name, zstd::stream::decode_all(&raw[..]).unwrap()) else {
            continue;
        };
        out.start_file(name, stored).unwrap();
        out.write_all(&zstd::bulk::compress(&bytes, 3).unwrap())
            .unwrap();
    }
    out.finish().unwrap();
}

/// An entry edit that sets `metadata.json`'s version to `version` and drops
/// the entries `drop` names.
fn relabel(
    version: u32,
    drop: impl Fn(&str) -> bool,
) -> impl Fn(&str, Vec<u8>) -> Option<(String, Vec<u8>)> {
    move |name, bytes| {
        if drop(name) {
            return None;
        }
        if name != "metadata.json.zst" {
            return Some((name.to_string(), bytes));
        }
        let mut json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        json["version"] = version.into();
        Some((name.to_string(), serde_json::to_vec(&json).unwrap()))
    }
}

/// An entry edit that makes a current-version `.sfmr` of the demo reconstruction,
/// which has no observation readings and no patch frames, the version 15
/// file of the same values: the version 15, and its one `float16` column,
/// the sharpness, widened to the `float32` column version 15 stores.
fn as_version_15(name: &str, bytes: Vec<u8>) -> Option<(String, Vec<u8>)> {
    if name == "metadata.json.zst" {
        return relabel(15, |_| false)(name, bytes);
    }
    if !name.contains(".float16.") {
        return Some((name.to_string(), bytes));
    }
    let widened: Vec<u8> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|h| half::f16::from_le_bytes(*h).to_f32().to_le_bytes())
        .collect();
    Some((name.replace(".float16.", ".float32."), widened))
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

/// The demo reconstruction, recording the fixture's workspace one directory
/// up from `sfmr/`.
fn reconstruction() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(12);
    recon.metadata.workspace = WorkspaceMetadata {
        absolute_path: String::new(),
        relative_path: "..".into(),
        contents: record(),
    };
    recon
}

/// The fixture: a version 2 workspace with a current `.sfmr`, `.matches` and
/// `.camrig`, and a version 15 `.sfmr`, a version 9 `.matches` and a version 1
/// `.camrig`, which the readers no longer read.
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("ws");
    let fixture = Fixture { dir, root };
    write_workspace_file(&fixture.root, 2);

    for name in ["current", "old"] {
        reconstruction()
            .save(
                &fixture.path(&format!("sfmr/{name}.sfmr")),
                &Progress::none(),
            )
            .unwrap();
    }
    rewrite_archive(&fixture.path("sfmr/old.sfmr"), as_version_15);

    std::fs::create_dir_all(fixture.path("matches")).unwrap();
    for name in ["current", "old"] {
        let path = fixture.path(&format!("matches/{name}.matches"));
        sfmtool_matches_format::write_matches(&path, &pairwise_matches(), 3).unwrap();
    }
    rewrite_archive(&fixture.path("matches/old.matches"), relabel(9, |_| false));

    write_rig(&fixture.path("old.camrig"));
    rewrite_archive(&fixture.path("old.camrig"), relabel(1, |_| false));
    write_rig(&fixture.path("current.camrig"));
    fixture
}

#[derive(PartialEq, Eq)]
struct Bytes(Vec<u8>);

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} bytes", self.0.len())
    }
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

fn run_on(path: &Path, dry_run: bool) -> Result<UpgradeWorkspaceReport, UpgradeWorkspaceError> {
    upgrade_workspace(
        path,
        &UpgradeWorkspaceOptions { dry_run },
        &Progress::none(),
    )
}

fn run(fixture: &Fixture, dry_run: bool) -> UpgradeWorkspaceReport {
    run_on(&fixture.root, dry_run).unwrap()
}

/// Each file of the fixture and the outcome a run gives it.
fn expected_outcomes(fixture: &Fixture) -> Vec<(PathBuf, UpgradeFileKind, &'static str)> {
    use UpgradeFileKind::*;
    vec![
        (fixture.path("sfmr/current.sfmr"), Sfmr, "already_current"),
        (fixture.path("sfmr/old.sfmr"), Sfmr, "not_converted"),
        (
            fixture.path("matches/current.matches"),
            Matches,
            "already_current",
        ),
        (
            fixture.path("matches/old.matches"),
            Matches,
            "not_converted",
        ),
        (fixture.path("current.camrig"), Camrig, "already_current"),
        (fixture.path("old.camrig"), Camrig, "not_converted"),
    ]
}

fn outcomes(report: &UpgradeWorkspaceReport) -> Vec<(PathBuf, UpgradeFileKind, &'static str)> {
    report
        .files
        .iter()
        .map(|f| {
            let outcome = match f.outcome {
                UpgradeOutcome::Converted => "converted",
                UpgradeOutcome::AlreadyCurrent => "already_current",
                UpgradeOutcome::NotConverted(_) => "not_converted",
                UpgradeOutcome::Failed(_) => "failed",
            };
            (f.path.clone(), f.kind, outcome)
        })
        .collect()
}

fn reason_of(report: &UpgradeWorkspaceReport, path: &Path) -> String {
    match &report
        .files
        .iter()
        .find(|f| f.path == path)
        .unwrap()
        .outcome
    {
        UpgradeOutcome::NotConverted(reason) | UpgradeOutcome::Failed(reason) => reason.clone(),
        other => panic!("{other:?}"),
    }
}

/// Each file is reported with what the run did to it. Every file in this
/// fixture is either already at the current version or below the oldest version
/// its reader accepts, so the run rewrites none: a refused file is left as it
/// is, with the reader's message.
#[test]
fn each_file_is_reported_and_one_the_reader_refuses_is_left_as_it_is() {
    let fixture = fixture();
    let before = snapshot(&fixture.root);
    let report = run(&fixture, false);
    assert_eq!(outcomes(&report), expected_outcomes(&fixture));
    assert_eq!(report.workspace_dir, fixture.root);
    assert_eq!(report.workspaces, vec![fixture.root.clone()]);
    assert!(report.nothing_to_do());
    // No `.sfmr` was rewritten, so no point ID changed.
    assert!(report.notes.is_empty(), "{:?}", report.notes);
    // A `.sfmr`, `.matches` or `.camrig` file below the oldest version read is
    // left as it is, with the reader's message, which names the release that
    // upgrades it.
    let old_sfmr = reason_of(&report, &fixture.path("sfmr/old.sfmr"));
    assert!(
        old_sfmr.starts_with("the current reader refuses it, so it is left as it is: "),
        "{old_sfmr}"
    );
    assert!(
        old_sfmr.contains(
            "is a version 15 .sfmr file, and this sfmtool reads .sfmr files of version 16 or \
             later. sfmtool 0.4.1 is the last release that reads version 15: install it, run \
             `sfm ws upgrade` on the workspace, then use the file with this release."
        ),
        "{old_sfmr}"
    );
    let old_matches = reason_of(&report, &fixture.path("matches/old.matches"));
    assert_eq!(
        old_matches,
        format!(
            "the current reader refuses it, so it is left as it is: {} is a version 9 \
             .matches file, and this sfmtool reads .matches files of version 10 or later. \
             sfmtool 0.4.1 is the last release that reads version 9: install it, run \
             `sfm ws upgrade` on the workspace, then use the file with this release.",
            fixture.path("matches").join("old.matches").display()
        )
    );
    let old_camrig = reason_of(&report, &fixture.path("old.camrig"));
    assert_eq!(
        old_camrig,
        format!(
            "the current reader refuses it, so it is left as it is: {} is a version 1 \
             .camrig file, and this sfmtool reads .camrig files of version 2 or later. \
             sfmtool 0.4.1 is the last release that reads version 1: install it, run \
             `sfm ws upgrade` on the workspace, then use the file with this release.",
            fixture.path("old.camrig").display()
        )
    );

    // Every file is byte for byte what it was.
    let after = snapshot(&fixture.root);
    assert_eq!(after, before);

    // A second run has nothing to do and changes nothing.
    let again = run(&fixture, false);
    assert!(again.nothing_to_do());
    assert!(again.notes.is_empty());
    assert_eq!(snapshot(&fixture.root), after);
}

#[test]
fn a_dry_run_reports_what_a_run_would_do_and_writes_nothing() {
    let fixture = fixture();
    let before = snapshot(&fixture.root);
    let report = run(&fixture, true);
    assert!(report.dry_run);
    assert_eq!(outcomes(&report), expected_outcomes(&fixture));
    assert!(report.notes.is_empty(), "{:?}", report.notes);
    assert_eq!(snapshot(&fixture.root), before);
}

#[test]
fn a_version_1_workspace_is_refused_before_anything_is_read() {
    let fixture = fixture();
    write_workspace_file(&fixture.root, 1);
    let before = snapshot(&fixture.root);
    let file = fixture.root.join(WORKSPACE_FILE_NAME);
    for dry_run in [true, false] {
        let error = run_on(&fixture.path("sfmr"), dry_run).unwrap_err();
        assert!(
            matches!(&error, UpgradeWorkspaceError::Version1Workspace { path } if *path == file),
            "{error}"
        );
        assert_eq!(
            error.to_string(),
            format!(
                "the workspace file {} is version 1, and this sfmtool upgrades a version 2 \
                 workspace. sfmtool 0.4.1 is the last release that reads the older .sift, \
                 .sfmr, .matches and .camrig versions a version 1 workspace holds: install it, \
                 run `sfm ws upgrade` on the workspace, then use the workspace with this \
                 release.",
                file.display()
            )
        );
    }
    assert_eq!(snapshot(&fixture.root), before);
}

#[test]
fn a_nested_version_1_workspace_is_refused_before_anything_is_written() {
    let fixture = fixture();
    let nested = fixture.path("nested");
    write_workspace_file(&nested, 1);
    let before = snapshot(&fixture.root);
    let error = run_on(&fixture.root, false).unwrap_err();
    let file = nested.join(WORKSPACE_FILE_NAME);
    assert!(
        matches!(&error, UpgradeWorkspaceError::Version1Workspace { path } if *path == file),
        "{error}"
    );
    assert_eq!(snapshot(&fixture.root), before);
}

#[test]
fn a_nested_workspace_s_files_are_read_with_the_root_s() {
    let fixture = fixture();
    let nested = fixture.path("nested");
    write_workspace_file(&nested, 2);
    std::fs::rename(fixture.path("old.camrig"), nested.join("old.camrig")).unwrap();
    let report = run(&fixture, false);
    assert_eq!(
        report.workspaces,
        vec![fixture.root.clone(), nested.clone()]
    );
    let row = report
        .files
        .iter()
        .find(|f| f.path == nested.join("old.camrig"))
        .unwrap();
    assert!(
        matches!(&row.outcome, UpgradeOutcome::NotConverted(reason)
            if reason.contains("is a version 1 .camrig file")),
        "{:?}",
        row.outcome
    );

    // Run in the nested workspace, the tree is that workspace alone.
    let report = run_on(&nested, true).unwrap();
    assert_eq!(report.workspace_dir, nested);
    assert_eq!(report.workspaces, vec![nested.clone()]);
    assert!(report.files.iter().all(|f| f.path.starts_with(&nested)));
}

#[test]
fn each_error_says_what_stopped_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let error = run_on(dir.path(), true).unwrap_err();
    assert!(
        matches!(error, UpgradeWorkspaceError::NoWorkspace { .. }),
        "{error}"
    );
    assert!(error
        .to_string()
        .contains("no workspace file .sfm-workspace.json"));

    let fixture = fixture();
    let file = fixture.root.join(WORKSPACE_FILE_NAME);
    for (text, problem) in [
        ("{", "is not valid JSON"),
        ("{}", "has no integer 'version'"),
        (r#"{"version": 3}"#, "is version 3, and this build upgrades"),
    ] {
        std::fs::write(&file, text).unwrap();
        let error = run_on(&fixture.root, true).unwrap_err();
        assert!(
            matches!(error, UpgradeWorkspaceError::WorkspaceFile { .. }),
            "{error}"
        );
        assert!(error.to_string().contains(problem), "{error}");
    }
}

#[test]
fn a_file_the_reader_cannot_parse_is_left_and_named() {
    let fixture = fixture();
    let broken = fixture.path("sfmr/broken.sfmr");
    std::fs::write(&broken, b"not a zip").unwrap();
    let report = run(&fixture, false);
    assert!(reason_of(&report, &broken).contains("the current reader refuses it"));
    assert_eq!(std::fs::read(&broken).unwrap(), b"not a zip");
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
    let outside = fixture.dir.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    write_rig(&outside.join("rig.camrig"));
    let link = fixture.path("linked");
    link_directory(&outside, &link);
    let before = snapshot(&outside);
    let report = run(&fixture, false);
    assert_eq!(report.linked_directories, vec![link.clone()]);
    assert!(report.files.iter().all(|f| !f.path.starts_with(&link)));
    assert_eq!(snapshot(&outside), before);
    // A junction is removed as a directory, which leaves its target.
    #[cfg(windows)]
    std::fs::remove_dir(&link).unwrap();
}

#[test]
fn a_run_stopped_part_way_finishes_when_run_again() {
    let fixture = fixture();
    let before = snapshot(&fixture.root);
    let flag = AtomicBool::new(false);
    // Stop once the first file of the rewrite is counted.
    let sink = |event: Event<'_>| {
        if matches!(
            event,
            Event::Count {
                done: 1,
                unit: "file",
                ..
            }
        ) {
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
    assert_eq!(snapshot(&fixture.root), before);

    // Run again, it goes through every file.
    let report = run(&fixture, false);
    assert_eq!(outcomes(&report), expected_outcomes(&fixture));
}

#[test]
fn a_version_16_sfmr_is_upgraded_to_the_current_version() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("ws");
    let fixture = Fixture { dir, root };
    write_workspace_file(&fixture.root, 2);

    let v16 = fixture.path("sfmr/v16.sfmr");
    reconstruction().save(&v16, &Progress::none()).unwrap();
    rewrite_archive(&v16, relabel(16, |_| false));
    assert_eq!(
        sfmtool_sfmr_format::read_sfmr_metadata(&v16)
            .unwrap()
            .version,
        16
    );

    let before = snapshot(&fixture.root);
    let dry = run(&fixture, true);
    assert_eq!(
        outcomes(&dry),
        vec![(v16.clone(), UpgradeFileKind::Sfmr, "converted")]
    );
    assert!(dry.notes.iter().any(|n| n.contains("content_xxh128")));
    assert_eq!(snapshot(&fixture.root), before);

    let report = run(&fixture, false);
    assert_eq!(
        outcomes(&report),
        vec![(v16.clone(), UpgradeFileKind::Sfmr, "converted")]
    );
    assert!(report.notes.iter().any(|n| n.contains("content_xxh128")));
    assert_eq!(
        sfmtool_sfmr_format::read_sfmr_metadata(&v16)
            .unwrap()
            .version,
        sfmtool_sfmr_format::SFMR_FORMAT_VERSION
    );
    assert_eq!(
        sfmtool_sfmr_format::verify_sfmr(&v16).unwrap(),
        (true, vec![])
    );

    let again = run(&fixture, false);
    assert!(again.nothing_to_do(), "{again:?}");
}

/// A version 16 `sift_files` file written without `tracks/keypoints_xy`, with
/// no `.sift` files to build the column from, fails, and a dry run reports it
/// failed with the reason the run gives, writing nothing either way.
#[test]
fn a_version_16_sfmr_without_keypoints_or_sift_files_fails_in_a_dry_run_as_in_a_run() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("ws");
    let fixture = Fixture { dir, root };
    write_workspace_file(&fixture.root, 2);

    let v16 = fixture.path("sfmr/v16.sfmr");
    reconstruction().save(&v16, &Progress::none()).unwrap();
    crate::reconstruction::data::rewrite_as_version_16_without_keypoints(&v16);
    let before = snapshot(&fixture.root);

    let dry = run(&fixture, true);
    let report = run(&fixture, false);
    let failed = vec![(v16.clone(), UpgradeFileKind::Sfmr, "failed")];
    assert_eq!(outcomes(&dry), failed);
    assert_eq!(outcomes(&report), failed);
    let reason = reason_of(&report, &v16);
    assert!(
        reason.starts_with("it could not be written: ")
            && reason.contains("tracks/keypoints_xy")
            && reason.contains(".sift: no such file")
            && reason.contains("Restore the .sift files"),
        "{reason}"
    );
    assert_eq!(reason_of(&dry, &v16), reason);
    assert_eq!(snapshot(&fixture.root), before);
}
