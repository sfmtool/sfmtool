// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `prune_mipmaps` on 300 x 200 synthetic photographs in temporary
//! workspaces whose `mipmap_options` sets a threshold of 64, with tiles of
//! 32, so each photograph stores levels 1 to 4, and a PNG also `-mip0`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::*;
use crate::camera::mipmap::check_mipmaps;
use crate::camera::mipmap::test_support::{
    encode_jpeg, encode_png, make_mipmap, synthetic_rgb, write_workspace_file, JpegOptions,
    TestWorkspace,
};

const TILE: u32 = 32;

fn options(dry_run: bool) -> PruneMipmapsOptions {
    PruneMipmapsOptions {
        dry_run,
        tile_size: TILE,
    }
}

fn prune(paths: &[&Path], dry_run: bool) -> MipmapPruneReport {
    prune_mipmaps(paths, &options(dry_run), &Progress::none()).unwrap()
}

/// Every file under `dir`, sorted.
fn files(dir: &Path) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries {
            let entry = entry.unwrap();
            let file_type = entry.file_type().unwrap();
            if file_type.is_dir() && !file_type.is_symlink() {
                stack.push(entry.path());
            } else {
                out.insert(entry.path());
            }
        }
    }
    out
}

fn png() -> Vec<u8> {
    encode_png(&synthetic_rgb(300, 200))
}

/// A region-readable JPEG, whose mipmap has no `-mip0`.
fn readable_jpeg() -> Vec<u8> {
    encode_jpeg(
        &synthetic_rgb(300, 200),
        JpegOptions {
            restart_interval: 19,
            ..JpegOptions::default()
        },
    )
}

fn set_age(path: &Path, age: Duration) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(SystemTime::now() - age)
        .unwrap();
}

/// A workspace holding each kind of orphan and each kind of file a prune
/// keeps, with the orphans and the kept files.
struct Scene {
    ws: TestWorkspace,
    orphans: BTreeSet<PathBuf>,
    kept: BTreeSet<PathBuf>,
    other_files: BTreeSet<PathBuf>,
    /// A file named like a mip level file outside the mipmap directory.
    outside: PathBuf,
}

fn scene() -> Scene {
    let ws = TestWorkspace::new(Some(95));
    let current = ws.write("images/current.png", &png());
    let readable = ws.write("images/readable.jpg", &readable_jpeg());
    let gone = ws.write("images/gone.png", &png());
    let changed = ws.write("images/changed.png", &png());
    let partial = ws.write("images/partial.png", &png());
    for photograph in [&current, &readable, &gone, &changed, &partial] {
        make_mipmap(photograph, TILE);
    }
    std::fs::remove_file(&gone).unwrap();
    std::fs::write(
        &changed,
        encode_jpeg(&synthetic_rgb(300, 200), JpegOptions::default()),
    )
    .unwrap();
    // A build that stopped before the coarsest file: the finer files are
    // current, and kept for the next build.
    std::fs::remove_file(ws.mipmap("images/partial.png-mip4-q95.jpg")).unwrap();
    let mip1 = std::fs::read(ws.mipmap("images/current.png-mip1-q95.jpg")).unwrap();
    let mip0 = std::fs::read(ws.mipmap("images/current.png-mip0-q95.jpg")).unwrap();

    let level_files = |name: &str, levels: &[usize]| -> Vec<PathBuf> {
        levels
            .iter()
            .map(|l| ws.mipmap(&format!("images/{name}-mip{l}-q95.jpg")))
            .collect()
    };
    let mut orphans: BTreeSet<PathBuf> = BTreeSet::new();
    orphans.extend(level_files("gone.png", &[0, 1, 2, 3, 4]));
    orphans.extend(level_files("changed.png", &[0, 1, 2, 3, 4]));
    // Another quality, a level the mipmap does not store, a `-mip0` beside a
    // region-readable original, and an old temporary file.
    let other_quality = ws.write(".sfm-mipmaps/images/current.png-mip1-q90.jpg", &mip1);
    let beyond = ws.write(".sfm-mipmaps/images/current.png-mip5-q95.jpg", &mip1);
    let readable_mip0 = ws.write(".sfm-mipmaps/images/readable.jpg-mip0-q95.jpg", &mip0);
    let old_temporary = ws.write(
        ".sfm-mipmaps/images/.current.png-mip1-q95.jpg.123-4.tmp",
        b"partial",
    );
    set_age(&old_temporary, Duration::from_secs(2 * 60 * 60));
    // A photograph in core, whose files are left from a larger one.
    ws.write("images/small.png", &encode_png(&synthetic_rgb(60, 40)));
    let small = ws.write(".sfm-mipmaps/images/small.png-mip1-q95.jpg", &mip1);
    orphans.extend([other_quality, beyond, readable_mip0, old_temporary, small]);

    let mut kept: BTreeSet<PathBuf> = BTreeSet::new();
    kept.extend(level_files("current.png", &[0, 1, 2, 3, 4]));
    kept.extend(level_files("readable.jpg", &[1, 2, 3, 4]));
    kept.extend(level_files("partial.png", &[0, 1, 2, 3]));
    let new_temporary = ws.write(
        ".sfm-mipmaps/images/.current.png-mip2-q95.jpg.123-5.tmp",
        b"partial",
    );
    kept.insert(new_temporary);

    let other_files: BTreeSet<PathBuf> = [
        ws.mipmap("CACHEDIR.TAG"),
        ws.write(".sfm-mipmaps/notes.txt", b"notes"),
        ws.write(".sfm-mipmaps/images/current.png-mip1-q95.jpg.bak", &mip1),
        ws.write(".sfm-mipmaps/images/current.png-mip01-q95.jpg", &mip1),
        ws.write(".sfm-mipmaps/images/current.png-mip1-q101.jpg", &mip1),
        ws.write(".sfm-mipmaps/images/.old.tmp", b"x"),
    ]
    .into();
    let outside = ws.write("images/gone.png-mip1-q95.jpg", &mip1);
    Scene {
        ws,
        orphans,
        kept,
        other_files,
        outside,
    }
}

fn paths(report: &[(PathBuf, String)]) -> BTreeSet<PathBuf> {
    report.iter().map(|(p, _)| p.clone()).collect()
}

/// A prune deletes each kind of orphan and keeps the current files, a new
/// temporary file, the files that are not mip level files and every file
/// outside the mipmap directory; the photographs are then current or
/// missing, never stale.
#[test]
fn a_prune_deletes_the_orphans_and_keeps_everything_else() {
    let scene = scene();
    let root = scene.ws.root();
    let before = files(&root);

    let report = prune(&[root.as_path()], false);

    assert_eq!(paths(&report.orphans), scene.orphans);
    assert_eq!(
        report.kept.iter().cloned().collect::<BTreeSet<_>>(),
        scene.kept
    );
    assert_eq!(
        report.other_files.iter().cloned().collect::<BTreeSet<_>>(),
        scene.other_files
    );
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    let after = files(&root);
    assert_eq!(after, &before - &scene.orphans);
    assert!(scene.outside.is_file());
    let why = |name: &str| {
        let path = scene.ws.mipmap(&format!("images/{name}"));
        report
            .orphans
            .iter()
            .find(|(p, _)| *p == path)
            .unwrap()
            .1
            .clone()
    };
    assert_eq!(
        why("gone.png-mip1-q95.jpg"),
        "its photograph images/gone.png no longer exists"
    );
    assert!(why("changed.png-mip1-q95.jpg").starts_with("it is not current: "));
    assert_eq!(
        why("current.png-mip1-q90.jpg"),
        "its quality 90 is not the workspace's jpeg_quality, 95"
    );
    assert_eq!(
        why("current.png-mip5-q95.jpg"),
        "the mipmap of images/current.png stores levels 1 to 4"
    );
    assert_eq!(
        why("small.png-mip1-q95.jpg"),
        "its photograph images/small.png is in core"
    );
    assert!(why(".current.png-mip1-q95.jpg.123-4.tmp").starts_with("it is a temporary file"));

    let photographs: Vec<PathBuf> = ["current.png", "readable.jpg", "partial.png"]
        .iter()
        .map(|name| root.join("images").join(name))
        .collect();
    let photographs: Vec<&Path> = photographs.iter().map(PathBuf::as_path).collect();
    let check = check_mipmaps(&photographs, TILE, &Progress::none()).unwrap();
    assert_eq!(check.current.len(), 2);
    assert_eq!(check.missing, vec![root.join("images/partial.png")]);
    assert!(check.stale.is_empty());

    // A second prune finds nothing more.
    assert!(prune(&[root.as_path()], false).orphans.is_empty());
}

/// A dry run reports the same orphans and deletes nothing.
#[test]
fn a_dry_run_reports_the_orphans_and_deletes_nothing() {
    let scene = scene();
    let root = scene.ws.root();
    let before = files(&root);
    let report = prune(&[root.as_path()], true);
    assert_eq!(paths(&report.orphans), scene.orphans);
    assert_eq!(files(&root), before);
}

/// In a workspace whose file has no `mipmap_options` every mip level file is
/// an orphan, and the other files stay.
#[test]
fn a_workspace_without_the_object_loses_every_mip_level_file() {
    let ws = TestWorkspace::new(Some(95));
    let photograph = ws.write("images/a.png", &png());
    make_mipmap(&photograph, TILE);
    let notes = ws.write(".sfm-mipmaps/notes.txt", b"notes");
    ws.set_quality(None);
    let root = ws.root();

    let report = prune(&[root.as_path()], false);

    assert_eq!(report.orphans.len(), 5);
    assert!(report
        .orphans
        .iter()
        .all(|(_, why)| why == "the workspace's file has no mipmap_options"));
    assert_eq!(
        files(&root.join(MIPMAP_DIR_NAME)),
        [ws.mipmap("CACHEDIR.TAG"), notes].into()
    );
    assert!(photograph.is_file());
}

/// A path searches only the part of the mipmap directory that mirrors it:
/// a directory everything under its mirror, a photograph's path, existing or
/// not, only that photograph's files.
#[test]
fn a_prune_searches_only_the_mirror_of_its_paths() {
    let ws = TestWorkspace::new(Some(95));
    let mut by_photograph = Vec::new();
    for relative in ["a/one.png", "a/sub/two.png", "b/three.png", "b/four.png"] {
        let photograph = ws.write(relative, &png());
        make_mipmap(&photograph, TILE);
        by_photograph.push(photograph);
    }
    ws.set_quality(None);
    let root = ws.root();
    let count = |prefix: &str| {
        files(&root.join(MIPMAP_DIR_NAME))
            .iter()
            .filter(|p| p.to_string_lossy().contains(prefix))
            .count()
    };

    let report = prune(&[root.join("a").as_path()], false);
    assert_eq!(report.orphans.len(), 10);
    assert_eq!((count("one.png"), count("two.png")), (0, 0));
    assert_eq!((count("three.png"), count("four.png")), (5, 5));

    std::fs::remove_file(&by_photograph[2]).unwrap();
    let report = prune(&[by_photograph[2].as_path()], false);
    assert_eq!(report.orphans.len(), 5);
    assert_eq!((count("three.png"), count("four.png")), (0, 5));
}

/// A path in no workspace is reported and searches nothing.
#[test]
fn a_path_in_no_workspace_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let report = prune(&[dir.path()], false);
    assert_eq!(report.outside_workspaces, vec![dir.path().to_path_buf()]);
    assert!(report.orphans.is_empty() && report.failed.is_empty());
}

/// A workspace whose `mipmap_options` is invalid is not searched.
#[test]
fn an_invalid_object_searches_nothing() {
    let ws = TestWorkspace::new(Some(95));
    let photograph = ws.write("a.png", &png());
    make_mipmap(&photograph, TILE);
    let stray = ws.write(".sfm-mipmaps/gone.png-mip1-q95.jpg", b"");
    let root = ws.root();
    std::fs::write(
        root.join(crate::workspace::WORKSPACE_FILE_NAME),
        r#"{"version": 2, "mipmap_options": {"jpeg_quality": 95}}"#,
    )
    .unwrap();
    let report = prune(&[root.as_path()], false);
    assert!(report.orphans.is_empty());
    let [(path, why)] = &report.failed[..] else {
        panic!("{:?}", report.failed);
    };
    assert_eq!(path, &root.join(crate::workspace::WORKSPACE_FILE_NAME));
    assert!(
        why.starts_with("its mipmap directory was not searched: "),
        "{why}"
    );
    assert!(stray.is_file());
}

/// A workspace nested below a path has its own mipmap directory searched,
/// and a file in the enclosing workspace's directory for a photograph of the
/// nested one is an orphan there.
#[test]
fn a_nested_workspace_is_searched_and_keeps_its_own_mipmaps() {
    let ws = TestWorkspace::new(Some(95));
    let root = ws.root();
    write_workspace_file(&root.join("inner").tap_mkdir(), Some(95));
    let photograph = ws.write("inner/a.png", &png());
    make_mipmap(&photograph, TILE);
    let inner_mipmaps = root.join("inner").join(MIPMAP_DIR_NAME);
    let mip1 = std::fs::read(inner_mipmaps.join("a.png-mip1-q95.jpg")).unwrap();
    let misplaced = ws.write(".sfm-mipmaps/inner/a.png-mip1-q95.jpg", &mip1);
    let inner_orphan = ws.write("inner/.sfm-mipmaps/gone.png-mip1-q95.jpg", &mip1);

    let report = prune(&[root.as_path()], false);

    assert_eq!(
        paths(&report.orphans),
        [misplaced.clone(), inner_orphan].into()
    );
    let why = &report
        .orphans
        .iter()
        .find(|(p, _)| *p == misplaced)
        .unwrap()
        .1;
    assert_eq!(
        why,
        &format!(
            "the mipmap of inner/a.png is kept in {}",
            Path::new("inner").join(MIPMAP_DIR_NAME).display()
        )
    );
    assert_eq!(report.kept.len(), 5);
}

trait TapMkdir {
    fn tap_mkdir(self) -> Self;
}

impl TapMkdir for PathBuf {
    fn tap_mkdir(self) -> Self {
        std::fs::create_dir_all(&self).unwrap();
        self
    }
}

/// Make `link` a link to the directory `target`: a junction on Windows,
/// which needs no privilege, a symbolic link elsewhere.
fn link_dir(target: &Path, link: &Path) {
    #[cfg(windows)]
    {
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "mklink /J failed");
    }
    #[cfg(not(windows))]
    std::os::unix::fs::symlink(target, link).unwrap();
}

/// A link inside the mipmap directory, or a mipmap directory that is one,
/// is neither entered nor deleted, so the files it leads to stay, even in a
/// workspace without the object, where every mip level file is an orphan.
#[test]
fn links_are_neither_entered_nor_deleted() {
    let ws = TestWorkspace::new(None);
    let root = ws.root();
    let elsewhere = tempfile::tempdir().unwrap();
    let target = elsewhere.path().join("x.png-mip1-q95.jpg");
    std::fs::write(&target, b"not ours").unwrap();
    let images = ws.mipmap("images").tap_mkdir();
    let link = images.join("linked");
    link_dir(elsewhere.path(), &link);
    let own = ws.write(".sfm-mipmaps/images/a.png-mip1-q95.jpg", b"");

    let report = prune(&[root.as_path()], false);
    assert_eq!(paths(&report.orphans), [own].into());
    assert_eq!(report.links, vec![link.clone()]);
    assert!(target.is_file());
    assert!(std::fs::symlink_metadata(&link).is_ok());

    // A path through the link searches nothing.
    let report = prune(&[root.join("images/linked/x.png").as_path()], false);
    assert!(report.orphans.is_empty());
    assert_eq!(report.links, vec![link]);
    assert!(target.is_file());

    // A mipmap directory that is a link is not entered.
    let other = TestWorkspace::new(None);
    link_dir(elsewhere.path(), &other.root().join(MIPMAP_DIR_NAME));
    let report = prune(&[other.root().as_path()], false);
    assert!(report.orphans.is_empty());
    assert_eq!(report.links, vec![other.root().join(MIPMAP_DIR_NAME)]);
    assert!(target.is_file());
}

/// A symbolic link named like a mip level file is not deleted, nor is the
/// file it points at.
#[cfg(unix)]
#[test]
fn a_link_to_a_file_is_not_deleted() {
    let ws = TestWorkspace::new(None);
    let elsewhere = tempfile::tempdir().unwrap();
    let target = elsewhere.path().join("x.jpg");
    std::fs::write(&target, b"not ours").unwrap();
    let link = ws.mipmap("a.png-mip1-q95.jpg");
    std::fs::create_dir_all(link.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let report = prune(&[ws.root().as_path()], false);
    assert!(report.orphans.is_empty());
    assert_eq!(report.links, vec![link.clone()]);
    assert!(target.is_file() && std::fs::symlink_metadata(&link).is_ok());
}

/// The last check before a delete refuses a file whose directory on disk is
/// not inside the mipmap directory, as a path through a link is.
#[test]
fn a_delete_refuses_a_file_outside_the_mipmap_directory() {
    let ws = TestWorkspace::new(None);
    let mipmaps = ws.mipmap("").tap_mkdir();
    let elsewhere = tempfile::tempdir().unwrap();
    let target = elsewhere.path().join("x.png-mip1-q95.jpg");
    std::fs::write(&target, b"not ours").unwrap();
    let link = mipmaps.join("linked");
    link_dir(elsewhere.path(), &link);
    assert!(delete(&mipmaps, &link.join("x.png-mip1-q95.jpg")).is_err());
    assert!(delete(&mipmaps, &target).is_err());
    assert!(target.is_file());
    let own = ws.write(".sfm-mipmaps/a.png-mip1-q95.jpg", b"");
    delete(&mipmaps, &own).unwrap();
    assert!(!own.exists());
}

#[test]
fn names_are_read_as_a_mip_level_file_s_or_a_temporary_file_s() {
    assert_eq!(
        parse_name("a-mip1-q95.jpg-mip2-q90.jpg"),
        Named::MipLevel {
            photograph: "a-mip1-q95.jpg",
            level: 2,
            quality: 90
        }
    );
    assert_eq!(
        parse_name("IMG.JPG-mip0-q100.jpg"),
        Named::MipLevel {
            photograph: "IMG.JPG",
            level: 0,
            quality: 100
        }
    );
    assert_eq!(parse_name(".a.png-mip1-q95.jpg.77-0.tmp"), Named::Temporary);
    for other in [
        "-mip1-q95.jpg",
        "a.png-mip1-q95.JPG",
        "a.png-mip-q95.jpg",
        "a.png-mip1-q0.jpg",
        "a.png-mip1-q095.jpg",
        "a.png-mip1-q95.jpg.tmp",
        ".a.png-mip1-q95.jpg.77.tmp",
        ".a.png.77-0.tmp",
        "CACHEDIR.TAG",
    ] {
        assert_eq!(parse_name(other), Named::Other, "{other}");
    }
}
