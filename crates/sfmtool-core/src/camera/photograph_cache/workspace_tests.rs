// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The workspace setting the cache reads its photographs under: each
//! workspace root's `mipmap_options`, read once and again when the workspace
//! file changes, and every photograph of a workspace whose file is invalid
//! refused until the file is fixed.

use std::time::{Duration, UNIX_EPOCH};

use crate::camera::sampler::{read_source_rect, test_pool, LevelRect};
use crate::progress::Progress;

use super::tests::{with_budget, write_png};
use super::*;

/// A `mipmap_options` whose `jpeg_quality` is out of range.
const INVALID: &str = r#"{"mipmap_options": {"out_of_core_long_side": 4096,
    "jpeg_quality": 0, "full_resolution_jpeg_quality": 95}}"#;

/// A valid `mipmap_options` under which every test photograph is in core.
fn valid(threshold: u32) -> String {
    format!(
        r#"{{"mipmap_options": {{"out_of_core_long_side": {threshold},
        "jpeg_quality": 95, "full_resolution_jpeg_quality": 95}}}}"#
    )
}

/// Write `text` as the workspace file at `root`, its modification time set to
/// `seconds` after the epoch, so each write in a test has a stamp of its own
/// on every file system.
fn write_workspace(root: &Path, text: &str, seconds: u64) {
    let path = root.join(WORKSPACE_FILE_NAME);
    std::fs::write(&path, text).unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(UNIX_EPOCH + Duration::from_secs(seconds))
        .unwrap();
}

/// Whether `message` names the key at fault in [`INVALID`].
fn names_the_key(message: &str) -> bool {
    message.contains("mipmap_options.jpeg_quality")
}

/// Level 0 of `photograph`, read whole through a render entry point.
fn level_0(photograph: &TiledPhotograph) -> Result<Vec<u8>, PhotographReadError> {
    let (x1, y1) = photograph.layout().level_size(0);
    let rect = LevelRect {
        x0: 0,
        y0: 0,
        x1,
        y1,
    };
    let read = read_source_rect(photograph, test_pool(), 0, rect, &Progress::none())?;
    Ok(read.image().data().to_vec())
}

#[test]
fn an_invalid_mipmap_options_refuses_every_photograph_of_its_workspace() {
    let key = TileKey {
        level: 0,
        col: 0,
        row: 0,
    };
    for budget in [GIB, 0] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_workspace(root, INVALID, 1);
        std::fs::create_dir(root.join("images")).unwrap();
        let a = write_png(&root.join("images"), "a.png", 48, 32);
        let b = write_png(&root.join("images"), "b.png", 40, 24);
        let cache = PhotographCache::new(with_budget(budget));

        match cache.get(&a, &Progress::none()) {
            Err(GetError::Workspace(e)) => assert!(names_the_key(&e.to_string()), "{e}"),
            other => panic!("budget {budget}: {:?}", other.map(|_| ())),
        }
        let (found, tally) = cache.get_many(&[&a, &b], &Progress::none()).unwrap();
        assert!(found.iter().all(Option::is_none), "budget {budget}");
        assert_eq!(tally.unreadable, 2, "budget {budget}");

        cache.request_layout(&b);
        match cache.layout(&b) {
            PeekedLayout::Unreadable(GetError::Workspace(e)) => {
                assert!(names_the_key(&e.to_string()), "{e}")
            }
            other => panic!("budget {budget}: {other:?}"),
        }
        match cache.try_decode_tiles(&b, &[key], &Progress::none()) {
            Err(TileError::Unreadable {
                path,
                why: GetError::Workspace(e),
            }) => {
                assert_eq!(path, b);
                assert!(names_the_key(&e.to_string()), "{e}");
            }
            Err(e) => panic!("budget {budget}: {e}"),
            Ok(_) => panic!("budget {budget}: answered tiles"),
        }
        match cache.get_tile(&a, key, &Progress::none()) {
            Err(TileError::Unreadable {
                why: GetError::Workspace(e),
                ..
            }) => assert!(names_the_key(&e.to_string()), "{e}"),
            Err(e) => panic!("budget {budget}: {e}"),
            Ok(_) => panic!("budget {budget}: answered a tile"),
        }
        assert_eq!(cache.stats().decodes, 0, "budget {budget}");
    }
}

#[test]
fn a_fixed_workspace_file_reads_its_photographs_and_a_broken_one_refuses_them() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_workspace(root, INVALID, 1);
    let a = write_png(root, "a.png", 48, 32);
    let cache = PhotographCache::new(with_budget(GIB));
    assert!(matches!(
        cache.get(&a, &Progress::none()),
        Err(GetError::Workspace(_))
    ));

    // Fixed: the next `get` decodes the photograph, its own pixels.
    write_workspace(root, &valid(4096), 2);
    let photograph = cache.get(&a, &Progress::none()).unwrap();
    assert_eq!(cache.stats().decodes, 1);
    assert_eq!(
        level_0(&photograph).unwrap(),
        ImageU8::read_rgb(&a).unwrap().data()
    );

    // Broken again: the photograph decoded under the valid file is refused,
    // by `get`, by `get_many` and by a read through the photograph answered
    // before, and its tiles are dropped.
    write_workspace(root, INVALID, 3);
    assert!(matches!(
        cache.get(&a, &Progress::none()),
        Err(GetError::Workspace(_))
    ));
    assert_eq!(cache.stats().tiles, 0);
    let (found, _) = cache.get_many(&[&a], &Progress::none()).unwrap();
    assert!(found[0].is_none());
    match level_0(&photograph) {
        Err(PhotographReadError::Unreadable {
            path: Some(path),
            why,
        }) => {
            assert_eq!(path, a);
            assert!(names_the_key(&why), "{why}");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    assert_eq!(cache.stats().decodes, 1);
}

#[test]
fn the_workspace_file_is_read_once_per_root_and_again_when_it_changes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_workspace(root, &valid(4096), 1);
    let photos: Vec<PathBuf> = (0..5)
        .map(|i| write_png(root, &format!("{i}.png"), 48, 32))
        .collect();
    let paths: Vec<&Path> = photos.iter().map(PathBuf::as_path).collect();
    let cache = PhotographCache::new(with_budget(GIB));
    let reads = || cache.shared.workspace_reads.load(Ordering::Relaxed);

    for path in &paths {
        cache.get(path, &Progress::none()).unwrap();
        cache.request_layout(path);
    }
    cache.get_many(&paths, &Progress::none()).unwrap();
    assert_eq!((reads(), cache.stats().decodes), (1, 5));

    // Written again with the same object: read again, once, and no
    // photograph is decoded again, since the setting is the same.
    write_workspace(root, &valid(4096), 2);
    cache.get_many(&paths, &Progress::none()).unwrap();
    assert_eq!((reads(), cache.stats().decodes), (2, 5));

    // Another object: read once more, and every photograph is decoded again
    // under it.
    write_workspace(root, &valid(2000), 3);
    cache.get_many(&paths, &Progress::none()).unwrap();
    assert_eq!((reads(), cache.stats().decodes), (3, 10));
}

#[test]
fn a_layout_read_notes_the_first_large_photograph_of_a_workspace_without_mipmap_options() {
    let long = DEFAULT_OUT_OF_CORE_LONG_SIDE;
    let without = tempfile::tempdir().unwrap();
    write_workspace(without.path(), "{}", 1);
    let at_threshold = write_png(without.path(), "at.png", long, 1);
    let a = write_png(without.path(), "a.png", 1, long + 1);
    let b = write_png(without.path(), "b.png", long + 1, 1);
    let cache = PhotographCache::new(with_budget(GIB));

    cache.request_layout(&at_threshold);
    assert_eq!(cache.large_photograph_without_mipmap_options(), None);
    cache.request_layout(&a);
    assert_eq!(
        cache.large_photograph_without_mipmap_options(),
        Some(a.as_path())
    );
    // The first one found stays the answer, and the blocking read notes it too.
    cache.read_layout(&b).unwrap();
    cache.clear();
    assert_eq!(
        cache.large_photograph_without_mipmap_options(),
        Some(a.as_path())
    );
    let blocking = PhotographCache::new(with_budget(GIB));
    blocking.read_layout(&b).unwrap();
    assert_eq!(
        blocking.large_photograph_without_mipmap_options(),
        Some(b.as_path())
    );

    // A workspace whose file has the object, even one whose threshold keeps
    // the photograph in core, is not one without it.
    let with = tempfile::tempdir().unwrap();
    write_workspace(with.path(), &valid(4096), 1);
    let c = write_png(with.path(), "c.png", long + 1, 1);
    let cache = PhotographCache::new(with_budget(GIB));
    cache.request_layout(&c);
    cache.read_layout(&c).unwrap();
    assert_eq!(cache.large_photograph_without_mipmap_options(), None);
}
