// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An out-of-core photograph read from its mip level files: 300 x 200
//! synthetic photographs in a temporary workspace whose `mipmap_options`
//! sets a threshold of 64, read with tiles of 32, so each is out of core with
//! levels 1 to 4 stored, and its mipmap made with `build_mipmaps`.

use std::time::Duration;

use jpeg_encoder::SamplingFactor;

use crate::camera::mipmap::test_support::{encode_jpeg, synthetic_rgb, JpegOptions};
use crate::camera::mipmap::{
    build_mipmaps, mipmap_location, ExpectedMipLevel, MipLevelFile, MipmapOptions, OriginalRegions,
    MIPMAP_DIR_NAME,
};
use crate::camera::sampler::{read_source_rect, test_pool, LevelRect};
use crate::camera::tiled_photograph::cut_level;
use crate::progress::Progress;

use super::tests::every_key;
use super::*;

pub(super) const TILE: u32 = 32;

pub(super) fn options() -> MipmapOptions {
    MipmapOptions {
        out_of_core_long_side: 64,
        jpeg_quality: 95,
        full_resolution_jpeg_quality: 95,
    }
}

/// A temporary workspace whose file has [`options`].
pub(super) struct Workspace {
    pub(super) dir: tempfile::TempDir,
}

impl Workspace {
    pub(super) fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let o = options();
        std::fs::write(
            dir.path().join(WORKSPACE_FILE_NAME),
            format!(
                r#"{{"version": 2, "mipmap_options": {{"out_of_core_long_side": {},
                "jpeg_quality": {}, "full_resolution_jpeg_quality": {}}}}}"#,
                o.out_of_core_long_side, o.jpeg_quality, o.full_resolution_jpeg_quality
            ),
        )
        .unwrap();
        Self { dir }
    }

    /// Write `bytes` at `name` in the workspace's `images` and return its
    /// path.
    pub(super) fn photograph(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let images = self.dir.path().join("images");
        std::fs::create_dir_all(&images).unwrap();
        let path = images.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// Every file under the mipmap directory, with its bytes, sorted by path.
    pub(super) fn mipmap_files(&self) -> Vec<(PathBuf, Vec<u8>)> {
        let mut out = Vec::new();
        let mut stack = vec![self.dir.path().join(MIPMAP_DIR_NAME)];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let bytes = std::fs::read(&path).unwrap();
                    out.push((path, bytes));
                }
            }
        }
        out.sort();
        out
    }
}

/// Make `photograph`'s mipmap, writing its files.
pub(super) fn build(photograph: &Path) {
    let built = build_mipmaps(photograph, None, &options(), TILE, &Progress::none()).unwrap();
    assert_eq!(built.not_written, None);
}

/// Make `photograph`'s mipmap again with another record in every file: the
/// files are removed, since a build keeps a level whose file is current, and
/// the original's modification time is moved while they are made, and put
/// back, so the files are current for the original and their records name
/// another time.
fn rebuild_with_another_record(photograph: &Path) {
    let dir = mipmap_location(photograph).unwrap().dir().to_path_buf();
    std::fs::remove_dir_all(dir).unwrap();
    let time = std::fs::metadata(photograph).unwrap().modified().unwrap();
    let set = |t| {
        std::fs::File::options()
            .write(true)
            .open(photograph)
            .unwrap()
            .set_modified(t)
            .unwrap()
    };
    set(time + Duration::from_secs(10));
    build(photograph);
    set(time);
}

/// A region-readable JPEG: 4:2:0 with an interval of the 19 MCUs of a row.
pub(super) fn readable_jpeg() -> Vec<u8> {
    encode_jpeg(
        &synthetic_rgb(300, 200),
        JpegOptions {
            sampling: SamplingFactor::F_2_2,
            restart_interval: 19,
            ..JpegOptions::default()
        },
    )
}

/// A JPEG with no restart markers, which gets a `-mip0`.
pub(super) fn restartless_jpeg() -> Vec<u8> {
    encode_jpeg(&synthetic_rgb(300, 200), JpegOptions::default())
}

pub(super) fn cache_of(budget_bytes: u64) -> PhotographCache {
    PhotographCache::new(PhotographCacheOptions {
        budget_bytes,
        build_budget_bytes: GIB,
        tile_size: TILE,
    })
}

/// The out-of-core layout of the test photographs.
pub(super) fn out_of_core_layout() -> PyramidLayout {
    PyramidLayout::for_size(300, 200, Some(&options()), TILE).unwrap()
}

/// The path of `photograph`'s mip level file of `level`.
pub(super) fn level_file(photograph: &Path, level: usize) -> PathBuf {
    mipmap_location(photograph)
        .unwrap()
        .level_path(level, options().quality_for_level(level))
}

/// Every level of `photograph`'s pyramid as its files hold it: level 0 the
/// region decode of the original, or the decode of its `-mip0`; levels 1 to
/// `K` the region decodes of their files, whole; and each level below `K`
/// box-filtered from the one above it.
pub(super) fn files_levels(photograph: &Path) -> Vec<ImageU8> {
    let layout = out_of_core_layout();
    let relative = mipmap_location(photograph).unwrap().relative().to_string();
    let open = |level: usize| {
        let expected = ExpectedMipLevel::new(&layout, &relative, level).unwrap();
        MipLevelFile::open(&level_file(photograph, level), &expected).unwrap()
    };
    let (w, h) = layout.level_size(0);
    let bytes: Arc<[u8]> = std::fs::read(photograph).unwrap().into();
    let level0 = match OriginalRegions::from_bytes(bytes) {
        Ok(original) => original.decode_region(0, 0, w, h).unwrap(),
        Err(_) => open(0).decode_region(0, 0, w, h).unwrap(),
    };
    let mut levels = vec![level0];
    let k = layout.stored_levels();
    for level in 1..=k {
        let (w, h) = layout.level_size(level);
        levels.push(open(level).decode_region(0, 0, w, h).unwrap());
    }
    for _ in k + 1..layout.levels() {
        let next = levels.last().unwrap().downsample_2x();
        levels.push(next);
    }
    levels
}

/// Every tile of `layout` cut from `levels`, its whole levels.
pub(super) fn tiles_of(
    layout: &PyramidLayout,
    levels: &[ImageU8],
) -> HashMap<TileKey, Arc<PhotographTile>> {
    let mut tiles = HashMap::new();
    for (level, image) in levels.iter().enumerate() {
        cut_level(layout, level, image, &mut tiles);
    }
    tiles
}

pub(super) fn same_tile(a: &PhotographTile, b: &PhotographTile) -> bool {
    a.key == b.key && a.origin == b.origin && a.size == b.size && a.pixels.data() == b.pixels.data()
}

/// Level `level` of `photograph`, read whole through a render entry point.
pub(super) fn read_level(
    photograph: &TiledPhotograph,
    level: usize,
) -> Result<Vec<u8>, PhotographReadError> {
    let (x1, y1) = photograph.layout().level_size(level);
    let rect = LevelRect {
        x0: 0,
        y0: 0,
        x1,
        y1,
    };
    let read = read_source_rect(photograph, test_pool(), level, rect, &Progress::none())?;
    Ok(read.image().data().to_vec())
}

/// Every tile, level and region of an out-of-core photograph whose mipmap is
/// current is the same pixels as its files hold, with nothing decoded whole:
/// levels 1 to `K` from their files, the levels below `K` box-filtered from
/// level `K`, and level 0 from the original when it is region-readable or
/// from its `-mip0` otherwise; with a budget, and with one of 0, under which
/// each request reads the files again. Its layout is the out-of-core one.
#[test]
fn a_current_mipmap_is_read_from_its_files() {
    for (name, jpeg) in [
        ("readable.jpg", readable_jpeg()),
        ("restartless.jpg", restartless_jpeg()),
    ] {
        let ws = Workspace::new();
        let path = ws.photograph(name, &jpeg);
        build(&path);
        let levels = files_levels(&path);
        let layout = out_of_core_layout();
        let expected = tiles_of(&layout, &levels);
        assert_eq!(expected.len(), 102, "every tile of the 8 levels");
        // The in-core pixels of level 1 are not the file's, so the test
        // tells the two apart.
        let in_core = ImageU8::read_rgb(&path).unwrap().downsample_2x();
        assert_ne!(in_core.data(), levels[1].data(), "{name}");
        for budget in [GIB, 0] {
            let cache = cache_of(budget);
            cache.request_layout(&path);
            match cache.layout(&path) {
                PeekedLayout::Known(known) => assert_eq!(*known, layout, "{name}"),
                other => panic!("{name}: {other:?}"),
            }
            for key in every_key(&layout) {
                let tile = cache.get_tile(&path, key, &Progress::none()).unwrap();
                assert!(
                    same_tile(&tile, &expected[&key]),
                    "{name} {key:?}, {budget}"
                );
            }
            let photograph = cache.get(&path, &Progress::none()).unwrap();
            assert_eq!(*photograph.layout(), layout);
            for (level, image) in levels.iter().enumerate() {
                assert_eq!(read_level(&photograph, level).unwrap(), image.data());
                let whole = cache.get_level(&path, level, &Progress::none()).unwrap();
                assert_eq!(whole.data(), image.data(), "{name} level {level}");
            }
            let stats = cache.stats();
            assert_eq!(stats.decodes, 0, "{name}: nothing decoded whole");
            if budget > 0 {
                assert_eq!(stats.tiles, 102);
                // Levels 0 to K, the original standing for level 0 when it
                // is region-readable, its `-mip0` otherwise.
                assert_eq!(stats.files, 5, "{name}");
                let tile_bytes: u64 = expected.values().map(|t| tile_bytes(t)).sum();
                assert!(stats.bytes > tile_bytes, "the files are counted");
            }
        }
    }
}

/// Under a budget that holds the coarse tiles and little else, reading every
/// tile of levels 0 and 1 after them evicts tiles and file entries read
/// later than the coarse tiles, and never a tile of level `K` or a level
/// below it.
#[test]
fn the_coarsest_level_outlives_every_other_entry() {
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    build(&path);
    let layout = out_of_core_layout();
    let k = layout.stored_levels();
    let budget = 60_000;
    let cache = cache_of(budget);
    let coarse = |level: usize| TileKey {
        level: level as u8,
        col: 0,
        row: 0,
    };
    cache.get_tile(&path, coarse(k), &Progress::none()).unwrap();
    for key in every_key(&layout).into_iter().filter(|key| key.level <= 1) {
        cache.get_tile(&path, key, &Progress::none()).unwrap();
    }
    let stats = cache.stats();
    assert!(stats.bytes <= budget, "{stats:?}");
    assert!(stats.tiles < 90, "the finer tiles were evicted: {stats:?}");
    for level in k..layout.levels() {
        assert!(
            matches!(cache.peek_tile(&path, coarse(level)), PeekedTile::Ready(_)),
            "level {level} is kept"
        );
    }
}

/// A kernel's read of a file replaced, after the read first opened it, by
/// one with another record fails with `Changed` naming the file, both a read
/// through the photograph `get` answered (from one cache) and a `get_tile`
/// (from another); after that the slot's mipmap is not current, so the next
/// `get_tile` answers `Changed` naming the photograph, decoding nothing,
/// until a `get` makes the mipmap again, since a read found a file of it
/// not current, which encodes nothing over the new files, current, and then
/// reads them. A read through a photograph after
/// its original changed fails with `Changed` naming the original. Under a
/// budget too small to keep a file, so each read opens it again, and under a
/// budget of 0, where a `get_tile` decides from the files as they are, the
/// rebuilt files being current.
#[test]
fn a_file_changed_during_a_kernel_s_read_fails_it_with_changed() {
    let key = TileKey {
        level: 1,
        col: 0,
        row: 0,
    };
    // A read of level 2, which keeps only its own tile under a budget of 1,
    // evicts level 1's.
    let level_2 = TileKey {
        level: 2,
        col: 0,
        row: 0,
    };
    for budget in [1, 0] {
        let ws = Workspace::new();
        let path = ws.photograph("a.jpg", &readable_jpeg());
        build(&path);
        let reader = cache_of(budget);
        let photograph = reader.get(&path, &Progress::none()).unwrap();
        assert!(read_level(&photograph, 1).is_ok());
        reader.get_tile(&path, level_2, &Progress::none()).unwrap();
        let cache = cache_of(budget);
        let before = cache.get_tile(&path, key, &Progress::none()).unwrap();
        cache.get_tile(&path, level_2, &Progress::none()).unwrap();

        rebuild_with_another_record(&path);
        let mip1 = level_file(&path, 1);
        match read_level(&photograph, 1) {
            Err(PhotographReadError::Changed { path: named }) => assert_eq!(named, mip1),
            other => panic!("budget {budget}: {other:?}"),
        }
        match cache.get_tile(&path, key, &Progress::none()) {
            Err(TileError::Changed { path: named }) => {
                assert!(budget > 0);
                assert_eq!(named, mip1);
                match cache.get_tile(&path, key, &Progress::none()) {
                    Err(TileError::Changed { path: named }) => assert_eq!(named, path),
                    other => panic!("not read in core: {:?}", other.err()),
                }
                let on_disk = ws.mipmap_files();
                cache.get(&path, &Progress::none()).unwrap();
                assert_eq!(ws.mipmap_files(), on_disk, "nothing written over them");
                let tile = cache.get_tile(&path, key, &Progress::none()).unwrap();
                assert!(same_tile(&tile, &before), "the new files' pixels");
                let stats = cache.stats();
                assert_eq!((stats.decodes, stats.mipmaps_made), (0, 1));
            }
            // The rebuilt files hold the same pixels.
            Ok(tile) => {
                assert_eq!(budget, 0);
                assert!(same_tile(&tile, &before));
                assert_eq!(cache.stats().decodes, 0);
            }
            Err(other) => panic!("budget {budget}: {other:?}"),
        }

        // The original changed: every file is stale for it.
        let photograph = cache_of(budget).get(&path, &Progress::none()).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 3;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        match read_level(&photograph, 0) {
            Err(PhotographReadError::Changed { path: named }) => assert_eq!(named, path),
            other => panic!("budget {budget}: {other:?}"),
        }
    }
}

/// An in-core photograph of the same workspace reads as before: decoded
/// whole once, its tiles those of the decode's pyramid, its layout in core,
/// and no mipmap made.
#[test]
fn an_in_core_photograph_in_the_same_workspace_reads_as_before() {
    let ws = Workspace::new();
    let small = synthetic_rgb(60, 40);
    let path = ws.photograph("small.jpg", &encode_jpeg(&small, JpegOptions::default()));
    let cut = cut_image(ImageU8::read_rgb(&path).unwrap(), TILE);
    let cache = cache_of(GIB);
    cache.request_layout(&path);
    match cache.layout(&path) {
        PeekedLayout::Known(layout) => {
            assert!(!layout.is_out_of_core());
            assert_eq!(*layout, cut.layout);
        }
        other => panic!("{other:?}"),
    }
    for key in every_key(&cut.layout) {
        let tile = cache.get_tile(&path, key, &Progress::none()).unwrap();
        assert!(same_tile(&tile, &cut.tiles[&key]), "{key:?}");
    }
    assert_eq!(cache.stats().decodes, 1);
    assert_eq!(cache.stats().files, 0);
    assert!(ws.mipmap_files().is_empty());
}

/// `request_layout` answers `Building` for an out-of-core photograph whose
/// mipmap is missing, each of its tiles `Building` to `peek_tile` (with a
/// budget; under a budget of 0 nothing is kept), and its out-of-core layout
/// once a `get_many` has made the mipmap. That layout has the levels, sizes
/// and tile grid of the photograph's in-core layout, so a panel asks for the
/// same keys whichever it holds. With a budget and with one of 0.
#[test]
fn request_layout_answers_building_until_the_mipmap_is_made() {
    let key = TileKey {
        level: 1,
        col: 0,
        row: 0,
    };
    for budget in [GIB, 0] {
        let ws = Workspace::new();
        let path = ws.photograph("a.jpg", &readable_jpeg());
        let cache = cache_of(budget);
        cache.request_layout(&path);
        assert_eq!(cache.layout(&path), PeekedLayout::Building, "{budget}");
        let building = matches!(cache.peek_tile(&path, key), PeekedTile::Building);
        assert_eq!(building, budget > 0, "{budget}");
        assert!(ws.mipmap_files().is_empty(), "{budget}: no mipmap made");

        cache
            .get_many(&[path.as_path()], &Progress::none())
            .unwrap();
        let out = match cache.layout(&path) {
            PeekedLayout::Known(layout) => layout,
            other => panic!("{budget}: {other:?}"),
        };
        assert!(out.is_out_of_core(), "{budget}");
        assert!(!matches!(cache.peek_tile(&path, key), PeekedTile::Building));
        let in_core = cache.layout_for(300, 200);
        assert_eq!(in_core.levels(), out.levels());
        for level in 0..out.levels() {
            assert_eq!(in_core.level_size(level), out.level_size(level));
            assert_eq!(in_core.tiles(level), out.tiles(level));
        }
    }
}

/// A panel's decode of an out-of-core photograph whose mipmap is missing,
/// or whose files are replaced with another record after its first tiles
/// were answered, or whose level-1 file is removed then, reads nothing in
/// core: it answers its tiles from the files while they are current, and
/// otherwise `Changed` (naming the photograph, or the file found not
/// current), decoding nothing, after which the photograph is `Building`. It
/// makes no mipmap. Under a budget too small to keep a file, and under a
/// budget of 0, where each request decides from the files as they are.
#[test]
fn a_panel_s_decode_reads_nothing_in_core_and_leaves_the_photograph_building() {
    let key = |level: u8, col: u32| TileKey { level, col, row: 0 };
    for budget in [1, 0] {
        // Missing.
        let ws = Workspace::new();
        let path = ws.photograph("a.jpg", &readable_jpeg());
        let cache = cache_of(budget);
        cache.request_layout(&path);
        let answer = cache.try_decode_tiles(&path, &[key(1, 0)], &Progress::none());
        assert_eq!(
            answer.err(),
            Some(TileError::Changed { path: path.clone() }),
            "budget {budget}"
        );
        assert_eq!(cache.layout(&path), PeekedLayout::Building);
        assert_eq!(cache.stats().decodes, 0);
        assert!(ws.mipmap_files().is_empty(), "no mipmap made");
        assert_eq!(cache.stats().mipmaps_made, 0);

        for replace in ["another record", "level 1 removed"] {
            let ws = Workspace::new();
            let path = ws.photograph("a.jpg", &readable_jpeg());
            build(&path);
            let cache = cache_of(budget);
            cache.request_layout(&path);
            let files = files_levels(&path);
            let from_files = tiles_of(&out_of_core_layout(), &files);
            let Ok(TryDecode::Decoded(tiles)) =
                cache.try_decode_tiles(&path, &[key(1, 0)], &Progress::none())
            else {
                panic!("{replace}, budget {budget}: decoded");
            };
            assert!(same_tile(&tiles[0], &from_files[&key(1, 0)]));
            match replace {
                "another record" => rebuild_with_another_record(&path),
                _ => std::fs::remove_file(level_file(&path, 1)).unwrap(),
            }
            let on_disk = ws.mipmap_files();
            let answer = cache.try_decode_tiles(&path, &[key(1, 1)], &Progress::none());
            // The rebuilt files are current, so a request that decides from
            // the files as they are reads them.
            if budget == 0 && replace == "another record" {
                let Ok(TryDecode::Decoded(tiles)) = answer else {
                    panic!("{replace}, budget {budget}: decoded");
                };
                assert!(same_tile(&tiles[0], &from_files[&key(1, 1)]));
                assert!(matches!(cache.layout(&path), PeekedLayout::Known(_)));
            } else {
                assert_eq!(
                    answer.err(),
                    Some(TileError::Changed {
                        path: level_file(&path, 1)
                    }),
                    "{replace}, budget {budget}"
                );
                assert_eq!(cache.layout(&path), PeekedLayout::Building, "{replace}");
                // Asked again, it is still `Building`, decoding nothing.
                let again = cache.try_decode_tiles(&path, &[key(1, 1)], &Progress::none());
                assert!(matches!(again, Err(TileError::Changed { .. })), "{replace}");
                assert_eq!(cache.layout(&path), PeekedLayout::Building, "{replace}");
            }
            assert_eq!(cache.stats().decodes, 0, "{replace}, {budget}");
            assert_eq!(ws.mipmap_files(), on_disk, "no mipmap made");
            assert_eq!(cache.stats().mipmaps_made, 0);
        }
    }
}
