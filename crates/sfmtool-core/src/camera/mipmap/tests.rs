// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The mip level files on 300 x 200 synthetic photographs, in a temporary
//! workspace whose `mipmap_options` sets a threshold of 64, with tiles of 32,
//! so each photograph is out of core with levels 1 to 4 stored.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime};

use jpeg_encoder::SamplingFactor;

use super::file::encode_mip_level;
use super::test_support::{encode_jpeg, encode_png, synthetic_rgb, JpegOptions};
use super::*;
use crate::camera::image::ImageU8;
use crate::camera::photograph_cache::{GetError, PhotographCache};
use crate::progress::Progress;
use crate::workspace::WORKSPACE_FILE_NAME;

const TILE: u32 = 32;

fn options(quality: u8) -> MipmapOptions {
    MipmapOptions {
        out_of_core_long_side: 64,
        jpeg_quality: quality,
        full_resolution_jpeg_quality: quality,
    }
}

/// A temporary workspace whose file has `options`, or no `mipmap_options`.
struct Workspace {
    dir: tempfile::TempDir,
}

impl Workspace {
    fn new(options: Option<MipmapOptions>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let ws = Self { dir };
        ws.set_options(options);
        ws
    }

    fn root(&self) -> PathBuf {
        crate::workspace::find_workspace_for_path(&self.dir.path().join("x")).unwrap()
    }

    fn set_options(&self, options: Option<MipmapOptions>) {
        let object = options.map_or(String::new(), |o| {
            format!(
                r#", "mipmap_options": {{"out_of_core_long_side": {}, "jpeg_quality": {},
                    "full_resolution_jpeg_quality": {}}}"#,
                o.out_of_core_long_side, o.jpeg_quality, o.full_resolution_jpeg_quality
            )
        });
        std::fs::write(
            self.dir.path().join(WORKSPACE_FILE_NAME),
            format!(r#"{{"version": 2{object}}}"#),
        )
        .unwrap();
    }

    /// Write `bytes` at `relative` and return its path.
    fn photograph(&self, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = self.dir.path().join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn mipmaps(&self) -> PathBuf {
        self.dir.path().join(MIPMAP_DIR_NAME)
    }
}

/// A region-readable JPEG: 4:2:0 with an interval of the 19 MCUs of a row.
fn readable_jpeg() -> Vec<u8> {
    encode_jpeg(
        &synthetic_rgb(300, 200),
        JpegOptions {
            sampling: SamplingFactor::F_2_2,
            restart_interval: 19,
            ..JpegOptions::default()
        },
    )
}

fn build(photograph: &Path, options: &MipmapOptions) -> BuiltMipmap {
    build_mipmaps(photograph, None, options, TILE, &Progress::none()).unwrap()
}

/// [`check_mipmap`] of the photograph as its workspace's file sets it, with
/// the source it was checked against.
fn check_with(photograph: &Path) -> (Result<bool, MipmapError>, MipmapSource) {
    let (layout, location) = photograph_layout(photograph, TILE).unwrap();
    let location = location.expect("out of core");
    let source = MipmapSource::read(photograph, &location).unwrap();
    (check_mipmap(&source, &location, &layout), source)
}

fn check(photograph: &Path) -> Result<bool, MipmapError> {
    check_with(photograph).0
}

fn location(photograph: &Path) -> MipmapLocation {
    mipmap_location(photograph).unwrap()
}

/// Every file under `dir`, with its bytes, sorted by path.
fn files_under(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
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

/// Rewrite the mip level file at `path` with its record changed by `edit`,
/// re-encoding its own pixels, so a test can make any record on any file
/// system.
fn rewrite_record(path: &Path, edit: impl FnOnce(&mut MipLevelRecord)) {
    let bytes = std::fs::read(path).unwrap();
    let header = super::region::read_header(&bytes).unwrap();
    let segment = header.app15[0].clone();
    let mut record = MipLevelRecord::from_segment(&bytes[segment]).unwrap();
    let expected = ExpectedMipLevel {
        source: record.source.clone(),
        source_width: record.source_width,
        source_height: record.source_height,
        level: record.level as usize,
        width: record.width,
        height: record.height,
        jpeg_quality: record.jpeg_quality,
    };
    let image = MipLevelFile::open(path, &expected)
        .unwrap()
        .decode()
        .unwrap();
    edit(&mut record);
    let encoded = encode_mip_level(&image, false, record.jpeg_quality, &record).unwrap();
    std::fs::write(path, encoded).unwrap();
}

fn set_modified(path: &Path, time: SystemTime) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(time)
        .unwrap();
}

fn assert_stale(result: Result<bool, MipmapError>, contains: &str) {
    match result {
        Err(MipmapError::Stale { why, .. }) => assert!(why.contains(contains), "{why}"),
        other => panic!("expected stale ({contains}), got {other:?}"),
    }
}

#[test]
fn a_first_build_writes_the_named_levels_and_a_cachedir_tag() {
    let ws = Workspace::new(Some(options(95)));
    let photograph = ws.photograph("images/IMG_0001.JPG", &readable_jpeg());
    let built = build(&photograph, &options(95));
    assert!(built.region_readable);
    assert_eq!(built.not_written, None);
    assert_eq!(built.layout.stored_levels(), 4);

    let images = ws.root().join(MIPMAP_DIR_NAME).join("images");
    let mut names: Vec<String> = std::fs::read_dir(&images)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        (1..=4)
            .map(|l| format!("IMG_0001.JPG-mip{l}-q95.jpg"))
            .collect::<Vec<_>>()
    );
    let tag = std::fs::read_to_string(ws.mipmaps().join("CACHEDIR.TAG")).unwrap();
    assert!(tag.starts_with("Signature: 8a477f597d28d172789f06886806bc55\n"));
    assert!(
        !files_under(ws.dir.path())
            .iter()
            .any(|(p, _)| p.file_name().unwrap() == ".gitignore"),
        "sfmtool writes no .gitignore"
    );
    // The levels in memory are what the files hold, and decode to their
    // sizes.
    for level in 1..=4 {
        let file = built.level(level).unwrap();
        assert_eq!(
            std::fs::read(location(&photograph).level_path(level, 95)).unwrap(),
            file.bytes().as_ref()
        );
        let image = file.decode().unwrap();
        assert_eq!(
            (image.width(), image.height()),
            built.layout.level_size(level)
        );
        assert_eq!(file.record().components, 3);
    }
    assert!(check(&photograph).unwrap());
}

#[test]
fn originals_that_cannot_be_read_by_region_get_a_mip0() {
    let ws = Workspace::new(Some(options(95)));
    let image = synthetic_rgb(300, 200);
    let progressive = JpegOptions {
        progressive: true,
        restart_interval: 19,
        ..JpegOptions::default()
    };
    // 1040 px at 4:4:4 with one interval per row: 1040 px wide.
    let wide = encode_jpeg(
        &synthetic_rgb(1040, 16),
        JpegOptions {
            sampling: SamplingFactor::F_1_1,
            restart_interval: 130,
            ..JpegOptions::default()
        },
    );
    for (name, bytes) in [
        ("a.b.png", encode_png(&image)),
        ("progressive.jpg", encode_jpeg(&image, progressive)),
        ("no_dri.jpg", encode_jpeg(&image, JpegOptions::default())),
        ("wide.jpg", wide),
    ] {
        let photograph = ws.photograph(&format!("images/{name}"), &bytes);
        let built = build(&photograph, &options(95));
        assert!(!built.region_readable, "{name}");
        let mip0 = ws
            .mipmaps()
            .join("images")
            .join(format!("{name}-mip0-q95.jpg"));
        assert!(mip0.is_file(), "{name}");
        // `-mip0` is the original's whole decode, encoded once.
        let level0 = built.level(0).unwrap().decode().unwrap();
        let original = ImageU8::read_rgb(&photograph).unwrap();
        assert_eq!(
            (level0.width(), level0.height()),
            (original.width(), original.height())
        );
        assert!(!check(&photograph).unwrap(), "{name}");
        // A missing `-mip0` makes such a mipmap stale.
        std::fs::remove_file(&mip0).unwrap();
        assert_stale(check(&photograph), "cannot be read");
    }
    // The region-readable original has none.
    let readable = ws.photograph("images/r.jpg", &readable_jpeg());
    assert!(build(&readable, &options(95)).level(0).is_none());
    assert!(!ws.mipmaps().join("images/r.jpg-mip0-q95.jpg").exists());
}

#[test]
fn two_builds_give_the_same_bytes() {
    let ws = Workspace::new(Some(options(95)));
    let photograph = ws.photograph("images/a.png", &encode_png(&synthetic_rgb(300, 200)));
    build(&photograph, &options(95));
    let first = files_under(&ws.mipmaps());
    build(&photograph, &options(95));
    assert_eq!(files_under(&ws.mipmaps()), first);
    assert_eq!(first.len(), 6, "CACHEDIR.TAG, -mip0 and four levels");
}

#[test]
fn each_staleness_case_is_found_and_made_again() {
    let ws = Workspace::new(Some(options(95)));
    let original = readable_jpeg();
    let photograph = ws.photograph("images/IMG_0001.JPG", &original);
    let mip = |level| location(&photograph).level_path(level, 95);
    let fresh = |ws: &Workspace| {
        ws.photograph("images/IMG_0001.JPG", &original);
        build(&photograph, &options(95));
        assert!(check(&photograph).is_ok());
    };
    fresh(&ws);
    let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);

    // Another length.
    let mut longer = original.clone();
    longer.extend_from_slice(b"more");
    ws.photograph("images/IMG_0001.JPG", &longer);
    assert_stale(check(&photograph), "bytes");
    fresh(&ws);

    // Another time with an equal hash: current, hashing once for the whole
    // mipmap.
    set_modified(&photograph, t0);
    let (result, source) = check_with(&photograph);
    assert!(result.is_ok());
    assert_eq!(source.hashes(), 1);

    // Another time and another hash, of the same length.
    let mut changed = original.clone();
    let at = changed.len() - 10;
    changed[at] ^= 0x01;
    ws.photograph("images/IMG_0001.JPG", &changed);
    set_modified(&photograph, t0 + Duration::from_secs(5));
    assert_stale(check(&photograph), "bytes differ");
    fresh(&ws);

    // A record naming another photograph.
    rewrite_record(&mip(2), |r| r.source = "images/IMG_0002.JPG".to_string());
    assert_stale(check(&photograph), "made from images/IMG_0002.JPG");
    fresh(&ws);

    // A record whose quality or level disagrees with the name.
    rewrite_record(&mip(3), |r| r.jpeg_quality = 90);
    assert_stale(check(&photograph), "quality 90");
    fresh(&ws);
    rewrite_record(&mip(3), |r| r.level = 2);
    assert_stale(check(&photograph), "level 2");
    fresh(&ws);

    // Another region answer than the other levels'.
    rewrite_record(&mip(1), |r| r.region_readable_original = false);
    assert_stale(check(&photograph), "region answer");
    fresh(&ws);

    // Cut short.
    let bytes = std::fs::read(mip(1)).unwrap();
    std::fs::write(mip(1), &bytes[..bytes.len() - 40]).unwrap();
    assert_stale(check(&photograph), "cut short");
    fresh(&ws);

    // No record: a plain JPEG at the name.
    ImageU8::new(150, 100, 3, vec![128; 150 * 100 * 3])
        .write(&mip(1), 95)
        .unwrap();
    assert_stale(check(&photograph), "no record");
    fresh(&ws);

    // Missing.
    std::fs::remove_file(mip(4)).unwrap();
    assert_stale(check(&photograph), "cannot be read");
    fresh(&ws);
}

#[test]
fn a_source_differing_only_in_case_is_decided_by_the_hash() {
    let ws = Workspace::new(Some(options(95)));
    let original = readable_jpeg();
    let photograph = ws.photograph("images/IMG_0001.JPG", &original);
    let mips = |p: &Path| {
        (1..=4)
            .map(|l| location(p).level_path(l, 95))
            .collect::<Vec<_>>()
    };
    let lower_case = |r: &mut MipLevelRecord| r.source = "images/img_0001.jpg".to_string();

    // One original under two spellings: length, time and hash match.
    build(&photograph, &options(95));
    for mip in mips(&photograph) {
        rewrite_record(&mip, lower_case);
    }
    let (result, source) = check_with(&photograph);
    assert!(result.unwrap());
    assert_eq!(source.hashes(), 1, "hashed once for the whole mipmap");
    let (layout, location) = photograph_layout(&photograph, TILE).unwrap();
    check_mipmap(&source, location.as_ref().unwrap(), &layout).unwrap();
    assert_eq!(source.hashes(), 1, "and not again for the same spelling");

    // Two originals sharing a cache file, of other lengths.
    build(&photograph, &options(95));
    for mip in mips(&photograph) {
        rewrite_record(&mip, |r| {
            lower_case(r);
            r.source_size += 1;
        });
    }
    assert_stale(check(&photograph), "bytes");
    build(&photograph, &options(95));
    assert!(check(&photograph).is_ok());

    // Two originals of equal length and time and another hash.
    for mip in mips(&photograph) {
        rewrite_record(&mip, |r| {
            lower_case(r);
            r.source_xxh128 = "0".repeat(32);
        });
    }
    assert_stale(check(&photograph), "bytes differ");
    build(&photograph, &options(95));
    assert!(check(&photograph).is_ok());
}

#[test]
fn a_quality_change_makes_new_names_and_leaves_the_old_ones_current() {
    let ws = Workspace::new(Some(options(95)));
    let photograph = ws.photograph("images/a.jpg", &readable_jpeg());
    build(&photograph, &options(95));
    let at_95 = files_under(&ws.mipmaps());

    ws.set_options(Some(options(90)));
    assert_stale(check(&photograph), "cannot be read");
    build(&photograph, &options(90));
    assert!(check(&photograph).is_ok());
    assert!(location(&photograph).level_path(1, 90).is_file());

    // Back at 95: the old files are there and current, and nothing is made.
    ws.set_options(Some(options(95)));
    let before = files_under(&ws.mipmaps());
    assert!(check(&photograph).is_ok());
    assert_eq!(files_under(&ws.mipmaps()), before);
    for (path, bytes) in &at_95 {
        assert_eq!(&std::fs::read(path).unwrap(), bytes);
    }
}

#[test]
fn a_record_of_another_threshold_is_still_current() {
    let ws = Workspace::new(Some(options(95)));
    let photograph = ws.photograph("images/a.jpg", &readable_jpeg());
    build(&photograph, &options(95));
    ws.set_options(Some(MipmapOptions {
        out_of_core_long_side: 100,
        ..options(95)
    }));
    assert!(check(&photograph).unwrap());
}

#[test]
fn a_photograph_in_core_writes_nothing() {
    let jpeg = readable_jpeg();
    // A workspace without the object.
    let ws = Workspace::new(None);
    let photograph = ws.photograph("images/a.jpg", &jpeg);
    let (layout, location) = photograph_layout(&photograph, TILE).unwrap();
    assert!(!layout.is_out_of_core());
    assert_eq!(location, None);
    // A photograph with no workspace.
    let bare = tempfile::tempdir().unwrap();
    let alone = bare.path().join("a.jpg");
    std::fs::write(&alone, &jpeg).unwrap();
    let (layout, location) = photograph_layout(&alone, TILE).unwrap();
    assert!(!layout.is_out_of_core());
    assert_eq!(location, None);

    // With no workspace there is nowhere to write, whatever options a caller
    // passes.
    assert!(matches!(
        build_mipmaps(&alone, None, &options(95), TILE, &Progress::none()),
        Err(MipmapError::NotOutOfCore { .. })
    ));
    for path in [&photograph, &alone] {
        let cache = PhotographCache::new(1 << 30);
        assert!(cache.get(path).is_ok());
    }
    assert!(!ws.mipmaps().exists());
    assert_eq!(files_under(bare.path()).len(), 1);
}

/// The CRC-32 a PNG chunk ends with.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A PNG whose header says `width` x `height` and whose pixel data is a few
/// bytes of nothing, so only its header can be read.
fn header_only_png(width: u32, height: u32) -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        png.extend((data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        png.extend_from_slice(&body);
        png.extend(crc32(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend(width.to_be_bytes());
    ihdr.extend(height.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &[0, 1, 2, 3]);
    chunk(b"IEND", &[]);
    png
}

#[test]
fn a_side_of_65536_is_refused_from_the_header_in_every_workspace_kind() {
    let small = Workspace::new(Some(options(95)));
    let fits = small.photograph("images/fits.png", &header_only_png(65535, 100));
    let (layout, location) = photograph_layout(&fits, TILE).unwrap();
    assert!(layout.is_out_of_core());
    assert!(location.is_some());

    let without = Workspace::new(None);
    let bare = tempfile::tempdir().unwrap();
    let big = header_only_png(65536, 100);
    let paths = [
        small.photograph("images/big.png", &big),
        without.photograph("images/big.png", &big),
        {
            let p = bare.path().join("big.png");
            std::fs::write(&p, &big).unwrap();
            p
        },
    ];
    let cache = PhotographCache::new(1 << 30);
    for path in &paths {
        let message = format!(
            "{} is 65536 x 100, and a photograph with a side over 65535 px cannot be read",
            path.display()
        );
        assert_eq!(
            photograph_layout(path, TILE).unwrap_err().to_string(),
            message
        );
        assert_eq!(cache.get(path).err().unwrap().to_string(), message);
        match photograph_layout(path, TILE) {
            Err(MipmapError::TooLarge {
                path: named,
                width: 65536,
                height: 100,
            }) => assert_eq!(&named, path),
            other => panic!("{}: {other:?}", path.display()),
        }
        match cache.get(path) {
            Err(GetError::TooLarge {
                path: named,
                width: 65536,
                height: 100,
            }) => assert_eq!(&named, path),
            Err(other) => panic!("{}: {other}", path.display()),
            Ok(_) => panic!("{}: decoded", path.display()),
        }
    }
}

#[test]
fn a_build_answers_in_core_or_too_large_from_the_header_without_decoding() {
    let ws = Workspace::new(Some(options(95)));
    let build = |relative: &str, width: u32, height: u32| {
        let path = ws.photograph(relative, &header_only_png(width, height));
        build_mipmaps(&path, None, &options(95), TILE, &Progress::none())
    };
    // The pixel data of these files does not decode, so the first two
    // answers come from the header alone.
    assert!(matches!(
        build("images/small.png", 64, 40),
        Err(MipmapError::NotOutOfCore { .. })
    ));
    assert!(matches!(
        build("images/big.png", 65536, 100),
        Err(MipmapError::TooLarge {
            width: 65536,
            height: 100,
            ..
        })
    ));
    assert!(matches!(
        build("images/out.png", 300, 200),
        Err(MipmapError::Source { .. })
    ));
    assert!(!ws.mipmaps().exists());
}

#[test]
fn an_unwritable_mipmap_directory_gives_the_in_memory_equivalent() {
    let bytes = encode_png(&synthetic_rgb(300, 200));
    let writable = Workspace::new(Some(options(95)));
    let on_disk = writable.photograph("images/a.png", &bytes);
    let written = build(&on_disk, &options(95));

    // A file where the mipmap directory belongs, so nothing can be made in it.
    let blocked = Workspace::new(Some(options(95)));
    let in_memory = blocked.photograph("images/a.png", &bytes);
    std::fs::write(blocked.mipmaps(), b"not a directory").unwrap();
    let built = build(&in_memory, &options(95));
    assert!(built.not_written.is_some());
    assert_eq!(built.files.len(), written.files.len());
    for (a, b) in built.files.iter().zip(&written.files) {
        assert_eq!(a.record().level, b.record().level);
        assert_eq!(a.decode().unwrap().data(), b.decode().unwrap().data());
    }
    assert!(blocked.mipmaps().is_file());
}

#[test]
fn a_cancelled_build_writes_nothing() {
    let ws = Workspace::new(Some(options(95)));
    let photograph = ws.photograph("images/a.jpg", &readable_jpeg());
    let flag = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    assert!(matches!(
        build_mipmaps(&photograph, None, &options(95), TILE, &progress),
        Err(MipmapError::Cancelled)
    ));
    assert!(!ws.mipmaps().exists());
}

#[test]
fn a_grey_photograph_has_grey_mip_level_files() {
    let ws = Workspace::new(Some(options(95)));
    let grey = ImageU8::new(300, 200, 3, {
        let rgb = synthetic_rgb(300, 200);
        rgb.data()
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[1]; 3])
            .collect()
    });
    let photograph = ws.photograph("images/g.png", &encode_png(&grey));
    let built = build(&photograph, &options(95));
    assert!(built.files.iter().all(|f| f.record().components == 1));
    let level0 = built.level(0).unwrap().decode().unwrap();
    assert!(level0
        .data()
        .as_chunks::<3>()
        .0
        .iter()
        .all(|&[r, g, b]| r == g && g == b));
    assert!(check(&photograph).is_ok());
}

#[test]
fn decoding_bytes_equals_reading_the_file() {
    let dir = tempfile::tempdir().unwrap();
    for (name, bytes) in [
        ("a.jpg", readable_jpeg()),
        ("a.png", encode_png(&synthetic_rgb(300, 200))),
    ] {
        let path = dir.path().join(name);
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(
            ImageU8::decode_rgb(&bytes).unwrap().data(),
            ImageU8::read_rgb(&path).unwrap().data()
        );
    }
}

#[test]
fn two_threads_building_at_once_leave_current_files() {
    let ws = Workspace::new(Some(options(95)));
    let photograph = ws.photograph("images/a.png", &encode_png(&synthetic_rgb(300, 200)));
    let builds: Vec<BuiltMipmap> = std::thread::scope(|s| {
        let a = s.spawn(|| build(&photograph, &options(95)));
        let b = s.spawn(|| build(&photograph, &options(95)));
        vec![a.join().unwrap(), b.join().unwrap()]
    });
    for built in &builds {
        assert_eq!(built.not_written, None);
    }
    assert!(!check(&photograph).unwrap());
    for file in &builds[0].files {
        let level = file.record().level as usize;
        assert_eq!(
            std::fs::read(location(&photograph).level_path(level, 95)).unwrap(),
            file.bytes().as_ref()
        );
    }
    // No temporary file is left behind.
    assert!(files_under(&ws.mipmaps())
        .iter()
        .all(|(p, _)| !p.to_string_lossy().ends_with(".tmp")));
}

/// The environment variable that makes [`child_build`] build the photograph
/// it names; unset, that test does nothing.
const CHILD_BUILD_VAR: &str = "SFMTOOL_TEST_MIPMAP_CHILD_BUILD";

/// Run by [`two_processes_building_at_once_leave_current_files`] in child
/// processes of the test binary; it is ignored in a normal run.
#[test]
#[ignore = "run as a child process by two_processes_building_at_once_leave_current_files"]
fn child_build() {
    let Some(photograph) = std::env::var_os(CHILD_BUILD_VAR) else {
        return;
    };
    let photograph = PathBuf::from(photograph);
    let (layout, _) = photograph_layout(&photograph, TILE).unwrap();
    let options = *layout.options().unwrap();
    for _ in 0..3 {
        let built = build(&photograph, &options);
        assert_eq!(built.not_written, None);
    }
}

#[test]
fn two_processes_building_at_once_leave_current_files() {
    let ws = Workspace::new(Some(options(95)));
    let photograph = ws.photograph("images/a.png", &encode_png(&synthetic_rgb(300, 200)));
    let built = build(&photograph, &options(95));
    // Hold a file open while the children rename over it: a reader keeps an
    // open file, and that must not block another process's write. The held
    // file is made stale first, so a rename it blocked would not count as
    // written and the children would report it in `not_written`.
    let held_path = location(&photograph).level_path(1, 95);
    rewrite_record(&held_path, |r| r.source_xxh128 = "0".repeat(32));
    let held = std::fs::File::open(&held_path).unwrap();
    let exe = std::env::current_exe().unwrap();
    let children: Vec<_> = (0..2)
        .map(|_| {
            std::process::Command::new(&exe)
                .args(["camera::mipmap::tests::child_build", "--exact", "--ignored"])
                .env(CHILD_BUILD_VAR, &photograph)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    for child in children {
        let out = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    drop(held);
    assert!(!check(&photograph).unwrap());
    for file in &built.files {
        let level = file.record().level as usize;
        assert_eq!(
            std::fs::read(location(&photograph).level_path(level, 95)).unwrap(),
            file.bytes().as_ref()
        );
    }
}
