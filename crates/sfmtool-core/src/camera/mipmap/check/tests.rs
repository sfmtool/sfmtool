// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `check_mipmaps` on 300 x 200 synthetic photographs in a temporary
//! workspace whose `mipmap_options` sets a threshold of 64, with tiles of 32.

use std::path::{Path, PathBuf};

use super::*;
use crate::camera::image::ImageU8;
use crate::camera::mipmap::test_support::{encode_png, make_mipmap, synthetic_rgb, TestWorkspace};

const TILE: u32 = 32;

/// Every file under `dir`, with its bytes, sorted by path.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push((path.clone(), std::fs::read(&path).unwrap()));
            }
        }
    }
    out.sort();
    out
}

/// A 300 x 200 photograph other than `synthetic_rgb`'s.
fn other_rgb() -> ImageU8 {
    let image = synthetic_rgb(300, 200);
    let data = image.data().iter().map(|v| v ^ 0x55).collect();
    ImageU8::new(300, 200, 3, data)
}

/// Each photograph is reported in the one list its files put it in, and
/// nothing is written, not even a mipmap directory for the missing ones.
#[test]
fn check_mipmaps_reports_each_photograph_and_writes_nothing() {
    let ws = TestWorkspace::new(Some(95));
    let png = encode_png(&synthetic_rgb(300, 200));
    let current = ws.write("images/current.png", &png);
    let missing = ws.write("images/missing.png", &png);
    let stale = ws.write("images/stale.png", &png);
    let small = ws.write("images/small.png", &encode_png(&synthetic_rgb(60, 40)));
    let gone = ws.root().join("images/gone.png");
    make_mipmap(&current, TILE);
    make_mipmap(&stale, TILE);
    // Another photograph of the same size at the same path.
    std::fs::write(&stale, encode_png(&other_rgb())).unwrap();
    let before = snapshot(ws.dir.path());

    let paths = [
        current.as_path(),
        missing.as_path(),
        stale.as_path(),
        small.as_path(),
        gone.as_path(),
    ];
    let report = check_mipmaps(&paths, TILE, &Progress::none()).unwrap();

    assert_eq!(report.current, vec![current]);
    assert_eq!(report.missing, vec![missing]);
    let [(stale_path, why)] = &report.stale[..] else {
        panic!("one stale photograph: {:?}", report.stale);
    };
    assert_eq!(stale_path, &stale);
    assert!(
        why.starts_with("stale.png-mip4-q95.jpg is not current: "),
        "{why}"
    );
    assert_eq!(report.in_core, vec![small]);
    let [(failed, _)] = &report.failed[..] else {
        panic!("one failure: {:?}", report.failed);
    };
    assert_eq!(failed, &gone);
    assert_eq!(snapshot(ws.dir.path()), before);
}

/// A photograph whose mipmap was made at another quality is missing, and
/// one in a workspace without the object is in core.
#[test]
fn a_mipmap_of_another_quality_is_missing_and_no_object_is_in_core() {
    let ws = TestWorkspace::new(Some(90));
    let photograph = ws.write("a.png", &encode_png(&synthetic_rgb(300, 200)));
    make_mipmap(&photograph, TILE);
    ws.set_quality(Some(95));
    let paths = [photograph.as_path()];
    let report = check_mipmaps(&paths, TILE, &Progress::none()).unwrap();
    assert_eq!(report.missing, vec![photograph.clone()]);

    ws.set_quality(None);
    let report = check_mipmaps(&paths, TILE, &Progress::none()).unwrap();
    assert_eq!(report.in_core, vec![photograph]);
}

/// A cancelled check answers `Cancelled`.
#[test]
fn a_cancelled_check_answers_cancelled() {
    let ws = TestWorkspace::new(Some(95));
    let photograph = ws.write("a.png", &encode_png(&synthetic_rgb(300, 200)));
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&cancel);
    assert!(matches!(
        check_mipmaps(&[photograph.as_path()], TILE, &progress),
        Err(Cancelled)
    ));
}
