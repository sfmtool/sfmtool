// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use sfmtool_core::camera::image::ImageU8;

use super::*;
use crate::state::PYRAMID_LEVELS;

/// A demo reconstruction whose workspace is `dir`, with a PNG written for
/// image 0 and none for image 1.
fn demo_in(dir: &Path) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(8);
    assert!(recon.image_count() > 1, "the demo has two images");
    recon.workspace_dir = dir.to_path_buf();
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name = format!("photo_{i:02}.png");
    }
    image::RgbImage::from_fn(6, 4, |x, y| image::Rgb([x as u8, y as u8, 3]))
        .save(dir.join("photo_00.png"))
        .unwrap();
    recon
}

/// Wait for the decode of `path` that `requests` started to end.
fn wait_for(requests: &PhotographRequests, path: &Path) {
    let start = Instant::now();
    while requests.is_in_flight(path) {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the decode never ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_miss_is_decoded_off_the_calling_thread_and_found_on_a_later_look() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(1 << 30, PYRAMID_LEVELS));
    let requests = PhotographRequests::default();
    let ctx = egui::Context::default();
    let path = photograph_path(&recon, 0).unwrap();

    // Asked twice, perhaps before it lands: one decode either way.
    for _ in 0..2 {
        assert!(
            !matches!(
                display_photograph(&cache, &requests, &recon, 0, &ctx),
                DisplayPhotograph::Unreadable
            ),
            "the photograph is on disk"
        );
    }
    wait_for(&requests, &path);
    match display_photograph(&cache, &requests, &recon, 0, &ctx) {
        DisplayPhotograph::Decoded(pyramid) => assert_eq!(pyramid.level(0).width(), 6),
        _ => panic!("the decode landed in the cache"),
    }
    assert_eq!(cache.stats().misses, 1, "one decode for every look");
}

#[test]
fn a_file_that_cannot_be_read_ends_as_unreadable_rather_than_decoding_forever() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(1 << 30, PYRAMID_LEVELS));
    let requests = PhotographRequests::default();
    let ctx = egui::Context::default();
    let path = photograph_path(&recon, 1).unwrap();

    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 1, &ctx),
        DisplayPhotograph::Decoding | DisplayPhotograph::Unreadable
    ));
    wait_for(&requests, &path);
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 1, &ctx),
        DisplayPhotograph::Unreadable
    ));
    assert!(!requests.is_in_flight(&path), "no second request");
    assert_eq!(cache.stats().misses, 1);

    // An index past the image table has no file to ask for.
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, recon.image_count(), &ctx),
        DisplayPhotograph::Unreadable
    ));
}

#[test]
fn a_cached_photograph_is_answered_without_a_request() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(1 << 30, PYRAMID_LEVELS));
    let path = photograph_path(&recon, 1).unwrap();
    let pyramid = Arc::new(ImageU8Pyramid::from_image(
        ImageU8::new(2, 2, 3, vec![9; 12]),
        PYRAMID_LEVELS,
    ));
    cache.insert(&path, Arc::clone(&pyramid));
    let requests = PhotographRequests::default();
    match display_photograph(&cache, &requests, &recon, 1, &egui::Context::default()) {
        DisplayPhotograph::Decoded(found) => assert!(Arc::ptr_eq(&found, &pyramid)),
        _ => panic!("the inserted pyramid"),
    }
    assert!(!requests.is_in_flight(&path));
}

/// A cache that keeps nothing would never show a background decode to the
/// next look, so the photograph is decoded where it is asked for.
#[test]
fn a_cache_that_keeps_nothing_decodes_on_the_calling_thread() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(0, PYRAMID_LEVELS));
    let requests = PhotographRequests::default();
    let ctx = egui::Context::default();
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 0, &ctx),
        DisplayPhotograph::Decoded(_)
    ));
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 1, &ctx),
        DisplayPhotograph::Unreadable
    ));
    assert!(!requests.is_in_flight(&photograph_path(&recon, 0).unwrap()));
}
