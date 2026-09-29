// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use crate::camera::PhotographCache;
use crate::progress::Progress;
use crate::SfmrReconstruction;

use super::{render_display_patch_bitmaps, DISPLAY_PYRAMID_LEVELS};

/// The Seoul bull ground truth: patch frames and inline keypoints, no bitmap
/// column, and its 17 photographs checked in beside it.
fn seoul_bull() -> SfmrReconstruction {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr");
    SfmrReconstruction::load(&path, &Progress::none()).expect("the ground truth loads")
}

/// A render reads every photograph through the cache it is given, and a second
/// render through the same cache decodes nothing and draws the same column. A
/// cache that keeps nothing draws it too.
#[test]
fn a_second_render_reads_its_photographs_from_the_cache() {
    let recon = seoul_bull();
    assert!(recon.point_set.patch_bitmaps_y_x_rgba.is_none());
    let n = recon.image_count();
    let cache = PhotographCache::new(1 << 30, DISPLAY_PYRAMID_LEVELS);

    let first = render_display_patch_bitmaps(&recon, &cache, &Progress::none())
        .expect("not cancelled")
        .expect("the photographs are readable");
    assert_eq!(first.shape()[0], recon.point_count());
    let decoded = cache.stats();
    assert_eq!(decoded.entries, n, "every photograph stays decoded");
    assert_eq!(decoded.misses as usize, n);

    let second = render_display_patch_bitmaps(&recon, &cache, &Progress::none())
        .expect("not cancelled")
        .expect("the photographs are readable");
    assert_eq!(
        cache.stats().misses,
        decoded.misses,
        "the second render decoded nothing"
    );
    assert_eq!(first, second, "the same pixels, the same column");

    let uncached = PhotographCache::new(0, DISPLAY_PYRAMID_LEVELS);
    let third = render_display_patch_bitmaps(&recon, &uncached, &Progress::none())
        .expect("not cancelled")
        .expect("the photographs are readable");
    assert_eq!(
        first, third,
        "a cache that keeps nothing draws the same column"
    );
    assert_eq!(uncached.stats().entries, 0);
}

/// With no photograph readable there is nothing to draw, and a photograph that
/// is not the size its camera says counts as unreadable.
#[test]
fn a_render_with_no_usable_photograph_draws_nothing() {
    let mut recon = seoul_bull();
    recon.workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("no-such-workspace");
    let cache = PhotographCache::new(1 << 30, DISPLAY_PYRAMID_LEVELS);
    let rendered =
        render_display_patch_bitmaps(&recon, &cache, &Progress::none()).expect("not cancelled");
    assert!(rendered.is_none(), "no photograph was readable");

    let mut resized = seoul_bull();
    for camera in &mut resized.image_table.cameras {
        camera.width += 1;
    }
    let rendered =
        render_display_patch_bitmaps(&resized, &cache, &Progress::none()).expect("not cancelled");
    assert!(
        rendered.is_none(),
        "photographs of the wrong size were sampled"
    );
}
