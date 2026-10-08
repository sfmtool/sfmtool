// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use crate::camera::PhotographCache;
use crate::progress::Progress;
use crate::SfmrReconstruction;

use super::{render_display_patch_bitmaps, render_patch_bitmap_column, DISPLAY_PYRAMID_LEVELS};

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

/// The column form draws the same bitmaps and names, for each row that is a
/// reference view's render, the observation in that point's track whose
/// image it was rendered from; the ground truth's well-seen points have one.
#[test]
fn the_column_names_each_rows_reference_observation() {
    let recon = seoul_bull();
    let cache = PhotographCache::new(1 << 30, DISPLAY_PYRAMID_LEVELS);
    let column = render_patch_bitmap_column(&recon, &cache, &Progress::none())
        .expect("not cancelled")
        .expect("the photographs are readable");
    let display = render_display_patch_bitmaps(&recon, &cache, &Progress::none())
        .expect("not cancelled")
        .expect("the photographs are readable");
    assert_eq!(column.bitmaps, display);
    assert_eq!(column.reference_observations.len(), recon.point_count());
    let counts = &recon.point_set.observation_counts;
    let mut picked = 0;
    for (point, &k) in column.reference_observations.iter().enumerate() {
        if k >= 0 {
            assert!((k as u32) < counts[point], "point {point}: {k}");
            picked += 1;
        }
    }
    assert!(
        picked * 2 > recon.point_count(),
        "most points pick a reference view, {picked} of {}",
        recon.point_count()
    );
}

/// With some images' views missing, each point's reference still names the
/// observation within its own track, not its place among the views that were
/// to hand: the observation a render over just those views picks.
#[test]
fn a_missing_view_leaves_each_reference_naming_its_own_observation() {
    use crate::geometry::RigidTransform;
    use crate::patch::keypoint_subpixel::KeypointSubpixelParams;
    use crate::patch::normal_refine::ProjectedImage;
    use crate::patch::stored_bitmap::{
        render_patch_bitmap, render_patch_cloud_bitmaps, UnreferencedPoints,
    };
    use crate::patch::PatchCloud;

    let recon = seoul_bull();
    let images = &recon.image_table.images;
    let paths: Vec<PathBuf> = images
        .iter()
        .map(|i| recon.workspace_dir.join(&i.name))
        .collect();
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let cache = PhotographCache::new(1 << 30, DISPLAY_PYRAMID_LEVELS);
    let (found, _) = cache
        .get_many(&refs, &Progress::none())
        .expect("not cancelled");
    let pyramids: Vec<_> = found
        .into_iter()
        .map(|p| p.expect("every photograph is checked in"))
        .collect();
    let poses: Vec<RigidTransform> = images
        .iter()
        .map(|image| {
            let q = image.quaternion_wxyz;
            let t = image.translation_xyz;
            RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
        })
        .collect();
    let dropped = [0usize, 3, 7];
    let views: Vec<Option<ProjectedImage<'_>>> = (0..images.len())
        .map(|i| {
            (!dropped.contains(&i)).then(|| ProjectedImage {
                camera: &recon.image_table.cameras[images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
        })
        .collect();
    let present: Vec<ProjectedImage<'_>> = views.iter().flatten().copied().collect();
    let mut slot = vec![None; views.len()];
    let mut next = 0u32;
    for (i, view) in views.iter().enumerate() {
        if view.is_some() {
            slot[i] = Some(next);
            next += 1;
        }
    }

    let cloud = PatchCloud::from_stored_frames(&recon).expect("the file has patch frames");
    let params = KeypointSubpixelParams::default();
    let column = render_patch_cloud_bitmaps(
        &cloud,
        &recon,
        &views,
        &params,
        UnreferencedPoints::Pick,
        None,
        &Progress::none(),
    )
    .expect("not cancelled");

    let offsets = &recon.point_set.observation_offsets;
    let tracks = &recon.point_set.tracks;
    let keypoints_xy = recon.keypoints_xy().expect("inline keypoints");
    let mut checked = 0;
    for (patch, &pid) in cloud.patches.iter().zip(&cloud.point_indexes) {
        let p = pid as usize;
        let range = offsets[p]..offsets[p + 1];
        // Only a track with a missing view before some kept one tells the
        // two indexings apart.
        if !range
            .clone()
            .any(|j| dropped.contains(&(tracks[j].image_index as usize)))
        {
            continue;
        }
        let kept: Vec<usize> = range
            .filter(|&j| slot[tracks[j].image_index as usize].is_some())
            .collect();
        let view_set: Vec<u32> = kept
            .iter()
            .map(|&j| slot[tracks[j].image_index as usize].unwrap())
            .collect();
        let keypoints: Vec<[f64; 2]> = kept
            .iter()
            .map(|&j| {
                [
                    f64::from(keypoints_xy[[j, 0]]),
                    f64::from(keypoints_xy[[j, 1]]),
                ]
            })
            .collect();
        let rendered = render_patch_bitmap(
            patch,
            &present,
            &view_set,
            &keypoints,
            &params,
            &Progress::none(),
        );
        let want = rendered
            .and_then(|b| b.reference)
            .map_or(-1, |r| (kept[r] - offsets[p]) as i32);
        assert_eq!(column.reference_observations[p], want, "point {p}");
        if want >= 0 && kept[0] > offsets[p] {
            checked += 1;
        }
        if checked >= 20 {
            break;
        }
    }
    assert!(checked > 0, "no track had a missing view before its pick");
}

/// The render reads the reconstruction's references: a point with a stored
/// reference is rendered from that observation and keeps it, so storing the
/// rule's picks and rendering again draws the same column, and storing other
/// observations draws their tiles under those references. Only a point at
/// `-1` gets the rule's pick.
#[test]
fn a_stored_reference_is_rendered_from_and_kept() {
    let mut recon = seoul_bull();
    let count = recon.point_count();
    recon.point_set.reference_observations =
        Some(vec![sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION; count]);
    let cache = PhotographCache::new(1 << 30, DISPLAY_PYRAMID_LEVELS);
    let render = |recon: &SfmrReconstruction| {
        render_patch_bitmap_column(recon, &cache, &Progress::none())
            .expect("not cancelled")
            .expect("the photographs are readable")
    };
    let picked = render(&recon);

    // The rule's picks, stored and rendered again: the same column.
    recon.point_set.reference_observations = Some(picked.reference_observations.clone());
    let again = render(&recon);
    assert_eq!(again.reference_observations, picked.reference_observations);
    assert_eq!(again.bitmaps, picked.bitmaps);

    // Another observation for every point that picked one, and -1 for half
    // of the rest of the points with a pick: the stored ones come back as
    // they are, the -1 ones come back with the rule's pick.
    let counts = recon.point_set.observation_counts.clone();
    let mut stored = picked.reference_observations.clone();
    let mut moved = 0;
    for (p, r) in stored.iter_mut().enumerate() {
        if *r < 0 {
            continue;
        }
        if p % 2 == 0 {
            *r = NO_REFERENCE;
        } else {
            *r = (*r + 1) % counts[p] as i32;
            moved += 1;
        }
    }
    assert!(moved > 0);
    recon.point_set.reference_observations = Some(stored.clone());
    let shifted = render(&recon);
    let row_len = shifted.bitmaps.len() / count;
    let flat_shifted = shifted.bitmaps.as_slice().unwrap();
    let flat_picked = picked.bitmaps.as_slice().unwrap();
    let mut differs = 0;
    for (p, &want) in stored.iter().enumerate() {
        let rows = p * row_len..(p + 1) * row_len;
        if want >= 0 {
            assert_eq!(shifted.reference_observations[p], want, "point {p}");
            differs += usize::from(flat_shifted[rows.clone()] != flat_picked[rows]);
        } else {
            assert_eq!(
                shifted.reference_observations[p], picked.reference_observations[p],
                "point {p}"
            );
            assert_eq!(flat_shifted[rows.clone()], flat_picked[rows], "point {p}");
        }
    }
    assert!(
        differs * 2 > moved,
        "most moved references draw another tile: {differs} of {moved}"
    );
}

const NO_REFERENCE: i32 = sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION;

/// The column form of [`render_patch_cloud_bitmaps`] with references stored,
/// and with photographs missing. A point whose stored reference is in a
/// missing image gets a zero row and keeps that reference rather than a render
/// of another observation; with [`UnreferencedPoints::Skip`] a point at `-1`
/// gets a zero row and stays `-1`; a point with a zero patch frame has no
/// patch in the cloud, gets a zero row, and keeps its reference.
///
/// [`render_patch_cloud_bitmaps`]: crate::patch::stored_bitmap::render_patch_cloud_bitmaps
/// [`UnreferencedPoints::Skip`]: crate::patch::stored_bitmap::UnreferencedPoints::Skip
#[test]
fn stored_references_survive_missing_views_skipped_points_and_zero_frames() {
    use crate::geometry::RigidTransform;
    use crate::patch::keypoint_subpixel::KeypointSubpixelParams;
    use crate::patch::normal_refine::ProjectedImage;
    use crate::patch::stored_bitmap::{render_patch_cloud_bitmaps, UnreferencedPoints};
    use crate::patch::PatchCloud;
    use sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION;

    let mut recon = seoul_bull();
    let count = recon.point_count();
    let images = &recon.image_table.images;
    let paths: Vec<PathBuf> = images
        .iter()
        .map(|i| recon.workspace_dir.join(&i.name))
        .collect();
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let cache = PhotographCache::new(1 << 30, DISPLAY_PYRAMID_LEVELS);
    let (found, _) = cache
        .get_many(&refs, &Progress::none())
        .expect("not cancelled");
    let pyramids: Vec<_> = found
        .into_iter()
        .map(|p| p.expect("every photograph is checked in"))
        .collect();
    let poses: Vec<RigidTransform> = images
        .iter()
        .map(|image| {
            let q = image.quaternion_wxyz;
            let t = image.translation_xyz;
            RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
        })
        .collect();

    // Every even point names its first observation; every odd point is -1.
    let stored: Vec<i32> = (0..count)
        .map(|p| {
            if p % 2 == 0 {
                0
            } else {
                NO_REFERENCE_OBSERVATION
            }
        })
        .collect();
    recon.point_set.reference_observations = Some(stored.clone());
    // Point 0 loses its patch frame.
    for half in [
        recon.point_set.patch_u_halfvec_xyz.as_mut(),
        recon.point_set.patch_v_halfvec_xyz.as_mut(),
    ] {
        half.expect("the file has patch frames")
            .row_mut(0)
            .fill(0.0);
    }
    let cloud = PatchCloud::from_stored_frames(&recon).expect("the file has patch frames");
    assert!(
        !cloud.point_indexes.contains(&0),
        "a zero frame is no patch"
    );

    let dropped = [0usize, 3, 7];
    let images = &recon.image_table.images;
    let views: Vec<Option<ProjectedImage<'_>>> = (0..images.len())
        .map(|i| {
            (!dropped.contains(&i)).then(|| ProjectedImage {
                camera: &recon.image_table.cameras[images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
        })
        .collect();
    let column = render_patch_cloud_bitmaps(
        &cloud,
        &recon,
        &views,
        &KeypointSubpixelParams::default(),
        UnreferencedPoints::Skip,
        None,
        &Progress::none(),
    )
    .expect("not cancelled");

    assert_eq!(
        column.reference_observations, stored,
        "every reference kept"
    );
    let tracks = &recon.point_set.tracks;
    let offsets = &recon.point_set.observation_offsets;
    let blank = |p: usize| {
        column
            .bitmaps
            .index_axis(ndarray::Axis(0), p)
            .iter()
            .all(|&v| v == 0)
    };
    assert!(blank(0), "a point with no patch has no bitmap");
    let (mut missing, mut rendered) = (0, 0);
    for &pid in &cloud.point_indexes {
        let p = pid as usize;
        if p % 2 == 1 {
            assert!(blank(p), "a skipped point {p} has no bitmap");
        } else if dropped.contains(&(tracks[offsets[p]].image_index as usize)) {
            assert!(blank(p), "point {p}'s reference photograph is missing");
            missing += 1;
        } else if !blank(p) {
            rendered += 1;
        }
    }
    assert!(
        missing > 0 && rendered > 0,
        "{missing} missing, {rendered} rendered"
    );
}
