// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Display patch bitmaps: the bitmap column a reconstruction with patch frames
//! and no bitmaps is drawn with, rendered from its photographs.
//!
//! SfM Explorer's open renders this column for such a file and marks it
//! [`PointSet::patch_bitmaps_for_display`](crate::PointSet::patch_bitmaps_for_display);
//! `sfm web-export` renders it to fill its patch atlas. Both call
//! [`render_display_patch_bitmaps`], so the two draw the same patches.

use std::path::{Path, PathBuf};

use ndarray::Array4;

use crate::camera::sampler::RegionPool;
use crate::camera::{PhotographCache, PhotographReadError, TiledPhotograph};
use crate::geometry::RigidTransform;
use crate::patch::keypoint_subpixel::KeypointSubpixelParams;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::stored_bitmap::{
    render_patch_cloud_bitmaps, CloudRenderError, PatchBitmapColumn, UnreferencedPoints,
};
use crate::patch::PatchCloud;
use crate::progress::{Cancelled, Progress};
use crate::{progress_note, SfmrReconstruction};

/// Render `recon`'s patch bitmap column at its stored frames and keypoints,
/// moving nothing.
///
/// Two stages under `progress`: `decode photographs`, each image's photograph
/// (`workspace_dir` joined with its name) read through `photographs`
/// ([`PhotographCache::get_many`], which decodes the misses in parallel), and
/// `render`, the whole-cloud form of the one render of the stored bitmap the
/// bench commit and `--add-patch-bitmaps` use ([`render_patch_cloud_bitmaps`]):
/// each point's tile from the reference observation `recon` stores for it, or,
/// for a point at `-1`, from the observation the reference-view rule picks. A
/// photograph that cannot be read, is not the size its camera says, or whose
/// pyramid in the cache does not hold every level whole (one made with
/// [`ImageU8Pyramid::from_regions`](crate::camera::image::ImageU8Pyramid::from_regions)
/// and inserted by a caller), is left out of every patch's views rather than
/// failing the operation; a point whose
/// reference observation's photograph is left out, or a point at `-1` that two
/// readable views do not see, gets a zero row.
///
/// SfM Explorer passes its own cache, so the photographs this decodes stay
/// for its panels and later operations; a caller with no cache to share, such
/// as `sfm web-export`, passes `PhotographCache::new(0)`, which keeps nothing.
/// The render reads the photographs through `pool`, the region pool the
/// caller made for this call. Every photograph it reads holds each of its
/// levels whole, so no read of its pixels fails.
///
/// `Ok(None)` when `recon` has no patch frames it can read (none, or columns
/// that are not one row per point, which
/// [`PatchCloud::from_stored_patch_frames`] refuses) or no inline keypoints, or when
/// not one photograph could be read, since a column of zero rows would draw
/// nothing.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled.
pub fn render_display_patch_bitmaps(
    recon: &SfmrReconstruction,
    photographs: &PhotographCache,
    pool: &RegionPool,
    progress: &Progress<'_>,
) -> Result<Option<Array4<u8>>, Cancelled> {
    // A caller that only draws the column has no use for the references; the
    // ones it read from `recon` it already holds.
    Ok(
        render_patch_bitmap_column(recon, photographs, pool, progress)?
            .map(|column| column.bitmaps),
    )
}

/// [`render_display_patch_bitmaps`] with the reference observation each row's
/// render came from: `recon`'s stored reference where it is `≥ 0`, the rule's
/// pick (or `-1`) where it is `-1`. SfM Explorer's conversion to embedded
/// patches stores the column as the reconstruction's own and records the
/// references with it; its open keeps the column for display and marks which
/// references only the display render picked.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled.
pub fn render_patch_bitmap_column(
    recon: &SfmrReconstruction,
    photographs: &PhotographCache,
    pool: &RegionPool,
    progress: &Progress<'_>,
) -> Result<Option<PatchBitmapColumn>, Cancelled> {
    if recon.keypoints_xy().is_none() {
        return Ok(None);
    }
    let Ok(Some(cloud)) = PatchCloud::from_stored_patch_frames(recon) else {
        return Ok(None);
    };
    let [decode, render] = progress.split([1.0, 3.0]);
    let images = &recon.image_set.images;
    let total = images.len();
    // Each image is sampled through its intrinsics: its camera's with its
    // adjustment applied, which keeps the camera's size.
    let intrinsics = recon.image_set.all_image_intrinsics();
    let sources = {
        let mut phase = decode.phase("decode photographs");
        let paths: Vec<PathBuf> = images
            .iter()
            .map(|image| recon.workspace_dir.join(&image.name))
            .collect();
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let (found, tally) = photographs.get_many(&refs, &phase)?;
        // A photograph that is not the size its camera says would be sampled
        // at the wrong pixels, so it is left out like an unreadable one, and
        // so is one whose pyramid does not hold every level whole, which a
        // read could ask for pixels it lacks. The rest are read through their
        // whole pyramids until step 11 of
        // `specs/drafts/out-of-core-photograph-pyramid.md` fetches tiles.
        let sources: Vec<Option<TiledPhotograph>> = found
            .into_iter()
            .zip(&intrinsics)
            .map(|(pyramid, camera)| {
                pyramid
                    .filter(|pyramid| {
                        let level = pyramid.level(0);
                        level.width() == camera.width && level.height() == camera.height
                    })
                    .map(TiledPhotograph::from_pyramid)
                    .filter(holds_every_level)
            })
            .collect();
        let read = sources.iter().filter(|p| p.is_some()).count();
        if tally.reused == 0 {
            progress_note!(phase, "{read} of {total} read");
        } else {
            progress_note!(
                phase,
                "{read} of {total} read, {} reused from the cache",
                tally.reused
            );
        }
        sources
    };
    if sources.iter().all(Option::is_none) {
        return Ok(None);
    }
    let poses: Vec<RigidTransform> = images
        .iter()
        .map(|image| {
            let q = image.quaternion_wxyz;
            RigidTransform::from_wxyz_translation(
                [q.w, q.i, q.j, q.k],
                [
                    image.translation_xyz.x,
                    image.translation_xyz.y,
                    image.translation_xyz.z,
                ],
            )
        })
        .collect();
    let views: Vec<Option<ProjectedImage<'_>>> = intrinsics
        .iter()
        .zip(&poses)
        .zip(&sources)
        .map(|((camera, cam_from_world), source)| {
            source.as_ref().map(|source| ProjectedImage {
                camera,
                cam_from_world,
                source,
            })
        })
        .collect();
    let mut phase = render.phase("render");
    let column = render_patch_cloud_bitmaps(
        &cloud,
        recon,
        &views,
        pool,
        &KeypointSubpixelParams::default(),
        UnreferencedPoints::Pick,
        None,
        &phase,
    )
    .map_err(|e| match e {
        CloudRenderError::Cancelled
        | CloudRenderError::Photograph(PhotographReadError::Cancelled) => Cancelled,
        // The cloud, the keypoints checked above and one view per image all
        // come from `recon`, so they agree with it, and every photograph is
        // a whole pyramid, which holds every pixel a read asks for.
        e => unreachable!("a display render of a reconstruction's own frames: {e}"),
    })?;
    progress_note!(
        phase,
        "{} patches at {} px",
        cloud.len(),
        column.bitmaps.shape()[1]
    );
    Ok(Some(column))
}

/// Whether `source` holds each level of its layout whole, so that no read of
/// its pixels can ask for one it lacks.
fn holds_every_level(source: &TiledPhotograph) -> bool {
    let pyramid = source.held_pyramid();
    let layout = source.layout();
    (0..layout.levels()).all(|l| {
        let (w, h) = layout.level_size(l);
        let image = pyramid.level(l);
        pyramid.level_origin(l) == [0, 0] && image.width() == w && image.height() == h
    })
}

#[cfg(test)]
mod tests;
