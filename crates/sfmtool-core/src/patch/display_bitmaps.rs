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

use crate::camera::PhotographCache;
use crate::geometry::RigidTransform;
use crate::patch::keypoint_subpixel::{fuse_patch_cloud_bitmaps, KeypointSubpixelParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::PatchCloud;
use crate::progress::{Cancelled, Progress};
use crate::{progress_note, SfmrReconstruction};

/// Levels in the photograph pyramids the display patch bitmaps are fused from,
/// and the level count SfM Explorer builds every photograph pyramid with.
pub const DISPLAY_PYRAMID_LEVELS: usize = 6;

/// Render `recon`'s patch bitmap column at its stored frames and keypoints,
/// moving nothing.
///
/// Two stages under `progress`: `decode photographs`, each image's photograph
/// (`workspace_dir` joined with its name) read through `photographs`
/// ([`PhotographCache::get_many`], which decodes the misses in parallel), and
/// `fuse`, the whole-cloud form of the one fuse the bench commit and
/// `--add-patch-bitmaps` use ([`fuse_patch_cloud_bitmaps`]). A photograph that cannot be read, or is not
/// the size its camera says, is left out of every patch's views rather than
/// failing the operation; a point that two readable views do not see gets a
/// zero row.
///
/// `photographs` should build pyramids of [`DISPLAY_PYRAMID_LEVELS`] levels.
/// SfM Explorer passes its own cache, so the photographs this decodes stay
/// for its panels and later operations; a caller with no cache to share, such
/// as `sfm web-export`, passes `PhotographCache::new(0, DISPLAY_PYRAMID_LEVELS)`,
/// which keeps nothing.
///
/// `Ok(None)` when `recon` has no patch frames or no inline keypoints, or when
/// not one photograph could be read, since a column of zero rows would draw
/// nothing.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled.
pub fn render_display_patch_bitmaps(
    recon: &SfmrReconstruction,
    photographs: &PhotographCache,
    progress: &Progress<'_>,
) -> Result<Option<Array4<u8>>, Cancelled> {
    if recon.keypoints_xy().is_none() {
        return Ok(None);
    }
    let Some(cloud) = PatchCloud::from_stored_frames(recon) else {
        return Ok(None);
    };
    let [decode, fuse] = progress.split([1.0, 3.0]);
    let images = &recon.image_table.images;
    let total = images.len();
    let pyramids = {
        let mut phase = decode.phase("decode photographs");
        let paths: Vec<PathBuf> = images
            .iter()
            .map(|image| recon.workspace_dir.join(&image.name))
            .collect();
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let (found, tally) = photographs.get_many(&refs, &phase)?;
        // A photograph that is not the size its camera says would be sampled
        // at the wrong pixels, so it is left out like an unreadable one.
        let pyramids: Vec<_> = found
            .into_iter()
            .zip(images)
            .map(|(pyramid, image)| {
                let camera = &recon.image_table.cameras[image.camera_index as usize];
                pyramid.filter(|pyramid| {
                    let level = pyramid.level(0);
                    level.width() == camera.width && level.height() == camera.height
                })
            })
            .collect();
        let read = pyramids.iter().filter(|p| p.is_some()).count();
        if tally.reused == 0 {
            progress_note!(phase, "{read} of {total} read");
        } else {
            progress_note!(
                phase,
                "{read} of {total} read, {} reused from the cache",
                tally.reused
            );
        }
        pyramids
    };
    if pyramids.iter().all(Option::is_none) {
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
    let views: Vec<Option<ProjectedImage<'_>>> = images
        .iter()
        .zip(&poses)
        .zip(&pyramids)
        .map(|((image, cam_from_world), pyramid)| {
            pyramid.as_ref().map(|pyramid| ProjectedImage {
                camera: &recon.image_table.cameras[image.camera_index as usize],
                cam_from_world,
                pyramid,
            })
        })
        .collect();
    let mut phase = fuse.phase("fuse");
    let column = fuse_patch_cloud_bitmaps(
        &cloud,
        recon,
        &views,
        &KeypointSubpixelParams::default(),
        None,
        &phase,
    )?;
    progress_note!(phase, "{} patches at {} px", cloud.len(), column.shape()[1]);
    Ok(Some(column))
}

#[cfg(test)]
mod tests;
