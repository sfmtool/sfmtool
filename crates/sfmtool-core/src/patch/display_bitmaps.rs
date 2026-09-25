// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Display patch bitmaps: the bitmap column a reconstruction with patch frames
//! and no bitmaps is drawn with, rendered from its photographs.
//!
//! SfM Explorer's open renders this column for such a file and marks it
//! [`PointSet::patch_bitmaps_for_display`](crate::PointSet::patch_bitmaps_for_display);
//! `sfm web-export` renders it to fill its patch atlas. Both call
//! [`render_display_patch_bitmaps`], so the two draw the same patches.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array4;
use rayon::prelude::*;

use crate::camera::remap::{ImageU8, ImageU8Pyramid};
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
/// (`workspace_dir` joined with its name) read and built into a pyramid of
/// [`DISPLAY_PYRAMID_LEVELS`] levels in parallel, and `fuse`, the whole-cloud form of the
/// one fuse the bench commit and `--add-patch-bitmaps` use
/// ([`fuse_patch_cloud_bitmaps`]). A photograph that cannot be read, or is not
/// the size its camera says, is left out of every patch's views rather than
/// failing the operation; a point that two readable views do not see gets a
/// zero row.
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
    let pyramids: Vec<Option<ImageU8Pyramid>> = {
        let mut phase = decode.phase("decode photographs");
        let landed = AtomicUsize::new(0);
        let pyramids: Vec<Option<ImageU8Pyramid>> = images
            .par_iter()
            .map(|image| {
                if phase.is_cancelled() {
                    return None;
                }
                let camera = &recon.image_table.cameras[image.camera_index as usize];
                let decoded = ImageU8::read_rgb(&recon.workspace_dir.join(&image.name))
                    .ok()
                    .filter(|decoded| {
                        decoded.width() == camera.width && decoded.height() == camera.height
                    })
                    .map(|decoded| ImageU8Pyramid::from_image(decoded, DISPLAY_PYRAMID_LEVELS));
                let n = landed.fetch_add(1, Ordering::Relaxed) + 1;
                phase.count(n as u64, Some(total as u64), "images");
                decoded
            })
            .collect();
        phase.check_cancel()?;
        let read = pyramids.iter().filter(|p| p.is_some()).count();
        progress_note!(phase, "{read} of {total} read");
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
