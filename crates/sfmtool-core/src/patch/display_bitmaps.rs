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
use crate::progress::Progress;
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
/// photograph that cannot be read, or is not the size its camera says, is left
/// out of every patch's views rather than failing the operation; a point whose
/// reference observation's photograph is left out, or a point at `-1` that two
/// readable views do not see, gets a zero row. A read of a photograph that
/// was handed back but lacks the pixels the read asks for fails the call.
///
/// SfM Explorer passes its own cache, so the photographs this decodes stay
/// for its panels and later operations; a caller with no cache to share, such
/// as `sfm web-export`, passes a cache whose `budget_bytes` is 0, which keeps
/// nothing.
/// The render reads the photographs through `pool`, the region pool the
/// caller made for this call, and a third stage, `read regions`, notes the
/// pool's counts: the region pyramids the reads assembled and reused, and the
/// reads that took the tile path.
///
/// `Ok(None)` when `recon` has no patch frames it can read (none, or columns
/// that are not one row per point, which
/// [`PatchCloud::from_stored_patch_frames`] refuses) or no inline keypoints, or when
/// not one photograph could be read, since a column of zero rows would draw
/// nothing.
///
/// # Errors
///
/// [`PhotographReadError::Cancelled`] when `progress` was cancelled, and the
/// first read of a photograph that failed.
pub fn render_display_patch_bitmaps(
    recon: &SfmrReconstruction,
    photographs: &PhotographCache,
    pool: &RegionPool,
    progress: &Progress<'_>,
) -> Result<Option<Array4<u8>>, PhotographReadError> {
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
/// As [`render_display_patch_bitmaps`].
pub fn render_patch_bitmap_column(
    recon: &SfmrReconstruction,
    photographs: &PhotographCache,
    pool: &RegionPool,
    progress: &Progress<'_>,
) -> Result<Option<PatchBitmapColumn>, PhotographReadError> {
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
        // at the wrong pixels, so it is left out like an unreadable one. The
        // size is the layout's, which a read's pixels are checked against.
        let sources: Vec<Option<TiledPhotograph>> = found
            .into_iter()
            .zip(&intrinsics)
            .map(|(pyramid, camera)| {
                pyramid
                    .map(TiledPhotograph::from_pyramid)
                    .filter(|source| source.layout().level_size(0) == (camera.width, camera.height))
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
    let rendered = render_patch_cloud_bitmaps(
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
        CloudRenderError::Cancelled => PhotographReadError::Cancelled,
        CloudRenderError::Photograph(e) => e,
        // The cloud, the keypoints checked above and one view per image all
        // come from `recon`, so they agree with it.
        e => unreachable!("a display render of a reconstruction's own frames: {e}"),
    });
    let column = match rendered {
        Ok(column) => column,
        Err(e) => {
            drop(phase);
            note_region_reads(&render, pool);
            return Err(e);
        }
    };
    progress_note!(
        phase,
        "{} patches at {} px",
        cloud.len(),
        column.bitmaps.shape()[1]
    );
    drop(phase);
    note_region_reads(&render, pool);
    Ok(Some(column))
}

/// The `read regions` stage after the render: `pool`'s counts of the region
/// pyramids the reads assembled and reused and of the reads that took the
/// tile path.
fn note_region_reads(progress: &Progress<'_>, pool: &RegionPool) {
    let mut phase = progress.phase("read regions");
    progress_note!(phase, "{}", pool.stats());
}

#[cfg(test)]
mod tests;
