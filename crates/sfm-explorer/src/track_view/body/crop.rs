// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The crop each row draws beside its tile, read from the row's track: the
//! values the shared crop functions ([`crate::patch_pictures::crop`]) take,
//! taken off the track and one of its observations.
//!
//! At the track stage the outline is the track's patch frame re-anchored where
//! the observation sits ([`crate::bench::observation_site`]), with the track's
//! point projected, or before it is triangulated its patch's centre. At the
//! cluster stage it is the observation's own parallelogram at the cluster's
//! radius.

use sfmtool_core::bench::{EditableTrack, Stage};
use sfmtool_core::camera::image::ImageU8Pyramid;
use sfmtool_core::SfmrReconstruction;

use crate::bench::geometry;
use crate::patch_pictures::crop::{self, CropPicture, Outline};
use crate::patch_pictures::marks::ProjectionOf;

/// Where the patch of `track` lands in the photograph of `observation`, or
/// `None` when there is nothing to outline: no patch yet at the track stage,
/// nothing saying where the observation sits, or no shape at the cluster
/// stage.
pub(super) fn outline(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
) -> Option<Outline> {
    let row = track.observations.get(observation)?;
    let site = crate::bench::observation_site(row)?;
    match &track.stage {
        Stage::Track(payload) => {
            let patch = payload.patch_frame.as_ref()?;
            let (camera, pose) = geometry::view_of(&recon.image_set, row.image as usize)?;
            // What the row's *Proj. err* cell measures to, as the tile's hover
            // view projects it.
            let (target, projection_of) = match payload.position {
                Some(position) => (position, ProjectionOf::Point),
                None => (patch.center, ProjectionOf::PatchCentre),
            };
            Some(crop::patch_outline(
                patch,
                &camera,
                &pose,
                site.pixel,
                target,
                projection_of,
            ))
        }
        Stage::Cluster(payload) => Some(crop::cluster_outline(
            site.pixel,
            site.shape?,
            payload.radius,
        )),
    }
}

/// The crop one row draws, or `None` when [`outline()`] has nothing to
/// outline or no sample of it landed.
pub(super) fn image(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
) -> Option<CropPicture> {
    crop::image(&outline(recon, track, observation)?, src)
}

/// One row's crop in context, its hover view ([`crop::context`]), or `None`
/// where [`image()`] would cut no crop.
pub(super) fn context(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
) -> Option<CropPicture> {
    crop::context(&outline(recon, track, observation)?, src)
}
