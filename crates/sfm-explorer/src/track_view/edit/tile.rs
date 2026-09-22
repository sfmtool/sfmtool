// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-observation tile the table draws: what one sighting of the track
//! actually looks like.
//!
//! The tile is the column the numbers beside it are about. A ZNCC of 0.42 is a
//! number; the tile that produced it is the thing a person can judge, and the
//! whole reason the bench exists is that a person judges.
//!
//! Which picture it is follows the stage, because the two stages register
//! different things:
//!
//! - At the **track stage** it is the patch re-rendered from this
//!   observation's own view, re-anchored where the observation sits -- the very
//!   tile Track View draws for a committed track, through
//!   that panel's own renderer
//!   ([`crate::track_view::view::patch_color_image`]), so a track on the
//!   bench and the point it came from cannot show one surface two ways.
//! - At the **cluster stage** there is no surface, so it is the observation's
//!   own grid: the `R x R` samples the refinement kernel reads at that place
//!   and its shape, through the kernel's own sampler
//!   (`sfmtool_core::patch::cluster_refine::sample_member_grid`), which is what
//!   makes the picture the thing the ZNCC beside it was computed over rather
//!   than a second opinion about it.
//!
//! **Where the observation sits is [`crate::bench::observation_site`]'s**, at
//! either stage: the keypoint a reading wrote, else the refined cluster
//! position, else the seed the step that proposed it left. A candidate a
//! descriptor search has just added has only that seed, and the tile is the
//! whole of what says whether the search found the right surface -- so it is
//! cut around the seed rather than left blank, or, at the track stage, cut
//! around wherever the bare projection of the point happens to land.

use sfmtool_core::bench::{EditableTrack, Stage};
use sfmtool_core::camera::remap::ImageU8Pyramid;
use sfmtool_core::patch::cluster_refine::{sample_member_grid, ClusterRefineParams};
use sfmtool_core::SfmrReconstruction;

/// The tile for one observation, uploaded, or `None` when [`image()`] had
/// nothing to render.
pub(super) fn render(
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
    name: String,
) -> Option<egui::TextureHandle> {
    let tile = image(recon, track, observation, src)?;
    Some(ctx.load_texture(name, tile, egui::TextureOptions::NEAREST))
}

/// The picture one row draws, or `None` when there is nothing to render: no
/// patch yet at the track stage, nothing saying where the observation sits, or
/// a degenerate shape at the cluster stage.
///
/// Pure, so a headless test can ask what a row shows rather than only whether
/// it showed something. `src` is the observation's own photograph as the node's
/// full-resolution cache holds it: the pyramid built at the decode, whose level
/// 0 is the photograph and whose lower levels are what the cluster sampler
/// mip-selects over.
pub(super) fn image(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
) -> Option<egui::ColorImage> {
    let row = track.observations.get(observation)?;
    let site = crate::bench::observation_site(row)?;
    let img_idx = row.image as usize;
    match &track.stage {
        Stage::Track(payload) => {
            let frame = payload.placement.as_ref()?;
            let image = recon.image_table.images.get(img_idx)?;
            let camera = recon.image_table.cameras.get(image.camera_index as usize)?;
            Some(crate::track_view::view::patch_color_image(
                frame,
                camera,
                &crate::scene::cam_from_world(image),
                Some(site.pixel),
                src.level(0),
            ))
        }
        Stage::Cluster(payload) => {
            // The cluster's own radius, always: it is what every one of its
            // shapes is written against, so the tile is the square the person
            // asked for before an evaluation and the square the ZNCC was
            // measured over after one. The grid is the template's when one has
            // been cut, and the kernel's default -- what the next evaluation
            // will use -- before that.
            let mut params = ClusterRefineParams {
                radius: payload.radius,
                ..ClusterRefineParams::default()
            };
            if let Some(template) = payload.template.as_ref() {
                params.resolution = template.samples.shape()[0] as u32;
            }
            let grid = sample_member_grid(src, site.pixel, site.shape?, &params)?;
            let resolution = params.resolution.max(2) as usize;
            let channels = grid.len() / (resolution * resolution);
            Some(color_image(&grid, resolution, channels))
        }
    }
}

/// The sampler's interleaved `R x R x C` samples as an RGBA image.
///
/// The samples are the source's own 0..255 range in `f32`, so they are rounded
/// and clamped rather than rescaled: a tile is a picture of the photograph, and
/// stretching its levels would make two tiles of one surface look different.
/// One channel is repeated across RGB, which is what a grey photograph is.
fn color_image(grid: &[f32], resolution: usize, channels: usize) -> egui::ColorImage {
    let mut rgba = Vec::with_capacity(resolution * resolution * 4);
    for texel in grid.chunks_exact(channels.max(1)) {
        let level = |c: usize| texel.get(c).copied().unwrap_or(texel[0]).clamp(0.0, 255.0) as u8;
        match channels {
            0 => rgba.extend_from_slice(&[0, 0, 0, 255]),
            1 => {
                let grey = level(0);
                rgba.extend_from_slice(&[grey, grey, grey, 255]);
            }
            _ => rgba.extend_from_slice(&[level(0), level(1), level(2), 255]),
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([resolution, resolution], &rgba)
}
