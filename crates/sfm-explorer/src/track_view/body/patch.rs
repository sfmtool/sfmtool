// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two pictures the body draws of a patch: a row's tile, warped out of
//! one photograph, and the track's own stored bitmap.
//!
//! [`patch_color_image`] warps a photograph through a surfel's frame,
//! re-anchored on the observation's keypoint first ([`render_frame`]), and
//! [`frame_color_image`] is its body for a frame already where it should be,
//! which the tile's hover view renders a wider frame through. The tile and its
//! hover view are rendered with the sampler the bench's evaluation renders the
//! same view with: its sampler choice (`crate::bench::sampler_choice`, the
//! sampler rule by default) applied to the view's Jacobian at `R`
//! ([`PatchJacobian::sampler`]), so a view the rule moves is drawn with the
//! anisotropic sampler and every other view with one bilinear sample per texel
//! from the mip level the warp's compression picks at that texel.
//! [`PatchJacobian`] holds the Jacobian at the tile's centre of the warp from
//! the patch grid at the reconstruction's patch resolution `R` -- not the
//! tile's display resolution -- which the table's *Zoom* column prints the
//! zoom of.
//! [`stored_patch_image`] turns a patch bitmap into an opaque picture, which
//! is what the header's patch slot draws at the track stage, and
//! [`track_patch_image`] picks the picture of a track's own patch for its
//! stage, for the header and the recent items strip alike.

use sfmtool_core::bench::{EditableTrack, Stage};
use sfmtool_core::camera::image::ImageU8Pyramid;
use sfmtool_core::camera::sampler::{minor_axis_loss, render_tile, Sampler};
use sfmtool_core::camera::warp_map::singular_values_2x2;
use sfmtool_core::camera::{CameraIntrinsics, WarpMap};
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;

/// Render resolution of a row's patch tile, in texels a side. The tile is
/// rendered crisp at this resolution and drawn scaled to the row's tile cell.
///
/// For display only: no reading, zoom, sort key or reported number depends
/// on it. Those are in patch-grid px at the reconstruction's patch resolution
/// (`crate::bench::patch_resolution`), or in photograph px or scene units.
pub(crate) const PATCH_RES: u32 = 64;

/// One observation's patch tile, as an RGBA picture: `src` warped through
/// `frame` re-anchored on `keypoint` ([`render_frame`]) at [`PATCH_RES`]
/// texels a side with `sampler`, so the tile shows the surface as *this*
/// sighting sees it rather than as the point's residual leaves it.
///
/// The warp itself, with no `egui::Context` in it, so what a tile shows is
/// testable without a texture manager -- which is what lets a headless test
/// assert that two ways of naming the same place produce the same tile.
pub(crate) fn patch_color_image(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
    keypoint: Option<[f64; 2]>,
    src: &ImageU8Pyramid,
    sampler: Sampler,
) -> egui::ColorImage {
    let frame = render_frame(frame, camera, cam_from_world, keypoint);
    frame_color_image(&frame, camera, cam_from_world, src, PATCH_RES, sampler)
}

/// `frame` warped out of `src` at `resolution` texels a side, as an RGBA
/// image: the body of [`patch_color_image`] for a frame that is already where
/// it should be.
///
/// Separate so that a wider frame can be rendered at the same sampling as the
/// tile -- a frame `k` times as wide at `k` times the resolution -- which is
/// what the tile shows in its hover view.
///
/// Under [`Sampler::BilinearMip`] each texel is one bilinear sample from the
/// pyramid level the warp's local compression picks, `round(log2(s_major))`
/// for the larger singular value `s_major` of the warp's Jacobian at that
/// texel, so the level is chosen texel by texel. Where a texel shrinks the
/// photograph by more than about 1.4 times (`s_major` of `sqrt(2)` or more), it
/// reads a level averaged down towards its own sampling rather than aliasing
/// the full-resolution pixels. Where no texel does, every texel reads level 0
/// and the picture is exactly plain bilinear. Under [`Sampler::Anisotropic`]
/// the level follows the smaller singular value and several samples are
/// averaged along the more compressed direction, so the less compressed one
/// keeps its detail.
pub(crate) fn frame_color_image(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
    src: &ImageU8Pyramid,
    resolution: u32,
    sampler: Sampler,
) -> egui::ColorImage {
    let mut map = WarpMap::from_patch(frame, camera, cam_from_world, resolution);
    let tile = render_tile(src, &mut map, sampler);
    // Expand 3-channel RGB (same channel count as the cached source) to RGBA.
    let (w, h) = (tile.width() as usize, tile.height() as usize);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for px in tile.data().as_chunks::<3>().0.iter() {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba)
}

/// The ratio of a patch Jacobian's smaller singular value to its larger at or
/// under which the patch has no zoom: the patch is seen edge on. A real
/// oblique view stays many orders of magnitude above it (a patch at 89.9° to
/// the line of sight reads about 2e-3), while an edge-on patch's finite
/// difference leaves a rounding residue near 1e-14 of the larger value, which
/// would otherwise print as a zoom of 10¹³.
const EDGE_ON_RATIO: f64 = 1e-9;

/// The 2x2 Jacobian, at the centre of a row's tile, of the warp from the
/// patch grid to the photograph, in photograph pixels per patch-grid px:
/// `[[dx/dcol, dx/drow], [dy/dcol, dy/drow]]`, the layout
/// `WarpMap::get_jacobian` uses. Core's `patch_grid_jacobian` reads it at the
/// reconstruction's patch resolution `R` ([`super::tile::patch_jacobian`]),
/// the grid the bench's shift and self-similarity radius are stated in, never
/// at the display tile's [`PATCH_RES`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PatchJacobian(pub(crate) [[f64; 2]; 2]);

impl PatchJacobian {
    /// How much the patch magnifies the photograph, in patch-grid px at `R`
    /// per photograph pixel: the least, along the direction the warp stretches most, then the
    /// most, `[1 / s_major, 1 / s_minor]` for the singular values
    /// (`singular_values_2x2`). Over 1 the patch grid is finer than the
    /// photograph's pixels there and under 1 it is coarser. `None` where either is not finite or the
    /// smaller is at most [`EDGE_ON_RATIO`] of the larger, as for a patch seen
    /// edge on, whose finite difference leaves a rounding residue across the
    /// collapsed axis rather than an exact zero.
    pub(crate) fn zoom_range(&self) -> Option<[f64; 2]> {
        let [major, minor] = singular_values_2x2(self.0);
        (major.is_finite() && major > 0.0 && minor > major * EDGE_ON_RATIO)
            .then(|| [1.0 / major, 1.0 / minor])
    }

    /// The geometric mean of [`Self::zoom_range`], which is `1 / sqrt(|det J|)`
    /// and what the *Zoom* column orders the rows by. `None` where
    /// [`Self::zoom_range`] is.
    pub(crate) fn mean_zoom(&self) -> Option<f64> {
        self.zoom_range().map(|[low, high]| (low * high).sqrt())
    }

    /// The sampler the view's tile is rendered with: the bench evaluation's
    /// sampler choice (`crate::bench::sampler_choice`) applied to this
    /// Jacobian, the choice the bench makes for the same view.
    pub(crate) fn sampler(&self) -> Sampler {
        crate::bench::sampler_choice().for_jacobian(Some(self.0))
    }

    /// How many times coarser `bilinear_mip` would read the view's less
    /// compressed axis than that axis needs, the `L` the sampler rule compares
    /// with its threshold ([`minor_axis_loss`]).
    pub(crate) fn minor_axis_loss(&self) -> f64 {
        minor_axis_loss(singular_values_2x2(self.0))
    }
}

/// The sampler a tile with no Jacobian is rendered with, as the bench's
/// kernels render such a view.
pub(crate) const FALLBACK_SAMPLER: Sampler = Sampler::BilinearMip;

/// The frame one observation's tile is rendered through: the point's patch
/// re-anchored so its centre projects onto `keypoint`, or the stored geometric
/// frame when the observation has no keypoint to anchor on or the keypoint's ray
/// cannot meet the patch (parallel to its plane, behind the camera under a
/// ray-path model, or pointing away from a direction patch).
pub(super) fn render_frame(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
    keypoint: Option<[f64; 2]>,
) -> OrientedPatch {
    keypoint
        .and_then(|kp| frame.anchored_at_keypoint(camera, cam_from_world, kp))
        .unwrap_or_else(|| frame.clone())
}

/// What a track's bitmap is, for the header's patch slot to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BitmapKind {
    /// The track's patch, the one a commit writes.
    Patch,
    /// The track's patch, kept only until the next render replaces it
    /// (`EditableTrack::bitmap_pending`): no row is scored against it.
    Pending,
    /// A bitmap for judging only (`TrackPayload::bitmap_for_judging`): not
    /// the track's patch, and never committed.
    Judging,
}

impl BitmapKind {
    /// What `track`'s bitmap is; [`Self::Patch`] at the cluster stage.
    pub(super) fn of(track: &EditableTrack) -> Self {
        if track.track().is_some_and(|p| p.bitmap_for_judging) {
            Self::Judging
        } else if track.bitmap_pending() {
            Self::Pending
        } else {
            Self::Patch
        }
    }
}

/// The picture of a track's own patch: the patch bitmap at the track
/// stage ([`stored_patch_image`]), the template at the cluster stage, `None`
/// where there is neither. A bitmap for judging
/// (`TrackPayload::bitmap_for_judging`) is drawn at a third of its
/// brightness, so it does not pass for the track's patch.
///
/// What the header's patch slot draws, and what the recent items strip draws
/// in each chip, so an item looks the same in both places.
pub(crate) fn track_patch_image(track: &EditableTrack) -> Option<egui::ColorImage> {
    match &track.stage {
        Stage::Track(payload) => {
            let mut image = stored_patch_image(payload.bitmap.as_ref()?.view())?;
            if payload.bitmap_for_judging {
                for pixel in &mut image.pixels {
                    let [r, g, b, a] = pixel.to_array();
                    *pixel = egui::Color32::from_rgba_unmultiplied(r / 3, g / 3, b / 3, a);
                }
            }
            Some(image)
        }
        Stage::Cluster(payload) => {
            let samples = &payload.template.as_ref()?.samples;
            let shape = samples.shape();
            let grid: Vec<f32> = samples.iter().copied().collect();
            Some(super::tile::color_image(&grid, shape[0], shape[2]))
        }
    }
}

/// A patch bitmap, `(rows, columns, channels)`, as an opaque RGBA image, or
/// `None` when it is all zero, which is how a point with no stored patch is
/// written.
///
/// One channel is repeated across RGB, three are RGB, and a fourth, the
/// alpha (on the photograph for a view's tile, the cross-view confidence for a
/// fused mean), is dropped for an opaque alpha. What the
/// header's patch slot draws at the track stage: the patch bitmap, which
/// for a track read from a point is that point's stored patch and which a
/// commit writes as it.
pub(crate) fn stored_patch_image(bitmap: ndarray::ArrayView3<'_, u8>) -> Option<egui::ColorImage> {
    let [h, w, channels] = [bitmap.shape()[0], bitmap.shape()[1], bitmap.shape()[2]];
    if channels == 0 || bitmap.iter().all(|&b| b == 0) {
        return None;
    }
    let mut rgba = Vec::with_capacity(h * w * 4);
    for row in 0..h {
        for col in 0..w {
            let level = |c: usize| bitmap[[row, col, c.min(channels - 1)]];
            if channels >= 3 {
                rgba.extend_from_slice(&[level(0), level(1), level(2), 255]);
            } else {
                rgba.extend_from_slice(&[level(0), level(0), level(0), 255]);
            }
        }
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}
