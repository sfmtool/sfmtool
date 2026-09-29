// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two pictures the body draws of a patch: a row's tile, warped out of
//! one photograph, and the track's own stored bitmap.
//!
//! [`patch_color_image`] warps a photograph through a surfel's frame,
//! re-anchored on the observation's keypoint first ([`render_frame`]), and
//! [`frame_color_image`] is its body for a frame already where it should be,
//! which the tile's hover view renders a wider frame through.
//! [`stored_patch_image`] turns a patch bitmap into an opaque picture, which
//! is what the header's patch slot draws at the track stage.

use sfmtool_core::camera::remap::{remap_bilinear, ImageU8};
use sfmtool_core::camera::{CameraIntrinsics, WarpMap};
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;

/// Render resolution of a row's patch tile, in texels a side. The tile is
/// rendered crisp at this resolution and drawn scaled to the row's tile cell.
pub(crate) const PATCH_RES: u32 = 64;

/// One observation's patch tile, as an RGBA picture: `src` warped through
/// `frame` re-anchored on `keypoint` ([`render_frame`]), so the tile shows the
/// surface as *this* sighting sees it rather than as the point's residual
/// leaves it.
///
/// The warp itself, with no `egui::Context` in it, so what a tile shows is
/// testable without a texture manager -- which is what lets a headless test
/// assert that two ways of naming the same place produce the same tile.
pub(crate) fn patch_color_image(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
    keypoint: Option<[f64; 2]>,
    src: &ImageU8,
) -> egui::ColorImage {
    let frame = render_frame(frame, camera, cam_from_world, keypoint);
    frame_color_image(&frame, camera, cam_from_world, src, PATCH_RES)
}

/// `frame` warped out of `src` at `resolution` texels a side, as an RGBA
/// image: the body of [`patch_color_image`] for a frame that is already where
/// it should be.
///
/// Separate so that a wider frame can be rendered at the same sampling as the
/// tile -- a frame `k` times as wide at `k` times the resolution -- which is
/// what the tile shows in its hover view.
pub(crate) fn frame_color_image(
    frame: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
    src: &ImageU8,
    resolution: u32,
) -> egui::ColorImage {
    let map = WarpMap::from_patch(frame, camera, cam_from_world, resolution);
    let tile = remap_bilinear(src, &map);
    // Expand 3-channel RGB (same channel count as the cached source) to RGBA.
    let (w, h) = (tile.width() as usize, tile.height() as usize);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for px in tile.data().as_chunks::<3>().0.iter() {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba)
}

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

/// A patch bitmap, `(rows, columns, channels)`, as an opaque RGBA image, or
/// `None` when it is all zero, which is how a point with no stored patch is
/// written.
///
/// One channel is repeated across RGB, three are RGB, and a fourth, the
/// per-texel cross-view confidence, is dropped for an opaque alpha. What the
/// header's patch slot draws at the track stage: the consensus bitmap, which
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
