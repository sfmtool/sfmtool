// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where a keypoint puts the patch: the frame a view's search starts from, and
//! the keypoint's distance from the point's projection on the patch grid.

use nalgebra::{Rotation3, Vector3};

use super::KeypointLocalizeParams;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::ProjectedImage;

/// Where `keypoint` sits relative to the point's own projection, in **patch-grid
/// px** on the `params.resolution` grid.
///
/// Published so that a caller can tell how far a keypoint has moved from the
/// point's projection on the grid the search runs on, which is what
/// [`KeypointLocalizeParams::max_shift_px`] judges in source px. The distance
/// is the move of the patch slid onto the keypoint
/// ([`OrientedPatch::keypoint_slide_offset`]), which is across the line of
/// sight, read on the patch's axes turned by the least rotation that takes its
/// normal onto the direction to the camera, so its length is the move's.
/// `None` on the slide's refusals (a ray at or past a right
/// angle to the line of sight, a ray pointing away from a direction patch) and
/// on a degenerate patch with no extent.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::patch::keypoint_localize::{keypoint_grid_offset, KeypointLocalizeParams};
/// # fn run(
/// #     patch: &sfmtool_core::patch::cloud::OrientedPatch,
/// #     view: &sfmtool_core::patch::normal_refine::ProjectedImage<'_>,
/// #     keypoint: [f64; 2],
/// # ) {
/// let params = KeypointLocalizeParams::default();
/// if let Some(off) = keypoint_grid_offset(patch, view, keypoint, &params) {
///     println!("{:.2} grid px from the projection", off[0].hypot(off[1]));
/// }
/// # }
/// ```
pub fn keypoint_grid_offset(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: [f64; 2],
    params: &KeypointLocalizeParams,
) -> Option<[f64; 2]> {
    let resolution = f64::from(params.resolution.max(2));
    seed_offset(
        patch,
        view,
        keypoint,
        2.0 * patch.half_extent[0] / resolution,
        2.0 * patch.half_extent[1] / resolution,
    )
}

/// The move of the patch slid onto `keypoint`
/// ([`OrientedPatch::keypoint_slide_offset`]) in patch-grid px, `[u, v]` with
/// grid rows counting downward from `+v̂`.
///
/// For a finite patch the move is perpendicular to the line of sight, and in
/// general not in the patch's plane, so it is read on the patch's axes turned
/// by the least rotation that takes the patch's normal onto the direction to
/// the camera: the axes of the plane facing the camera that are nearest the
/// patch's own. Its length is the move's length, and two keypoints of one view
/// read this way are as far apart as their slid centres. For a direction patch
/// (`w == 0`) the move is in the tangent plane and is read on the patch's own
/// axes.
///
/// `None` on the slide's refusals and on a degenerate patch whose zero extent
/// makes `wpp` zero.
pub(in crate::patch) fn seed_offset(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: [f64; 2],
    wpp_u: f64,
    wpp_v: f64,
) -> Option<[f64; 2]> {
    if wpp_u <= 0.0 || wpp_v <= 0.0 {
        return None;
    }
    let off = patch.keypoint_slide_offset(view.camera, view.cam_from_world, keypoint)?;
    let (u, v) = facing_axes(patch, view);
    // Grid rows count downward from `+v̂` (they map to `−v_axis`), so the
    // v-grid coordinate negates the `v̂` component — as `shifted_center` does.
    Some([off.dot(&u) / wpp_u, -off.dot(&v) / wpp_v])
}

/// The patch's `u` and `v` axes turned by the least rotation that takes its
/// normal onto the unit direction from its centre to `view`'s camera; the
/// axes themselves for a direction patch, or where the patch faces exactly
/// away from the camera.
fn facing_axes(patch: &OrientedPatch, view: &ProjectedImage<'_>) -> (Vector3<f64>, Vector3<f64>) {
    let own = (patch.u_axis, patch.v_axis);
    if patch.w == 0.0 {
        return own;
    }
    let eye = view.cam_from_world.inverse_translation_origin();
    let Some(to_camera) = (eye - patch.center).try_normalize(1e-12) else {
        return own;
    };
    match Rotation3::rotation_between(&patch.normal(), &to_camera) {
        Some(turn) => (turn * patch.u_axis, turn * patch.v_axis),
        None => own,
    }
}

/// The patch frame one view's search is stated on: `patch` slid onto the
/// view's starting `keypoint` ([`OrientedPatch::anchored_at_keypoint`]), with
/// whether that keypoint was followed.
///
/// Every render of the view and every offset its search reads is on this
/// frame, offsets counting from its centre in its own plane on the patch's
/// grid, so the search starts at the keypoint's own tile and a found offset
/// `d` is written back as the projection of `shifted_center(frame, d)`. A view
/// with no keypoint starts at the patch itself, as one whose keypoint the slide
/// refuses does; that one reports `false`, so the caller can treat it as
/// starting at the projection rather than at its keypoint.
pub(in crate::patch) fn seed_frame(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: Option<[f64; 2]>,
) -> (OrientedPatch, bool) {
    let Some(keypoint) = keypoint else {
        return (patch.clone(), true);
    };
    match patch.anchored_at_keypoint(view.camera, view.cam_from_world, keypoint) {
        Some(slid) => (slid, true),
        None => (patch.clone(), false),
    }
}
