// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where a keypoint puts the patch centre on the patch grid.

use super::KeypointLocalizeParams;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::ProjectedImage;

/// Where `keypoint` sits relative to the point's own projection, in **patch-grid
/// px** on the `params.resolution` grid: the offset
/// [`super::localize_patch_keypoints`] would seed that view's search at.
///
/// Published so that a caller can tell how far a keypoint has moved from the
/// point's projection on the grid the search runs on, which is what
/// [`KeypointLocalizeParams::max_shift_px`] judges in source px. `None` on the
/// same refusals the seeding itself makes: a ray parallel
/// to the plane, a hit behind the camera, a ray pointing away from a direction
/// patch, or a degenerate patch with no extent.
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

/// Unproject a starting keypoint onto the patch plane and express the in-plane
/// offset of its hit point (from the patch centre) in patch-grid px.
///
/// The world-space unprojection itself is
/// [`OrientedPatch::anchored_at_keypoint`]'s — the same offset the renderer
/// re-anchors a frame by — so a seed and a re-anchored frame can never disagree
/// about where a keypoint puts the patch. `None` on each of that method's
/// refusals (ray parallel to the plane, a ray-path hit behind the camera, a ray
/// pointing away from a direction patch), and on a degenerate patch whose zero
/// extent makes `wpp` zero: seeding at the projection (`acc = 0`) beats
/// propagating a NaN/inf offset.
///
/// [`keypoint_grid_offset`] is the same question in a caller's terms: a params'
/// own grid resolution rather than a `wpp` pair.
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
    let off = patch.keypoint_plane_offset(view.camera, view.cam_from_world, keypoint)?;
    // Grid rows count downward from `+v̂` (they map to `−v_axis`), so the
    // v-grid coordinate negates the in-plane `v̂` component — the inverse of
    // `shifted_center`.
    Some([
        off.dot(&patch.u_axis) / wpp_u,
        -off.dot(&patch.v_axis) / wpp_v,
    ])
}
