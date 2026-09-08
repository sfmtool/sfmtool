// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Oriented-patch construction and texture rendering for the selected point.
//!
//! Four pieces, all specific to embedded-patches reconstructions:
//! [`build_patch_frame`] recovers the point's oriented frame from the stored
//! half-vectors (it gates the table's "Patch" column),
//! [`build_stored_patch_texture`] turns the stored bitmap into the header tile,
//! [`render_frame`] re-anchors the point's frame on one observation's stored
//! keypoint, and [`PointTrackDetail::ensure_rendered_patch`] warps that
//! observation's full-res image through the re-anchored frame to produce the
//! per-row tiles.

use std::collections::HashMap;

use nalgebra::Vector3;
use ndarray::Axis;
use sfmtool_core::camera::remap::{remap_bilinear, ImageU8};
use sfmtool_core::camera::{CameraIntrinsics, WarpMap};
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::SfmrReconstruction;

use super::PointTrackDetail;
use crate::scene::ImageRef;

/// Render resolution of per-observation patch tiles (rendered crisp at this
/// resolution, displayed scaled to [`super::PATCH_TILE`]).
const PATCH_RES: u32 = 64;

impl PointTrackDetail {
    /// Render the patch tile for one observation if not already cached: warp
    /// the observation's full-res image through the selected point's patch
    /// frame (`WarpMap::from_patch` + `remap_bilinear`), **re-anchored so the
    /// frame's centre projects onto this observation's stored keypoint**
    /// ([`OrientedPatch::anchored_at_keypoint`]). The tiles of a photometrically
    /// aligned track then show the same content whatever the point's geometric
    /// residual is; that residual is the Error column's business, not the
    /// tile's. The stored geometric frame is used unchanged when the
    /// reconstruction carries no keypoints (a `sift_files` reconstruction, which
    /// never reaches this column) or when the keypoint's ray cannot meet the
    /// patch. A patch not visible in this view warps to an all-black tile and is
    /// drawn as such. A missing source image is not cached (the dock pre-caches
    /// full-res images, so this only happens transiently).
    pub(super) fn ensure_rendered_patch(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        image_ref: ImageRef,
        full_res_cache: &HashMap<ImageRef, Option<ImageU8>>,
    ) {
        if self.rendered_patch_textures.contains_key(&image_ref) {
            return;
        }
        let img_idx = image_ref.index();
        let keypoint = self.observation_keypoint(recon, img_idx);
        let Some(frame) = self.patch_frame.as_ref() else {
            return;
        };
        let Some(src) = full_res_cache.get(&image_ref).and_then(|o| o.as_ref()) else {
            return;
        };
        let image = &recon.image_table.images[img_idx];
        let camera = &recon.image_table.cameras[image.camera_index as usize];
        let q = image.quaternion_wxyz.quaternion();
        let cam_from_world = RigidTransform::from_wxyz_translation(
            [q.w, q.i, q.j, q.k],
            [
                image.translation_xyz.x,
                image.translation_xyz.y,
                image.translation_xyz.z,
            ],
        );
        let frame = render_frame(frame, camera, &cam_from_world, keypoint);
        let map = WarpMap::from_patch(&frame, camera, &cam_from_world, PATCH_RES);
        let tile = remap_bilinear(src, &map);
        // Expand 3-channel RGB (same channel count as the cached source) to RGBA.
        let (w, h) = (tile.width() as usize, tile.height() as usize);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in tile.data().as_chunks::<3>().0.iter() {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        let point_idx = self.prepared_point.map(|p| p.index()).unwrap_or(0);
        let texture = ctx.load_texture(
            format!("track_patch_{point_idx}_{img_idx}"),
            color_image,
            egui::TextureOptions::NEAREST,
        );
        self.rendered_patch_textures.insert(image_ref, texture);
    }

    /// The keypoint this track stores in image `img_idx`, in source-image
    /// pixels, or `None` when the reconstruction has no stored keypoints at all
    /// (`sift_files`, where the prepared `feature_xy` is a SIFT feature position
    /// rather than a keypoint of this patch) or the image is not in the track.
    ///
    /// A track should observe an image once; if one somehow observes it twice,
    /// the first prepared row wins — two tiles cannot be drawn in one cell
    /// anyway, and picking a row beats panicking on the surprise.
    pub(super) fn observation_keypoint(
        &self,
        recon: &SfmrReconstruction,
        img_idx: usize,
    ) -> Option<[f64; 2]> {
        recon.keypoints_xy()?;
        self.observations
            .iter()
            .find(|obs| obs.image_index == img_idx)
            .map(|obs| [obs.feature_xy[0] as f64, obs.feature_xy[1] as f64])
    }
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

/// Build the selected point's oriented patch frame from the stored patch
/// half-vectors, or `None` when the reconstruction carries no frame or the
/// point's `u` half-vector is zero (no patch for this point). The stored
/// arrays are half-*vectors* (`axis * half_extent`); split them into unit
/// axis and half-extent like `PatchCloud::from_halfvec_arrays`. For a point
/// at infinity the stored u/v are already the tangent frame — the same frame
/// applies, just re-marked with `w = 0`.
pub(super) fn build_patch_frame(
    recon: &SfmrReconstruction,
    point_idx: usize,
) -> Option<OrientedPatch> {
    let u_arr = recon.point_set.patch_u_halfvec_xyz.as_ref()?;
    let v_arr = recon.point_set.patch_v_halfvec_xyz.as_ref()?;
    if point_idx >= u_arr.nrows() || point_idx >= v_arr.nrows() {
        return None;
    }
    let u = Vector3::new(
        u_arr[[point_idx, 0]] as f64,
        u_arr[[point_idx, 1]] as f64,
        u_arr[[point_idx, 2]] as f64,
    );
    let hu = u.norm();
    if hu <= 1e-12 {
        return None;
    }
    let v = Vector3::new(
        v_arr[[point_idx, 0]] as f64,
        v_arr[[point_idx, 1]] as f64,
        v_arr[[point_idx, 2]] as f64,
    );
    let hv = v.norm();
    let u_axis = u / hu;
    let v_axis = if hv > 1e-12 { v / hv } else { v };
    let point = &recon.point_set.points[point_idx];
    let mut patch = OrientedPatch::new(point.position, u_axis, v_axis, [hu, hv]);
    if point.w == 0.0 {
        patch.w = 0.0;
    }
    Some(patch)
}

/// Build the stored-patch header texture for the selected point from
/// `patch_bitmaps_y_x_rgba`, or `None` when the array is absent or the
/// point's bitmap is all-zero (no stored patch). Displays RGB only: the
/// alpha channel (per-texel cross-view confidence) is forced opaque.
pub(super) fn build_stored_patch_texture(
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    point_idx: usize,
) -> Option<egui::TextureHandle> {
    let bitmaps = recon.point_set.patch_bitmaps_y_x_rgba.as_ref()?;
    if point_idx >= bitmaps.shape()[0] {
        return None;
    }
    let bitmap = bitmaps.index_axis(Axis(0), point_idx);
    let h = bitmap.shape()[0];
    let w = bitmap.shape()[1];
    let mut rgba: Vec<u8> = if let Some(slice) = bitmap.as_slice() {
        slice.to_vec()
    } else {
        bitmap.iter().copied().collect()
    };
    if rgba.iter().all(|&b| b == 0) {
        return None;
    }
    for px in rgba.as_chunks_mut::<4>().0.iter_mut() {
        px[3] = 255;
    }
    let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
    Some(ctx.load_texture(
        format!("stored_patch_{point_idx}"),
        image,
        egui::TextureOptions::NEAREST,
    ))
}
