// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Keypoint affine shape derived from a reconstruction's patch frames.
//!
//! Projection algebra rather than field access: an observation's patch frame
//! `(u, v)` is projected into the observing camera to recover the local affine
//! shape the `.sift` frame carries for a `sift_files` reconstruction, and the
//! per-point feature size reduces over it. See
//! `specs/formats/sfmr-file-format.md` ("Deriving keypoint shape, scale, and
//! orientation").

use nalgebra::Vector3;

use super::SfmrReconstruction;

impl SfmrReconstruction {
    /// Derive the local **affine shape** of an observation's keypoint — the
    /// counterpart of the `.sift` affine frame carried by `sift_files` — by
    /// projecting the point's patch frame `(u, v)` into the observing camera.
    /// The returned 2×2 matrix has the projected patch half-axes as its columns,
    /// so (like a `.sift` shape) it maps the unit circle to the keypoint's image
    /// footprint: an overlay recovers scale from the column norms, orientation
    /// from their angle, and anisotropy from their ratio.
    ///
    /// `keypoint_xy` anchors the frame in-plane (the patch-plane point that
    /// projects to it); a grazing view falls back to the point centre. A point
    /// at infinity is handled too: its patch is tangent to the direction sphere,
    /// so the frame is evaluated at the stored direction and its corners project
    /// as directions (the camera translation folds out), giving a roughly
    /// circular shape. Returns `None` when the reconstruction carries no patch
    /// frame, this point has no patch (zero frame), or the projection is
    /// degenerate or behind the camera. See `specs/formats/sfmr-file-format.md`
    /// ("Deriving keypoint shape, scale, and orientation").
    pub fn observation_affine_shape(
        &self,
        point_idx: usize,
        image_index: usize,
        keypoint_xy: [f32; 2],
    ) -> Option<[[f32; 2]; 2]> {
        let u_arr = self.patch_u_halfvec_xyz.as_ref()?;
        let v_arr = self.patch_v_halfvec_xyz.as_ref()?;
        let point = self.points.get(point_idx)?;
        let u = Vector3::new(
            u_arr[[point_idx, 0]] as f64,
            u_arr[[point_idx, 1]] as f64,
            u_arr[[point_idx, 2]] as f64,
        );
        let v = Vector3::new(
            v_arr[[point_idx, 0]] as f64,
            v_arr[[point_idx, 1]] as f64,
            v_arr[[point_idx, 2]] as f64,
        );
        if u.norm_squared() == 0.0 || v.norm_squared() == 0.0 {
            return None; // no patch for this point
        }

        let image = self.images.get(image_index)?;
        let camera = self.cameras.get(image.camera_index as usize)?;
        let r = image.quaternion_wxyz.to_rotation_matrix();

        // Where to evaluate the frame. For a point at infinity the patch is
        // tangent to the direction sphere: the anchor is the stored direction
        // and corners are directions (projected with `w = 0`). For a finite
        // point the anchor is where the keypoint's back-projected ray meets the
        // patch plane (fall back to the point centre for a grazing view).
        let anchor = if point.is_at_infinity() {
            point.position.coords
        } else {
            let normal = u.cross(&v);
            let n_norm = normal.norm();
            if n_norm == 0.0 {
                return None;
            }
            let normal = normal / n_norm;
            let center = image.camera_center();
            let ray_cam = camera.pixel_to_ray(keypoint_xy[0] as f64, keypoint_xy[1] as f64);
            let ray_world = r.transpose() * Vector3::new(ray_cam[0], ray_cam[1], ray_cam[2]);
            let denom = ray_world.dot(&normal);
            if denom.abs() < 1e-9 {
                point.position.coords
            } else {
                let lambda = (point.position.coords - center.coords).dot(&normal) / denom;
                center.coords + lambda * ray_world
            }
        };

        // Project the anchor and the two half-axis tips. `w` folds out the
        // translation for a point at infinity, so its corners project as
        // directions.
        let w = point.w;
        let project = |world: Vector3<f64>| -> Option<(f64, f64)> {
            let p_cam = r * world + w * image.translation_xyz;
            camera.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z])
        };
        let k = project(anchor)?;
        let pu = project(anchor + u)?;
        let pv = project(anchor + v)?;

        // Columns are the projected half-axes: u -> column 0, v -> column 1.
        Some([
            [(pu.0 - k.0) as f32, (pv.0 - k.0) as f32],
            [(pu.1 - k.1) as f32, (pv.1 - k.1) as f32],
        ])
    }

    /// Per-point maximum keypoint feature size (px) for an `embedded_patches`
    /// reconstruction, derived from each observation's projected patch frame.
    ///
    /// For every observation the point's patch frame is projected into the
    /// observing camera (`observation_affine_shape`) and its size taken as the
    /// mean of the two projected half-axis column norms — the same size measure
    /// the Track View reports and that `.sift` affine shapes yield for a
    /// `sift_files` reconstruction. The per-point value is the maximum over its
    /// observations. Returns `None` for a `sift_files` source (no inline
    /// keypoints); a point with no patch, or whose every observation projects
    /// degenerately, gets `0.0`.
    pub fn max_embedded_feature_size_per_point(&self) -> Option<Vec<f32>> {
        let keypoints_xy = self.keypoints_xy()?;
        let mut out = vec![0.0f32; self.point_count()];
        for (point_idx, slot) in out.iter_mut().enumerate() {
            let start = self.observation_offsets[point_idx];
            let mut max_size = 0.0f32;
            for (k, obs) in self.observations_for_point(point_idx).iter().enumerate() {
                let obs_global = start + k;
                let xy = [keypoints_xy[[obs_global, 0]], keypoints_xy[[obs_global, 1]]];
                if let Some(a) =
                    self.observation_affine_shape(point_idx, obs.image_index as usize, xy)
                {
                    let col0 = (a[0][0] * a[0][0] + a[1][0] * a[1][0]).sqrt();
                    let col1 = (a[0][1] * a[0][1] + a[1][1] * a[1][1]).sqrt();
                    max_size = max_size.max(0.5 * (col0 + col1));
                }
            }
            *slot = max_size;
        }
        Some(out)
    }
}
