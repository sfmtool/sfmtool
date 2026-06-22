// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Oriented 3D patches (surfels) and patch clouds.
//!
//! See `specs/core/patch-cloud.md`. An [`OrientedPatch`] is a small planar
//! surface element in world space; [`crate::camera::warp_map::WarpMap::from_patch`]
//! projects a camera's image onto one to render its canonical appearance.

use nalgebra::{Matrix3, Point3, Vector3};
use ndarray::Array2;

use crate::geometry::rigid_transform::RigidTransform;
use crate::reconstruction::SfmrReconstruction;
use crate::spatial::PointCloud;

/// Errors from [`PatchCloud::from_reconstruction`].
#[derive(Debug)]
pub enum PatchCloudError {
    /// [`PatchExtent::FeatureSize`] could not read a keypoint scale for any
    /// observation of point `point_id` — its `.sift` files were unreadable (or,
    /// degenerately, every observation coincides with a camera centre) — so the
    /// patch has no defined world size.
    MissingFeatureScale { point_id: u32 },
}

impl std::fmt::Display for PatchCloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatchCloudError::MissingFeatureScale { point_id } => write!(
                f,
                "no readable keypoint scale for any observation of point {point_id}; \
                 cannot size its patch from FeatureSize"
            ),
        }
    }
}

impl std::error::Error for PatchCloudError {}

/// An oriented planar patch (surfel) in world space.
///
/// The plane is spanned by orthonormal in-plane axes `u_axis` and `v_axis`; the
/// outward normal is `u_axis × v_axis`. The patch covers the world points
/// `center + s·half_extent[0]·u_axis + t·half_extent[1]·v_axis` for
/// `(s, t) ∈ [-1, 1]²`.
#[derive(Debug, Clone)]
pub struct OrientedPatch {
    pub center: Point3<f64>,
    /// Unit, in-plane.
    pub u_axis: Vector3<f64>,
    /// Unit, in-plane, perpendicular to `u_axis`.
    pub v_axis: Vector3<f64>,
    /// World-space half-size along `(u, v)`.
    pub half_extent: [f64; 2],
}

impl OrientedPatch {
    pub fn new(
        center: Point3<f64>,
        u_axis: Vector3<f64>,
        v_axis: Vector3<f64>,
        half_extent: [f64; 2],
    ) -> Self {
        Self {
            center,
            u_axis,
            v_axis,
            half_extent,
        }
    }

    /// Outward normal (`u_axis × v_axis`, normalized).
    pub fn normal(&self) -> Vector3<f64> {
        self.u_axis.cross(&self.v_axis).normalize()
    }

    /// World point for a normalized patch coordinate `(s, t) ∈ [-1, 1]²`.
    pub fn to_world(&self, s: f64, t: f64) -> Point3<f64> {
        self.center
            + self.u_axis * (s * self.half_extent[0])
            + self.v_axis * (t * self.half_extent[1])
    }

    /// Build from a center, a normal, and an `up_hint` that pins the in-plane
    /// rotation about the normal.
    ///
    /// `u_axis` is `up_hint` projected onto the plane and normalized; if
    /// `up_hint` is (near-)parallel to the normal, an arbitrary in-plane axis is
    /// chosen instead. `v_axis = normal × u_axis`, so `u_axis × v_axis` recovers
    /// the (normalized) normal.
    pub fn from_center_normal(
        center: Point3<f64>,
        normal: Vector3<f64>,
        up_hint: Vector3<f64>,
        half_extent: [f64; 2],
    ) -> Self {
        let n = normalize_or(normal, Vector3::z());
        let proj = up_hint - n * up_hint.dot(&n);
        let u = if proj.norm() > 1e-9 {
            proj.normalize()
        } else {
            any_orthonormal(&n)
        };
        let v = n.cross(&u);
        Self {
            center,
            u_axis: u,
            v_axis: v,
            half_extent,
        }
    }

    /// Whether the `cam_from_world` camera looks at the patch's front face — its
    /// outward normal points toward the camera centre.
    pub fn is_front_facing(&self, cam_from_world: &RigidTransform) -> bool {
        let cam_center = cam_from_world.inverse_translation_origin();
        (cam_center - self.center).dot(&self.normal()) > 0.0
    }
}

fn normalize_or(v: Vector3<f64>, fallback: Vector3<f64>) -> Vector3<f64> {
    let n = v.norm();
    if n > 1e-12 {
        v / n
    } else {
        fallback
    }
}

/// An arbitrary unit vector orthogonal to the (unit) vector `n`.
fn any_orthonormal(n: &Vector3<f64>) -> Vector3<f64> {
    let a = if n.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    (a - n * a.dot(n)).normalize()
}

/// A collection of oriented patches, optionally linked to reconstruction points.
#[derive(Debug, Clone, Default)]
pub struct PatchCloud {
    pub patches: Vec<OrientedPatch>,
    /// Optional per-patch link back to a reconstruction 3D-point id (empty if
    /// unused). When present, parallel to `patches`.
    pub point_ids: Vec<u32>,
}

impl PatchCloud {
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    pub fn patch(&self, i: usize) -> &OrientedPatch {
        &self.patches[i]
    }

    /// Build a patch cloud from a reconstruction's finite 3D points.
    ///
    /// One patch per finite point: center = position, in-plane up from the
    /// first observing camera, normal and half-size per the given policies.
    /// `point_ids` records the source point index for each patch.
    ///
    /// Errors with [`PatchCloudError::MissingFeatureScale`] under
    /// [`PatchExtent::FeatureSize`] if a point has no readable keypoint scale in
    /// any view.
    pub fn from_reconstruction(
        recon: &SfmrReconstruction,
        normal: PatchNormal,
        extent: PatchExtent,
    ) -> Result<Self, PatchCloudError> {
        let finite: Vec<usize> = (0..recon.points.len())
            .filter(|&i| !recon.points[i].is_at_infinity())
            .collect();
        let cam_centers: Vec<Point3<f64>> =
            recon.images.iter().map(|im| im.camera_center()).collect();

        // Spatial index over finite positions, needed for geometric normals and
        // spacing-relative extent.
        let need_spatial = matches!(normal, PatchNormal::Geometric { .. })
            || matches!(extent, PatchExtent::RelativeToSpacing(_));
        let cloud = if need_spatial && !finite.is_empty() {
            let mut flat = Vec::with_capacity(finite.len() * 3);
            for &i in &finite {
                let p = recon.points[i].position;
                flat.extend_from_slice(&[p.x, p.y, p.z]);
            }
            Some(PointCloud::<f64, 3>::new(&flat, finite.len()))
        } else {
            None
        };

        let spacing_half = if let PatchExtent::RelativeToSpacing(factor) = extent {
            let mut d: Vec<f64> = cloud
                .as_ref()
                .map(|c| c.nearest_neighbor_distances())
                .unwrap_or_default()
                .into_iter()
                .filter(|x| x.is_finite())
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if d.is_empty() { 1.0 } else { d[d.len() / 2] };
            median * factor
        } else {
            0.0
        };

        let geo_k = match normal {
            PatchNormal::Geometric { k_neighbors } => k_neighbors,
            _ => 0,
        };
        let geo_neighbors = if geo_k > 0 {
            cloud.as_ref().map(|c| c.self_nearest_k(geo_k))
        } else {
            None
        };

        // FeatureSize: per-finite-point world half-size from the observing
        // keypoints' scales, read from the workspace `.sift` files. A point with
        // no readable scale in any view is an error.
        //
        // A σ-pixel keypoint subtends an angle ≈ σ/f, which at ray-distance
        // d = ‖X − C‖ from the camera spans ≈ σ·d/f world units. Using the ray
        // distance d (always positive) rather than the optical-axis depth z makes
        // this camera-agnostic: a fisheye (FoV > 180°) sees points past 90° off
        // axis at z ≤ 0, where the old pinhole `σ·z/f` (gated on z > 0) could not
        // size them. On-axis d = z, so this reduces to the pinhole formula.
        let feature_half: Vec<f64> = if let PatchExtent::FeatureSize { factor, across } = extent {
            let img_scales: Vec<Option<Vec<f64>>> = (0..recon.images.len())
                .map(|i| read_image_scales(recon, i))
                .collect();
            // Feature scales index `.sift` files, so this sizing applies to
            // sift_files reconstructions; an embedded_patches recon has no scales
            // and falls through to the "no readable scale" error below.
            let feature_indexes = recon.feature_indexes();
            let mut halves = vec![f64::NAN; finite.len()];
            for (fi, &p) in finite.iter().enumerate() {
                let center = recon.points[p].position;
                let start = recon.observation_offsets[p];
                let obs = &recon.tracks[start..recon.observation_offsets[p + 1]];
                let mut sizes: Vec<f64> = Vec::new();
                for (k, o) in obs.iter().enumerate() {
                    let im = &recon.images[o.image_index as usize];
                    let Some(feature_index) = feature_indexes.map(|f| f[start + k]) else {
                        continue;
                    };
                    if let Some(Some(scales)) = img_scales.get(o.image_index as usize) {
                        if let Some(&sigma) = scales.get(feature_index as usize) {
                            let d =
                                (im.quaternion_wxyz * center.coords + im.translation_xyz).norm();
                            if d > 1e-6 {
                                let (fx, _) =
                                    recon.cameras[im.camera_index as usize].focal_lengths();
                                sizes.push(sigma * d / fx);
                            }
                        }
                    }
                }
                if sizes.is_empty() {
                    return Err(PatchCloudError::MissingFeatureScale { point_id: p as u32 });
                }
                halves[fi] = factor * reduce(&mut sizes, across);
            }
            halves
        } else {
            Vec::new()
        };

        let mut patches = Vec::with_capacity(finite.len());
        let mut point_ids = Vec::with_capacity(finite.len());

        for (fi, &p) in finite.iter().enumerate() {
            let pt = &recon.points[p];
            let center = pt.position;
            let obs = &recon.tracks[recon.observation_offsets[p]..recon.observation_offsets[p + 1]];
            let view_centers: Vec<Point3<f64>> = obs
                .iter()
                .map(|o| cam_centers[o.image_index as usize])
                .collect();
            let mean_view = mean_viewing_normal(&center, &view_centers);

            let n = match normal {
                PatchNormal::Stored => {
                    let en = pt.normal;
                    let v = Vector3::new(en.x as f64, en.y as f64, en.z as f64);
                    if v.norm() > 1e-6 {
                        v.normalize()
                    } else {
                        mean_view
                    }
                }
                PatchNormal::MeanViewing => mean_view,
                PatchNormal::Geometric { .. } => {
                    let mut pts = vec![center];
                    if let Some(neigh) = &geo_neighbors {
                        for j in 0..geo_k {
                            let idx = neigh[fi * geo_k + j];
                            if idx != u32::MAX {
                                pts.push(recon.points[finite[idx as usize]].position);
                            }
                        }
                    }
                    let g = pca_plane_normal(&pts);
                    if g.dot(&mean_view) < 0.0 {
                        -g
                    } else {
                        g
                    }
                }
            };

            let up = match obs.first() {
                Some(o0) => {
                    recon.images[o0.image_index as usize]
                        .quaternion_wxyz
                        .inverse()
                        * Vector3::new(0.0, -1.0, 0.0)
                }
                None => Vector3::y(),
            };

            let half = match extent {
                PatchExtent::Fixed(w) => w,
                PatchExtent::RelativeToSpacing(_) => spacing_half,
                PatchExtent::FeatureSize { .. } => feature_half[fi],
                PatchExtent::PixelRadius { radius_px, across } => {
                    // World half-size that projects to `radius_px` px in each
                    // observing view, reduced across views.
                    if obs.is_empty() {
                        radius_px
                    } else {
                        let mut hs: Vec<f64> = obs
                            .iter()
                            .map(|o| {
                                let im = &recon.images[o.image_index as usize];
                                let p_cam = im.quaternion_wxyz * center.coords + im.translation_xyz;
                                let depth = p_cam.z.abs().max(1e-6);
                                let (fx, _) =
                                    recon.cameras[im.camera_index as usize].focal_lengths();
                                radius_px * depth / fx
                            })
                            .collect();
                        reduce(&mut hs, across)
                    }
                }
            };

            patches.push(OrientedPatch::from_center_normal(
                center,
                n,
                up,
                [half, half],
            ));
            point_ids.push(p as u32);
        }

        Ok(PatchCloud { patches, point_ids })
    }

    /// Serialize to the per-point in-plane half-extent vector arrays
    /// (`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`) stored beside the other
    /// `points3d/` arrays.
    ///
    /// The arrays are **per 3D point** (parallel to the points arrays), so the
    /// patches are scattered into `point_count` rows by their `point_ids`; every
    /// row this cloud has no patch for is left as a zero row (a row is "present"
    /// iff its `u` is non-zero). The center is not stored (it is the point's own
    /// position — a Euclidean point for a finite point, a direction for a point
    /// at infinity); each unit axis and its half-extent are folded into one
    /// half-extent vector (`u = u_axis · half_extent[0]`). Bitmaps are not
    /// produced here.
    pub fn to_halfvec_arrays(&self, point_count: usize) -> (Array2<f32>, Array2<f32>) {
        let mut u_xyz = Array2::<f32>::zeros((point_count, 3));
        let mut v_xyz = Array2::<f32>::zeros((point_count, 3));
        for (k, patch) in self.patches.iter().enumerate() {
            let i = self.point_ids.get(k).copied().unwrap_or(k as u32) as usize;
            if i >= point_count {
                continue;
            }
            let u = patch.u_axis * patch.half_extent[0];
            let v = patch.v_axis * patch.half_extent[1];
            u_xyz[[i, 0]] = u.x as f32;
            u_xyz[[i, 1]] = u.y as f32;
            u_xyz[[i, 2]] = u.z as f32;
            v_xyz[[i, 0]] = v.x as f32;
            v_xyz[[i, 1]] = v.y as f32;
            v_xyz[[i, 2]] = v.z as f32;
        }
        (u_xyz, v_xyz)
    }

    /// Reconstruct a patch cloud from the per-point half-extent vector arrays,
    /// keeping only the present rows (non-zero `u`) and recording each one's
    /// point index in `point_ids`. `centers[i]` supplies the patch center for
    /// point `i` (its position). The half-extent vectors are split back into a
    /// unit axis and a half-size.
    pub fn from_halfvec_arrays(
        half_u_xyz: &Array2<f32>,
        half_v_xyz: &Array2<f32>,
        centers: &[Point3<f64>],
    ) -> Self {
        let mut patches = Vec::new();
        let mut point_ids = Vec::new();
        for i in 0..half_u_xyz.shape()[0] {
            let u = Vector3::new(
                half_u_xyz[[i, 0]] as f64,
                half_u_xyz[[i, 1]] as f64,
                half_u_xyz[[i, 2]] as f64,
            );
            // A zero `u` marks a point with no patch (e.g. a point at infinity).
            let hu = u.norm();
            if hu <= 1e-12 {
                continue;
            }
            let v = Vector3::new(
                half_v_xyz[[i, 0]] as f64,
                half_v_xyz[[i, 1]] as f64,
                half_v_xyz[[i, 2]] as f64,
            );
            let hv = v.norm();
            let u_axis = u / hu;
            let v_axis = if hv > 1e-12 { v / hv } else { v };
            let center = centers.get(i).copied().unwrap_or_else(Point3::origin);
            patches.push(OrientedPatch::new(center, u_axis, v_axis, [hu, hv]));
            point_ids.push(i as u32);
        }
        PatchCloud { patches, point_ids }
    }
}

/// How to choose each patch's surface normal in [`PatchCloud::from_reconstruction`].
#[derive(Debug, Clone, Copy)]
pub enum PatchNormal {
    /// Use the reconstruction's stored estimated normal (whatever is in the
    /// `.sfmr`, not necessarily the mean viewing direction). Falls back to the
    /// mean viewing direction if the stored value is zero/degenerate.
    Stored,
    /// Normalized mean of the unit directions from the point to each observing
    /// camera centre.
    MeanViewing,
    /// Local plane fit (PCA) over the `k_neighbors` nearest 3D points, oriented
    /// toward the mean viewing direction.
    Geometric { k_neighbors: usize },
}

/// How to choose each patch's world-space half-size in
/// [`PatchCloud::from_reconstruction`].
#[derive(Debug, Clone, Copy)]
pub enum PatchExtent {
    /// Same world half-size for every patch.
    Fixed(f64),
    /// `factor` × the median nearest-neighbor spacing of the point cloud.
    RelativeToSpacing(f64),
    /// Back-project `radius_px` pixels to a world half-size in each observing
    /// view, then reduce across views by `across`. `Min` (the default choice)
    /// keeps the patch within the pixel budget in *every* view — it is sized to
    /// `radius_px` in the view where the point appears largest and smaller in the
    /// rest; `Max` is sized to the smallest-appearing view, so the patch can be
    /// much larger in close views.
    PixelRadius { radius_px: f64, across: ViewReduce },
    /// `factor` × the projected world feature size: each observation's keypoint
    /// scale `sigma_i` back-projected to world (`sigma_i · z_i / f_i`), reduced
    /// across views by `across` (`Median` recommended — the scales are
    /// consistent across views). Reads the workspace `.sift` files; a point whose
    /// scale can't be read in any view yields a
    /// [`PatchCloudError::MissingFeatureScale`].
    FeatureSize { factor: f64, across: ViewReduce },
}

impl Default for PatchExtent {
    /// `FeatureSize { factor: 5.0, across: Median }` — size each patch from the
    /// observing keypoints' scales (the default sizing policy).
    fn default() -> Self {
        PatchExtent::FeatureSize {
            factor: 5.0,
            across: ViewReduce::Median,
        }
    }
}

/// Per-feature keypoint scales (column-0 norm of the affine shape) read from an
/// image's `.sift` file, or `None` if it cannot be read.
fn read_image_scales(recon: &SfmrReconstruction, image_index: usize) -> Option<Vec<f64>> {
    let read_count = *recon.max_track_feature_index.get(image_index)? as usize + 1;
    let path = recon.sift_path_for_image(image_index);
    let data = sift_format::read_sift_partial(&path, read_count).ok()?;
    let aff = &data.affine_shapes;
    Some(
        (0..aff.shape()[0])
            .map(|i| {
                let a00 = aff[[i, 0, 0]] as f64;
                let a10 = aff[[i, 1, 0]] as f64;
                (a00 * a00 + a10 * a10).sqrt()
            })
            .collect(),
    )
}

/// How to reduce a per-view quantity to one value across a point's observing
/// views.
#[derive(Debug, Clone, Copy)]
pub enum ViewReduce {
    Min,
    Max,
    Median,
    Mean,
}

fn reduce(values: &mut [f64], how: ViewReduce) -> f64 {
    match how {
        ViewReduce::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        ViewReduce::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ViewReduce::Mean => values.iter().sum::<f64>() / values.len().max(1) as f64,
        ViewReduce::Median => {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = values.len();
            if n == 0 {
                f64::NAN
            } else if n % 2 == 1 {
                values[n / 2]
            } else {
                0.5 * (values[n / 2 - 1] + values[n / 2])
            }
        }
    }
}

/// Normalized mean of the unit directions from `center` to each camera centre.
pub fn mean_viewing_normal(center: &Point3<f64>, camera_centers: &[Point3<f64>]) -> Vector3<f64> {
    let mut acc = Vector3::zeros();
    for c in camera_centers {
        let d = c - center;
        let n = d.norm();
        if n > 1e-12 {
            acc += d / n;
        }
    }
    normalize_or(acc, Vector3::z())
}

/// Plane normal (smallest-variance PCA direction) of a set of points.
pub fn pca_plane_normal(points: &[Point3<f64>]) -> Vector3<f64> {
    let n = points.len();
    if n < 3 {
        return Vector3::z();
    }
    let mut centroid = Vector3::zeros();
    for p in points {
        centroid += p.coords;
    }
    centroid /= n as f64;
    let mut cov = Matrix3::zeros();
    for p in points {
        let d = p.coords - centroid;
        cov += d * d.transpose();
    }
    let eig = cov.symmetric_eigen();
    let mut min_i = 0;
    for i in 1..3 {
        if eig.eigenvalues[i] < eig.eigenvalues[min_i] {
            min_i = i;
        }
    }
    eig.eigenvectors.column(min_i).into_owned().normalize()
}

#[cfg(test)]
mod tests;
