// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Oriented 3D patches (surfels) and patch clouds.
//!
//! See `specs/core/patch-cloud.md`. An [`OrientedPatch`] is a small planar
//! surface element in world space; [`crate::warp_map::WarpMap::from_patch`]
//! projects a camera's image onto one to render its canonical appearance.

use nalgebra::{Matrix3, Point3, Vector3};

use crate::reconstruction::SfmrReconstruction;
use crate::rigid_transform::RigidTransform;
use crate::spatial::PointCloud;

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
    pub fn from_reconstruction(
        recon: &SfmrReconstruction,
        normal: PatchNormal,
        extent: PatchExtent,
    ) -> Self {
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
                    let en = pt.estimated_normal;
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

        PatchCloud { patches, point_ids }
    }
}

/// How to choose each patch's surface normal in [`PatchCloud::from_reconstruction`].
#[derive(Debug, Clone, Copy)]
pub enum PatchNormal {
    /// Use the reconstruction's stored normal (the mean viewing direction).
    /// Falls back to recomputing it if the stored value is zero/degenerate.
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
            values[values.len() / 2]
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
