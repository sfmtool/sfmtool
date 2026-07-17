// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Oriented 3D patches (surfels) and patch clouds.
//!
//! See `specs/core/patch-cloud.md`. An [`OrientedPatch`] is a small planar
//! surface element in world space; [`crate::camera::WarpMap::from_patch`]
//! projects a camera's image onto one to render its canonical appearance.

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use ndarray::Array2;

use crate::geometry::RigidTransform;
use crate::reconstruction::SfmrReconstruction;
use crate::spatial::PointCloud;

/// Errors from [`PatchCloud::from_reconstruction`].
#[derive(Debug)]
pub enum PatchCloudError {
    /// [`PatchExtent::FeatureSize`] could not derive a world size for point
    /// `point_index` because none of its observations yielded a usable keypoint
    /// scale. The counts break the `observations` down by cause so the message
    /// can name the actual failure:
    ///
    /// - `unreadable_scale` — the observation's `.sift` keypoint scale could not
    ///   be read (missing/stale `.sift`, or the observation carries no feature
    ///   index). This is an I/O problem.
    /// - `coincident_with_camera` — the observation's viewing distance
    ///   `d = ‖X − C‖` is ~0, i.e. the point sits on top of the camera centre.
    ///   A pixel scale `σ` maps to a world size `σ·d/f`, which vanishes at
    ///   `d ≈ 0`, so the size is undefined. This is a degenerate *reconstruction*
    ///   (e.g. a run of frames whose poses collapsed to one point), not an I/O
    ///   problem — the scales are perfectly readable. Only the finite-point path
    ///   scales by `d`, so this is always `0` for a point at infinity.
    MissingFeatureScale {
        point_index: u32,
        observations: usize,
        unreadable_scale: usize,
        coincident_with_camera: usize,
    },
}

impl std::fmt::Display for PatchCloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatchCloudError::MissingFeatureScale {
                point_index,
                observations,
                unreadable_scale,
                coincident_with_camera,
            } => write!(
                f,
                "cannot size point {point_index}'s patch from FeatureSize: none of its \
                 {observations} observation(s) gave a usable keypoint scale \
                 ({coincident_with_camera} coincident with the camera centre / zero viewing \
                 distance, {unreadable_scale} with an unreadable .sift scale). A point \
                 coincident with its cameras is a degenerate reconstruction artifact (its \
                 frames' poses have collapsed onto the point); an unreadable scale usually \
                 means the .sift files are missing or stale."
            ),
        }
    }
}

impl std::error::Error for PatchCloudError {}

/// An oriented planar patch (surfel) in world space.
///
/// The plane is spanned by orthonormal in-plane axes `u_axis` and `v_axis`; the
/// frame is right-handed with the outward normal `u_axis × v_axis`. The patch
/// covers the world points
/// `center + s·half_extent[0]·u_axis + t·half_extent[1]·v_axis` for
/// `(s, t) ∈ [-1, 1]²`.
///
/// The frame is right-handed (`u × v` is the geometric outward normal), but the
/// image raster increases its row index *downward*, so [`WarpMap::from_patch`]
/// steps `col` along `+u_axis` and `row` along **`−v_axis`**. That reversal is
/// what makes the baked bitmap un-mirrored — the same chirality the camera sees:
/// walking `+v` for the raster row (as if `v` were "screen-down") would render
/// the back face, i.e. a mirror image. So `v_axis` points "up" in the patch
/// plane while patch pixel rows count downward from it.
///
/// [`WarpMap::from_patch`]: crate::camera::WarpMap::from_patch
///
/// `w` is the homogeneous weight of the patch's anchor, mirroring the source 3D
/// point: `1.0` for a finite point (`center` is a Euclidean position) and `0.0`
/// for a point at infinity (`center` is a direction `d`, the patch is tangent to
/// the unit sphere around `d`, and corners are themselves directions). Rendering
/// and visibility branch on it; see [`Self::corner_homogeneous`].
#[derive(Debug, Clone)]
pub struct OrientedPatch {
    pub center: Point3<f64>,
    /// Unit, in-plane.
    pub u_axis: Vector3<f64>,
    /// Unit, in-plane, perpendicular to `u_axis`.
    pub v_axis: Vector3<f64>,
    /// World-space half-size along `(u, v)`.
    pub half_extent: [f64; 2],
    /// Homogeneous weight of the anchor: `1.0` finite, `0.0` at infinity.
    pub w: f64,
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
            w: 1.0,
        }
    }

    /// Outward normal (`u_axis × v_axis`, normalized). The frame is right-handed;
    /// see the type docs for how the raster reverses `v` (not the normal) to
    /// render un-mirrored.
    pub fn normal(&self) -> Vector3<f64> {
        self.u_axis.cross(&self.v_axis).normalize()
    }

    /// World point for a normalized patch coordinate `(s, t) ∈ [-1, 1]²`.
    ///
    /// This is the **Euclidean** corner and is only meaningful for a finite
    /// patch (`w == 1`); for the homogeneous corner that also covers points at
    /// infinity, use [`Self::corner_homogeneous`].
    pub fn to_world(&self, s: f64, t: f64) -> Point3<f64> {
        self.center
            + self.u_axis * (s * self.half_extent[0])
            + self.v_axis * (t * self.half_extent[1])
    }

    /// The patch corner for `(s, t) ∈ [-1, 1]²` as a homogeneous world point
    /// `(xyz, w)`. For a finite patch (`w == 1`) `xyz` is the Euclidean point
    /// `center + s·u + t·v`; for a point at infinity (`w == 0`) `center` is a
    /// direction `d` and the corner `d + s·u + t·v` is again a direction. Project
    /// it by transforming with [`RigidTransform::transform_point_homogeneous`]
    /// (which applies the translation only when `w == 1`) and calling
    /// `ray_to_pixel`.
    pub fn corner_homogeneous(&self, s: f64, t: f64) -> (Vector3<f64>, f64) {
        let xyz = self.center.coords
            + self.u_axis * (s * self.half_extent[0])
            + self.v_axis * (t * self.half_extent[1]);
        (xyz, self.w)
    }

    /// Build from a center, a normal, and an `up_hint` that pins the in-plane
    /// rotation about the normal. The result is a **finite** patch (`w == 1`).
    ///
    /// `v_axis` is `up_hint` projected onto the plane and normalized (the "up"
    /// axis — [`WarpMap::from_patch`] steps the raster row along `−v_axis`, so
    /// `up_hint` maps to the top of the rendered patch); if `up_hint` is
    /// (near-)parallel to the normal, an arbitrary in-plane axis is chosen
    /// instead. `u_axis = v_axis × normal` is the in-plane "right" axis (the
    /// raster steps `col` along `+u_axis`), and `u_axis × v_axis` recovers the
    /// (normalized) normal — a right-handed frame that renders un-mirrored and
    /// un-rotated (see the type docs).
    ///
    /// [`WarpMap::from_patch`]: crate::camera::WarpMap::from_patch
    pub fn from_center_normal(
        center: Point3<f64>,
        normal: Vector3<f64>,
        up_hint: Vector3<f64>,
        half_extent: [f64; 2],
    ) -> Self {
        let n = normalize_or(normal, Vector3::z());
        let proj = up_hint - n * up_hint.dot(&n);
        let v = if proj.norm() > 1e-9 {
            proj.normalize()
        } else {
            any_orthonormal(&n)
        };
        // "Right" axis; `u × v = n` keeps the outward normal, and the renderer
        // draws `col` along `+u` / `row` along `−v`, so `up_hint` → tile up.
        let u = v.cross(&n);
        Self {
            center,
            u_axis: u,
            v_axis: v,
            half_extent,
            w: 1.0,
        }
    }

    /// Build the tangent-sphere frame for a **point at infinity** with direction
    /// `d` (`w == 0`): outward normal `normalize(-d)`, `u, v ⊥ d` with `u × v`
    /// along `-d`, the in-plane rotation pinned by `up_hint`. `center` stores the
    /// direction `d` itself. Per the `.sfmr` format's infinity-patch convention
    /// (see `specs/formats/sfmr-file-format.md`).
    pub fn from_infinity_direction(
        direction: Point3<f64>,
        up_hint: Vector3<f64>,
        half_extent: [f64; 2],
    ) -> Self {
        let mut p = Self::from_center_normal(direction, -direction.coords, up_hint, half_extent);
        p.w = 0.0;
        p
    }

    /// Whether the `cam_from_world` camera looks at the patch's front face — its
    /// outward normal points toward the camera centre.
    ///
    /// A point at infinity (`w == 0`) is always front-facing: its outward normal
    /// is `normalize(-d)` and every viewing ray is parallel to `d`, so the front
    /// face faces every observer. Whether the camera actually looks toward `+d`
    /// (cheirality) is enforced by the projection (`ray_to_pixel`), not here.
    pub fn is_front_facing(&self, cam_from_world: &RigidTransform) -> bool {
        if self.w == 0.0 {
            return true;
        }
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
    pub point_indexes: Vec<u32>,
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

    /// Build a patch cloud from a reconstruction's 3D points.
    ///
    /// One patch per finite point: center = position, in-plane up from the
    /// first observing camera, normal and half-size per the given policies.
    /// `point_indexes` records the source point index for each patch.
    ///
    /// When `exclude_points_at_infinity` is `false`, each point at infinity
    /// (`w = 0`) additionally gets a tangent-sphere frame (`w = 0` patch) around
    /// its direction `d` — outward normal `normalize(-d)`, `u, v ⊥ d`, with an
    /// angular half-size from `extent` (the distance-free form of each policy);
    /// see the format's infinity-patch convention. Every patch operation handles
    /// these, so `false` is the default at the binding layer. Pass `true` to emit
    /// finite points only — needed by an operation that scatters per-point results
    /// back and must leave infinity points untouched (e.g. normal refinement's
    /// normal write-back), or that wants the historical finite-only behavior.
    ///
    /// Errors with [`PatchCloudError::MissingFeatureScale`] under
    /// [`PatchExtent::FeatureSize`] if a point (finite, or at infinity when
    /// included) has no observation that yields a usable size — either its
    /// keypoint scale is unreadable in every view, or (finite points only) it
    /// coincides with every observing camera centre so the distance-scaled size
    /// vanishes. The error carries the per-cause breakdown.
    pub fn from_reconstruction(
        recon: &SfmrReconstruction,
        normal: PatchNormal,
        extent: PatchExtent,
        exclude_points_at_infinity: bool,
    ) -> Result<Self, PatchCloudError> {
        // Per-point geometry.
        let positions: Vec<Point3<f64>> = recon.points.iter().map(|p| p.position).collect();
        let weights: Vec<f64> = recon.points.iter().map(|p| p.w).collect();
        let stored_normals: Vec<Vector3<f64>> = recon
            .points
            .iter()
            .map(|p| Vector3::new(p.normal.x as f64, p.normal.y as f64, p.normal.z as f64))
            .collect();

        // Per-observation image index (flat, grouped by point — `recon.tracks` is
        // sorted by point then image, so `observation_offsets` groups it).
        let obs_images: Vec<u32> = recon.tracks.iter().map(|o| o.image_index).collect();

        // Per-image pose + focal length (fx of the image's camera).
        let cam_quats: Vec<UnitQuaternion<f64>> =
            recon.images.iter().map(|im| im.quaternion_wxyz).collect();
        let cam_translations: Vec<Vector3<f64>> =
            recon.images.iter().map(|im| im.translation_xyz).collect();
        let cam_focals: Vec<f64> = recon
            .images
            .iter()
            .map(|im| recon.cameras[im.camera_index as usize].focal_lengths().0)
            .collect();

        // FeatureSize is the one policy that reads the workspace `.sift` files:
        // resolve every observation's keypoint scale (σ = column-0 norm of the
        // affine shape) into a per-observation `Option` (`None` = unreadable),
        // exactly as the historical inline lookup did — the `.sift` files are read
        // once per image and shared across the finite and infinity paths. Feature
        // scales index `.sift` files, so FeatureSize applies to sift_files
        // reconstructions; an embedded_patches recon has no scales, so every
        // observation resolves to `None` and the "no readable scale" error fires.
        //
        // A σ-pixel keypoint subtends an angle ≈ σ/f, which at ray-distance
        // d = ‖X − C‖ from the camera spans ≈ σ·d/f world units. Using the ray
        // distance d (always positive) rather than the optical-axis depth z makes
        // this camera-agnostic: a fisheye (FoV > 180°) sees points past 90° off
        // axis at z ≤ 0, where the old pinhole `σ·z/f` (gated on z > 0) could not
        // size them. On-axis d = z, so this reduces to the pinhole formula.
        let obs_scales: Vec<Option<f64>> = if matches!(extent, PatchExtent::FeatureSize { .. }) {
            let feature_indexes = recon.feature_indexes();
            let img_scales: Vec<Option<Vec<f64>>> = (0..recon.images.len())
                .map(|i| read_image_scales(recon, i))
                .collect();
            (0..recon.tracks.len())
                .map(|j| {
                    let img = recon.tracks[j].image_index as usize;
                    feature_indexes.map(|f| f[j]).and_then(|feature_index| {
                        img_scales
                            .get(img)
                            .and_then(|s| s.as_ref())
                            .and_then(|scales| scales.get(feature_index as usize).copied())
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        let scene = PatchScene {
            positions: &positions,
            weights: &weights,
            stored_normals: &stored_normals,
            obs_offsets: &recon.observation_offsets,
            obs_images: &obs_images,
            obs_scales: &obs_scales,
            cam_quats: &cam_quats,
            cam_translations: &cam_translations,
            cam_focals: &cam_focals,
        };
        build_patch_cloud(&scene, normal, extent, exclude_points_at_infinity)
    }

    /// Build a patch cloud from in-memory arrays instead of a reconstruction.
    ///
    /// The array counterpart of [`Self::from_reconstruction`]: the same normal and
    /// extent policies, up-hint (first observing camera), infinity tangent-sphere
    /// frames, sizing, and error taxonomy — one shared routine, fed by
    /// caller-supplied geometry rather than a solved `SfmrReconstruction`.
    ///
    /// - `positions` / `weights` are per point (`P`): `weights[i] == 0.0` marks a
    ///   point at infinity (its `positions[i]` is a direction).
    /// - `obs_offsets` (`P + 1`, prefix sum) groups the flat per-observation
    ///   `obs_images` (`M`) by point; point `p`'s observations are
    ///   `obs_images[obs_offsets[p]..obs_offsets[p + 1]]`.
    /// - `cam_quats` / `cam_translations` / `cam_focals` (`N`) give each image's
    ///   `cam_from_world` rotation, translation, and focal length (fx).
    /// - `stored_normals` (`P`) is read only by [`PatchNormal::Stored`]; a
    ///   zero/missing row falls back to the mean viewing direction.
    /// - `obs_scales` (`M`, `NaN` = unreadable) supplies the keypoint scale that
    ///   [`PatchExtent::FeatureSize`] otherwise reads from `.sift`; a `NaN` entry
    ///   counts as an unreadable scale, so the [`PatchCloudError::MissingFeatureScale`]
    ///   error and its per-cause breakdown are unchanged.
    ///
    /// The resulting patches' `point_indexes` are row indexes into `positions`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_tracks(
        positions: &[Point3<f64>],
        weights: &[f64],
        stored_normals: Option<&[Vector3<f64>]>,
        obs_offsets: &[usize],
        obs_images: &[u32],
        obs_scales: Option<&[f64]>,
        cam_quats: &[UnitQuaternion<f64>],
        cam_translations: &[Vector3<f64>],
        cam_focals: &[f64],
        normal: PatchNormal,
        extent: PatchExtent,
        exclude_points_at_infinity: bool,
    ) -> Result<Self, PatchCloudError> {
        // A NaN scale counts as an unreadable scale, mirroring the `.sift` path's
        // "missing scale" so the MissingFeatureScale taxonomy is unchanged.
        let obs_scales_opt: Vec<Option<f64>> = obs_scales
            .map(|s| {
                s.iter()
                    .map(|&v| if v.is_nan() { None } else { Some(v) })
                    .collect()
            })
            .unwrap_or_default();
        let scene = PatchScene {
            positions,
            weights,
            stored_normals: stored_normals.unwrap_or(&[]),
            obs_offsets,
            obs_images,
            obs_scales: &obs_scales_opt,
            cam_quats,
            cam_translations,
            cam_focals,
        };
        build_patch_cloud(&scene, normal, extent, exclude_points_at_infinity)
    }

    /// Serialize to the per-point in-plane half-extent vector arrays
    /// (`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`) stored beside the other
    /// `points3d/` arrays.
    ///
    /// The arrays are **per 3D point** (parallel to the points arrays), so the
    /// patches are scattered into `point_count` rows by their `point_indexes`; every
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
            let i = self.point_indexes.get(k).copied().unwrap_or(k as u32) as usize;
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
    /// point index in `point_indexes`. `centers[i]` supplies the patch center for
    /// point `i` (its position). The half-extent vectors are split back into a
    /// unit axis and a half-size.
    ///
    /// Every patch is built **finite** (`w = 1`): the half-vector arrays don't
    /// encode the homogeneous weight, which lives in the points' `w`. A caller
    /// that knows which points are at infinity (e.g. the `recon.patches` getter)
    /// sets `patch.w = 0` on those rows afterward.
    pub fn from_halfvec_arrays(
        half_u_xyz: &Array2<f32>,
        half_v_xyz: &Array2<f32>,
        centers: &[Point3<f64>],
    ) -> Self {
        let mut patches = Vec::new();
        let mut point_indexes = Vec::new();
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
            point_indexes.push(i as u32);
        }
        PatchCloud {
            patches,
            point_indexes,
        }
    }
}

/// The reconstruction-independent inputs shared by
/// [`PatchCloud::from_reconstruction`] and [`PatchCloud::from_tracks`]: per-point
/// geometry, grouped observations, per-image poses/focal lengths, and a
/// per-observation keypoint-scale source. One routine ([`build_patch_cloud`])
/// implements the sizing policies, up-hint, infinity frames, and error taxonomy
/// over it, so both entry points share a single implementation.
struct PatchScene<'a> {
    /// Per-point position (finite) or direction (at infinity). `P` entries.
    positions: &'a [Point3<f64>],
    /// Per-point homogeneous weight (`1.0` finite, `0.0` at infinity). `P` entries.
    weights: &'a [f64],
    /// Per-point stored normal (read only by [`PatchNormal::Stored`]; a zero or
    /// missing row falls back to the mean viewing direction). `P` entries or empty.
    stored_normals: &'a [Vector3<f64>],
    /// Prefix-sum offsets into the flat observation arrays. `P + 1` entries.
    obs_offsets: &'a [usize],
    /// Per-observation image index (flat, grouped by point). `M` entries.
    obs_images: &'a [u32],
    /// Per-observation keypoint scale `σ` (`None` = unreadable), or empty when the
    /// extent policy reads no scales. `M` entries or empty.
    obs_scales: &'a [Option<f64>],
    /// Per-image `cam_from_world` rotation. `N` entries.
    cam_quats: &'a [UnitQuaternion<f64>],
    /// Per-image `cam_from_world` translation. `N` entries.
    cam_translations: &'a [Vector3<f64>],
    /// Per-image focal length (fx). `N` entries.
    cam_focals: &'a [f64],
}

impl PatchScene<'_> {
    /// In-plane "up" hint: the first observing camera's up axis (canonical camera
    /// `+y` — image up) rotated into world, or world `+y` when the point has no
    /// observation. Pins the in-plane rotation identically for finite and infinity
    /// patches.
    fn up_hint(&self, first_image_index: Option<u32>) -> Vector3<f64> {
        match first_image_index {
            Some(idx) => self.cam_quats[idx as usize].inverse() * Vector3::new(0.0, 1.0, 0.0),
            None => Vector3::y(),
        }
    }

    /// The stored normal for point `p`, or the zero vector when none is supplied.
    fn stored_normal(&self, p: usize) -> Vector3<f64> {
        self.stored_normals
            .get(p)
            .copied()
            .unwrap_or_else(Vector3::zeros)
    }
}

/// The shared body of [`PatchCloud::from_reconstruction`] /
/// [`PatchCloud::from_tracks`]: one finite patch per finite point (center =
/// position, in-plane up from the first observing camera, normal and half-size
/// per the given policies) plus, when `exclude_points_at_infinity` is `false`, one
/// tangent-sphere frame per point at infinity.
fn build_patch_cloud(
    scene: &PatchScene<'_>,
    normal: PatchNormal,
    extent: PatchExtent,
    exclude_points_at_infinity: bool,
) -> Result<PatchCloud, PatchCloudError> {
    let n_points = scene.positions.len();
    let finite: Vec<usize> = (0..n_points).filter(|&i| scene.weights[i] != 0.0).collect();

    // Camera centres in world (`C = -R^T · t`).
    let cam_centers: Vec<Point3<f64>> = (0..scene.cam_quats.len())
        .map(|i| {
            let r = scene.cam_quats[i].to_rotation_matrix();
            Point3::from(-(r.transpose() * scene.cam_translations[i]))
        })
        .collect();

    // Spatial index over finite positions, needed for geometric normals and
    // spacing-relative extent.
    let need_spatial = matches!(normal, PatchNormal::Geometric { .. })
        || matches!(extent, PatchExtent::RelativeToSpacing(_));
    let cloud = if need_spatial && !finite.is_empty() {
        let mut flat = Vec::with_capacity(finite.len() * 3);
        for &i in &finite {
            let p = scene.positions[i];
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

    // FeatureSize: per-finite-point world half-size from the observing keypoints'
    // scales. A point with no readable scale in any view — or coincident with
    // every camera centre (`d ≈ 0`) — is an error.
    let feature_half: Vec<f64> = if let PatchExtent::FeatureSize { factor, across } = extent {
        let mut halves = vec![f64::NAN; finite.len()];
        for (fi, &p) in finite.iter().enumerate() {
            let center = scene.positions[p];
            let start = scene.obs_offsets[p];
            let end = scene.obs_offsets[p + 1];
            let mut sizes: Vec<f64> = Vec::new();
            // Per-cause tallies so a failure can name why every observation was
            // rejected (unreadable scale vs. zero viewing distance).
            let mut unreadable_scale = 0usize;
            let mut coincident_with_camera = 0usize;
            for obs in start..end {
                let img = scene.obs_images[obs] as usize;
                let Some(sigma) = scene.obs_scales.get(obs).copied().flatten() else {
                    unreadable_scale += 1;
                    continue;
                };
                let d = (scene.cam_quats[img] * center.coords + scene.cam_translations[img]).norm();
                if d > 1e-6 {
                    sizes.push(sigma * d / scene.cam_focals[img]);
                } else {
                    coincident_with_camera += 1;
                }
            }
            if sizes.is_empty() {
                return Err(PatchCloudError::MissingFeatureScale {
                    point_index: p as u32,
                    observations: end - start,
                    unreadable_scale,
                    coincident_with_camera,
                });
            }
            halves[fi] = factor * reduce(&mut sizes, across);
        }
        halves
    } else {
        Vec::new()
    };

    let mut patches = Vec::with_capacity(finite.len());
    let mut point_indexes = Vec::with_capacity(finite.len());

    for (fi, &p) in finite.iter().enumerate() {
        let center = scene.positions[p];
        let start = scene.obs_offsets[p];
        let end = scene.obs_offsets[p + 1];
        let view_centers: Vec<Point3<f64>> = (start..end)
            .map(|o| cam_centers[scene.obs_images[o] as usize])
            .collect();
        let mean_view = mean_viewing_normal(&center, &view_centers);

        let n = match normal {
            PatchNormal::Stored => {
                let v = scene.stored_normal(p);
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
                            pts.push(scene.positions[finite[idx as usize]]);
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

        let up = scene.up_hint((start < end).then(|| scene.obs_images[start]));

        let half = match extent {
            PatchExtent::Fixed(w) => w,
            PatchExtent::RelativeToSpacing(_) => spacing_half,
            PatchExtent::FeatureSize { .. } => feature_half[fi],
            PatchExtent::PixelRadius { radius_px, across } => {
                // World half-size that projects to `radius_px` px in each observing
                // view, reduced across views.
                if start == end {
                    radius_px
                } else {
                    let mut hs: Vec<f64> = (start..end)
                        .map(|o| {
                            let img = scene.obs_images[o] as usize;
                            let p_cam =
                                scene.cam_quats[img] * center.coords + scene.cam_translations[img];
                            let depth = p_cam.z.abs().max(1e-6);
                            radius_px * depth / scene.cam_focals[img]
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
        point_indexes.push(p as u32);
    }

    let mut cloud_out = PatchCloud {
        patches,
        point_indexes,
    };
    if !exclude_points_at_infinity {
        push_infinity_patches(&mut cloud_out, scene, extent, spacing_half)?;
    }
    Ok(cloud_out)
}

/// Push a tangent-sphere patch for each point at infinity (`w = 0`), the
/// counterpart to the finite patches [`build_patch_cloud`] emits (it calls this
/// when `exclude_points_at_infinity` is `false`).
///
/// Per the `.sfmr` format's infinity-patch convention (see
/// `specs/formats/sfmr-file-format.md`, "Per-point patch frame"): a point at
/// infinity with direction `d` gets a frame tangent to the unit sphere around `d`
/// — `u` and `v` are `⊥ d` and `u × v` points along `-d`, so the implied normal
/// `normalize(u × v)` is `normalize(-d)`. The in-plane rotation is pinned by the
/// first observing camera's up, matching the finite path. The patch's `w` is `0`.
///
/// The angular half-size (the tangent vectors' magnitude) follows `extent`:
/// [`PatchExtent::FeatureSize`] and [`PatchExtent::PixelRadius`] have a natural
/// distance-free angular form (`σ_i / f_i` and `radius_px / f_i` per view, reduced
/// across views — the finite formulas with the ray distance dropped), while
/// [`PatchExtent::Fixed`] and [`PatchExtent::RelativeToSpacing`] reuse their world
/// half-size as the tangent magnitude. Errors with
/// [`PatchCloudError::MissingFeatureScale`] under [`PatchExtent::FeatureSize`] if
/// an infinity point has no readable keypoint scale in any view (its angular size
/// is distance-free, so the coincident-camera cause never applies here).
fn push_infinity_patches(
    cloud: &mut PatchCloud,
    scene: &PatchScene<'_>,
    extent: PatchExtent,
    spacing_half: f64,
) -> Result<(), PatchCloudError> {
    let infinity: Vec<usize> = (0..scene.positions.len())
        .filter(|&i| scene.weights[i] == 0.0)
        .collect();
    if infinity.is_empty() {
        return Ok(());
    }
    cloud.patches.reserve(infinity.len());
    cloud.point_indexes.reserve(infinity.len());
    for &p in &infinity {
        let dir = scene.positions[p];
        let start = scene.obs_offsets[p];
        let end = scene.obs_offsets[p + 1];

        let half = match extent {
            PatchExtent::Fixed(w) => w,
            PatchExtent::RelativeToSpacing(_) => spacing_half,
            PatchExtent::PixelRadius { radius_px, across } => {
                if start == end {
                    radius_px
                } else {
                    let mut angles: Vec<f64> = (start..end)
                        .map(|o| radius_px / scene.cam_focals[scene.obs_images[o] as usize])
                        .collect();
                    reduce(&mut angles, across)
                }
            }
            PatchExtent::FeatureSize { factor, across } => {
                let mut angles: Vec<f64> = Vec::new();
                for obs in start..end {
                    let img = scene.obs_images[obs] as usize;
                    if let Some(sigma) = scene.obs_scales.get(obs).copied().flatten() {
                        angles.push(sigma / scene.cam_focals[img]);
                    }
                }
                if angles.is_empty() {
                    // An infinity patch's angular size is `σ/f` (no viewing
                    // distance), so the only failure cause is an unreadable scale —
                    // `coincident_with_camera` never applies here.
                    return Err(PatchCloudError::MissingFeatureScale {
                        point_index: p as u32,
                        observations: end - start,
                        unreadable_scale: end - start,
                        coincident_with_camera: 0,
                    });
                }
                factor * reduce(&mut angles, across)
            }
        };

        // Implied normal `normalize(-d)`; the in-plane rotation follows the first
        // observing camera's up. `center` is the direction itself.
        let up = scene.up_hint((start < end).then(|| scene.obs_images[start]));
        cloud.patches.push(OrientedPatch::from_infinity_direction(
            dir,
            up,
            [half, half],
        ));
        cloud.point_indexes.push(p as u32);
    }
    Ok(())
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
    /// `FeatureSize { factor: 2.5, across: Median }` — size each patch from the
    /// observing keypoints' scales (the default sizing policy). `factor` is a
    /// half-extent multiplier, so the full patch edge is `5 ×` the projected
    /// feature size.
    fn default() -> Self {
        PatchExtent::FeatureSize {
            factor: 2.5,
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
