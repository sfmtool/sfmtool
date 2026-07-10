// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for sfmtool-core oriented patches.

use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::{OrientedPatch, PatchCloud, PatchExtent, PatchNormal, ViewReduce};
use sfmtool_core::patch::keypoint_localize::{
    localize_patch_cloud_keypoints, KeypointLocalizeParams,
    SearchStrategy as LocalizeSearchStrategy,
};
use sfmtool_core::patch::keypoint_subpixel::{ConsensusRefresh, KeypointSubpixelParams};
use sfmtool_core::patch::localizability::{
    score_localizability_stack, window_weights as localizability_window_weights,
};
use sfmtool_core::patch::normal_refine::{
    refine_patch_cloud_normals, view_indices_from_reconstruction, CacheMode, NormalRefineParams,
    Objective, PatchWindow, ProjectedImage, Sampler,
};
use sfmtool_core::patch::view_selection::{select_patch_cloud_views, ViewSelectParams};

use crate::flow::warp::extract_image_u8;
use crate::geometry::rigid_transform::PyRigidTransform;
use crate::py_progress::ProgressCounter;
use crate::py_sfmr_reconstruction::PySfmrReconstruction;

/// An oriented planar patch (surfel) in world space.
///
/// The plane is spanned by orthonormal in-plane axes ``u_axis`` and ``v_axis``;
/// the frame is right-handed with outward normal ``u_axis × v_axis``. A
/// ``(col, row)`` render steps ``col`` along ``+u_axis`` and ``row`` along
/// ``−v_axis`` (rows count downward), so the front face renders un-mirrored.
/// The patch covers the world points
/// ``center + s·half_extent[0]·u_axis + t·half_extent[1]·v_axis`` for
/// ``(s, t)`` in ``[-1, 1]²``. Pair with :meth:`WarpMap.from_patch` to render its
/// appearance in a camera. See ``specs/core/patch-cloud.md``.
#[pyclass(name = "OrientedPatch", module = "sfmtool")]
pub struct PyOrientedPatch {
    pub(crate) inner: OrientedPatch,
}

#[pymethods]
impl PyOrientedPatch {
    /// Construct from an explicit center, in-plane axes, and per-axis half-extent.
    #[new]
    #[pyo3(signature = (center, u_axis, v_axis, half_extent))]
    fn new(center: [f64; 3], u_axis: [f64; 3], v_axis: [f64; 3], half_extent: [f64; 2]) -> Self {
        Self {
            inner: OrientedPatch::new(
                Point3::new(center[0], center[1], center[2]),
                Vector3::new(u_axis[0], u_axis[1], u_axis[2]),
                Vector3::new(v_axis[0], v_axis[1], v_axis[2]),
                half_extent,
            ),
        }
    }

    /// Build from a center, a normal, and an ``up_hint`` that pins the in-plane
    /// rotation about the normal.
    ///
    /// Args:
    ///     center: Patch center in world coordinates.
    ///     normal: Surface normal (need not be unit length).
    ///     up_hint: The "up" reference direction; projected onto the plane to set
    ///         ``v_axis`` (``u_axis = v_axis × normal`` is the in-plane "right"
    ///         axis). If parallel to the normal an arbitrary axis is used.
    ///     half_extent: World-space half-size along ``(u, v)``.
    #[staticmethod]
    #[pyo3(signature = (center, normal, up_hint, half_extent))]
    fn from_center_normal(
        center: [f64; 3],
        normal: [f64; 3],
        up_hint: [f64; 3],
        half_extent: [f64; 2],
    ) -> Self {
        Self {
            inner: OrientedPatch::from_center_normal(
                Point3::new(center[0], center[1], center[2]),
                Vector3::new(normal[0], normal[1], normal[2]),
                Vector3::new(up_hint[0], up_hint[1], up_hint[2]),
                half_extent,
            ),
        }
    }

    /// Build the tangent-sphere frame for a **point at infinity** with direction
    /// ``direction`` (``w == 0``): outward normal ``normalize(-direction)``,
    /// ``u, v`` perpendicular to the direction, the in-plane rotation pinned by
    /// ``up_hint``. ``center`` stores the direction itself. Pair with
    /// :meth:`WarpMap.from_patch`, which projects the (direction-valued) corners
    /// without applying the camera translation.
    #[staticmethod]
    #[pyo3(signature = (direction, up_hint, half_extent))]
    fn from_infinity_direction(
        direction: [f64; 3],
        up_hint: [f64; 3],
        half_extent: [f64; 2],
    ) -> Self {
        Self {
            inner: OrientedPatch::from_infinity_direction(
                Point3::new(direction[0], direction[1], direction[2]),
                Vector3::new(up_hint[0], up_hint[1], up_hint[2]),
                half_extent,
            ),
        }
    }

    #[getter]
    fn center(&self) -> [f64; 3] {
        let c = self.inner.center;
        [c.x, c.y, c.z]
    }

    #[getter]
    fn u_axis(&self) -> [f64; 3] {
        let v = self.inner.u_axis;
        [v.x, v.y, v.z]
    }

    #[getter]
    fn v_axis(&self) -> [f64; 3] {
        let v = self.inner.v_axis;
        [v.x, v.y, v.z]
    }

    #[getter]
    fn half_extent(&self) -> [f64; 2] {
        self.inner.half_extent
    }

    /// Homogeneous weight of the anchor: ``1.0`` for a finite patch (``center``
    /// is a position), ``0.0`` for a point at infinity (``center`` is a direction
    /// and the patch is tangent to the unit sphere).
    #[getter]
    fn w(&self) -> f64 {
        self.inner.w
    }

    /// Outward normal (``u_axis × v_axis``).
    #[getter]
    fn normal(&self) -> [f64; 3] {
        let n = self.inner.normal();
        [n.x, n.y, n.z]
    }

    /// Whether the ``cam_from_world`` camera looks at the patch's front face.
    fn is_front_facing(&self, cam_from_world: &PyRigidTransform) -> bool {
        self.inner.is_front_facing(&cam_from_world.inner)
    }
}

/// `numpy.median` of a non-empty slice: the middle value for an odd count, the
/// mean of the two central values for an even count. Sorts `v` in place.
fn np_median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Map a window name + sigma to the shared [`PatchWindow`] kernel.
fn parse_patch_window(window: &str, sigma: f64) -> PyResult<PatchWindow> {
    match window {
        "uniform" => Ok(PatchWindow::Uniform),
        "gaussian" => Ok(PatchWindow::Gaussian { sigma }),
        "gaussian_disk" => Ok(PatchWindow::GaussianDisk { sigma }),
        other => Err(PyValueError::new_err(format!(
            "unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"
        ))),
    }
}

fn parse_reduce(s: &str) -> PyResult<ViewReduce> {
    match s {
        "min" => Ok(ViewReduce::Min),
        "max" => Ok(ViewReduce::Max),
        "median" => Ok(ViewReduce::Median),
        "mean" => Ok(ViewReduce::Mean),
        other => Err(PyValueError::new_err(format!(
            "unknown reduce: {other:?} (expected min|max|median|mean)"
        ))),
    }
}

/// A collection of oriented patches built from a reconstruction's 3D points.
///
/// See :meth:`from_reconstruction` and ``specs/core/patch-cloud.md``.
#[pyclass(name = "PatchCloud", module = "sfmtool")]
pub struct PyPatchCloud {
    pub(crate) inner: PatchCloud,
}

/// Full pyramid depth for a source image: down to ~1 px on the short side.
/// The formula is shared by the list path and [`PyImagePyramidSet`] so a
/// prebuilt set is level-for-level identical to a per-call build.
fn pyramid_levels(src: &ImageU8) -> usize {
    let min_dim = src.width().min(src.height()).max(1);
    ((min_dim as f32).log2().floor() as usize).max(1) + 1
}

/// Validate that image `i` (dimensions `w × h`) matches its camera's resolution.
fn check_image_matches_camera(
    recon: &sfmtool_core::SfmrReconstruction,
    i: usize,
    w: u32,
    h: u32,
) -> PyResult<()> {
    let cam = &recon.cameras[recon.images[i].camera_index as usize];
    if w != cam.width || h != cam.height {
        return Err(PyValueError::new_err(format!(
            "image {i} is {w}x{h} but its camera is {}x{}; \
             pass full-resolution images",
            cam.width, cam.height
        )));
    }
    Ok(())
}

/// Extract each numpy image and build its full pyramid, validating dimensions
/// against the reconstruction's cameras. The extraction runs under the GIL; the
/// pyramid build (the expensive part) runs GIL-free and rayon-parallel.
fn build_pyramids_from_arrays(
    py: Python<'_>,
    recon: &sfmtool_core::SfmrReconstruction,
    images: &[Bound<'_, PyAny>],
) -> PyResult<Vec<ImageU8Pyramid>> {
    if images.len() != recon.images.len() {
        return Err(PyValueError::new_err(format!(
            "images must be parallel to the reconstruction's {} images, got {}",
            recon.images.len(),
            images.len()
        )));
    }
    let srcs: Vec<ImageU8> = images
        .iter()
        .enumerate()
        .map(|(i, im)| {
            let src = extract_image_u8(im)?;
            check_image_matches_camera(recon, i, src.width(), src.height())?;
            Ok(src)
        })
        .collect::<PyResult<_>>()?;
    Ok(py.detach(|| {
        srcs.par_iter()
            .map(|src| ImageU8Pyramid::build(src, pyramid_levels(src)))
            .collect()
    }))
}

/// The per-image pyramids a kernel call reads: either owned (built from a numpy
/// image list for this one call) or shared (an [`PyImagePyramidSet`] handle,
/// built once and reused across calls).
enum PyramidSet {
    Owned(Vec<ImageU8Pyramid>),
    Shared(Arc<Vec<ImageU8Pyramid>>),
}

impl PyramidSet {
    fn as_slice(&self) -> &[ImageU8Pyramid] {
        match self {
            PyramidSet::Owned(v) => v,
            PyramidSet::Shared(a) => a,
        }
    }
}

/// Resolve the `images` argument of a PatchCloud kernel method — either a
/// prebuilt [`PyImagePyramidSet`] (validated against `recon`, shared) or a list
/// of numpy images (pyramids built for this call) — plus one camera pose per
/// reconstruction image. Shared by `refine_normals`, `select_views`,
/// `localize_keypoints`, and `refine_keypoints` so they handle imagery
/// identically.
fn build_pyramids_and_poses(
    recon: &sfmtool_core::SfmrReconstruction,
    images: &Bound<'_, PyAny>,
) -> PyResult<(PyramidSet, Vec<RigidTransform>)> {
    let pyramids = if let Ok(set) = images.cast::<PyImagePyramidSet>() {
        let set = set.get();
        if set.pyramids.len() != recon.images.len() {
            return Err(PyValueError::new_err(format!(
                "ImagePyramidSet holds {} images but the reconstruction has {}; \
                 build the set from this reconstruction's image list",
                set.pyramids.len(),
                recon.images.len()
            )));
        }
        for (i, pyr) in set.pyramids.iter().enumerate() {
            let l0 = pyr.level(0);
            check_image_matches_camera(recon, i, l0.width(), l0.height())?;
        }
        PyramidSet::Shared(Arc::clone(&set.pyramids))
    } else if let Ok(list) = images.extract::<Vec<Bound<'_, PyAny>>>() {
        PyramidSet::Owned(build_pyramids_from_arrays(images.py(), recon, &list)?)
    } else {
        return Err(PyTypeError::new_err(
            "images must be a list of HxW[xC] uint8 numpy arrays or an ImagePyramidSet",
        ));
    };
    let poses: Vec<RigidTransform> = recon
        .images
        .iter()
        .map(|im| {
            let q = im.quaternion_wxyz;
            RigidTransform::from_wxyz_translation(
                [q.w, q.i, q.j, q.k],
                [
                    im.translation_xyz.x,
                    im.translation_xyz.y,
                    im.translation_xyz.z,
                ],
            )
        })
        .collect();
    Ok((pyramids, poses))
}

/// Per-image source pyramids prebuilt **once** and shared across PatchCloud
/// kernel calls.
///
/// Every :class:`PatchCloud` kernel method (:meth:`PatchCloud.refine_normals`,
/// :meth:`PatchCloud.select_views`, :meth:`PatchCloud.localize_keypoints`,
/// :meth:`PatchCloud.refine_keypoints`) accepts one of these anywhere it accepts
/// the list of numpy images: rather than rebuilding a full image pyramid per
/// image on **every** call (the per-call ``images`` list path, kept for
/// back-compat), build the set once and pass it to each call. The pyramids are
/// identical to the per-call build (same levels, same box-filter downsample), so
/// results are unchanged; the build itself is rayon-parallel.
///
/// Args:
///     recon: The reconstruction the images belong to. Each image is validated
///         against its camera's resolution at build time; kernel calls
///         re-validate the set against *their* reconstruction (image count and
///         per-image camera dimensions), so a set built here works for any
///         reconstruction with the same images (e.g. across ``embed-patches``
///         rounds).
///     images: One source image (HxWxC uint8 numpy array) per reconstruction
///         image, parallel to ``recon`` (index = image index).
#[pyclass(name = "ImagePyramidSet", module = "sfmtool", frozen)]
pub struct PyImagePyramidSet {
    pub(crate) pyramids: Arc<Vec<ImageU8Pyramid>>,
}

#[pymethods]
impl PyImagePyramidSet {
    #[new]
    #[pyo3(signature = (recon, images))]
    fn new(
        py: Python<'_>,
        recon: &PySfmrReconstruction,
        images: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pyramids = build_pyramids_from_arrays(py, &recon.inner, &images)?;
        Ok(Self {
            pyramids: Arc::new(pyramids),
        })
    }

    /// Number of per-image pyramids in the set.
    fn __len__(&self) -> usize {
        self.pyramids.len()
    }
}

#[pymethods]
impl PyPatchCloud {
    /// Build one oriented patch per finite 3D point of a reconstruction.
    ///
    /// Args:
    ///     recon: The reconstruction.
    ///     normal: Normal policy — ``"stored"`` (default; the reconstruction's
    ///         stored estimated normal, whatever is in the ``.sfmr``, falling back
    ///         to the mean viewing direction where that is zero/degenerate),
    ///         ``"mean_viewing"`` (mean direction to the observing cameras), or
    ///         ``"geometric"`` (local PCA plane fit over ``k_neighbors`` points).
    ///     k_neighbors: Neighbor count for the ``"geometric"`` policy.
    ///     extent: Half-size policy — ``"feature_size"`` (default; ``extent_value``
    ///         × each observation's keypoint scale back-projected to world, reduced
    ///         by ``feature_reduce``; reads the ``.sift`` files and raises
    ///         ``ValueError`` if a point has no readable scale in any view),
    ///         ``"fixed"`` (world
    ///         units = ``extent_value``), ``"relative_spacing"`` (``extent_value`` ×
    ///         median point spacing), or ``"pixel_radius"`` (back-project
    ///         ``extent_value`` px in each observing view, reduced by ``pixel_reduce``).
    ///     extent_value: The scalar for the chosen extent policy (default 2.5; for
    ///         ``"feature_size"`` the keypoint-scale half-extent multiplier, so the
    ///         full patch edge is ``5 ×`` the projected feature size).
    ///     pixel_reduce: For ``"pixel_radius"``, the view reduce — ``"min"``
    ///         (default), ``"max"``, ``"median"``, or ``"mean"``.
    ///     feature_reduce: For ``"feature_size"``, the view reduce (default
    ///         ``"median"``).
    ///     exclude_points_at_infinity: When ``False`` (default), each point at
    ///         infinity also gets a tangent-sphere frame (``w = 0``) around its
    ///         direction; every patch operation handles these. Pass ``True`` to
    ///         emit patches for finite points only (e.g. an operation that scatters
    ///         per-point results back and must leave infinity points untouched).
    #[staticmethod]
    #[pyo3(signature = (
        recon, normal="stored", k_neighbors=12,
        extent="feature_size", extent_value=2.5,
        pixel_reduce="min", feature_reduce="median",
        exclude_points_at_infinity=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_reconstruction(
        recon: &PySfmrReconstruction,
        normal: &str,
        k_neighbors: usize,
        extent: &str,
        extent_value: f64,
        pixel_reduce: &str,
        feature_reduce: &str,
        exclude_points_at_infinity: bool,
    ) -> PyResult<Self> {
        let normal = match normal {
            "stored" => PatchNormal::Stored,
            "mean_viewing" | "mean" => PatchNormal::MeanViewing,
            "geometric" => PatchNormal::Geometric { k_neighbors },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown normal policy: {other:?} (expected stored|mean_viewing|geometric)"
                )))
            }
        };
        let extent = match extent {
            "fixed" => PatchExtent::Fixed(extent_value),
            "relative_spacing" => PatchExtent::RelativeToSpacing(extent_value),
            "pixel_radius" => PatchExtent::PixelRadius {
                radius_px: extent_value,
                across: parse_reduce(pixel_reduce)?,
            },
            "feature_size" => PatchExtent::FeatureSize {
                factor: extent_value,
                across: parse_reduce(feature_reduce)?,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown extent policy: {other:?} \
                     (expected fixed|relative_spacing|pixel_radius|feature_size)"
                )))
            }
        };
        let inner = PatchCloud::from_reconstruction(
            &recon.inner,
            normal,
            extent,
            exclude_points_at_infinity,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a patch cloud from per-point half-extent vector arrays and centers.
    ///
    /// The inverse of the per-point ``patch_u_halfvec_xyz`` / ``patch_v_halfvec_xyz``
    /// layout: each present row (non-zero ``u``) becomes one patch whose
    /// ``point_indexes`` entry is that row index, with the half-extent vectors split
    /// back into a unit axis and a half-size. Use this to assemble a renumbered or
    /// culled cloud (e.g. after dropping points) to hand to
    /// ``SfmrReconstruction.clone_with_changes(patches=...)``.
    ///
    /// Args:
    ///     half_u_xyz: ``(N, 3)`` float32 in-plane half-extent vectors ``u``.
    ///     half_v_xyz: ``(N, 3)`` float32 in-plane half-extent vectors ``v``.
    ///     centers: ``(N, 3)`` float64 patch centers (each point's position).
    #[staticmethod]
    #[pyo3(signature = (half_u_xyz, half_v_xyz, centers))]
    fn from_halfvec_arrays(
        half_u_xyz: PyReadonlyArray2<'_, f32>,
        half_v_xyz: PyReadonlyArray2<'_, f32>,
        centers: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let u = half_u_xyz.as_array();
        let v = half_v_xyz.as_array();
        let c = centers.as_array();
        let n = u.shape()[0];
        if u.shape()[1] != 3 || v.shape()[1] != 3 || c.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "from_halfvec_arrays: half_u_xyz, half_v_xyz, centers must each have shape (N, 3)",
            ));
        }
        if v.shape()[0] != n || c.shape()[0] != n {
            return Err(PyValueError::new_err(format!(
                "from_halfvec_arrays: row counts must match (u={}, v={}, centers={})",
                n,
                v.shape()[0],
                c.shape()[0]
            )));
        }
        let centers_vec: Vec<Point3<f64>> = (0..n)
            .map(|i| Point3::new(c[[i, 0]], c[[i, 1]], c[[i, 2]]))
            .collect();
        let inner = PatchCloud::from_halfvec_arrays(&u.to_owned(), &v.to_owned(), &centers_vec);
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The oriented patch at index `i`.
    fn __getitem__(&self, i: usize) -> PyResult<PyOrientedPatch> {
        if i >= self.inner.len() {
            return Err(PyIndexError::new_err("patch index out of range"));
        }
        Ok(PyOrientedPatch {
            inner: self.inner.patch(i).clone(),
        })
    }

    /// Source 3D-point index for each patch (parallel to the cloud).
    #[getter]
    fn point_indexes(&self) -> Vec<u32> {
        self.inner.point_indexes.clone()
    }

    /// Refine every patch's normal in place by photometric consistency across
    /// the reconstruction's observing views (see
    /// ``specs/core/patch-normal-refinement.md``).
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides cameras,
    ///         poses, and the per-point observing-image lists via ``point_indexes``).
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images (decode the
    ///         pyramids once, share them across kernel calls).
    ///     resolution: The R×R patch grid the consensus is scored on.
    ///     objective: ``"robust"`` (IRLS-weighted consensus, default) or
    ///         ``"mean"`` (unweighted all-pairs consensus).
    ///     window: Per-pixel scoring weight — ``"gaussian_disk"`` (default),
    ///         ``"gaussian"``, or ``"uniform"``.
    ///     sampler: How to sample the source pyramids — ``"bilinear"`` (default;
    ///         fastest, and the found normal barely differs) or ``"anisotropic"``
    ///         (anti-aliased oblique views; keeps the reported Φ/confidence
    ///         unbiased, ~1.6-3x slower).
    ///     point_indexes: If given, refine only the patches with these source point
    ///         ids (the rest keep their input normal) — cheap when refining a few
    ///         patches out of a large cloud. ``None`` refines every patch.
    ///     view_indices: If given, a per-patch list of image indices to refine
    ///         that patch over (parallel to the cloud's patches), *overriding* the
    ///         reconstruction's track-based observing-view lists. This lets a
    ///         caller refine against an arbitrary view set — e.g. every image that
    ///         geometrically sees the point, not just the ones that matched a
    ///         feature there (an MVS-style expansion). Indices must be in range for
    ///         the reconstruction; duplicates within a patch are ignored (the
    ///         consensus counts each view once). ``None`` (default) uses the track
    ///         observations. Combines with ``point_indexes`` (which still selects
    ///         *which* patches to refine).
    ///     obliquity_weight_power: Exponent ``p`` of the multiplicative obliquity
    ///         view-weight ``|v̂·n|^p`` folded into the robust consensus (use A).
    ///         ``0.0`` (default) disables it — the consensus is byte-for-byte the
    ///         prior-free result. ``2.0`` is the ``cos²θ`` foreshortening weight:
    ///         it softly down-weights a view the more obliquely it sees the surfel,
    ///         a continuous alternative to a hard grazing-view cut (only affects the
    ///         robust ``objective``).
    ///     fronto_prior_weight: Weight ``λ`` of the additive fronto-parallel prior
    ///         ``λ·mean_v (v̂·n)²`` on each candidate normal when ranking (use B).
    ///         ``0.0`` (default) disables it. It rewards normals that face the
    ///         observing cameras, supplying the missing constraint on a low-parallax
    ///         point (flat ``Φ``) so the normal settles fronto-parallel instead of
    ///         drifting to a photometrically-equivalent tilt; where real parallax
    ///         curves ``Φ`` the small prior is overruled. With it active the
    ///         reported ``photoconsistency`` can dip below ``init_photoconsistency``
    ///         by up to the prior gap (a more-frontal normal winning a near-tie).
    ///     max_refine_views: Cap on the per-patch **refinement basis**: when
    ///         ``> 0`` and a patch has more views than this, refine over only the
    ///         ``K`` most normal-informative views — a D-optimal geometric pick
    ///         (least-oblique appearance anchor plus a greedy information-matrix
    ///         determinant fill; see
    ///         ``specs/core/patch-normal-refine-view-subset.md``). ``0`` (default)
    ///         disables the cap — byte-for-byte the uncapped behavior. The cap is
    ///         floored at ``min_views`` internally, and ignored per-patch when the
    ///         subset would under-constrain one tilt DOF (the conditioning
    ///         fallback). Only the refinement basis shrinks — no observation is
    ///         dropped from the reconstruction.
    ///     use_stored_keypoints: When ``True`` (the default), anchor each
    ///         view's patch at that observation's stored per-observation 2D
    ///         keypoint (the inline keypoint an ``embedded_patches`` recon
    ///         carries) — the stored anchor gives a cleaner cross-view
    ///         consensus than the reprojected point center. A view with no
    ///         stored keypoint — either a candidate from ``view_indices`` /
    ///         ``select_views`` that the SIFT track didn't observe, or any
    ///         view on a ``sift_files`` recon (which has no inline keypoints
    ///         at all) — falls back per-view to the reprojected center.
    ///         When ``False``, every view is anchored at the reprojected
    ///         center regardless of what the recon carries — useful for
    ///         callers (e.g. ``sfm compare --strips``) that want a defined
    ///         comparison reference independent of recon kind. Works with
    ///         the fronto ``cache`` (each view's base is rendered at its
    ///         anchored center), so there is no speed penalty either way.
    ///     render_bitmaps: If true, also render each refined patch's RGBA
    ///         representative texture at the found normal and return them scattered
    ///         to per-3D-point rows (see ``bitmaps`` below). Costs one extra
    ///         full-grid source render per kept view per patch, so it is off by
    ///         default.
    ///
    /// Returns a dict of per-patch results (numpy arrays parallel to the cloud):
    /// ``normal`` (Nx3), ``photoconsistency`` (N), ``init_photoconsistency`` (N),
    /// ``confidence`` (N), ``valid_view_count`` (N). The cloud's patches are
    /// updated to the refined normals in place. When ``render_bitmaps`` is true,
    /// also ``bitmaps``: a ``(P, R, R, 4)`` uint8 array of fused RGBA patch
    /// textures scattered to **per-3D-point** rows (``P`` = ``recon`` point count,
    /// ``R`` = ``resolution``), zero rows for points with no refined patch — ready
    /// to pass straight to ``clone_with_changes(patch_bitmaps=...)``.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        recon, images, *, resolution=24, angular_range_deg=25.0, init_steps=7,
        refine_levels=3, objective="robust", robust_iters=3, window="gaussian_disk",
        window_sigma=0.6, min_valid_fraction=0.6, min_views=3, sampler="bilinear",
        cache="fronto", cache_supersample=2.0, compute_confidence=false,
        search_robust_iters=None, obliquity_weight_power=0.0, fronto_prior_weight=0.0,
        max_refine_views=0, point_indexes=None, view_indices=None,
        use_stored_keypoints=true, render_bitmaps=false, progress=None
    ))]
    fn refine_normals<'py>(
        &mut self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        images: &Bound<'py, PyAny>,
        resolution: u32,
        angular_range_deg: f64,
        init_steps: u32,
        refine_levels: u32,
        objective: &str,
        robust_iters: u32,
        window: &str,
        window_sigma: f64,
        min_valid_fraction: f64,
        min_views: u32,
        sampler: &str,
        cache: &str,
        cache_supersample: f64,
        compute_confidence: bool,
        search_robust_iters: Option<u32>,
        obliquity_weight_power: f64,
        fronto_prior_weight: f64,
        max_refine_views: u32,
        point_indexes: Option<Vec<u32>>,
        view_indices: Option<Vec<Vec<u32>>>,
        use_stored_keypoints: bool,
        render_bitmaps: bool,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let recon = &recon.inner;
        // The cloud must have been built from this reconstruction: its per-patch
        // point_indexes index `recon`'s points. This is a range check only — it
        // catches a too-small recon (and would-be panics in the core), but cannot
        // detect a *different* recon with at least as many points; the caller is
        // responsible for passing the recon the cloud was built from.
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_indexes
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_indexes are out of range for this reconstruction \
                 (was the cloud built from a different recon?)",
            ));
        }
        let objective = match objective {
            "mean" | "mean_pairwise" => Objective::MeanPairwise,
            "robust" | "robust_weighted" => Objective::RobustWeighted {
                iters: robust_iters,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown objective: {other:?} (expected mean|robust)"
                )))
            }
        };
        let window = match window {
            "uniform" => PatchWindow::Uniform,
            "gaussian" => PatchWindow::Gaussian {
                sigma: window_sigma,
            },
            "gaussian_disk" => PatchWindow::GaussianDisk {
                sigma: window_sigma,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"
                )))
            }
        };
        let sampler = match sampler {
            "bilinear" => Sampler::Bilinear,
            "anisotropic" => Sampler::Anisotropic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler: {other:?} (expected bilinear|anisotropic)"
                )))
            }
        };
        let cache = match cache {
            "off" => CacheMode::Off,
            "fronto" => CacheMode::FrontoParallel,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown cache: {other:?} (expected off|fronto)"
                )))
            }
        };
        if cache_supersample < 1.0 {
            return Err(PyValueError::new_err(format!(
                "cache_supersample must be >= 1.0, got {cache_supersample}"
            )));
        }
        let params = NormalRefineParams {
            angular_range_deg,
            init_steps,
            refine_levels,
            objective,
            window,
            min_valid_fraction,
            min_views,
            sampler,
            cache,
            cache_supersample,
            compute_confidence,
            search_robust_iters,
            obliquity_weight_power,
            fronto_prior_weight,
            render_bitmap: render_bitmaps,
            max_refine_views,
        };

        // Build one pyramid + pose per reconstruction image; the ProjectedImages
        // borrow these for the duration of the refinement. The warp validity /
        // sampling assume each image matches its camera's resolution, so the
        // helper rejects mismatched sizes.
        let (pyramid_set, poses) = build_pyramids_and_poses(recon, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<ProjectedImage<'_>> = (0..recon.images.len())
            .map(|i| ProjectedImage {
                camera: &recon.cameras[recon.images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
            .collect();
        // The per-patch view sets the consensus is scored over. By default these
        // are the reconstruction's track observations; `view_indices` overrides
        // them with an explicit per-patch list (e.g. every geometrically-visible
        // image, for an MVS-style refinement).
        let mut patch_views = match view_indices {
            Some(mut views) => {
                if views.len() != self.inner.len() {
                    return Err(PyValueError::new_err(format!(
                        "view_indices must be parallel to the cloud's {} patches, got {}",
                        self.inner.len(),
                        views.len()
                    )));
                }
                let n_images = recon.images.len() as u32;
                if views.iter().flatten().any(|&i| i >= n_images) {
                    return Err(PyValueError::new_err(
                        "view_indices contains an image index out of range for this \
                         reconstruction",
                    ));
                }
                // Drop repeated views within a patch: the consensus weights each
                // view once, so a duplicate would silently double-count it. Order
                // is preserved (it does not affect the consensus).
                let mut seen = std::collections::HashSet::new();
                for pv in &mut views {
                    seen.clear();
                    pv.retain(|&i| seen.insert(i));
                }
                views
            }
            None => view_indices_from_reconstruction(recon, &self.inner),
        };
        // Optional subset: refine only patches whose point id is listed. Cleared
        // view-lists make `refine_patch_normal` skip a patch immediately (it sees
        // too few views), so a handful of patches can be refined out of a large
        // cloud cheaply — the rest keep their input normal.
        if let Some(ids) = point_indexes {
            let keep: std::collections::HashSet<u32> = ids.into_iter().collect();
            for (pv, &pid) in patch_views.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    pv.clear();
                }
            }
        }

        // Optional stored-keypoint anchoring: position each view's patch at that
        // observation's stored per-observation keypoint instead of the reprojected
        // point center. Only an `embedded_patches` recon carries inline keypoints;
        // a `sift_files` recon (or any other view without a stored keypoint) falls
        // through per-view to the reprojected center.
        let patch_view_keypoints: Option<Vec<Vec<Option<[f64; 2]>>>> = use_stored_keypoints
            .then(|| recon.keypoints_xy())
            .flatten()
            .map(|keypoints_xy| {
                // (point_index, image_index) -> stored keypoint. `keypoints_xy` is
                // parallel to `recon.tracks`, so walk the tracks once and index by
                // observation row. Duplicate (point, image) observations all key to
                // the same map entry, so any duplicate view resolves to the same
                // keypoint (last write wins, harmless).
                let mut kp_map: std::collections::HashMap<(u32, u32), [f64; 2]> =
                    std::collections::HashMap::with_capacity(recon.tracks.len());
                for (j, obs) in recon.tracks.iter().enumerate() {
                    kp_map.insert(
                        (obs.point_index, obs.image_index),
                        [keypoints_xy[[j, 0]] as f64, keypoints_xy[[j, 1]] as f64],
                    );
                }
                // Build per-patch keypoints parallel to `patch_views`: for each
                // patch's (point_index, image_index) view, the stored keypoint when
                // present (else `None`, which the core treats as "anchor at the
                // reprojected center for this view").
                patch_views
                    .iter()
                    .zip(&self.inner.point_indexes)
                    .map(|(pv, &pidx)| {
                        pv.iter()
                            .map(|&img| kp_map.get(&(pidx, img)).copied())
                            .collect()
                    })
                    .collect()
            });

        // Hold a handle to the shared counter across the detach so a Python
        // poller thread can read it while this pass runs with the GIL released.
        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            refine_patch_cloud_normals(
                &mut self.inner,
                &views,
                &patch_views,
                resolution,
                &params,
                patch_view_keypoints.as_deref(),
                progress_handle.as_deref(),
            )
        });

        let n = results.len();
        let mut normals = Vec::with_capacity(n);
        let mut photo = Vec::with_capacity(n);
        let mut init_photo = Vec::with_capacity(n);
        let mut conf = Vec::with_capacity(n);
        let mut vvc = Vec::with_capacity(n);
        for r in &results {
            let nrm = r.patch.normal();
            normals.push(vec![nrm.x, nrm.y, nrm.z]);
            photo.push(r.photoconsistency);
            init_photo.push(r.init_photoconsistency);
            conf.push(r.confidence);
            vvc.push(r.valid_view_count);
        }

        let out = PyDict::new(py);
        out.set_item("normal", PyArray2::from_vec2(py, &normals).unwrap())?;
        out.set_item("photoconsistency", photo.into_pyarray(py))?;
        out.set_item("init_photoconsistency", init_photo.into_pyarray(py))?;
        out.set_item("confidence", conf.into_pyarray(py))?;
        out.set_item("valid_view_count", vvc.into_pyarray(py))?;

        // Scatter the per-patch representative textures into per-3D-point rows so
        // the result drops straight into `clone_with_changes(patch_bitmaps=...)`.
        // Points with no refined patch (and patches that produced no render) keep
        // their zero rows, mirroring `PatchCloud::to_halfvec_arrays`.
        if render_bitmaps {
            let r = resolution as usize;
            let npoints = recon.points.len();
            let stride = r * r * 4;
            let mut flat = vec![0u8; npoints * stride];
            for (res, &pid) in results.iter().zip(&self.inner.point_indexes) {
                if let Some(rep) = &res.representative {
                    let pid = pid as usize;
                    if pid < npoints && rep.len() == stride {
                        flat[pid * stride..(pid + 1) * stride].copy_from_slice(rep);
                    }
                }
            }
            let arr = ndarray::Array4::from_shape_vec((npoints, r, r, 4), flat)
                .expect("bitmap scatter shape matches");
            out.set_item("bitmaps", arr.into_pyarray(py))?;
        }
        Ok(out)
    }

    /// Select, per patch, the **view set** ``G`` that photometrically sees it:
    /// the point's track views plus every other image that geometrically sees the
    /// surfel and whose windowed ZNCC to a robust reference appearance (fused from
    /// the track views) clears ``min_relative_zncc`` × the track's own
    /// self-agreement. Track views are always admitted. See
    /// ``specs/core/patch-view-selection.md``.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides cameras,
    ///         poses, and the per-point track view lists via ``point_indexes``).
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images (decode the
    ///         pyramids once, share them across kernel calls).
    ///     min_relative_zncc: Admit a candidate whose ZNCC to the reference clears
    ///         this fraction of the track's self-agreement (default 0.7).
    ///     resolution: The R×R patch grid the reference / ZNCC are scored on.
    ///     window: Per-pixel scoring weight — ``"gaussian_disk"`` (default),
    ///         ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"bilinear"`` (default) or ``"anisotropic"``.
    ///     min_valid_fraction: Per-view floor on the window-weighted valid-pixel
    ///         fraction; a view below it does not cover enough of the patch.
    ///     min_track_views: Minimum number of *valid* track views to build a
    ///         reference from; a track below this admits its views verbatim with no
    ///         candidate vetting.
    ///     robust_iters: IRLS passes for the robust reference consensus.
    ///     min_self_agreement: Trust gate on the track's self-agreement (default
    ///         0.3). When the track views' mean ZNCC to the reference is below this,
    ///         there is no trustworthy reference, so the track is admitted verbatim
    ///         with no candidate expansion. At or above it, the admission bar is
    ///         ``min_relative_zncc`` × self-agreement.
    ///     point_indexes: If given, select only for the patches with these source
    ///         point ids; ``None`` (default) selects for every patch.
    ///
    /// Returns a list of per-patch dicts (parallel to the cloud's patches, in
    /// cloud order): ``point_index`` (int), ``admitted`` (1-D int32 numpy array of
    /// image indices — the track views first, then the vetted candidates in
    /// ascending order), ``scores`` (1-D float64 numpy array of the per-admitted
    /// ZNCC to the reference, parallel to ``admitted``; NaN where a view could not
    /// be scored), and ``self_agreement`` (float; NaN when no reference could be
    /// built). Patches excluded by ``point_indexes`` are omitted from the list.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        recon, images, *, min_relative_zncc=0.7, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear", min_valid_fraction=0.6, min_track_views=2,
        robust_iters=3, min_self_agreement=0.3, point_indexes=None, progress=None
    ))]
    fn select_views<'py>(
        &self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        images: &Bound<'py, PyAny>,
        min_relative_zncc: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        min_valid_fraction: f64,
        min_track_views: u32,
        robust_iters: u32,
        min_self_agreement: f64,
        point_indexes: Option<Vec<u32>>,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let recon = &recon.inner;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_indexes
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_indexes are out of range for this reconstruction \
                 (was the cloud built from a different recon?)",
            ));
        }

        let window = match window {
            "uniform" => PatchWindow::Uniform,
            "gaussian" => PatchWindow::Gaussian {
                sigma: window_sigma,
            },
            "gaussian_disk" => PatchWindow::GaussianDisk {
                sigma: window_sigma,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"
                )))
            }
        };
        let sampler = match sampler {
            "bilinear" => Sampler::Bilinear,
            "anisotropic" => Sampler::Anisotropic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler: {other:?} (expected bilinear|anisotropic)"
                )))
            }
        };
        let params = ViewSelectParams {
            min_relative_zncc,
            resolution,
            window,
            sampler,
            min_valid_fraction,
            min_track_views,
            robust_iters,
            min_self_agreement,
        };

        let (pyramid_set, poses) = build_pyramids_and_poses(recon, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<ProjectedImage<'_>> = (0..recon.images.len())
            .map(|i| ProjectedImage {
                camera: &recon.cameras[recon.images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
            .collect();

        // Per-patch track view lists from the reconstruction; an empty list makes
        // a patch's selection trivially empty, so `point_indexes` selects a subset by
        // clearing the rest.
        let mut track_views = view_indices_from_reconstruction(recon, &self.inner);
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (tv, &pid) in track_views.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    tv.clear();
                }
            }
        }

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            select_patch_cloud_views(
                &self.inner,
                &views,
                &track_views,
                &params,
                progress_handle.as_deref(),
            )
        });

        let mut out = Vec::new();
        for (res, &pid) in results.iter().zip(&self.inner.point_indexes) {
            if let Some(keep) = &selected_mask {
                if !keep.contains(&pid) {
                    continue;
                }
            }
            let d = PyDict::new(py);
            d.set_item("point_index", pid)?;
            d.set_item("admitted", res.admitted.clone().into_pyarray(py))?;
            d.set_item("scores", res.scores.clone().into_pyarray(py))?;
            d.set_item("self_agreement", res.self_agreement)?;
            out.push(d);
        }
        Ok(out)
    }

    /// Refine, per patch, the per-view 2D keypoints by group-wise translation
    /// registration (**congealing**): each round renders every view's patch tile
    /// at its accumulated in-plane offset (a single resample of the source),
    /// builds the robust consensus, and searches each view's residual shift
    /// against the **leave-one-out** consensus of the others, dropping views that
    /// drift too far, leave the frame, or stop agreeing. See
    /// ``specs/core/patch-keypoint-localization.md``.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (cameras, poses, and
    ///         the per-point track view lists via ``point_indexes``).
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images (decode the
    ///         pyramids once, share them across kernel calls).
    ///     view_sets: Optional mapping ``point_index -> [image_index, ...]`` giving the
    ///         view set to refine per point (typically the output of
    ///         :meth:`select_views`). Points absent from the map fall back to their
    ///         track; ``None`` (default) uses the track for every point.
    ///     max_iters: Max congealing rounds (stops early at convergence).
    ///     search: Max total per-view drift in patch-grid px (bounds runaway; also
    ///         the context-tile margin).
    ///     max_shift_px: Drop a view whose refined keypoint sits more than this many
    ///         source-image px from the point's projection.
    ///     min_relative_zncc: Drop a view whose leave-one-out ZNCC falls below this
    ///         fraction of the views' median leave-one-out ZNCC.
    ///     min_grazing_cos: Grazing cutoff; drop a view whose ray is near-parallel
    ///         to the patch plane (``|d·n|`` below this).
    ///     resolution: The R×R patch grid the consensus / ZNCC are scored on.
    ///     window: ``"gaussian_disk"`` (default), ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"bilinear"`` (default) or ``"anisotropic"``.
    ///     robust_iters: IRLS passes for the robust consensus.
    ///     convergence_px: Stop once a round's mean round-over-round change of
    ///         the per-view refined positions is below this many patch-grid px.
    ///     point_indexes: If given, localize only for the patches with these source
    ///         point ids; ``None`` (default) localizes for every patch.
    ///     search_resolution_multiplier: ``m`` for the discrete cross-view search;
    ///         the search runs at resolution ``R_s = round(m·R)``. ``m = 1.0``
    ///         (default) is the no-op; ``m > 1`` (the supersampled grid) resolves
    ///         sub-pixel offsets directly at a cost that grows ~``m²``. See
    ///         ``specs/core/keypoint-localization-search-cache.md``.
    ///
    /// Returns:
    ///     A list of per-point dicts ``{point_index, views (uint32[K]),
    ///     keypoints (float64[K, 2]), offsets_px (float64[K]),
    ///     loo_zncc (float64[K])}`` over the **kept** views. ``loo_zncc`` is NaN for
    ///     a view no round scored (a lone input view, or a view kept by the two-view
    ///     floor before any consensus was built), so guard before reducing it.
    #[pyo3(signature = (
        recon, images, *, view_sets=None, max_iters=5, search=6.0, max_shift_px=3.0,
        min_relative_zncc=0.7, min_grazing_cos=0.1, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear", robust_iters=3, convergence_px=0.05,
        point_indexes=None, search_resolution_multiplier=1.0,
        search_strategy="plus_descent", progress=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn localize_keypoints<'py>(
        &self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        images: &Bound<'py, PyAny>,
        view_sets: Option<std::collections::HashMap<u32, Vec<u32>>>,
        max_iters: u32,
        search: f64,
        max_shift_px: f64,
        min_relative_zncc: f64,
        min_grazing_cos: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        robust_iters: u32,
        convergence_px: f64,
        point_indexes: Option<Vec<u32>>,
        search_resolution_multiplier: f32,
        search_strategy: &str,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let recon = &recon.inner;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_indexes
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_indexes are out of range for this reconstruction \
                 (was the cloud built from a different recon?)",
            ));
        }

        let window = match window {
            "uniform" => PatchWindow::Uniform,
            "gaussian" => PatchWindow::Gaussian {
                sigma: window_sigma,
            },
            "gaussian_disk" => PatchWindow::GaussianDisk {
                sigma: window_sigma,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"
                )))
            }
        };
        let sampler = match sampler {
            "bilinear" => Sampler::Bilinear,
            "anisotropic" => Sampler::Anisotropic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler: {other:?} (expected bilinear|anisotropic)"
                )))
            }
        };
        if !(search_resolution_multiplier.is_finite() && search_resolution_multiplier > 0.0) {
            return Err(PyValueError::new_err(format!(
                "search_resolution_multiplier must be > 0, got {search_resolution_multiplier}"
            )));
        }
        let search_strategy = match search_strategy {
            "exhaustive" => LocalizeSearchStrategy::Exhaustive,
            "plus_descent" => LocalizeSearchStrategy::PlusDescent,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown search_strategy: {other:?} (expected exhaustive|plus_descent)"
                )))
            }
        };
        let params = KeypointLocalizeParams {
            max_iters,
            search,
            max_shift_px,
            min_relative_zncc,
            min_grazing_cos,
            resolution,
            window,
            sampler,
            robust_iters,
            convergence_px,
            search_resolution_multiplier,
            search_strategy,
        };

        let (pyramid_set, poses) = build_pyramids_and_poses(recon, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<ProjectedImage<'_>> = (0..recon.images.len())
            .map(|i| ProjectedImage {
                camera: &recon.cameras[recon.images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
            .collect();

        // Per-patch view sets: the supplied map where present, else the track. An
        // empty view set makes a patch's localization trivially empty, so
        // `point_indexes` selects a subset by clearing the rest.
        let mut sets = view_indices_from_reconstruction(recon, &self.inner);
        if let Some(map) = &view_sets {
            // Reject out-of-range image indices up front so the kernel never indexes
            // `views` out of bounds (which would surface as an opaque panic rather
            // than a clean error). The kernel dedups, so duplicates are fine here.
            let n_images = recon.images.len() as u32;
            for vs in map.values() {
                if let Some(&bad) = vs.iter().find(|&&i| i >= n_images) {
                    return Err(PyValueError::new_err(format!(
                        "view_sets contains image index {bad} out of range for this \
                         reconstruction's {n_images} images"
                    )));
                }
            }
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_indexes) {
                if let Some(vs) = map.get(&pid) {
                    *set = vs.clone();
                }
            }
        }
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    set.clear();
                }
            }
        }

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            localize_patch_cloud_keypoints(
                &self.inner,
                &views,
                &sets,
                None,
                &params,
                progress_handle.as_deref(),
            )
        });

        let mut out = Vec::new();
        for (res, &pid) in results.iter().zip(&self.inner.point_indexes) {
            if let Some(keep) = &selected_mask {
                if !keep.contains(&pid) {
                    continue;
                }
            }
            // Flat (K, 2) keypoint array, built with an explicit shape so the
            // no-kept-views case yields a clean (0, 2) array rather than failing
            // column inference.
            let flat: Vec<f64> = res.keypoints.iter().flat_map(|k| [k[0], k[1]]).collect();
            let kpts = ndarray::Array2::from_shape_vec((res.keypoints.len(), 2), flat)
                .expect("keypoints shape matches");
            let d = PyDict::new(py);
            d.set_item("point_index", pid)?;
            d.set_item("views", res.views.clone().into_pyarray(py))?;
            d.set_item("keypoints", kpts.into_pyarray(py))?;
            d.set_item("offsets_px", res.offsets_px.clone().into_pyarray(py))?;
            d.set_item("loo_zncc", res.loo_zncc.clone().into_pyarray(py))?;
            out.push(d);
        }
        Ok(out)
    }

    /// Refine, per patch, the per-view 2D keypoints to **sub-pixel** by a local
    /// continuous photometric solve: forward-additive ECC (Enhanced Correlation
    /// Coefficient) Gauss–Newton against a single **frozen** robust cross-view
    /// consensus (the cheapest spec variant). This is the high-accuracy reference
    /// refiner — it does no grid search, changes no view membership, and is
    /// **never worse than the seed** (a step is accepted only if it raises the ECC
    /// score and stays in frame). Points at infinity (``w == 0``) are refined like
    /// finite ones, not skipped. The seed must already be close (≲ 1 px) — putting
    /// it in the basin is the caller's job (e.g. :meth:`localize_keypoints`). See
    /// ``specs/core/keypoint-subpixel-refinement.md``.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (cameras, poses, and
    ///         the per-point track view lists via ``point_indexes``).
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images (decode the
    ///         pyramids once, share them across kernel calls).
    ///     view_sets: Optional mapping ``point_index -> [image_index, ...]`` giving the
    ///         view set to refine per point. Points absent fall back to their track;
    ///         ``None`` (default) uses the track for every point.
    ///     resolution: The R×R patch grid the consensus / ECC are scored on.
    ///     window: ``"gaussian_disk"`` (default), ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"bilinear"`` (default) or ``"anisotropic"`` (the MVP uses a
    ///         bilinear finite-difference Jacobian for both).
    ///     robust_iters: IRLS passes for the robust consensus.
    ///     max_outer_sweeps: Max outer sweeps of the alternating loop (refresh
    ///         consensus → move every view). ``1`` (default) is the
    ///         single-pass-frozen variant — build the consensus once at the seed,
    ///         hold it fixed. ``> 1`` enables per-sweep refresh: each subsequent
    ///         sweep re-renders the views at their current offsets and rebuilds the
    ///         consensus from those. Exits early once the mean per-view move of a
    ///         sweep falls below ``outer_convergence_px``.
    ///     outer_convergence_px: Stop the outer (consensus-refresh) loop once the
    ///         mean per-view move across a sweep is below this many patch-grid px.
    ///         Ignored when ``max_outer_sweeps == 1``.
    ///     consensus_refresh: Within-sweep consensus refresh granularity.
    ///         ``"per_sweep"`` (default) holds the consensus fixed for the
    ///         duration of a sweep (current behavior). ``"per_move"`` is the
    ///         spec's Gauss–Seidel incremental variant: after each view's GN
    ///         solve, its z-normalized core delta-updates a running weighted
    ///         sum, and the next view aligns to a freshly-incrementalized
    ///         **shared** consensus ``normalize(S)``. (The spec's leave-one-out
    ///         alternative was measured-and-rejected on real-track view counts
    ///         — see the Rust ``ConsensusRefresh::PerMove`` doc.) IRLS weights
    ///         are refreshed only at the per-sweep boundary either way.
    ///         **Limitation:** at ``N = 2`` views ``per_move`` underestimates
    ///         the relative offset by ~3% (the moved view's own contribution
    ///         dominates the shared ``T``); ``N ≥ 3`` is recommended.
    ///     max_gn_steps: Max forward-additive Gauss–Newton steps per view per outer
    ///         sweep.
    ///     convergence_px: Stop a view's solve once an accepted step is below this
    ///         many patch-grid px.
    ///     max_offset_px: Max total per-view drift from the seed, in patch-grid px.
    ///     point_indexes: If given, refine only the patches with these source point
    ///         indexes; ``None`` (default) refines every patch.
    ///     starting_keypoints: Optional explicit per-view seed overrides:
    ///         ``point_index -> [[x, y], ...]`` in **source-image** pixels,
    ///         parallel to that point's entry in ``view_sets`` (one ``[x, y]``
    ///         per view, in order). When ``view_sets`` is also given, each
    ///         point's ``starting_keypoints`` length must match its
    ///         ``view_sets`` length. These seeds **override** the recon-default
    ///         seeds for those points and let the refiner align to keypoints
    ///         produced by an upstream localizer (e.g.
    ///         :meth:`localize_keypoints`) rather than the per-observation
    ///         seeds the recon already carries.
    ///
    ///         For a point absent from this map (or for the whole map when
    ///         ``starting_keypoints=None``, the default), seeds come from the
    ///         **recon**: each view's seed is the per-observation inline keypoint
    ///         stored on the embedded_patches recon for that ``(point_index,
    ///         image_index)``, and views with no inline observation seed at their
    ///         own projection ``project_i(X_p)``.
    ///
    ///         **Refinement requires starting keypoints.** Calling
    ///         ``refine_keypoints`` on a ``sift_files`` reconstruction without
    ///         supplying ``starting_keypoints`` raises ``ValueError`` — the
    ///         projection alone isn't a "real" keypoint for the purposes of a
    ///         local refiner. Either run ``sfm xform --to-embedded-patches``
    ///         first or pass explicit ``starting_keypoints`` covering every
    ///         point to refine.
    ///     render_bitmaps: If true, also fuse each point's RGBA representative
    ///         texture at the **final** refined keypoints (final IRLS view
    ///         weights, anisotropic sampling) and return it per point (see
    ///         ``bitmap`` below). Points at infinity take the same render path
    ///         (they are refined, not skipped). Costs one extra full-grid source
    ///         render per view per point, so it is off by default.
    ///
    /// Returns:
    ///     A list of per-point dicts ``{point_index, views (uint32[K]),
    ///     keypoints (float64[K, 2]), offsets_px (float64[K]),
    ///     scores (float64[K])}`` over the views, in **input order** (the view set
    ///     is unchanged; a guard-failed view keeps its seed). ``scores`` is the
    ///     final ECC score (channel-averaged windowed ZNCC), NaN for a view with no
    ///     consensus (fewer than two views). When ``render_bitmaps`` is true each
    ///     dict also carries ``bitmap``: an ``(R, R, 4)`` uint8 RGBA texture fused
    ///     at the final keypoints, or ``None`` when the point produced **no valid
    ///     cross-view consensus** (fewer than two views rendered at their final
    ///     offsets) — the uniform culled-point signal, finite and infinity alike.
    #[pyo3(signature = (
        recon, images, *, view_sets=None, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear", robust_iters=3, max_outer_sweeps=1,
        outer_convergence_px=0.005, max_gn_steps=10, convergence_px=0.01,
        max_offset_px=2.0, consensus_refresh="per_sweep", point_indexes=None,
        starting_keypoints=None, render_bitmaps=false, progress=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn refine_keypoints<'py>(
        &self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        images: &Bound<'py, PyAny>,
        view_sets: Option<std::collections::HashMap<u32, Vec<u32>>>,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        robust_iters: u32,
        max_outer_sweeps: u32,
        outer_convergence_px: f64,
        max_gn_steps: u32,
        convergence_px: f64,
        max_offset_px: f64,
        consensus_refresh: &str,
        point_indexes: Option<Vec<u32>>,
        starting_keypoints: Option<std::collections::HashMap<u32, Vec<[f64; 2]>>>,
        render_bitmaps: bool,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let recon = &recon.inner;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_indexes
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_indexes are out of range for this reconstruction \
                 (was the cloud built from a different recon?)",
            ));
        }
        // `refine_keypoints` is a *local* refiner: it needs a starting
        // keypoint in the basin of the true optimum, and the projection
        // alone isn't a "real" keypoint for that purpose. Require either an
        // embedded_patches recon (which carries inline per-observation
        // keypoints) or explicit ``starting_keypoints`` from the caller.
        // Fail fast before any pyramid decode.
        if starting_keypoints.is_none() && recon.keypoints_xy().is_none() {
            return Err(PyValueError::new_err(
                "refine_keypoints requires starting keypoints — either an \
                 embedded_patches reconstruction (which carries inline \
                 per-observation keypoints; run `sfm xform --to-embedded-patches` \
                 first) or explicit `starting_keypoints` covering every point \
                 to refine. This recon is sift_files and no `starting_keypoints` \
                 were provided.",
            ));
        }

        let window = match window {
            "uniform" => PatchWindow::Uniform,
            "gaussian" => PatchWindow::Gaussian {
                sigma: window_sigma,
            },
            "gaussian_disk" => PatchWindow::GaussianDisk {
                sigma: window_sigma,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"
                )))
            }
        };
        let sampler = match sampler {
            "bilinear" => Sampler::Bilinear,
            "anisotropic" => Sampler::Anisotropic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler: {other:?} (expected bilinear|anisotropic)"
                )))
            }
        };
        let consensus_refresh = match consensus_refresh {
            "per_sweep" => ConsensusRefresh::PerSweep,
            "per_move" => ConsensusRefresh::PerMove,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown consensus_refresh: {other:?} (expected per_sweep|per_move)"
                )))
            }
        };
        let params = KeypointSubpixelParams {
            resolution,
            window,
            sampler,
            robust_iters,
            max_outer_sweeps,
            outer_convergence_px,
            max_gn_steps,
            convergence_px,
            max_offset_px,
            consensus_refresh,
            render_bitmaps,
            ..Default::default()
        };

        let (pyramid_set, poses) = build_pyramids_and_poses(recon, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<ProjectedImage<'_>> = (0..recon.images.len())
            .map(|i| ProjectedImage {
                camera: &recon.cameras[recon.images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
            .collect();

        let mut sets = view_indices_from_reconstruction(recon, &self.inner);
        if let Some(map) = &view_sets {
            let n_images = recon.images.len() as u32;
            for vs in map.values() {
                if let Some(&bad) = vs.iter().find(|&&i| i >= n_images) {
                    return Err(PyValueError::new_err(format!(
                        "view_sets contains image index {bad} out of range for this \
                         reconstruction's {n_images} images"
                    )));
                }
            }
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_indexes) {
                if let Some(vs) = map.get(&pid) {
                    *set = vs.clone();
                }
            }
        }
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_indexes) {
                if !keep.contains(&pid) {
                    set.clear();
                }
            }
        }

        // Per-view seeds in source-image px, one per view in the (final) view
        // set, in order. Sourced in priority:
        //   1. Explicit `starting_keypoints[pid]` from the caller — wraps every
        //      slot in `Some(...)`, overriding any recon-default for that point.
        //   2. Otherwise, when the recon has inline per-observation keypoints
        //      (an embedded_patches recon), each view's slot is `Some(stored)`
        //      where the track has an observation for that (pid, image_index),
        //      and `None` (= project for that view) where it doesn't.
        //   3. Otherwise (a sift-files recon, no inline keypoints), the whole
        //      seed slice is `None` (= project every view).
        //
        // The caller's explicit overrides are validated up front; recon-default
        // seeds are built lazily in the per-patch loop.
        if let Some(seed_map) = &starting_keypoints {
            // Build a point_index -> array-position map once so every per-pid
            // lookup below is O(1); the seed map may cover every patch, in
            // which case the previous per-pid linear scan was O(N²).
            let pid_to_idx: std::collections::HashMap<u32, usize> = self
                .inner
                .point_indexes
                .iter()
                .enumerate()
                .map(|(i, &p)| (p, i))
                .collect();
            for (pid, seeds) in seed_map {
                let Some(&idx) = pid_to_idx.get(pid) else {
                    return Err(PyValueError::new_err(format!(
                        "starting_keypoints[{pid}] is not a point in this patch cloud",
                    )));
                };
                if let Some(keep) = &selected_mask {
                    if !keep.contains(pid) {
                        return Err(PyValueError::new_err(format!(
                            "starting_keypoints[{pid}] is excluded by point_indexes; \
                             drop the entry or include {pid} in point_indexes",
                        )));
                    }
                }
                let set_len = sets[idx].len();
                if seeds.len() != set_len {
                    return Err(PyValueError::new_err(format!(
                        "starting_keypoints[{pid}] has {} seeds but the view set has {} views",
                        seeds.len(),
                        set_len,
                    )));
                }
            }
        }

        // Build the (point_index, image_index) -> stored keypoint lookup once
        // for embedded_patches recons; sift-files recons return None and the
        // per-patch loop just falls through to the projection-seed path. The
        // map is keyed identically to the one in `refine_normals`'s
        // `use_stored_keypoints` path; duplicate observations of the same
        // (point, image) all hash to the same slot (last write wins, harmless).
        let stored_kp_map: Option<std::collections::HashMap<(u32, u32), [f64; 2]>> =
            recon.keypoints_xy().map(|keypoints_xy| {
                let mut m = std::collections::HashMap::with_capacity(recon.tracks.len());
                for (j, obs) in recon.tracks.iter().enumerate() {
                    m.insert(
                        (obs.point_index, obs.image_index),
                        [keypoints_xy[[j, 0]] as f64, keypoints_xy[[j, 1]] as f64],
                    );
                }
                m
            });

        // This binding inlines its own per-patch loop (for lazy per-point seed
        // construction) rather than calling `refine_patch_cloud_keypoints`, so it
        // brackets the shared remap-sampler counters itself; under
        // `SFMTOOL_PROFILE` this reports the stage's value + GN-gradient render
        // taps instead of leaving them unaccounted.
        sfmtool_core::patch::keypoint_subpixel::prof::reset();
        let subpixel_wall = std::time::Instant::now();
        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            use sfmtool_core::patch::keypoint_subpixel::refine_patch_keypoints;
            self.inner
                .patches
                .par_iter()
                .enumerate()
                .map(|(i, patch)| {
                    let pid = self.inner.point_indexes[i];
                    let set = &sets[i];
                    let per_view_seeds: Option<Vec<Option<[f64; 2]>>> = if let Some(user_seeds) =
                        starting_keypoints.as_ref().and_then(|m| m.get(&pid))
                    {
                        Some(user_seeds.iter().map(|&kp| Some(kp)).collect())
                    } else {
                        stored_kp_map
                            .as_ref()
                            .map(|m| set.iter().map(|&img| m.get(&(pid, img)).copied()).collect())
                    };
                    let out = refine_patch_keypoints(
                        patch,
                        &views,
                        set,
                        per_view_seeds.as_deref(),
                        &params,
                    );
                    // Bump the shared work counter per patch for a Python progress poller.
                    if let Some(c) = &progress_handle {
                        c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    out
                })
                .collect::<Vec<_>>()
        });
        sfmtool_core::patch::keypoint_subpixel::prof::report(
            self.inner.patches.len(),
            subpixel_wall.elapsed().as_secs_f64(),
        );

        let mut out = Vec::new();
        for (res, &pid) in results.iter().zip(&self.inner.point_indexes) {
            if let Some(keep) = &selected_mask {
                if !keep.contains(&pid) {
                    continue;
                }
            }
            let flat: Vec<f64> = res.keypoints.iter().flat_map(|k| [k[0], k[1]]).collect();
            let kpts = ndarray::Array2::from_shape_vec((res.keypoints.len(), 2), flat)
                .expect("keypoints shape matches");
            let d = PyDict::new(py);
            d.set_item("point_index", pid)?;
            d.set_item("views", res.views.clone().into_pyarray(py))?;
            d.set_item("keypoints", kpts.into_pyarray(py))?;
            d.set_item("offsets_px", res.offsets_px.clone().into_pyarray(py))?;
            d.set_item("scores", res.scores.clone().into_pyarray(py))?;
            if render_bitmaps {
                // `bitmap` is the point's fused RGBA representative at the final
                // keypoints; `None` marks a point with no valid cross-view
                // consensus (the culled-point signal `embed-patches` drops on).
                match &res.representative {
                    Some(rep) => {
                        let r = resolution.max(2) as usize;
                        let arr = ndarray::Array3::from_shape_vec((r, r, 4), rep.clone())
                            .expect("representative is R*R*4");
                        d.set_item("bitmap", arr.into_pyarray(py))?;
                    }
                    None => d.set_item("bitmap", py.None())?,
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// Score each point's **keypoint positional uncertainty** ``σ_pos`` (source-
    /// image px) from its cross-view consensus ``patch_bitmaps`` and the recon
    /// geometry — the patch-localizability score (see
    /// ``specs/core/patch-localizability.md``). No source images are read.
    ///
    /// The noise-normalized structure tensor of each consensus patch gives the
    /// weak-axis uncertainty in patch-grid px (``σ_pos_grid = σ_noise /
    /// √λ₂_sum``); that is mapped to source px per observing view by the patch's
    /// projected scale (``half_extent / (R/2) · f / depth``, canonical ``depth =
    /// −(R·X + t)_z``) and reduced by the **median** over the point's views.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides poses,
    ///         intrinsics, point positions, and the per-point observing-view lists
    ///         via its tracks). The cloud's ``point_indexes`` must index it.
    ///     patch_bitmaps: An ``(N, R, R, C)`` uint8 consensus stack scattered per
    ///         **source 3D point** (``N`` = ``recon`` point count; zero rows for
    ///         points with no consensus) — e.g. ``recon.patch_bitmaps`` or the
    ///         ``bitmaps`` from a ``render_bitmaps`` refine pass. Luminance
    ///         (``0.299R + 0.587G + 0.114B``) is scored; alpha is used only to tell
    ///         a covered flat patch from an empty (culled) one.
    ///     sigma_noise: Global consensus photometric-noise constant (intensity
    ///         units, ~3 gray levels for a u8 consensus). Sets the absolute px
    ///         scale of ``σ_pos``; the per-recon ranking is scale-free.
    ///     window: Scoring window — ``"gaussian_disk"`` (default), ``"gaussian"``,
    ///         or ``"uniform"`` — matched to the consensus render window.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///
    /// Returns a dict of per-**source-point** numpy arrays (length ``N``, parallel
    /// to ``recon``'s points): ``sigma_pos_grid`` (grid px — **the cull quantity**;
    /// intrinsic and resolution-independent, so a fixed threshold transfers across
    /// datasets; NaN where the patch is empty), ``sigma_pos_px`` (source px, via the
    /// grid→px map, median over views — a diagnostic that does *not* transfer;
    /// additionally NaN where no view gives a finite depth), ``lam1``/``lam2``
    /// (structure-tensor eigenvalues, summed form), and ``theta`` (radians; the
    /// weak-axis / slide direction).
    #[pyo3(signature = (
        recon, patch_bitmaps, *, sigma_noise=3.0, window="gaussian_disk", window_sigma=0.6
    ))]
    fn score_localizability<'py>(
        &self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        patch_bitmaps: PyReadonlyArray4<'py, u8>,
        sigma_noise: f64,
        window: &str,
        window_sigma: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let recon = &recon.inner;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_indexes
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_indexes are out of range for this reconstruction \
                 (was the cloud built from a different recon?)",
            ));
        }
        let bm = patch_bitmaps.as_array();
        let (n, r, r2, c) = (bm.shape()[0], bm.shape()[1], bm.shape()[2], bm.shape()[3]);
        if r != r2 {
            return Err(PyValueError::new_err(format!(
                "patch_bitmaps must be square R×R per point, got {r}×{r2}"
            )));
        }
        if n != recon.points.len() {
            return Err(PyValueError::new_err(format!(
                "patch_bitmaps has {n} rows but the reconstruction has {} points; \
                 pass a stack scattered per source 3D point",
                recon.points.len()
            )));
        }
        let window = parse_patch_window(window, window_sigma)?;

        // Logical (row-major) copy of the consensus stack as f32. `iter()` walks
        // the array in C order regardless of its physical layout, so the flat
        // buffer is the `(N, R, R, C)` the batch scorer expects.
        let flat: Vec<f32> = bm.iter().map(|&v| v as f32).collect();

        // Per-point half-extent (world) from the cloud frames, scattered to source
        // points; NaN for points the cloud has no patch for.
        let mut half = vec![f64::NAN; n];
        for (i, &pid) in self.inner.point_indexes.iter().enumerate() {
            half[pid as usize] = self.inner.patch(i).half_extent[0];
        }
        // One pose + mean focal per image.
        let poses: Vec<RigidTransform> = recon
            .images
            .iter()
            .map(|im| {
                let q = im.quaternion_wxyz;
                RigidTransform::from_wxyz_translation(
                    [q.w, q.i, q.j, q.k],
                    [
                        im.translation_xyz.x,
                        im.translation_xyz.y,
                        im.translation_xyz.z,
                    ],
                )
            })
            .collect();
        let focal: Vec<f64> = recon
            .images
            .iter()
            .map(|im| {
                let (fx, fy) = recon.cameras[im.camera_index as usize].focal_lengths();
                0.5 * (fx + fy)
            })
            .collect();

        // The structure-tensor scoring is the expensive, rayon-parallel part; run
        // it with the GIL released. The grid→source-px map below is a cheap
        // per-observation pass, so it stays on the GIL-holding thread.
        let scores = py.detach(|| score_localizability_stack(&flat, n, r, c, window, sigma_noise));

        // Median-over-views grid→source-px scale per point (canonical −Z depth:
        // `depth = -(R·X + t)_z`).
        let half_r = r as f64 / 2.0;
        let mut scale = vec![f64::NAN; n];
        for (p_idx, s) in scale.iter_mut().enumerate() {
            let h = half[p_idx];
            if !h.is_finite() {
                continue;
            }
            let point = &recon.points[p_idx];
            let mut vals: Vec<f64> = Vec::new();
            for obs in &recon.tracks
                [recon.observation_offsets[p_idx]..recon.observation_offsets[p_idx + 1]]
            {
                let im = obs.image_index as usize;
                // Homogeneous transform so a point at infinity (`w == 0`, whose
                // `position` is a direction) is rotated without translation —
                // `R·d` rather than `R·d + t` — giving a meaningful bearing depth.
                let depth = -poses[im]
                    .transform_point_homogeneous(point.position.coords, point.w)
                    .z;
                if depth > 1e-6 {
                    vals.push((h / half_r) * focal[im] / depth);
                }
            }
            if !vals.is_empty() {
                *s = np_median(&mut vals);
            }
        }

        let mut sigma_px = Vec::with_capacity(n);
        let mut sigma_grid = Vec::with_capacity(n);
        let mut lam1 = Vec::with_capacity(n);
        let mut lam2 = Vec::with_capacity(n);
        let mut theta = Vec::with_capacity(n);
        for (s, &sc) in scores.iter().zip(&scale) {
            sigma_px.push(s.sigma_pos_grid * sc);
            sigma_grid.push(s.sigma_pos_grid);
            lam1.push(s.lam1);
            lam2.push(s.lam2);
            theta.push(s.theta);
        }

        let out = PyDict::new(py);
        out.set_item("sigma_pos_px", sigma_px.into_pyarray(py))?;
        out.set_item("sigma_pos_grid", sigma_grid.into_pyarray(py))?;
        out.set_item("lam1", lam1.into_pyarray(py))?;
        out.set_item("lam2", lam2.into_pyarray(py))?;
        out.set_item("theta", theta.into_pyarray(py))?;
        Ok(out)
    }

    /// The scorer's `R×R` window weights (row-major), for `window` (default
    /// ``"gaussian_disk"``) at `window_sigma`. The same kernel
    /// [`score_localizability`](Self::score_localizability) accumulates the
    /// structure tensor over — exposed so callers score against it rather than
    /// reimplementing the window formula.
    #[staticmethod]
    #[pyo3(signature = (resolution, *, window="gaussian_disk", window_sigma=0.6))]
    fn window_weights<'py>(
        py: Python<'py>,
        resolution: usize,
        window: &str,
        window_sigma: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let w = parse_patch_window(window, window_sigma)?;
        let weights = localizability_window_weights(w, resolution as u32);
        let arr = Array2::from_shape_vec((resolution, resolution), weights)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }
}
