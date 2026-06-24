// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for sfmtool-core oriented patches.

use nalgebra::{Point3, Vector3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::camera::remap::ImageU8Pyramid;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::{OrientedPatch, PatchCloud, PatchExtent, PatchNormal, ViewReduce};
use sfmtool_core::patch::keypoint_localize::{
    localize_patch_cloud_keypoints, KeypointLocalizeParams,
};
use sfmtool_core::patch::normal_refine::{
    refine_patch_cloud, view_indices_from_reconstruction, CacheMode, NormalRefineParams, Objective,
    PatchWindow, ProjectedImage, Sampler,
};
use sfmtool_core::patch::view_selection::{select_patch_cloud_views, ViewSelectParams};

use crate::flow::warp::extract_image_u8;
use crate::py_rigid_transform::PyRigidTransform;
use crate::py_sfmr_reconstruction::PySfmrReconstruction;

/// An oriented planar patch (surfel) in world space.
///
/// The plane is spanned by orthonormal in-plane axes ``u_axis`` and ``v_axis``;
/// the outward normal is ``u_axis × v_axis``. The patch covers the world points
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
    ///     up_hint: In-plane reference direction; projected onto the plane to set
    ///         ``u_axis``. If parallel to the normal an arbitrary axis is used.
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

/// Build one source-image pyramid + camera pose per reconstruction image,
/// validating that each image matches its camera resolution. Shared by
/// `refine_normals` and `select_views` so they handle imagery identically.
fn build_pyramids_and_poses(
    recon: &sfmtool_core::SfmrReconstruction,
    images: &[Bound<'_, PyAny>],
) -> PyResult<(Vec<ImageU8Pyramid>, Vec<RigidTransform>)> {
    if images.len() != recon.images.len() {
        return Err(PyValueError::new_err(format!(
            "images must be parallel to the reconstruction's {} images, got {}",
            recon.images.len(),
            images.len()
        )));
    }
    let pyramids: Vec<ImageU8Pyramid> = images
        .iter()
        .enumerate()
        .map(|(i, im)| {
            let src = extract_image_u8(im)?;
            let cam = &recon.cameras[recon.images[i].camera_index as usize];
            if src.width() != cam.width || src.height() != cam.height {
                return Err(PyValueError::new_err(format!(
                    "image {i} is {}x{} but its camera is {}x{}; \
                     pass full-resolution images",
                    src.width(),
                    src.height(),
                    cam.width,
                    cam.height
                )));
            }
            let min_dim = src.width().min(src.height()).max(1);
            let max_levels = ((min_dim as f32).log2().floor() as usize).max(1) + 1;
            Ok(ImageU8Pyramid::build(&src, max_levels))
        })
        .collect::<PyResult<_>>()?;
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
    ///     extent_value: The scalar for the chosen extent policy (default 5.0; for
    ///         ``"feature_size"`` the keypoint-scale multiplier).
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
        extent="feature_size", extent_value=5.0,
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
    /// ``point_ids`` entry is that row index, with the half-extent vectors split
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
    fn point_ids(&self) -> Vec<u32> {
        self.inner.point_ids.clone()
    }

    /// Refine every patch's normal in place by photometric consistency across
    /// the reconstruction's observing views (see
    /// ``specs/core/patch-normal-refinement.md``).
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides cameras,
    ///         poses, and the per-point observing-image lists via ``point_ids``).
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index).
    ///     resolution: The R×R patch grid the consensus is scored on.
    ///     objective: ``"robust"`` (IRLS-weighted consensus, default) or
    ///         ``"mean"`` (unweighted all-pairs consensus).
    ///     window: Per-pixel scoring weight — ``"gaussian_disk"`` (default),
    ///         ``"gaussian"``, or ``"uniform"``.
    ///     sampler: How to sample the source pyramids — ``"bilinear"`` (default;
    ///         fastest, and the found normal barely differs) or ``"anisotropic"``
    ///         (anti-aliased oblique views; keeps the reported Φ/confidence
    ///         unbiased, ~1.6-3x slower).
    ///     point_ids: If given, refine only the patches with these source point
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
    ///         observations. Combines with ``point_ids`` (which still selects
    ///         *which* patches to refine).
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
        search_robust_iters=None, point_ids=None, view_indices=None, render_bitmaps=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn refine_normals<'py>(
        &mut self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        images: Vec<Bound<'py, PyAny>>,
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
        point_ids: Option<Vec<u32>>,
        view_indices: Option<Vec<Vec<u32>>>,
        render_bitmaps: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let recon = &recon.inner;
        // The cloud must have been built from this reconstruction: its per-patch
        // point_ids index `recon`'s points. This is a range check only — it
        // catches a too-small recon (and would-be panics in the core), but cannot
        // detect a *different* recon with at least as many points; the caller is
        // responsible for passing the recon the cloud was built from.
        if self.inner.point_ids.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_ids; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_ids
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_ids are out of range for this reconstruction \
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
            render_bitmap: render_bitmaps,
        };

        // Build one pyramid + pose per reconstruction image; the ProjectedImages
        // borrow these for the duration of the refinement. The warp validity /
        // sampling assume each image matches its camera's resolution, so the
        // helper rejects mismatched sizes.
        let (pyramids, poses) = build_pyramids_and_poses(recon, &images)?;
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
        if let Some(ids) = point_ids {
            let keep: std::collections::HashSet<u32> = ids.into_iter().collect();
            for (pv, &pid) in patch_views.iter_mut().zip(&self.inner.point_ids) {
                if !keep.contains(&pid) {
                    pv.clear();
                }
            }
        }

        let results = py.detach(|| {
            refine_patch_cloud(&mut self.inner, &views, &patch_views, resolution, &params)
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
            for (res, &pid) in results.iter().zip(&self.inner.point_ids) {
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
    ///         poses, and the per-point track view lists via ``point_ids``).
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index).
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
    ///     point_ids: If given, select only for the patches with these source
    ///         point ids; ``None`` (default) selects for every patch.
    ///
    /// Returns a list of per-patch dicts (parallel to the cloud's patches, in
    /// cloud order): ``point_id`` (int), ``admitted`` (1-D int32 numpy array of
    /// image indices — the track views first, then the vetted candidates in
    /// ascending order), ``scores`` (1-D float64 numpy array of the per-admitted
    /// ZNCC to the reference, parallel to ``admitted``; NaN where a view could not
    /// be scored), and ``self_agreement`` (float; NaN when no reference could be
    /// built). Patches excluded by ``point_ids`` are omitted from the list.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        recon, images, *, min_relative_zncc=0.7, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear", min_valid_fraction=0.6, min_track_views=2,
        robust_iters=3, min_self_agreement=0.3, point_ids=None
    ))]
    fn select_views<'py>(
        &self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        images: Vec<Bound<'py, PyAny>>,
        min_relative_zncc: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        min_valid_fraction: f64,
        min_track_views: u32,
        robust_iters: u32,
        min_self_agreement: f64,
        point_ids: Option<Vec<u32>>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let recon = &recon.inner;
        if self.inner.point_ids.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_ids; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_ids
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_ids are out of range for this reconstruction \
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

        let (pyramids, poses) = build_pyramids_and_poses(recon, &images)?;
        let views: Vec<ProjectedImage<'_>> = (0..recon.images.len())
            .map(|i| ProjectedImage {
                camera: &recon.cameras[recon.images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
            .collect();

        // Per-patch track view lists from the reconstruction; an empty list makes
        // a patch's selection trivially empty, so `point_ids` selects a subset by
        // clearing the rest.
        let mut track_views = view_indices_from_reconstruction(recon, &self.inner);
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_ids.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (tv, &pid) in track_views.iter_mut().zip(&self.inner.point_ids) {
                if !keep.contains(&pid) {
                    tv.clear();
                }
            }
        }

        let results =
            py.detach(|| select_patch_cloud_views(&self.inner, &views, &track_views, &params));

        let mut out = Vec::new();
        for (res, &pid) in results.iter().zip(&self.inner.point_ids) {
            if let Some(keep) = &selected_mask {
                if !keep.contains(&pid) {
                    continue;
                }
            }
            let d = PyDict::new(py);
            d.set_item("point_id", pid)?;
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
    ///         the per-point track view lists via ``point_ids``).
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index).
    ///     view_sets: Optional mapping ``point_id -> [image_index, ...]`` giving the
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
    ///     convergence_px: Stop once the mean per-view residual shift of a round is
    ///         below this many patch-grid px.
    ///     point_ids: If given, localize only for the patches with these source
    ///         point ids; ``None`` (default) localizes for every patch.
    ///
    /// Returns:
    ///     A list of per-point dicts ``{point_id, views (uint32[K]),
    ///     keypoints (float64[K, 2]), offsets_px (float64[K]),
    ///     loo_zncc (float64[K])}`` over the **kept** views. ``loo_zncc`` is NaN for
    ///     a view no round scored (a lone input view, or a view kept by the two-view
    ///     floor before any consensus was built), so guard before reducing it.
    #[pyo3(signature = (
        recon, images, *, view_sets=None, max_iters=5, search=6.0, max_shift_px=3.0,
        min_relative_zncc=0.7, min_grazing_cos=0.1, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="bilinear", robust_iters=3, convergence_px=0.05,
        point_ids=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn localize_keypoints<'py>(
        &self,
        py: Python<'py>,
        recon: &PySfmrReconstruction,
        images: Vec<Bound<'py, PyAny>>,
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
        point_ids: Option<Vec<u32>>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let recon = &recon.inner;
        if self.inner.point_ids.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_ids; rebuild it with from_reconstruction",
            ));
        }
        if self
            .inner
            .point_ids
            .iter()
            .any(|&p| p as usize >= recon.points.len())
        {
            return Err(PyValueError::new_err(
                "patch cloud point_ids are out of range for this reconstruction \
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
        };

        let (pyramids, poses) = build_pyramids_and_poses(recon, &images)?;
        let views: Vec<ProjectedImage<'_>> = (0..recon.images.len())
            .map(|i| ProjectedImage {
                camera: &recon.cameras[recon.images[i].camera_index as usize],
                cam_from_world: &poses[i],
                pyramid: &pyramids[i],
            })
            .collect();

        // Per-patch view sets: the supplied map where present, else the track. An
        // empty view set makes a patch's localization trivially empty, so
        // `point_ids` selects a subset by clearing the rest.
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
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_ids) {
                if let Some(vs) = map.get(&pid) {
                    *set = vs.clone();
                }
            }
        }
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_ids.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (set, &pid) in sets.iter_mut().zip(&self.inner.point_ids) {
                if !keep.contains(&pid) {
                    set.clear();
                }
            }
        }

        let results =
            py.detach(|| localize_patch_cloud_keypoints(&self.inner, &views, &sets, None, &params));

        let mut out = Vec::new();
        for (res, &pid) in results.iter().zip(&self.inner.point_ids) {
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
            d.set_item("point_id", pid)?;
            d.set_item("views", res.views.clone().into_pyarray(py))?;
            d.set_item("keypoints", kpts.into_pyarray(py))?;
            d.set_item("offsets_px", res.offsets_px.clone().into_pyarray(py))?;
            d.set_item("loo_zncc", res.loo_zncc.clone().into_pyarray(py))?;
            out.push(d);
        }
        Ok(out)
    }
}
