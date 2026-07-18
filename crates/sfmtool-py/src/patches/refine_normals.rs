// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.refine_normals`: photometric normal refinement.

use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::normal_refine::{
    refine_patch_cloud_normals, view_indices_from_reconstruction, CacheMode, NormalRefineParams,
    Objective, PatchWindow, ProjectedImage, Sampler,
};

use super::cloud::PyPatchCloud;
use super::views::{resolve_pyramids, resolve_scene};
use crate::ProgressCounter;

#[pymethods]
impl PyPatchCloud {
    /// Refine every patch's normal in place by photometric consistency across
    /// the reconstruction's observing views (see
    /// ``specs/core/patch-normal-refinement.md``).
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides cameras,
    ///         poses, and the per-point observing-image lists via ``point_indexes``),
    ///         **or** a :class:`CameraViews` — which carries no tracks, so
    ///         ``view_indices`` becomes required.
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
    ///         fastest, and the found normal barely differs), ``"bilinear_mip"``
    ///         (single bilinear tap from the nearest mip level; bounds aliasing on
    ///         cross-scale views at ~bilinear cost), or ``"anisotropic"``
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
        recon: &Bound<'py, PyAny>,
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
        // Resolve the scene: a reconstruction (track-derived per-patch view
        // defaults) or a bare CameraViews (no tracks — `view_indices` required).
        let (posed, recon_guard) = resolve_scene(recon)?;
        let recon_opt = recon_guard.as_ref().map(|r| &r.inner);
        let n_images = posed.len() as u32;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
        // The cloud's per-patch point_indexes index the reconstruction's points, so
        // this range check catches a too-small recon (and would-be panics in the
        // core). A CameraViews carries no points, so it does not apply there;
        // supplied view lists are still validated against the view count below.
        if let Some(recon) = recon_opt {
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
        }
        // Without tracks there is no default per-patch view list, so `view_indices`
        // is required. Fail fast before decoding any imagery.
        if recon_opt.is_none() && view_indices.is_none() {
            return Err(PyValueError::new_err(
                "view_indices is required when the first argument is a CameraViews \
                 (there are no tracks to derive per-patch views from)",
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
            "bilinear_mip" => Sampler::BilinearMip,
            "anisotropic" => Sampler::Anisotropic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler: {other:?} (expected bilinear|bilinear_mip|anisotropic)"
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

        // Build one pyramid + pose per image; the ProjectedImages borrow these for
        // the duration of the refinement. The warp validity / sampling assume each
        // image matches its camera's resolution, so the helper rejects mismatched
        // sizes.
        let pyramid_set = resolve_pyramids(&posed, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<ProjectedImage<'_>> = (0..posed.len())
            .map(|i| ProjectedImage {
                camera: &posed.cameras[i],
                cam_from_world: &posed.poses[i],
                pyramid: &pyramids[i],
            })
            .collect();
        // The per-patch view sets the consensus is scored over. In reconstruction
        // mode these default to the track observations; `view_indices` overrides
        // them with an explicit per-patch list (also the only source in views mode).
        let mut patch_views = match view_indices {
            Some(mut views) => {
                if views.len() != self.inner.len() {
                    return Err(PyValueError::new_err(format!(
                        "view_indices must be parallel to the cloud's {} patches, got {}",
                        self.inner.len(),
                        views.len()
                    )));
                }
                if views.iter().flatten().any(|&i| i >= n_images) {
                    return Err(PyValueError::new_err(
                        "view_indices contains an image index out of range for this scene",
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
            // Guaranteed `Some` in views mode by the early required-arg check.
            None => view_indices_from_reconstruction(recon_opt.unwrap(), &self.inner),
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
        // Only a reconstruction carries inline keypoints; a CameraViews has none,
        // so every view falls through per-view to the reprojected center.
        let patch_view_keypoints: Option<Vec<Vec<Option<[f64; 2]>>>> = if use_stored_keypoints {
            recon_opt.and_then(|recon| {
                recon.keypoints_xy().map(|keypoints_xy| {
                    // (point_index, image_index) -> stored keypoint. `keypoints_xy`
                    // is parallel to `recon.tracks`, so walk the tracks once and
                    // index by observation row. Duplicate (point, image)
                    // observations all key to the same map entry, so any duplicate
                    // view resolves to the same keypoint (last write wins, harmless).
                    let mut kp_map: std::collections::HashMap<(u32, u32), [f64; 2]> =
                        std::collections::HashMap::with_capacity(recon.tracks.len());
                    for (j, obs) in recon.tracks.iter().enumerate() {
                        kp_map.insert(
                            (obs.point_index, obs.image_index),
                            [keypoints_xy[[j, 0]] as f64, keypoints_xy[[j, 1]] as f64],
                        );
                    }
                    // Build per-patch keypoints parallel to `patch_views`: for each
                    // patch's (point_index, image_index) view, the stored keypoint
                    // when present (else `None`, which the core treats as "anchor at
                    // the reprojected center for this view").
                    patch_views
                        .iter()
                        .zip(&self.inner.point_indexes)
                        .map(|(pv, &pidx)| {
                            pv.iter()
                                .map(|&img| kp_map.get(&(pidx, img)).copied())
                                .collect()
                        })
                        .collect()
                })
            })
        } else {
            None
        };

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
            // Scatter per source 3D point. In views mode there is no reconstruction
            // point count, so size to the highest referenced point id.
            let npoints = recon_opt
                .map(|recon| recon.points.len())
                .unwrap_or_else(|| {
                    self.inner
                        .point_indexes
                        .iter()
                        .copied()
                        .max()
                        .map_or(0, |m| m as usize + 1)
                });
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
}
