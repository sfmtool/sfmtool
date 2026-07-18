// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.refine_keypoints`: sub-pixel LK/ECC keypoint refinement.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use sfmtool_core::patch::keypoint_subpixel::{ConsensusRefresh, KeypointSubpixelParams};
use sfmtool_core::patch::normal_refine::{
    view_indices_from_reconstruction, PatchWindow, ProjectedImage, Sampler,
};

use super::cloud::PyPatchCloud;
use super::views::{resolve_pyramids, resolve_scene};
use crate::ProgressCounter;

#[pymethods]
impl PyPatchCloud {
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
    ///         the per-point track view lists via ``point_indexes``), **or** a
    ///         :class:`CameraViews` — which carries no tracks, so ``view_sets``
    ///         becomes required (and, having no inline keypoints, so does
    ///         ``starting_keypoints``).
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
    ///     sampler: ``"bilinear"`` (default), ``"bilinear_mip"``, or
    ///         ``"anisotropic"`` (value and gradient are rendered with the same
    ///         sampler).
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
        recon: &Bound<'py, PyAny>,
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
        let (posed, recon_guard) = resolve_scene(recon)?;
        let recon_opt = recon_guard.as_ref().map(|r| &r.inner);
        let n_images = posed.len() as u32;
        if self.inner.point_indexes.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "patch cloud has no per-patch point_indexes; rebuild it with from_reconstruction",
            ));
        }
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
        // Without tracks there is no default per-patch view list, so `view_sets` is
        // required. Fail fast before decoding any imagery.
        if recon_opt.is_none() && view_sets.is_none() {
            return Err(PyValueError::new_err(
                "view_sets is required when the first argument is a CameraViews \
                 (there are no tracks to derive per-patch views from)",
            ));
        }
        // `refine_keypoints` is a *local* refiner: it needs a starting keypoint in
        // the basin of the true optimum, and the projection alone isn't a "real"
        // keypoint for that purpose. Require either an embedded_patches recon (which
        // carries inline per-observation keypoints) or explicit ``starting_keypoints``
        // from the caller. A CameraViews carries no keypoints, so it always requires
        // ``starting_keypoints``. Fail fast before any pyramid decode.
        if starting_keypoints.is_none() && recon_opt.is_none_or(|r| r.keypoints_xy().is_none()) {
            return Err(PyValueError::new_err(
                "refine_keypoints requires starting keypoints — either an \
                 embedded_patches reconstruction (which carries inline \
                 per-observation keypoints; run `sfm xform --to-embedded-patches` \
                 first) or explicit `starting_keypoints` covering every point \
                 to refine. This scene has no inline keypoints and no \
                 `starting_keypoints` were provided.",
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
            "bilinear_mip" => Sampler::BilinearMip,
            "anisotropic" => Sampler::Anisotropic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler: {other:?} (expected bilinear|bilinear_mip|anisotropic)"
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

        let pyramid_set = resolve_pyramids(&posed, images)?;
        let pyramids = pyramid_set.as_slice();
        let views: Vec<ProjectedImage<'_>> = (0..posed.len())
            .map(|i| ProjectedImage {
                camera: &posed.cameras[i],
                cam_from_world: &posed.poses[i],
                pyramid: &pyramids[i],
            })
            .collect();

        let mut sets = match recon_opt {
            Some(recon) => view_indices_from_reconstruction(recon, &self.inner),
            None => vec![Vec::new(); self.inner.len()],
        };
        if let Some(map) = &view_sets {
            for vs in map.values() {
                if let Some(&bad) = vs.iter().find(|&&i| i >= n_images) {
                    return Err(PyValueError::new_err(format!(
                        "view_sets contains image index {bad} out of range for this \
                         scene's {n_images} views"
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
        let stored_kp_map: Option<std::collections::HashMap<(u32, u32), [f64; 2]>> = recon_opt
            .and_then(|recon| {
                recon.keypoints_xy().map(|keypoints_xy| {
                    let mut m = std::collections::HashMap::with_capacity(recon.tracks.len());
                    for (j, obs) in recon.tracks.iter().enumerate() {
                        m.insert(
                            (obs.point_index, obs.image_index),
                            [keypoints_xy[[j, 0]] as f64, keypoints_xy[[j, 1]] as f64],
                        );
                    }
                    m
                })
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
}
