// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.localize_keypoints`: discrete keypoint search against the
//! reference render.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::keypoint_localize::{
    localize_patch_cloud_keypoints, KeypointLocalizeParams,
    SearchStrategy as LocalizeSearchStrategy,
};
use sfmtool_core::patch::normal_refine::{view_indices_from_reconstruction, ProjectedImage};

use super::args::{parse_patch_window, parse_sampler, reference_positions};
use super::cloud::PyPatchCloud;
use super::views::{resolve_patch_scene, resolve_pyramids};
use crate::ProgressCounter;
use sfmtool_core::progress::Progress;

#[pymethods]
impl PyPatchCloud {
    /// Refine, per patch, the per-view 2D keypoints by aligning every view to
    /// the point's **reference render**: the template is the reference
    /// observation's ``R×R`` tile at its own starting keypoint, and each other
    /// view is searched once around its starting keypoint for the shift whose
    /// tile best matches it (an integer search, then a sub-pixel step from a
    /// quadratic fit over the 3×3 cells around the peak). The reference's keypoint
    /// is returned exactly as given. Views that
    /// drift too far, leave the frame, do not pin a position or do not match
    /// the reference are dropped. See
    /// ``specs/core/patch/patch-keypoint-localization.md``.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (cameras, poses, and
    ///         the per-point track view lists via ``point_indexes``), **or** a
    ///         :class:`CameraViews` — which carries no tracks, so ``view_sets``
    ///         becomes required.
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images (decode the
    ///         pyramids once, share them across kernel calls).
    ///     view_sets: Optional mapping ``point_index -> [image_index, ...]`` giving the
    ///         view set to refine per point (typically the output of
    ///         :meth:`select_views`). Points absent from the map fall back to their
    ///         track; ``None`` (default) uses the track for every point.
    ///     search: The reach of each view's search around its starting keypoint,
    ///         in patch-grid px (also the context-tile margin).
    ///     max_shift_px: Drop a view whose refined keypoint sits more than this many
    ///         source-image px from the point's projection.
    ///     min_relative_zncc: Drop a view whose ZNCC against the reference render
    ///         falls below this fraction of the median over the point's other
    ///         views (the reference left out). ``0`` disables it exactly.
    ///     min_absolute_zncc: Drop a view whose ZNCC against the reference render
    ///         is finite and below this **absolute** floor. Default ``0.5``;
    ///         ``0`` disables it exactly.
    ///     max_member_zncc_self_similarity_radius: Drop a view whose **own**
    ///         rendered core tile does not pin a 2D position: its ZNCC
    ///         self-similarity radius, how far the core can slide over itself
    ///         and still match itself as well as a true match between two
    ///         views would (grid px of the search grid), is above this bar. A
    ///         flat sky tile or a lone straight edge matches itself a few
    ///         pixels away, so it is refused before it is scored. Not applied
    ///         to the reference observation. A ``NaN`` radius fails. The
    ///         radius reads at most ``3``, so a bar of ``3`` or more turns
    ///         nothing out, and ``0`` disables the gate exactly. Default
    ///         ``2.5`` (see ``specs/core/patch/patch-keypoint-localization.md``,
    ///         "The member gate's default").
    ///     min_grazing_cos: Grazing cutoff; drop a view whose ray is near-parallel
    ///         to the patch plane (``|d·n|`` below this).
    ///     resolution: The R×R patch grid the template and the ZNCC are scored on.
    ///     window: ``"gaussian_disk"`` (default), ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"per_view"`` (default: the sampler rule picks
    ///         ``"anisotropic"`` or ``"bilinear_mip"`` for each view from its
    ///         zoom), or one sampler for every view, ``"bilinear_mip"``,
    ///         ``"bilinear"`` or ``"anisotropic"``.
    ///     robust_iters: IRLS passes for the fused mean that is the template where
    ///         the reference-view rule picks no reference it would store.
    ///     point_indexes: If given, localize only for the patches with these source
    ///         point ids; ``None`` (default) localizes for every patch.
    ///     starting_keypoints: Optional explicit per-view seeds:
    ///         ``point_index -> [[x, y], ...]`` in **source-image** pixels,
    ///         parallel to that point's (final) view set — one entry per view, in
    ///         order. Same shape as :meth:`refine_keypoints`'s parameter of the
    ///         same name, with one addition: an entry may be ``None`` instead of
    ///         an ``[x, y]`` pair, seeding **that** view at the point's own
    ///         projection while its siblings keep their explicit seeds. A point
    ///         absent from the map (and every point when this is ``None``, the
    ///         default) seeds each of its views at the point's own projection
    ///         ``project_i(X_p)``, which is exactly today's behaviour — as does an
    ///         all-``None`` list.
    ///
    ///         Seeding localization around the caller's own keypoints rather
    ///         than around the projection starts the search from the appearance
    ///         the caller trusts (the observation that was actually matched)
    ///         rather than from a position carrying the point's reprojection
    ///         residual. The per-view ``None`` is what makes that usable on a
    ///         view set that mixes observed views with expansion candidates: the
    ///         candidates have no observation, hence no keypoint, and take the
    ///         projection.
    ///     search_strategy: ``"plus_descent"`` (default: climb from the starting
    ///         keypoint to the nearest correlation peak) or ``"exhaustive"``
    ///         (score every shift in the window and take the best).
    ///     reference_images: Optional mapping ``point_index -> image_index`` naming
    ///         each point's reference observation, the view whose render is the
    ///         template and whose keypoint is not moved. A point absent from the
    ///         map, mapped to ``None`` or ``-1``, or whose image is not in its view
    ///         set, has the reference-view rule pick one from the renders at the
    ///         starting keypoints. ``None`` (default) has the rule pick for every
    ///         point.
    ///         A point index the cloud does not have, or an image index that is
    ///         neither ``-1`` nor one of the scene's images, raises ``ValueError``.
    ///     progress: Optional :class:`ProgressCounter`, bumped once per patch.
    ///
    /// Returns:
    ///     A list of per-point dicts ``{point_index, views (uint32[K]),
    ///     keypoints (float64[K, 2]), offsets_px (float64[K]), zncc (float64[K]),
    ///     reference_image}`` over the **kept** views. ``zncc`` is each view's
    ///     plain ZNCC against the template at its integer peak: ``1.0`` for the
    ///     reference observation, and NaN for a view that was not searched
    ///     because there was no template, so guard before reducing it.
    ///     ``reference_image`` is the image index of the reference observation
    ///     the views were aligned to, or ``None`` where the rule picked none it
    ///     would store (the template was the fused mean of the views) or nothing
    ///     rendered. The reference is always kept when it renders; every other
    ///     view can be dropped, so ``K`` can be below two for the caller's
    ///     ``min_views`` cull to remove.
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (
        recon, images, *, view_sets=None, search=6.0, max_shift_px=3.0,
        min_relative_zncc=0.7, min_absolute_zncc=0.5, max_member_zncc_self_similarity_radius=2.5,
        min_grazing_cos=0.1, resolution=24, window="gaussian_disk",
        window_sigma=0.6, sampler="per_view", robust_iters=3,
        point_indexes=None, starting_keypoints=None,
        search_strategy="plus_descent", reference_images=None, progress=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn localize_keypoints<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        view_sets: Option<std::collections::HashMap<u32, Vec<u32>>>,
        search: f64,
        max_shift_px: f64,
        min_relative_zncc: f64,
        min_absolute_zncc: f64,
        max_member_zncc_self_similarity_radius: f64,
        min_grazing_cos: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        robust_iters: u32,
        point_indexes: Option<Vec<u32>>,
        starting_keypoints: Option<std::collections::HashMap<u32, Vec<Option<[f64; 2]>>>>,
        search_strategy: &str,
        reference_images: Option<std::collections::HashMap<u32, Option<i64>>>,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let (posed, recon_guard, n_images) = resolve_patch_scene(
            recon,
            &self.inner,
            view_sets.is_some(),
            "view_sets",
            "per-patch views",
        )?;
        let recon_opt = recon_guard.as_ref().map(|r| &r.inner);

        let window = parse_patch_window(window, window_sigma)?;
        let sampler = parse_sampler(sampler)?;
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
            search,
            max_shift_px,
            min_relative_zncc,
            min_absolute_zncc,
            max_member_zncc_self_similarity_radius,
            min_grazing_cos,
            resolution,
            window,
            sampler,
            robust_iters,
            search_strategy,
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

        // Per-patch view sets: the supplied map where present, else the track (in
        // views mode there is no track, so the base is empty and `view_sets` — which
        // is required — supplies every list). An empty view set makes a patch's
        // localization trivially empty, so `point_indexes` selects a subset by
        // clearing the rest.
        let mut sets = match recon_opt {
            Some(recon) => view_indices_from_reconstruction(recon, &self.inner),
            None => vec![Vec::new(); self.inner.len()],
        };
        if let Some(map) = &view_sets {
            // Reject out-of-range image indices up front so the kernel never indexes
            // `views` out of bounds (which would surface as an opaque panic rather
            // than a clean error). The kernel dedups, so duplicates are fine here.
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
        // Per-patch explicit seeds, parallel to `sets`. A point absent from the map
        // gets an EMPTY list, which the kernel reads as "unseeded" — that patch's
        // views seed at the projection, i.e. exactly the historical behaviour that
        // `starting_keypoints=None` keeps for the whole cloud; a `None` entry
        // inside a listed point's seeds says the same thing for that one view. A
        // length mismatch against the point's view set would silently mis-pair
        // seeds with views, so reject it up front (mirroring `refine_keypoints`).
        let seeds_per_patch: Option<Vec<Vec<Option<[f64; 2]>>>> = match &starting_keypoints {
            None => None,
            Some(map) => {
                let pid_to_idx: std::collections::HashMap<u32, usize> = self
                    .inner
                    .point_indexes
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (p, i))
                    .collect();
                let mut out = vec![Vec::new(); self.inner.len()];
                for (pid, seeds) in map {
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
                    if seeds.len() != sets[idx].len() {
                        return Err(PyValueError::new_err(format!(
                            "starting_keypoints[{pid}] has {} seeds but the view set has {} views",
                            seeds.len(),
                            sets[idx].len(),
                        )));
                    }
                    out[idx] = seeds.clone();
                }
                Some(out)
            }
        };

        let references = reference_positions(
            reference_images.as_ref(),
            &self.inner.point_indexes,
            &sets,
            n_images,
        )?;

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            localize_patch_cloud_keypoints(
                &self.inner,
                &views,
                &sets,
                seeds_per_patch.as_deref(),
                references.as_deref(),
                &params,
                progress_handle.as_deref(),
                &Progress::none(),
            )
            .expect("Progress::none never cancels")
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
            d.set_item("zncc", res.zncc.clone().into_pyarray(py))?;
            d.set_item("reference_image", res.reference)?;
            out.push(d);
        }
        Ok(out)
    }
}
