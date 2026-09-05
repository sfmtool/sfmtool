// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `PatchCloud.validate_member_coherence`: pairwise track-member agreement and
//! the verdict read off it.

use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::patch::member_coherence::{
    member_keypoints_from_reconstruction, member_views_from_reconstruction,
    validate_patch_cloud_member_coherence, MemberCoherenceParams, MemberVerdict,
};
use sfmtool_core::patch::normal_refine::ProjectedImage;

use super::args::{parse_patch_window, parse_sampler};
use super::cloud::PyPatchCloud;
use super::views::{resolve_pyramids, resolve_scene};
use crate::ProgressCounter;

/// The `verdict` string a [`MemberVerdict`] reports as.
fn verdict_name(v: MemberVerdict) -> &'static str {
    match v {
        MemberVerdict::KeepAll => "keep_all",
        MemberVerdict::Split => "split",
        MemberVerdict::Retire => "retire",
    }
}

#[pymethods]
impl PyPatchCloud {
    /// Validate each point's track against itself: render every member's patch
    /// through the point's own frame, correlate every **pair** of members, and
    /// read a verdict off that matrix. See
    /// ``specs/core/patch/member-coherence-validation.md``.
    ///
    /// The pairwise matrix sees a member that images a different surface even
    /// when the fused cross-view consensus does not — on a balanced split the
    /// consensus is a compromise blend that flatters both sides.
    ///
    /// Args:
    ///     recon: The reconstruction the cloud was built from (provides cameras,
    ///         poses, and the per-point member lists via ``point_indexes``),
    ///         **or** a :class:`CameraViews` — which carries no tracks, so
    ///         ``member_views`` becomes required.
    ///     images: One source image (HxWxC uint8 numpy array) per reconstruction
    ///         image, parallel to ``recon`` (index = image index), **or** an
    ///         :class:`ImagePyramidSet` prebuilt from those images.
    ///     bar: Pairwise ZNCC at or above which two members agree (default 0.65;
    ///         calibrated for the default window / sampler / resolution).
    ///     margin_gate: Separation-margin floor a cut must clear (default 0.05);
    ///         below it the track is a drift chain and is kept whole. An upper
    ///         bound — ``self_bar_k`` lowers it per track, see below.
    ///     self_bar_k: Strength of the **self-normalized admission bar** (default
    ///         1.5): how many units of a track's own core scatter the effective
    ///         bar sits below its core centre. ``bar`` and ``margin_gate`` are
    ///         absolute and catch a member imaging a *different* surface (0.2-0.5
    ///         against the core); they do not catch a member imaging an
    ///         *occluder in front of the same repeating texture*, which shares the
    ///         core's structure at 0.85-0.95 while the core agrees with itself at
    ///         0.99. So each track re-derives its own thresholds from the
    ///         intra-block agreement of the block admitted at ``bar``:
    ///         ``effective_bar = max(bar, min(centre - self_bar_k * scatter,
    ///         0.99))`` and ``effective_margin_gate = min(margin_gate, scatter)``,
    ///         one tighten pass, never iterated. A noisy or drifting track has a
    ///         large scatter and collapses back to the absolute pair. ``0``
    ///         disables the relative term entirely (the absolute rule, bit for
    ///         bit); a block of three or fewer members has too few pairs to
    ///         estimate from and leaves it inactive. It trades occlusion recall
    ///         against collateral on legitimately-marginal members (blur,
    ///         exposure, obliquity) trailing a tight core.
    ///     exoneration_ratio: Strength of **multi-scale exoneration** (default
    ///         0.90), which is what pays that trade back. A member the relative
    ///         term alone would evict has its agreement deficit re-measured on a
    ///         half-scale box-downsampled copy of the same renders; the quotient of
    ///         that to the full-scale deficit — the ``retained_deficit`` —
    ///         separates a *structural* disagreement, which survives the loss of
    ///         the detail (an occluder), from a *spectral* one, which is made of
    ///         it (a soft frame). The ratio runs high — the test is whether the
    ///         disagreement *survives* one halving, not whether it decays — so the
    ///         threshold does too. A member at or below this ratio is spared. Only
    ///         the relative term's evictions are exonerable: a member the absolute
    ///         ``bar`` rejects images a different thing, and blur is not a defence
    ///         against that. ``0`` disables it; it is inert whenever the relative
    ///         term is.
    ///     resolution: The R×R patch grid members are rendered and correlated on.
    ///     window: Per-pixel scoring weight — ``"gaussian_disk"`` (default),
    ///         ``"gaussian"``, or ``"uniform"``.
    ///     window_sigma: Window sigma for the gaussian windows.
    ///     sampler: ``"bilinear_mip"`` (default), ``"bilinear"``, or
    ///         ``"anisotropic"``.
    ///     min_valid_fraction: Per-member floor on the window-weighted valid-pixel
    ///         fraction; a member below it is left unscored.
    ///     min_support_pixels: Floor on the **common** support — the pixels valid
    ///         in every scoreable member, which every pairwise ZNCC is computed
    ///         over. A track below it is left entirely unscored (and so kept). The
    ///         default 8 is the floor the shared support builder already enforces,
    ///         so it changes nothing; raise it when vetting wide-baseline tracks,
    ///         where the intersection can shrink to a sliver of the R×R grid.
    ///     point_indexes: If given, validate only the patches with these source
    ///         point ids; ``None`` (default) validates every patch.
    ///     member_views: Optional mapping ``point_index -> [image_index, ...]``
    ///         giving each point's member list — the role the track observations
    ///         play in reconstruction mode. **Required** when the first argument is
    ///         a :class:`CameraViews`; with a reconstruction it *overrides* the
    ///         track-derived list for the points present in the map.
    ///     keypoint_anchor: Render each member **anchored at its stored keypoint**
    ///         (default ``True``) rather than at the point's reprojection — the
    ///         appearance that was actually matched in that image, so a member's
    ///         reprojection residual does not deflate the pairwise scores it takes
    ///         part in. Members with no stored keypoint (a ``sift_files``
    ///         reconstruction, a :class:`CameraViews` scene, or a ``member_views``
    ///         entry naming an image the point does not observe) fall back to
    ///         projection anchoring individually. ``False`` anchors every member at
    ///         its projection. ``bar`` is calibrated per anchoring — the two are not
    ///         the same score.
    ///     return_matrix: Also return the per-point ``zncc`` matrix (default
    ///         ``False`` — it is ``k×k`` per point).
    ///
    /// Returns a list of per-patch dicts (in cloud order, patches excluded by
    /// ``point_indexes`` omitted): ``point_index`` (int), ``members`` (1-D uint32
    /// numpy array of image indices, deduplicated first-seen-wins — every other
    /// per-member array is parallel to it), ``verdict`` (``"keep_all"`` /
    /// ``"split"`` / ``"retire"``), ``kept`` (1-D bool numpy array — the members
    /// the point keeps; all-``False`` on a retirement), ``block`` (1-D bool numpy
    /// array — the winning max-support block over the *scored* members;
    /// informational on a retirement), ``scored`` (1-D bool numpy array — which
    /// members carry pairwise evidence), ``support`` (int — the block size, ``0``
    /// when fewer than two members scored), ``n_support`` (int — the common-support
    /// pixel count every pairwise ZNCC was taken over; one number per point,
    /// because the support is frozen per point rather than per pair),
    /// ``margin``, ``min_intra``, ``max_cross`` (floats, NaN where undefined),
    /// ``effective_bar`` / ``effective_margin_gate`` (the thresholds the block
    /// sweep and the margin test actually ran at — equal to ``bar`` /
    /// ``margin_gate`` when the relative term is off or inactive, NaN when no
    /// sweep ran; ``effective_bar > bar`` is exactly "the relative term
    /// engaged"), ``core_center`` / ``core_scatter`` (the statistics they were
    /// derived from, NaN when inactive), ``relative_flagged`` / ``exonerated``
    /// (1-D bool numpy arrays — the members the relative term alone put outside
    /// the block, and the subset multi-scale exoneration spared; an exonerated
    /// member is in ``kept`` and not in ``block``), ``retained_deficit`` (1-D
    /// float64 — the flagged members' coarse-over-full deficit ratio, NaN
    /// elsewhere and where undefined), ``sharpness_deficit`` (1-D float64 —
    /// photometric sharpness relative to the track consensus, for **every**
    /// scored member: the part of its agreement deficit that exists only at fine
    /// scale, so ``0`` is scale-free and larger is softer), and — with
    /// ``return_matrix=True`` — ``zncc`` (``k×k`` float64 numpy array, unit
    /// diagonal, NaN for uncorrelatable pairs), ``zncc_coarse`` (a list of the
    /// same-shaped tables at the coarse scales, coarsest last) and
    /// ``coarse_factors`` (their downsampling factors).
    ///
    /// **Unscored members** — nothing rendered them, or nothing could be
    /// correlated with them — sit outside the decision rule entirely: the block
    /// sweep, the margin and the majority denominator all run over the scored
    /// members only, and an unscored member is passed through in ``kept`` (a
    /// ``"retire"`` still ships nothing: the point itself is refused).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        recon, images, *, bar=0.65, margin_gate=0.05, self_bar_k=1.5, exoneration_ratio=0.90,
        resolution=24,
        window="gaussian_disk", window_sigma=0.6, sampler="bilinear_mip",
        min_valid_fraction=0.6, min_support_pixels=8,
        point_indexes=None, member_views=None, keypoint_anchor=true, return_matrix=false,
        progress=None
    ))]
    fn validate_member_coherence<'py>(
        &self,
        py: Python<'py>,
        recon: &Bound<'py, PyAny>,
        images: &Bound<'py, PyAny>,
        bar: f64,
        margin_gate: f64,
        self_bar_k: f64,
        exoneration_ratio: f64,
        resolution: u32,
        window: &str,
        window_sigma: f64,
        sampler: &str,
        min_valid_fraction: f64,
        min_support_pixels: u32,
        point_indexes: Option<Vec<u32>>,
        member_views: Option<std::collections::HashMap<u32, Vec<u32>>>,
        keypoint_anchor: bool,
        return_matrix: bool,
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
        if recon_opt.is_none() && member_views.is_none() {
            return Err(PyValueError::new_err(
                "member_views is required when the first argument is a CameraViews \
                 (there are no tracks to derive per-patch member views from)",
            ));
        }

        let window = parse_patch_window(window, window_sigma)?;
        let sampler = parse_sampler(sampler)?;
        let params = MemberCoherenceParams {
            bar,
            margin_gate,
            resolution,
            window,
            sampler,
            min_valid_fraction,
            min_support_pixels,
            self_bar_k,
            exoneration_ratio,
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

        // Per-patch member lists. In reconstruction mode these default to the track
        // observations; `member_views` overrides the listed points (and is the only
        // source in views mode). An empty list makes a patch's validation trivially
        // empty, so `point_indexes` selects a subset by clearing the rest.
        let mut members = match recon_opt {
            Some(recon) => member_views_from_reconstruction(recon, &self.inner),
            None => vec![Vec::new(); self.inner.len()],
        };
        // Per-member stored keypoints, parallel to `members`. Only a
        // reconstruction has them; a `CameraViews` scene renders at projections.
        let mut keypoints: Option<Vec<Vec<Option<[f64; 2]>>>> = match (keypoint_anchor, recon_opt) {
            (true, Some(recon)) => Some(member_keypoints_from_reconstruction(recon, &self.inner)),
            _ => None,
        };
        if let Some(map) = &member_views {
            for vs in map.values() {
                if let Some(&bad) = vs.iter().find(|&&i| i >= n_images) {
                    return Err(PyValueError::new_err(format!(
                        "member_views contains image index {bad} out of range for this \
                         scene's {n_images} views"
                    )));
                }
            }
            for (i, &pid) in self.inner.point_indexes.iter().enumerate() {
                let Some(vs) = map.get(&pid) else { continue };
                // An overridden member list is re-keyed against the point's own
                // track: an image it observes contributes that observation's stored
                // keypoint, an image it does not is anchored at its projection.
                if let Some(kps) = keypoints.as_mut() {
                    let track = &members[i];
                    let own = &kps[i];
                    kps[i] = vs
                        .iter()
                        .map(|img| {
                            track
                                .iter()
                                .position(|t| t == img)
                                .and_then(|at| own.get(at).copied().flatten())
                        })
                        .collect();
                }
                members[i] = vs.clone();
            }
        }
        let selected_mask: Option<std::collections::HashSet<u32>> =
            point_indexes.map(|ids| ids.into_iter().collect());
        if let Some(keep) = &selected_mask {
            for (i, &pid) in self.inner.point_indexes.iter().enumerate() {
                if !keep.contains(&pid) {
                    members[i].clear();
                    if let Some(kps) = keypoints.as_mut() {
                        kps[i].clear();
                    }
                }
            }
        }

        let progress_handle = progress.as_ref().map(|p| p.handle());
        let results = py.detach(|| {
            validate_patch_cloud_member_coherence(
                &self.inner,
                &views,
                &members,
                keypoints.as_deref(),
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
            let k = res.matrix.len();
            let d = PyDict::new(py);
            d.set_item("point_index", pid)?;
            d.set_item("members", res.matrix.members.clone().into_pyarray(py))?;
            d.set_item("verdict", verdict_name(res.decision.verdict))?;
            d.set_item("kept", res.decision.kept.clone().into_pyarray(py))?;
            d.set_item("block", res.decision.block.clone().into_pyarray(py))?;
            d.set_item("scored", res.matrix.scored.clone().into_pyarray(py))?;
            d.set_item("support", res.decision.support)?;
            d.set_item("n_support", res.matrix.n_support)?;
            d.set_item("margin", res.decision.margin)?;
            d.set_item("min_intra", res.decision.min_intra)?;
            d.set_item("max_cross", res.decision.max_cross)?;
            d.set_item("effective_bar", res.decision.effective_bar)?;
            d.set_item("effective_margin_gate", res.decision.effective_margin_gate)?;
            d.set_item("core_center", res.decision.core_center)?;
            d.set_item("core_scatter", res.decision.core_scatter)?;
            d.set_item(
                "relative_flagged",
                res.decision.relative_flagged.clone().into_pyarray(py),
            )?;
            d.set_item(
                "exonerated",
                res.decision.exonerated.clone().into_pyarray(py),
            )?;
            d.set_item(
                "retained_deficit",
                res.decision.retained_deficit.clone().into_pyarray(py),
            )?;
            d.set_item(
                "sharpness_deficit",
                res.decision.sharpness_deficit.clone().into_pyarray(py),
            )?;
            if return_matrix {
                let arr = Array2::from_shape_vec((k, k), res.matrix.zncc.clone())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                d.set_item("zncc", arr.into_pyarray(py))?;
                let coarse: Vec<Bound<'py, _>> = res
                    .matrix
                    .zncc_coarse
                    .iter()
                    .map(|t| {
                        Array2::from_shape_vec((k, k), t.clone())
                            .map(|a| a.into_pyarray(py))
                            .map_err(|e| PyValueError::new_err(e.to_string()))
                    })
                    .collect::<PyResult<_>>()?;
                d.set_item("zncc_coarse", coarse)?;
                d.set_item("coarse_factors", res.matrix.coarse_factors.clone())?;
            }
            out.push(d);
        }
        Ok(out)
    }
}
